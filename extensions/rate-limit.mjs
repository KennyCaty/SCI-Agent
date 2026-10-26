import {setTimeout as delay} from 'node:timers/promises';
export function retryPolicy(value) {
  if(!value?.enabled)return {enabled:false};
  const policy={enabled:true,min_delay_seconds:1,max_delay_seconds:20,max_attempts:100,max_elapsed_seconds:900,...value};
  if(!Number.isInteger(policy.max_attempts)||policy.max_attempts<1||policy.max_attempts>100||
    !Number.isFinite(policy.min_delay_seconds)||policy.min_delay_seconds<0||
    !Number.isFinite(policy.max_delay_seconds)||policy.max_delay_seconds<policy.min_delay_seconds||
    !Number.isFinite(policy.max_elapsed_seconds)||policy.max_elapsed_seconds<=0)throw new Error('Invalid rate-limit retry policy');
  return policy;
}
export function classifyLimit(status,text) {
  const s=text.toLowerCase();
  if([401,402,403].includes(status))return null;
  if(/rate_limit_exceeded/.test(s)&&!/insufficient_quota|billing_hard_limit|invalid_api_key/.test(s))return 'rate_limit';
  // Quota exhaustion is not a transient rate limit.
  if([401,402,403].includes(status)||/quota|余额|balance|credit|billing|invalid[_ -]api[_ -]key|authentication|unauthorized|policy|content[_ -]filter|safety|moderation|potentially violating|invalid prompt flagged/.test(s))return null;
  if(status===0&&s==='transport_connection_failed')return 'transport_connection';
  if(/upstream_error/.test(s)&&/upstream request failed/.test(s))return 'server_retryable';
  if([500,502,503,504].includes(status)&&s.includes('an error occurred while processing your request')&&s.includes('you can retry your request'))return 'server_retryable';
  if(status===429)return 'rate_limit';
  if(status===524)return 'gateway_timeout';
  if(status===503&&/rate[_ :.-]*limit|too (?:many|frequent)|too frequently|频率|限流/.test(s))return 'rate_limit';
  if(status===503&&/overload|capacity|temporarily unavailable/.test(s))return 'overload';
  return null;
}
export function retryAfterSeconds(value,now=Date.now()) {
  if(!value)return 0;
  const numeric=Number(value);
  if(Number.isFinite(numeric)&&numeric>=0)return numeric;
  const timestamp=Date.parse(value);return Number.isFinite(timestamp)?Math.max(0,(timestamp-now)/1000):0;
}
export function jitterDelay(policy,_serverSeconds=0,random=Math.random) {
  // Use the configured wait interval; keep Retry-After for diagnostics.
  const low=policy.min_delay_seconds,high=policy.max_delay_seconds;
  return low+Math.max(0,Math.min(1,random()))*(high-low);
}
export async function peekStreamStart(reader) {
  const prefix=[],decoder=new TextDecoder();let text='',responses=false,completed=false;
  for(;;) {
    const {done,value}=await reader.read();
    if(done){if(responses&&!completed)throw new Error('Incomplete Responses stream');return {prefix};}
    prefix.push(value);text+=decoder.decode(value,{stream:true});
    let end;
    while((end=text.search(/\r?\n\r?\n/))>=0) {
      const block=text.slice(0,end),separator=text.slice(end).match(/^\r?\n\r?\n/)[0];text=text.slice(end+separator.length);
      const data=block.split(/\r?\n/).filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n');
      if(!data)continue;
      if(data==='[DONE]'){if(responses&&!completed)throw new Error('Incomplete Responses stream');return {prefix};}
      let frame;try{frame=JSON.parse(data);}catch{return {prefix};}
      if(frame.type?.startsWith('response.'))responses=true;
      if(frame.type==='response.completed')completed=true;
      if(frame.type==='response.incomplete')throw new Error('Incomplete Responses generation');
      const error=frame.error??frame.response?.error;
      if(error)return {prefix,retryHint:{status:503,text:JSON.stringify(error)}};
      // Buffer the stream so a retry cannot expose partial content or tool calls.
    }
  }
  return {prefix};
}
// Retry the same HTTP request without appending failed content to history.
export async function withRateLimitRetry(perform,config,{signal,onEvent=()=>{},onExcluded=()=>{},initialRetryState=null,random=Math.random,wait=delay,now=()=>performance.now()}={}) {
  const policy=retryPolicy(config),previousAttempts=initialRetryState?.attempts??0,previousSeconds=initialRetryState?.elapsed_seconds??0,started=now()-previousSeconds*1000;
  if(policy.enabled&&(previousAttempts>=policy.max_attempts||previousSeconds>=policy.max_elapsed_seconds))throw new Error('Saved identical-request retry budget exhausted; no new request');
  for(let attempt=previousAttempts+1;;attempt++) {
    signal?.throwIfAborted();
    const callStart=now();let item;
    try {item=await perform(attempt,policy.enabled?Math.max(.001,policy.max_elapsed_seconds-(now()-started)/1000):null);}
    catch(error) {
      if(!policy.enabled||signal?.aborted||(!error.api_timeout&&(error.name!=='TypeError'||!['fetch failed','terminated'].includes(error.message))))throw error;
      item={index:error.physical_index,settled:Promise.resolve(),transportFailure:true,
        retryHint:error.api_timeout?{status:524,text:'transport_request_timeout'}:{status:0,text:'transport_connection_failed'},
        response:new Response(JSON.stringify({error:{message:'Connection error: fetch failed'}}),{status:503,headers:{'content-type':'application/json','x-should-retry':'false'}})};
    }
    if(!policy.enabled||(item.response.ok&&!item.retryHint))return item.response;
    let text;
    try {text=item.retryHint?.text??await item.response.clone().text();}catch{return item.response;}
    const kind=classifyLimit(item.retryHint?.status??item.response.status,text);
    if(!kind)return item.response;
    const serverDelay=retryAfterSeconds(item.response.headers.get('retry-after'));
    const seconds=jitterDelay(policy,serverDelay,random);
    if(attempt>=policy.max_attempts||(now()-started)/1000+seconds>=policy.max_elapsed_seconds) {
      onEvent({type:'retry_exhausted',physical_index:item.index,attempt,kind,http_status:item.transportFailure?null:item.response.status,reason:attempt>=policy.max_attempts?'max_attempts':'max_elapsed_seconds'});
      return item.response;
    }
    await item.settled;
    await item.response.body?.cancel();
    const failedSeconds=(now()-callStart)/1000;
    onExcluded({kind:'limited_request',seconds:failedSeconds,physical_index:item.index});
    const record={type:'retry_wait',physical_index:item.index,attempt,kind,http_status:item.transportFailure?null:item.response.status,
      retry_after_seconds:serverDelay,retry_after_applied:false,delay_seconds:seconds,configured_min_seconds:policy.min_delay_seconds,configured_max_seconds:policy.max_delay_seconds,
      strategy:'uniform configured interval; Retry-After logged only; no exponential growth; identical request body'};
    onEvent(record);
    const waitStart=now();
    try {await wait(seconds*1000,undefined,{signal});}
    finally {onExcluded({kind:'retry_wait',seconds:(now()-waitStart)/1000,physical_index:item.index});}
  }
}
