import {parseArgs} from 'node:util';

export function validateInterval(value) {
  if(typeof value!=='number'||!Number.isFinite(value)||value<0)throw new Error('Request interval must be a finite number >= 0 seconds');
  return value;
}
export function applyRuntimeOptions(config,{requestIntervalSeconds,api}={}) {
  const effective={...config};
  if(requestIntervalSeconds!==undefined)effective.minimum_request_interval_seconds=validateInterval(requestIntervalSeconds);
  if(api!==undefined)effective.api=api;
  validateInterval(effective.minimum_request_interval_seconds??0);
  if(!['openai-completions','openai-responses'].includes(effective.api??'openai-completions'))throw new Error('Unsupported API: use openai-responses or openai-completions');
  if(effective.responses_system_role!==undefined&&!['system','developer'].includes(effective.responses_system_role))throw new Error('Invalid Responses system role');
  return effective;
}
export function parseRuntimeArgs(args,{batch=false}={}) {
  const options={'request-interval-seconds':{type:'string'},api:{type:'string'},
    ...(batch?{concurrency:{type:'string'}}:{resume:{type:'string'}})};
  const {values,positionals}=parseArgs({args,options,allowPositionals:true,strict:true});
  const runtime={};
  if(values['request-interval-seconds']!==undefined){
    if(!values['request-interval-seconds'].trim())throw new Error('Empty request interval');
    runtime.requestIntervalSeconds=validateInterval(Number(values['request-interval-seconds']));
  }
  if(values.api!==undefined)runtime.api=values.api;
  applyRuntimeOptions({},runtime);
  if(batch){
    if(positionals.length<1||positionals.length>2)throw new Error('Expected RUN_DIRECTORY [CONCURRENCY]');
    if(values.concurrency!==undefined&&positionals.length===2)throw new Error('Specify concurrency once, either positional or --concurrency');
    const concurrency=Number(values.concurrency??positionals[1]??1);
    if(!Number.isInteger(concurrency)||concurrency<1||concurrency>32)throw new Error('Concurrency must be 1..32');
    return {root:positionals[0],concurrency,...runtime};
  }
  if(positionals.length!==3)throw new Error('Expected CONFIG.json QUESTION.txt NEW_OUTPUT_DIRECTORY');
  return {positionals,resumeFrom:values.resume??null,...runtime};
}
