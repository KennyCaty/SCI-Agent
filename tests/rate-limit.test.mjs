import test from 'node:test';
import assert from 'node:assert/strict';
import {retryPolicy,classifyLimit,retryAfterSeconds,jitterDelay,withRateLimitRetry} from '../extensions/rate-limit.mjs';
import {BudgetClock} from '../extensions/budget.mjs';
const config={enabled:true,min_delay_seconds:45,max_delay_seconds:75,max_attempts:4,max_elapsed_seconds:900};
const response=(status,text,after)=>new Response(text,{status,headers:after?{'retry-after':after}:{}});
test('uniform jitter ignores advisory Retry-After and never grows exponentially',()=>{
 const defaults=retryPolicy({enabled:true});
 assert.equal(defaults.max_attempts,100);
 assert.equal(defaults.min_delay_seconds,1);assert.equal(defaults.max_delay_seconds,20);
 assert.deepEqual([0,.5,1].map(r=>jitterDelay(defaults,0,()=>r)),[1,10.5,20]);
 assert.deepEqual([0,.5,1].map(r=>jitterDelay(defaults,60,()=>r)),[1,10.5,20]);
 const p=retryPolicy(config);
 assert.deepEqual([0,.5,1].map(r=>jitterDelay(p,0,()=>r)),[45,60,75]);
 assert.deepEqual([0,.5,1].map(r=>jitterDelay(p,60,()=>r)),[45,60,75]);
 assert.equal(retryAfterSeconds('60'),60);
 assert.equal(retryAfterSeconds(new Date(65000).toUTCString(),5000),60);
 assert.throws(()=>retryPolicy({...config,max_attempts:0}));
});
test('only transient request limits or explicit overload retry; auth/quota/policy never do',()=>{
 assert.equal(classifyLimit(429,'too many requests'),'rate_limit');
 assert.equal(classifyLimit(503,'model:rate_limited'),'rate_limit');
 assert.equal(classifyLimit(503,'Our servers are currently overloaded'),'overload');
 assert.equal(classifyLimit(524,''),'gateway_timeout');
 assert.equal(classifyLimit(524,'origin_response_timeout'),'gateway_timeout');
 assert.equal(classifyLimit(524,'insufficient_quota'),null);
 for(const [status,text] of [[401,'rate limit'],[403,'rate limit'],[429,'insufficient_quota'],[429,'balance low'],[429,'Invalid prompt flagged potentially violating usage policy'],[500,'server error'],[400,'bad request']])assert.equal(classifyLimit(status,text),null);
});
test('524 retries identical call with bounded jitter and excludes timeout plus wait',async()=>{
 let clock=0;const events=[],excluded=[],waits=[];
 const result=await withRateLimitRetry(async n=>{
  clock+=n===1?120000:200;
  return {index:n,settled:Promise.resolve(),response:response(n===1?524:200,n===1?'origin_response_timeout':'ok','120')};
 },retryPolicy({enabled:true}),{now:()=>clock,random:()=>.5,wait:async ms=>{waits.push(ms);clock+=ms;},onEvent:e=>events.push(e),onExcluded:e=>excluded.push(e)});
 assert.equal(await result.text(),'ok');assert.deepEqual(waits,[10500]);
 assert.equal(events[0].kind,'gateway_timeout');assert.equal(events[0].retry_after_applied,false);
 assert.equal(excluded.reduce((sum,e)=>sum+e.seconds,0),130.5);
});
test('retry sequence is bounded, uses constant jitter, and retains each physical attempt',async()=>{
 let clock=0;const attempts=[],events=[],excluded=[],waits=[];
 const result=await withRateLimitRetry(async n=>{attempts.push(n);clock+=200;return {index:n,settled:Promise.resolve(),response:response(n<3?503:200,n<3?'model:rate_limited':'ok')};},config,
 {now:()=>clock,random:()=>.5,wait:async ms=>{waits.push(ms);clock+=ms;},onEvent:e=>events.push(e),onExcluded:r=>excluded.push(r)});
 assert.equal(await result.text(),'ok');assert.deepEqual(attempts,[1,2,3]);assert.deepEqual(waits,[60000,60000]);
 assert.equal(events.length,2);assert.equal(excluded.filter(e=>e.kind==='limited_request').length,2);
 let calls=0;const stopEvents=[];
 const failure=await withRateLimitRetry(async n=>{calls++;return {index:n,settled:Promise.resolve(),response:response(429,'rate limited')};},{...config,max_attempts:2,min_delay_seconds:0,max_delay_seconds:0},
 {wait:async()=>{},onEvent:e=>stopEvents.push(e)});
 assert.equal(failure.status,429);assert.equal(calls,2);assert.equal(stopEvents.at(-1).type,'retry_exhausted');
});
test('wall limit and cancellation stop retry without dispatching extra requests',async()=>{
 let calls=0;
 await withRateLimitRetry(async n=>{calls++;return {index:n,response:response(503,'rate_limited'),settled:Promise.resolve()};},{...config,max_elapsed_seconds:1});
 assert.equal(calls,1);
 calls=0;const stopped=[],delays=[];
 await withRateLimitRetry(async n=>{calls++;return {index:n,response:response(n===1?503:200,n===1?'rate_limited':'ok','60'),settled:Promise.resolve()};},retryPolicy({enabled:true}),{random:()=>.5,wait:async ms=>delays.push(ms),onEvent:e=>stopped.push(e)});
 assert.equal(calls,2);assert.deepEqual(delays,[10500]);assert.equal(stopped[0].retry_after_seconds,60);assert.equal(stopped[0].retry_after_applied,false);
 const controller=new AbortController();calls=0;
 await assert.rejects(withRateLimitRetry(async n=>{calls++;return {index:n,response:response(503,'rate_limited'),settled:Promise.resolve()};},config,
 {signal:controller.signal,wait:async()=>{controller.abort();controller.signal.throwIfAborted();}}),/abort/i);
 assert.equal(calls,1);
});
test('success and exhausted failure deduct retry spans exactly once',()=>{
 for(const outcome of ['stop','error']) {
  let time=0;const clock=new BudgetClock(1,()=>time);clock.start();clock.beginCall();
  time=200;clock.excludeRetrySpan({kind:'limited_request',seconds:.2});
  time=1000;clock.excludeRetrySpan({kind:'retry_wait',seconds:.8});
  time=1300;clock.finishCall(outcome);
  assert.ok(Math.abs(clock.snapshot().normal_seconds-(outcome==='stop'?.3:0))<1e-10);
  assert.equal(clock.failedSeconds,outcome==='stop'?1:1.3);
 }
});

test('fetch connection failure retries visibly and excludes failure plus backoff without retrying cancellation',async()=>{
 let calls=0,t=0;const excluded=[],events=[];
 const r=await withRateLimitRetry(async()=>{calls++;if(calls===1){t+=250;const e=new TypeError('fetch failed');e.physical_index=7;throw e;}return {response:new Response('ok')};},{...config,min_delay_seconds:1,max_delay_seconds:1},{now:()=>t,wait:async ms=>{t+=ms},onExcluded:e=>excluded.push(e),onEvent:e=>events.push(e)});
 assert.equal(await r.text(),'ok');assert.equal(calls,2);assert.equal(events[0].http_status,null);assert.equal(events[0].kind,'transport_connection');assert.equal(excluded.reduce((s,x)=>s+x.seconds,0),1.25);
 await assert.rejects(withRateLimitRetry(async()=>{throw new Error('Authentication failed')},config),/Authentication/);
 const c=new AbortController();c.abort();await assert.rejects(withRateLimitRetry(async()=>{throw new TypeError('fetch failed')},config,{signal:c.signal}));
});

test('midstream SSE classification preserves success bytes and never retries hard errors',async()=>{
 const {peekStreamStart}=await import('../extensions/rate-limit.mjs');
 const frame=x=>'data: '+JSON.stringify(x)+'\n\n';
 const first=frame({choices:[{delta:{reasoning_content:'some thought'},finish_reason:null}]});
 for(const message of ['Our servers are currently overloaded','insufficient_quota','Unauthorized','content_filter']){
  const bytes=first+frame({error:{message}})+'data: [DONE]\n\n';
  const result=await peekStreamStart(new Response(bytes).body.getReader());
  assert.equal(result.retryHint.text,JSON.stringify({message}));
  assert.equal(classifyLimit(result.retryHint.status,result.retryHint.text),message.includes('overloaded')?'overload':null);
 }
 const bytes=first+frame({choices:[{delta:{content:'final'},finish_reason:'stop'}]})+'data: [DONE]\n\n';
 const result=await peekStreamStart(new Response(bytes).body.getReader());
 assert.equal(result.retryHint,undefined);assert.equal(Buffer.concat(result.prefix).toString(),bytes);
});

test('request timeout retries visibly, inherits retry limits, and never retries caller cancellation',async()=>{
 let clock=0,calls=0;const events=[],excluded=[];
 const result=await withRateLimitRetry(async n=>{calls++;assert.equal(n,calls+2);if(calls===1){clock+=300000;const e=new DOMException('request timed out','TimeoutError');e.api_timeout=true;e.physical_index=1;throw e;}clock+=100;return {response:new Response('ok')};},{...config,min_delay_seconds:1,max_delay_seconds:1},{initialRetryState:{attempts:2,elapsed_seconds:310},now:()=>clock,wait:async ms=>{clock+=ms},onEvent:e=>events.push(e),onExcluded:e=>excluded.push(e)});
 assert.equal(await result.text(),'ok');assert.equal(calls,2);assert.equal(events[0].kind,'gateway_timeout');assert.equal(excluded.reduce((s,x)=>s+x.seconds,0),301);
 await assert.rejects(withRateLimitRetry(async()=>{throw new Error('must not dispatch')},config,{initialRetryState:{attempts:4,elapsed_seconds:2}}),/exhausted/);
 await assert.rejects(withRateLimitRetry(async()=>{throw new Error('must not dispatch')},config,{initialRetryState:{attempts:1,elapsed_seconds:900}}),/exhausted/);
 const c=new AbortController();await assert.rejects(withRateLimitRetry(async()=>{c.abort();const e=new DOMException('aborted','AbortError');e.api_timeout=true;throw e;},config,{signal:c.signal}),/aborted/);
});

test('explicit server retry invitation retries identically without treating policy errors as transient',async()=>{
 const text='An error occurred while processing your request. You can retry your request';
 assert.equal(classifyLimit(503,text),'server_retryable');assert.equal(classifyLimit(403,text),null);assert.equal(classifyLimit(503,text+' policy'),null);
 let calls=0;const exclusions=[];const r=await withRateLimitRetry(async()=>({response:++calls===1?response(503,text):response(200,'ok'),settled:Promise.resolve()}),config,{wait:async()=>{},onExcluded:e=>exclusions.push(e)});
 assert.equal(calls,2);assert.equal(await r.text(),'ok');assert.equal(exclusions.length,2);
});

// The socket can close after HTTP 200, during stream consumption.
test('retries a terminated response stream before exposing partial content',async()=>{
 let calls=0;const events=[];
 const r=await withRateLimitRetry(async()=>{
  calls++;if(calls===1){const e=new TypeError('terminated');e.physical_index=19;throw e;}
  return {response:new Response('recovered')};
 },{enabled:true,min_delay_seconds:0,max_delay_seconds:0,max_attempts:3,max_elapsed_seconds:30},{wait:async()=>{},onEvent:e=>events.push(e)});
 assert.equal(await r.text(),'recovered');assert.equal(calls,2);
 assert.equal(events[0].kind,'transport_connection');assert.equal(events[0].physical_index,19);
});

test('explicit upstream request failure retries while hard-error guards win',()=>{
 assert.equal(classifyLimit(503,'upstream_error: Upstream request failed'),'server_retryable');
 assert.equal(classifyLimit(503,'upstream_error: Upstream request failed; insufficient_quota'),null);
 assert.equal(classifyLimit(503,'upstream_error: Upstream request failed; content_filter'),null);
 assert.equal(classifyLimit(401,'upstream_error: Upstream request failed'),null);
});
