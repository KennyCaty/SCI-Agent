import {mkdirSync, writeFileSync, appendFileSync} from 'node:fs';
import {resolve} from 'node:path';
import assert from 'node:assert/strict';
import {CaseStore} from './storage.mjs';
import {withRateLimitRetry,retryPolicy,peekStreamStart} from './rate-limit.mjs';

// Record provider SSE before usage normalization; omit headers and keys.
export function installWireLog(folder, baseUrl, options={}) {
  const store=options.store ?? new CaseStore(folder),completions=[];
  const original = globalThis.fetch;
  const endpoint = baseUrl.replace(/\/$/,'') + (options.api==='openai-responses'?'/responses':'/chat/completions');
  let count = 0, logicalCount=0;
  const policy=retryPolicy(options.retry);
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (url !== endpoint) return original(input,init);
    // Disable SDK retries; request retries are handled here.
    const requestHeaders = new Headers(init?.headers ?? (input instanceof Request ? input.headers : {}));
    if (Number(requestHeaders.get('x-stainless-retry-count') ?? 0) > 0) {
      throw new Error('Automatic transport retry disabled for this CEP attempt');
    }
    const body = init?.body ?? (input instanceof Request ? await input.clone().text() : null);
    const logical=++logicalCount,projectionIndex=options.currentContext?.()??null;
    if(logical===1&&options.expectedFirstRequest){
      assert.deepEqual(JSON.parse(body),options.expectedFirstRequest,'Resumed first request differs from saved failed request');
      writeFileSync(resolve(folder,'RESUME_REQUEST_VERIFICATION.json'),JSON.stringify({status:'PASS',same_request:true,at:new Date().toISOString()}));
    }
    const inheritedSignal=init?.signal??(input instanceof Request?input.signal:undefined);
    const perform=async (attempt,remainingSeconds)=>{
      const index=++count;
      const dest = resolve(folder,'api',String(index).padStart(4,'0')); mkdirSync(dest,{recursive:true});
      let settled;
      const completion=new Promise(r=>{settled=r;});completions.push(completion);
      if (typeof body !== 'string') {settled();throw new Error('Expected JSON provider request');}
      writeFileSync(resolve(dest,'request.ref.json'),JSON.stringify(store.putBytes(Buffer.from(body))),{mode:0o600});
      writeFileSync(resolve(dest,'request.meta.json'),JSON.stringify({context_projection_index:projectionIndex,logical_request_index:logical,physical_attempt:attempt}));
      const started = Date.now();
      const timeout=policy.enabled?AbortSignal.timeout(Math.max(1,Math.ceil(Math.min(options.apiTimeoutSeconds??300,remainingSeconds)*1000))):null;
      try {
        const signal=timeout?(inheritedSignal?AbortSignal.any([inheritedSignal,timeout]):timeout):inheritedSignal;
        const requestInit=input instanceof Request?{method:input.method,headers:input.headers,...init,body,signal}:{...init,body,signal};
        const response = await original(input instanceof Request?input.url:input,requestInit);
        writeFileSync(resolve(dest,'status.json'),JSON.stringify({status:response.status,started_at:new Date(started).toISOString(),retry_after:response.headers.get('retry-after')}));
        const headers = new Headers(response.headers);
        if (!response.ok) headers.set('x-should-retry','false');
        if (!response.body) {settled();return {index,settled:completion,response:new Response(null,{status:response.status,statusText:response.statusText,headers})};}
        // Drain a separate stream branch to capture usage sent after an error event.
        const [recording,delivery]=response.body.tee();
        const reader = delivery.getReader(), rawReader=recording.getReader();
        const raw = resolve(dest,'response.sse');
        writeFileSync(raw,'',{mode:0o600});
        void (async()=>{
          try {
            for(;;){const {done,value}=await rawReader.read();if(done)break;appendFileSync(raw,value);}
            writeFileSync(resolve(dest,'END.json'),JSON.stringify({seconds:(Date.now()-started)/1000}));
          } catch(e) {
            writeFileSync(resolve(dest,'ERROR.json'),JSON.stringify({error:String(e),seconds:(Date.now()-started)/1000}));
          } finally {settled();}
        })();
        // Finish recording before exposing response deltas to Pi.
        const peek=(policy.enabled||options.api==='openai-responses')&&response.ok&&(headers.get('content-type')??'').includes('text/event-stream')
          ?await peekStreamStart(reader):{prefix:[]};
        let deliveryEnded=false;
        const stream = new ReadableStream({
          async pull(controller) {
            try {
              const {done,value} = peek.prefix.length?{done:false,value:peek.prefix.shift()}:await reader.read();
              if(deliveryEnded)return;
              if (done) {
                deliveryEnded=true;
                controller.close();
              } else { controller.enqueue(value); }
            } catch (e) {
              if(!deliveryEnded){deliveryEnded=true;controller.error(e);}
            }
          },
          async cancel(reason) {
            if(deliveryEnded)return;deliveryEnded=true;
            try {await reader.cancel(reason);}
            finally {writeFileSync(resolve(dest,'CANCELLED.json'),JSON.stringify({reason:String(reason),seconds:(Date.now()-started)/1000}));}
          },
        });
        return {index,settled:completion,retryHint:peek.retryHint,response:new Response(stream,{status:response.status,statusText:response.statusText,headers})};
      } catch (e) {
        writeFileSync(resolve(dest,'ERROR.json'),JSON.stringify({error:String(e),seconds:(Date.now()-started)/1000}));
        settled();
        e.physical_index=index;
        e.api_timeout=!!timeout?.aborted&&!inheritedSignal?.aborted;
        throw e;
      }
    };
    return withRateLimitRetry(perform,policy,{initialRetryState:logical===1?options.initialRetryState:null,signal:inheritedSignal,onExcluded:options.onRetryExcluded,
      onEvent:record=>{const event={...record,logical_request_index:logical,at:new Date().toISOString()};
        appendFileSync(resolve(folder,'rate_limit_retries.jsonl'),JSON.stringify(event)+'\n');options.onRetry?.(event);},
      random:options.retryRandom});
  };
  const restore=() => {globalThis.fetch=original;};
  restore.settled=()=>Promise.all(completions);
  return restore;
}
