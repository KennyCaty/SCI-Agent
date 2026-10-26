import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,readFileSync,writeFileSync,rmSync,existsSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {run} from '../extensions/run.mjs';
import {createBackend} from '../extensions/tools.mjs';
import {readApi,readTrajectory} from '../extensions/inspect.mjs';
import {readAccounting} from '../extensions/accounting.mjs';
import {prompt,schemas,ROOT} from '../extensions/paths.mjs';
import {classifyLimit,peekStreamStart} from '../extensions/rate-limit.mjs';

const usage={input_tokens:100,input_tokens_details:{cached_tokens:20,cache_write_tokens:70},output_tokens:10,output_tokens_details:{reasoning_tokens:8},total_tokens:110};
const reasoning={id:'rs_mock1',type:'reasoning',summary:[],encrypted_content:'synthetic-encrypted-content'};
const fc={id:'fc_mock1',type:'function_call',call_id:'call_mock1',name:'read',arguments:JSON.stringify({path:'evidence.txt'})};
const answer={id:'msg_mock2',type:'message',role:'assistant',status:'completed',content:[{type:'output_text',text:'Synthetic final answer.',annotations:[]}]};
const frame=(type,body)=>'event: '+type+'\ndata: '+JSON.stringify({type,...body})+'\n\n';
function frames(n,{model='gpt-6-luna',reasoningItem=reasoning,usageValue=usage}={}) {
  const response={id:'resp_mock'+n,model,status:'completed',usage:usageValue,output:n===1?[reasoningItem,fc]:[answer]};
  let text=frame('response.created',{response:{...response,status:'in_progress',usage:null,output:[]}});
  if(n===1){
    text+=frame('response.output_item.added',{output_index:0,item:{...reasoningItem,content:reasoningItem.content?[]:undefined,encrypted_content:undefined}});
    if(reasoningItem.content){
      text+=frame('response.content_part.added',{output_index:0,content_index:0,item_id:reasoningItem.id,part:{type:'reasoning_text',text:''}});
      text+=frame('response.reasoning_text.delta',{output_index:0,content_index:0,item_id:reasoningItem.id,delta:reasoningItem.content[0].text});
      text+=frame('response.reasoning_text.done',{output_index:0,content_index:0,item_id:reasoningItem.id,text:reasoningItem.content[0].text});
    }
    text+=frame('response.output_item.done',{output_index:0,item:reasoningItem});
    text+=frame('response.output_item.added',{output_index:1,item:{...fc,arguments:''}});
    text+=frame('response.function_call_arguments.delta',{output_index:1,item_id:fc.id,delta:fc.arguments});
    text+=frame('response.function_call_arguments.done',{output_index:1,item_id:fc.id,arguments:fc.arguments});
    text+=frame('response.output_item.done',{output_index:1,item:fc});
  }else{
    text+=frame('response.output_item.added',{output_index:0,item:{...answer,content:[]}});
    text+=frame('response.content_part.added',{output_index:0,content_index:0,item_id:answer.id,part:{type:'output_text',text:'',annotations:[]}});
    text+=frame('response.output_text.delta',{output_index:0,content_index:0,item_id:answer.id,delta:answer.content[0].text});
    text+=frame('response.output_item.done',{output_index:0,item:answer});
  }
  return text+frame('response.completed',{response});
}
const stream=text=>new Response(text,{status:200,headers:{'content-type':'text/event-stream'}});
const load=p=>JSON.parse(readFileSync(p,'utf8'));
async function fixture(fn,profileName='luna') {
  const root=mkdtempSync(resolve(tmpdir(),'cep-responses-test-')),original=globalThis.fetch;
  const cfg=resolve(root,'config.json'),question=resolve(root,'question.txt');
  const profiles={ds:'deepseek-v4.1-flash',qwen:'qwen3.8-27b',flash:'qwen3.8-flash',luna:'gpt-6-luna'};
  const profile=load(resolve(ROOT,'configs/models/'+profiles[profileName]+'.json'));
  assert.equal(profile.api,'openai-responses');assert.equal(profile.minimum_request_interval_seconds,profileName==='flash'?0:5);
  const config={...profile,api_key_env:'CEP_RESPONSES_TEST_KEY',base_url:'http://127.0.0.1:1/v1',rate_limit_retry:{enabled:false},
    pricing:{currency:'USD',tiers:[{input:.1,cache_read:.01,cache_write:.125,output:.5}]}};
  if(['qwen','flash'].includes(profileName))config.pricing=profile.pricing;
  if(profileName==='ds')config.pricing={currency:'USD',tiers:[{input:.15,cache_read:.003,output:.6}]};
  process.env.CEP_RESPONSES_TEST_KEY='synthetic-test-key';
  writeFileSync(cfg,JSON.stringify(config));writeFileSync(question,'Synthetic question.');
  const backendFactory=folder=>{
    const b=createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}});
    writeFileSync(resolve(folder,'workspace/evidence.txt'),'Synthetic evidence.');
    writeFileSync(resolve(folder,'state.json'),JSON.stringify({ids:[],nodes:{}}));
    return b;
  };
  try{await fn({root,cfg,question,config,backendFactory});}
  finally{globalThis.fetch=original;rmSync(root,{recursive:true,force:true});}
}

test('Responses full runner retains reasoning/tool IDs, raw accounting, archive and pacing override',async()=>fixture(async({root,cfg,question,backendFactory})=>{
  const calls=[],starts=[];
  globalThis.fetch=async(url,init)=>{
    assert.equal(String(url),'http://127.0.0.1:1/v1/responses');
    const p=JSON.parse(init.body);calls.push(p);starts.push(Date.now());assert(calls.length<=2);
    assert.equal(p.reasoning.effort,'high');assert.equal(p.max_output_tokens,32000);assert.equal(p.store,false);
    assert.equal(p.input.find(m=>['system','developer'].includes(m.role)).content,prompt);
    assert.deepEqual(p.tools.map(({name,description,parameters})=>({name,description,parameters})),schemas);
    assert.equal(p.stream_options,undefined);assert.equal(p.reasoning_effort,undefined);
    if(calls.length===2){
      assert(p.input.some(x=>x.type==='reasoning'&&x.encrypted_content===reasoning.encrypted_content));
      assert(p.input.some(x=>x.type==='function_call'&&x.id===fc.id&&x.call_id===fc.call_id));
      assert(p.input.some(x=>x.type==='function_call_output'&&x.call_id===fc.call_id&&x.output.includes('Synthetic evidence')));
    }
    return stream(frames(calls.length));
  };
  mkdirSync(resolve(root,'cases'));const folder=resolve(root,'cases/q0');
  const result=await run(cfg,question,folder,{backendFactory,requestIntervalSeconds:.05});
  assert.equal(result.status,'answered');assert.equal(result.decisions,2);assert.equal(result.tools,1);
  assert(starts[1]-starts[0]>=45);assert.equal(load(resolve(folder,'CONFIG.json')).minimum_request_interval_seconds,.05);
  assert.equal(load(cfg).minimum_request_interval_seconds,5);
  const accounting=load(resolve(folder,'TOKEN_USAGE.json'));
  assert.equal(accounting.physical_requests,2);assert.equal(accounting.usage_complete,true);
  assert.deepEqual([accounting.token_counts.input_total,accounting.token_counts.cache_hit,accounting.token_counts.cache_write,accounting.token_counts.cache_miss,accounting.token_counts.output_total,accounting.token_counts.reasoning_subset,accounting.token_counts.non_reasoning_output],[200,40,140,20,20,16,4]);
  assert.ok(Math.abs(accounting.total_cost-.0000299)<1e-15);
  assert.deepEqual(readAccounting(folder,load(resolve(folder,'CONFIG.json'))),accounting);
  assert.equal(load(resolve(folder,'STORAGE_VERIFICATION.json')).status,'PASS');
  assert.deepEqual(readApi(folder,2).request,calls[1]);
  assert.equal(existsSync(resolve(folder,'ACCOUNTING_ERROR.json')),false);
}));

test('Responses continuation resends the same failed request with pacing changed and keeps segment accounting',async()=>fixture(async({root,cfg,question,backendFactory})=>{
  const calls=[];let failed=true;
  globalThis.fetch=async(url,init)=>{
    const p=JSON.parse(init.body);calls.push(p);
    if(calls.length===1)return stream(frames(1));
    if(failed)return stream(frame('response.failed',{response:{id:'resp_failed',status:'failed',error:{code:'rate_limit_exceeded',message:'rate limit; see account/billing'}}}));
    return stream(frames(2));
  };
  const parent=resolve(root,'parent'),child=resolve(root,'child');
  const first=await run(cfg,question,parent,{backendFactory,requestIntervalSeconds:0});
  assert.equal(first.status,'model_error');assert.equal(first.decisions,1);assert.equal(first.tools,1);
  const originalUsage=readFileSync(resolve(parent,'TOKEN_USAGE.json'),'utf8');
  failed=false;
  const second=await run(cfg,question,child,{backendFactory,resumeFrom:parent,requestIntervalSeconds:.01});
  assert.equal(second.status,'answered');assert.equal(second.decisions,2);assert.equal(second.tools,1);
  assert.equal(calls.length,3);assert.deepEqual(calls[2],calls[1]);
  assert.equal(load(resolve(child,'RESUME_REQUEST_VERIFICATION.json')).status,'PASS');
  assert.deepEqual(load(resolve(child,'RESUME.json')).request_interval_seconds,{previous:0,current:.01});
  assert.equal(load(resolve(child,'TOKEN_USAGE.json')).physical_requests,1);
  assert.equal(load(resolve(child,'TOKEN_USAGE.json')).token_counts.input_total,100);
  assert.equal(readFileSync(resolve(parent,'TOKEN_USAGE.json'),'utf8'),originalUsage);
  await assert.rejects(run(cfg,question,resolve(root,'bad-protocol-resume'),{backendFactory,resumeFrom:parent,api:'openai-completions',requestIntervalSeconds:0}),/preserve model/);
}));

test('Responses SSE rate limit with billing URL retries identical payload and preserves failed usage',async()=>fixture(async({root,cfg,question,config,backendFactory})=>{
  config.rate_limit_retry={enabled:true,min_delay_seconds:0,max_delay_seconds:0,max_attempts:3,max_elapsed_seconds:10};
  writeFileSync(cfg,JSON.stringify(config));const calls=[];
  globalThis.fetch=async(url,init)=>{
    calls.push(JSON.parse(init.body));assert(calls.length<=2);
    if(calls.length===1)return stream(frame('response.failed',{response:{id:'resp_failed',status:'failed',usage,error:{code:'rate_limit_exceeded',message:'TPM exceeded. Visit https://platform.openai.com/account/billing.'}}}));
    return stream(frames(2));
  };
  const folder=resolve(root,'case');const result=await run(cfg,question,folder,{backendFactory,requestIntervalSeconds:0});
  assert.equal(result.status,'answered');assert.equal(result.decisions,1);assert.equal(calls.length,2);assert.deepEqual(calls[1],calls[0]);
  const accounting=load(resolve(folder,'TOKEN_USAGE.json'));
  assert.equal(accounting.physical_requests,2);assert.equal(accounting.token_counts.input_total,200);
  assert.equal(accounting.usage_complete,false);assert.equal(accounting.total_cost,null);
  assert.ok(Math.abs(accounting.partial_or_failed_known_usage_cost-.00001495)<1e-15);
  assert.equal(result.rate_limit_retries[0].kind,'rate_limit');
  assert.equal(classifyLimit(401,'rate_limit_exceeded'),null);
  assert.equal(classifyLimit(429,'insufficient_quota rate_limit_exceeded'),null);
}));

test('truncated Responses stream cannot silently become a completed answer with retries disabled',async()=>fixture(async({root,cfg,question,backendFactory})=>{
  const truncated=frame('response.created',{response:{id:'unfinished',status:'in_progress'}});
  await assert.rejects(()=>peekStreamStart(stream(truncated).body.getReader()),/Incomplete Responses stream/);
  globalThis.fetch=async()=>stream(truncated);
  const folder=resolve(root,'case');const result=await run(cfg,question,folder,{backendFactory,requestIntervalSeconds:0});
  assert.equal(result.status,'model_error');assert.equal(result.decisions,0);
  assert.equal(load(resolve(folder,'TOKEN_USAGE.json')).unknown_usage_calls,1);
  assert.equal(existsSync(resolve(folder,'final.txt')),false);
}));

for(const name of ['ds','qwen','flash'])test(name+' Responses: prompt role, plaintext reasoning replay, tools and usage accounting',async()=>fixture(async({root,cfg,question,config,backendFactory})=>{
  const calls=[],plain={id:'rs_plain',type:'reasoning',content:[{type:'reasoning_text',text:'Synthetic provider reasoning.'}]};
  const tokens={input_tokens:100,input_tokens_details:{cached_tokens:80},output_tokens:30,output_tokens_details:{reasoning_tokens:20},total_tokens:130};
  globalThis.fetch=async(url,init)=>{
    assert.equal(String(url),'http://127.0.0.1:1/v1/responses');const p=JSON.parse(init.body);calls.push(p);assert(calls.length<=2);
    assert.equal(p.model,config.model);assert.equal(p.input[0].role,name==='flash'?'developer':'system');assert.equal(p.input[0].content,prompt);
    assert.equal(p.reasoning.effort,name==='ds'?'high':'xhigh');assert.equal(p.max_output_tokens,32000);
    assert.equal(p.reasoning_effort,undefined);assert.equal(p.thinking,undefined);
    if(name==='qwen')assert.deepEqual(p.chat_template_kwargs,{enable_thinking:true});
    if(name==='flash')assert.equal(p.chat_template_kwargs,undefined);
    if(calls.length===2){
      assert.deepEqual(p.input.find(x=>x.type==='reasoning'),plain);
      assert(p.input.some(x=>x.type==='function_call_output'&&x.call_id===fc.call_id));
    }
    return stream(frames(calls.length,{model:config.model,reasoningItem:plain,usageValue:tokens}));
  };
  const folder=resolve(root,'case');const result=await run(cfg,question,folder,{backendFactory,requestIntervalSeconds:0});
  assert.equal(result.status,'answered');assert.equal(result.decisions,2);assert.equal(result.tools,1);
  const thinking=readTrajectory(folder).find(x=>x.role==='assistant').content.find(x=>x.type==='thinking');
  assert.equal(thinking.thinking,plain.content[0].text);assert.deepEqual(JSON.parse(thinking.thinkingSignature),plain);
  const report=load(resolve(folder,'TOKEN_USAGE.json'));
  assert.equal(report.usage_complete,true);assert.equal(report.token_counts.input_total,200);assert.equal(report.token_counts.cache_hit,160);
  assert.equal(report.token_counts.output_total,60);assert.equal(report.token_counts.reasoning_subset,40);
  assert.equal(report.token_counts.non_reasoning_output,20);
  if(name==='ds')assert.ok(Math.abs(report.total_cost-.00004248)<1e-15);
  else if(name==='flash')assert.ok(Math.abs(report.total_cost-.00003676)<1e-15);
  else assert.equal(report.total_cost,null);
  assert.deepEqual(readAccounting(folder,load(resolve(folder,'CONFIG.json'))),report);
  assert.equal(load(resolve(folder,'STORAGE_VERIFICATION.json')).status,'PASS');
},name));
