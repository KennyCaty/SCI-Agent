import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {resolve,dirname,relative,isAbsolute} from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readTrajectory,readContext,readApi} from './inspect.mjs';
import {classifyLimit} from './rate-limit.mjs';
import {CaseStore} from './storage.mjs';
export function restoreContinuation(parent,folder,config,question){
 parent=resolve(parent);folder=resolve(folder);
 const load=name=>JSON.parse(readFileSync(resolve(parent,name),'utf8'));
 const complete=load('COMPLETE.json'),old=load('CONFIG.json');
 assert.ok(['model_error','interrupted'].includes(complete.status),'Only explicitly failed or interrupted requests may resume');
 assert.equal(readFileSync(resolve(parent,'question.txt'),'utf8'),question);
 const {minimum_request_interval_seconds:spacing=0,...same}=config;const {minimum_request_interval_seconds:oldSpacing=0,...oldSame}=old;
 assert.ok(Number.isFinite(spacing)&&spacing>=0,'Invalid request pacing');assert.deepEqual(same,oldSame,'Continuation must preserve model, prompt, tools and budgets');
 const messages=readTrajectory(parent),error=messages.pop();
 assert.equal(error.role,'assistant');assert.ok(['error','aborted'].includes(error.stopReason));
 assert.ok(error.content.length===0 || (
   (error.stopReason==='aborted'||classifyLimit(503,error.errorMessage??'')) &&
   error.content.every(c=>['thinking','toolCall'].includes(c.type))
 ),'Only incomplete failed thinking/tool deltas may be removed from resumed memory');
 for(const c of error.content)if(c.type==='toolCall')assert.ok(!messages.some(m=>m.role==='toolResult'&&m.toolCallId===c.id),'Failed tool was already executed');
 assert.ok(['user','toolResult'].includes(messages.at(-1)?.role),'Incomplete tool batch cannot resume');
 const completedTools=new Set(messages.filter(m=>m.role==='toolResult').map(m=>m.toolCallId));
 for(const m of messages)if(m.role==='assistant')for(const c of m.content)if(c.type==='toolCall')assert.ok(completedTools.has(c.id),'Pending tool must not be repeated implicitly');
 assert.ok(complete.decisions<config.max_turns&&complete.normal_seconds<config.max_seconds);
 assert.equal(load('STORAGE_VERIFICATION.json').status,'PASS');
 const refs=load('retrieval_and_artifacts.json'),store=new CaseStore(parent);
 const toolResults=messages.filter(m=>m.role==='toolResult');
 const uncommittedInitialGraph=toolResults.length>0 && toolResults.every(m=>m.isError===true&&['graphsearch','graphexpand'].includes(m.toolName)) && !Object.keys(refs).some(n=>n.startsWith('workspace/'));
 // Only root state records committed acquisitions; event snapshots may be incomplete.
 assert.ok(refs['state.json'] || uncommittedInitialGraph || (complete.tools===0 && !toolResults.length),
   'Missing authoritative graph state after tool execution');
 const materialized=[];
 for(const [name,ref] of Object.entries(refs)){
  if(name!=='state.json'&&!name.startsWith('workspace/'))continue;
  const target=resolve(folder,name);assert.ok(!relative(folder,target).startsWith('..')&&!isAbsolute(relative(folder,target)));
  mkdirSync(dirname(target),{recursive:true});writeFileSync(target,store.readBytes(ref));materialized.push(name);
 }
 // Saved bash output may use an absolute Pi temporary path.
 const nativeOutputs=[];
 for(const [name,ref] of Object.entries(refs))if(/^\/tmp\/pi-bash-[^/]+$/.test(name)){
  const bytes=store.readBytes(ref);
  if(existsSync(name))assert.deepEqual(readFileSync(name),bytes);else writeFileSync(name,bytes,{mode:0o600});
  nativeOutputs.push(name);
 }
 mkdirSync(resolve(folder,'workspace/.tool_tmp'),{recursive:true});
 writeFileSync(resolve(folder,'workspace/.tool_tmp/native_outputs.json'),JSON.stringify(nativeOutputs));
 const contexts=readFileSync(resolve(parent,'context.jsonl'),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
 const last=contexts.length?readContext(parent,contexts.at(-1).index):{boundary_after:0};
 const hash=name=>createHash('sha256').update(readFileSync(resolve(parent,name))).digest('hex');
 writeFileSync(resolve(folder,'RESUME.json'),JSON.stringify({parent,parent_hashes:Object.fromEntries(['COMPLETE.json','CONFIG.json','trajectory.jsonl','objects.tar','TOKEN_USAGE.json'].map(n=>[n,hash(n)])),inherited_messages:messages.length,discarded_empty_failed_message:error.content.length===0,failed_partial_preserved_in_parent:true,discarded_unexecuted_partial_types:error.content.map(c=>c.type),inherited_decisions:complete.decisions,inherited_normal_seconds:complete.normal_seconds,parent_known_cost:load('TOKEN_USAGE.json').known_usage_cost,cost_scope:'TOKEN_USAGE is new physical requests only; sum parent chain for cumulative cost; unknown bills remain unknown',materialized},null,2)+'\n');
 const apiFile=resolve(parent,'api.jsonl');const apiRows=existsSync(apiFile)?readFileSync(apiFile,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];
 const lastApi=apiRows.length?readApi(parent,apiRows.at(-1).index):null;
 // An abort before fetch may leave a successful request as the last recorded call.
 const previousCompleted=lastApi?.response_sse?.split('\n').some(line=>{if(!line.startsWith('data:'))return false;try{const x=JSON.parse(line.slice(5));return x.type==='response.completed'||x.choices?.some(c=>['stop','tool_calls','length'].includes(c.finish_reason));}catch{return false;}});
 const expectedRequest=lastApi&&!previousCompleted?lastApi.request:null;
 let initialRetryState=null;
 if(expectedRequest){
  const meta=lastApi['request.meta.json'],logical=meta?.logical_request_index;
  const physical=apiRows.filter(x=>x['request.meta.json']?.logical_request_index===logical);
  const indices=new Set(physical.map(x=>x.index));
  const inherited=logical===1&&existsSync(resolve(parent,'RESUME.json'))?load('RESUME.json').initial_retry_state:null;
  const elapsed=physical.reduce((sum,x)=>sum+Math.max(...['END.json','ERROR.json','CANCELLED.json'].map(n=>x[n]?.seconds??0)),0)+(complete.retry_time_exclusions??[]).filter(x=>x.kind==='retry_wait'&&indices.has(x.physical_index)).reduce((sum,x)=>sum+x.seconds,0)+(inherited?.elapsed_seconds??0);
  initialRetryState={attempts:meta?.physical_attempt??physical.length,elapsed_seconds:elapsed};
 }
 const resumeFile=resolve(folder,'RESUME.json');const record=JSON.parse(readFileSync(resumeFile));record.initial_retry_state=initialRetryState;
 record.request_interval_seconds={previous:oldSpacing,current:spacing};writeFileSync(resumeFile,JSON.stringify(record,null,2)+'\n');
 return {parent,complete,messages,expectedRequest,initialRetryState,context:{boundary_after:last.boundary_after,prompt_cache_key:lastApi?.request?.prompt_cache_key,last_successful_message:messages.findLast(m=>m.role==='assistant')}};
}
