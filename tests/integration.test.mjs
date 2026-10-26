import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync,readdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import {createCepSession} from '../extensions/session.mjs';
import {createBackend,GraphBridge,summarizeToolTiming} from '../extensions/tools.mjs';
import {ToolHistoryCompactor} from '../extensions/extension.mjs';
import {prompt,schemas} from '../extensions/paths.mjs';
import {installWireLog} from '../extensions/wire-log.mjs';
import {BudgetClock,watchBudget} from '../extensions/budget.mjs';
import {CaseStore,compactCase} from '../extensions/storage.mjs';
import {readTrajectory,readTool,readContext,readApi} from '../extensions/inspect.mjs';
import {run} from '../extensions/run.mjs';

async function fixture(fn) {
  const root=mkdtempSync(resolve(tmpdir(),'cep-pi-test-'));
  try { await fn(root); } finally {rmSync(root,{recursive:true,force:true});}
}
async function serverFor(responseFor) {
  const requests=[];
  const server=createServer(async(req,res)=>{
    let body='';for await(const b of req)body+=b;
    const p=JSON.parse(body); requests.push(p);
    const delta=responseFor(p,requests.length);
    res.writeHead(200,{'content-type':'text/event-stream'});
    const base={id:`mock-${requests.length}`,object:'chat.completion.chunk',created:1,model:'test'};
    const emit=x=>res.write('data: '+JSON.stringify(x)+'\n\n');
    emit({...base,choices:[{index:0,delta:{role:'assistant',...delta},finish_reason:null}]});
    emit({...base,choices:[{index:0,delta:{},finish_reason:delta.tool_calls?'tool_calls':'stop'}],
      usage:{prompt_tokens:100,completion_tokens:10,total_tokens:110,prompt_cache_hit_tokens:80,prompt_cache_miss_tokens:20}});
    res.end('data: [DONE]\n\n');
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  return {requests,url:`http://127.0.0.1:${server.address().port}/v1`,close:()=>new Promise(r=>{server.close(r);server.closeAllConnections();})};
}
const config=url=>({model:'test',api_key_env:'CEP_PI_TEST_KEY',base_url:url,context_window:262144,
  max_output_tokens:32000,max_turns:100,max_seconds:3600});
const call=(index,id,name,args)=>({index,id,type:'function',function:{name,arguments:JSON.stringify(args)}});
process.env.CEP_PI_TEST_KEY='local-test-not-a-secret';

test('real Pi SDK: exact prompt/schema, parent isolation, parallel graph/read/bash, raw usage',async()=>fixture(async(root)=>{
  writeFileSync(resolve(root,'AGENTS.md'),'PARENT_SECRET_SENTINEL');
  const folder=resolve(root,'case');mkdirSync(folder);
  let release;
  const graphWait=new Promise(r=>{release=r;});
  let graphStarted=false, readFinishedWhileGraphPending=false;
  const graph={async execute(){graphStarted=true;await graphWait;return {text:'MAP fixture',details:{method:'CEP18'}};},async close(){release();}};
  const backend=createBackend(folder,{graph});
  writeFileSync(resolve(folder,'workspace/evidence.txt'),'A😀BC\nSecond line');
  const invoke=backend.execute.bind(backend);
  backend.execute=async(name,...args)=>{
    const result=await invoke(name,...args);
    if(name==='read'){readFinishedWhileGraphPending=graphStarted;release();}
    return result;
  };
  const server=await serverFor((p,n)=>n===1?{tool_calls:[
    call(0,'g','graphsearch',{queries:['example']}),
    call(1,'r','read',{path:'evidence.txt',charOffset:1,charLimit:2}),
    call(2,'b','bash',{command:'printf hello'}),
  ]}:{content:'Explanation: fixture.\nExact Answer: hello\nConfidence: 100%'});
  const restore=installWireLog(folder,server.url);
  let session;
  try{
    const created=await createCepSession(folder,config(server.url),backend);session=created.session;
    assert.equal(created.settingsManager.getRetryEnabled(),false);
    assert.equal(created.settingsManager.getCompactionEnabled(),false);
    assert.equal(created.settingsManager.getRuntimeContextManagementSettings().level,'level0');
    assert.deepEqual(created.loader.getAgentsFiles(),{agentsFiles:[]});
    assert.deepEqual(session.getActiveToolNames(),schemas.map(s=>s.name));
    await session.prompt('Question unchanged?');
    assert.equal(server.requests.length,2,JSON.stringify(session.messages));
    for(const p of server.requests){
      assert.equal(p.messages[0].content,prompt);
      assert.equal(JSON.stringify(p).includes('PARENT_SECRET_SENTINEL'),false);
      assert.deepEqual(p.tools.map(t=>({name:t.function.name,description:t.function.description,parameters:t.function.parameters})),schemas);
      assert.equal(p.max_completion_tokens,32000);assert.equal(p.reasoning_effort,'high');
      assert.equal('temperature' in p,false);assert.equal(p.parallel_tool_calls,true);
    }
    assert.equal(readFinishedWhileGraphPending,true);
    const results=session.messages.filter(m=>m.role==='toolResult');
    assert.deepEqual(results.map(r=>r.toolName),['graphsearch','read','bash']);
    assert.equal(results[1].content[0].text.startsWith('😀'),true);
    assert.equal(results[2].isError,false,JSON.stringify(results[2]));
    assert.equal(results[2].content[0].text,'hello');
    assert.equal(created.state.decisions,2);
    const raw=readFileSync(resolve(folder,'api/0001/response.sse'),'utf8');
    assert.match(raw,/prompt_cache_hit_tokens/);
    assert.equal(new CaseStore(folder).readBytes(JSON.parse(readFileSync(resolve(folder,'api/0001/request.ref.json')))).toString().includes('local-test-not-a-secret'),false);
  }finally{session?.dispose();await backend.close();restore();await server.close();}
}));

test('decision cap: Pi sends no 101st/extra final request and preserves tool results',async()=>fixture(async(root)=>{
  const backend={execute:async()=>({content:[{type:'text',text:'ok'}],details:{}})};
  const server=await serverFor(()=>({tool_calls:[call(0,'c','read',{path:'x'})]}));
  let session;
  try{
    const created=await createCepSession(root,{...config(server.url),max_turns:2},backend);session=created.session;
    await session.prompt('continue');
    assert.equal(server.requests.length,2);assert.equal(created.state.status,'turn_limit');
    assert.equal(session.messages.filter(m=>m.role==='toolResult').length,2);
  }finally{session?.dispose();await server.close();}
}));

test('context clearing preserves whole boundary group and full persisted history',()=>{
  const messages=[{role:'user',content:'question'}];
  for(let i=0;i<5;i++){
    const ids=[`${i}a`,`${i}b`,`${i}c`];
    messages.push({role:'assistant',content:ids.map(id=>({type:'toolCall',id,name:'read',arguments:{path:'x'}}))});
    messages.push(...ids.map(id=>({role:'toolResult',toolCallId:id,content:[{type:'text',text:'original'}]})));
  }
  const before=JSON.stringify(messages),c=new ToolHistoryCompactor(0,4),projected=c.project(messages);
  assert.equal(projected.filter(m=>m.role==='toolResult'&&m.content[0].text==='original').length,6);
  assert.equal(JSON.stringify(messages),before);
  c.threshold=Infinity;assert.deepEqual(c.project(messages),projected);
});

test('independent graph workers, abort kills owned bridge and rejects queued requests',async()=>fixture(async(root)=>{
  const script=resolve(root,'fixture.py');
  writeFileSync(script,`import os,json,sys,time\nprint(json.dumps({'ready':True,'pid':os.getpid()}),flush=True)\nfor line in sys.stdin:\n r=json.loads(line);time.sleep(r['args'].get('sleep',0));print(json.dumps({'id':r['id'],'ok':True,'result':{'text':'ok'}}),flush=True)\n`);
  const aDir=resolve(root,'a'),bDir=resolve(root,'b');mkdirSync(aDir);mkdirSync(bDir);
  const a=new GraphBridge(aDir,'python3',script),b=new GraphBridge(bDir,'python3',script);
  try{
    await Promise.all([a.start(),b.start()]);assert.notEqual(a.pid,b.pid);
    const abort=new AbortController();
    const pending=a.execute('graphsearch',{sleep:20},'a',abort.signal);
    const rejected=assert.rejects(pending,/exited/);
    assert.equal((await b.execute('graphsearch',{},'b')).text,'ok');
    abort.abort();await rejected;
  }finally{await a.close();await b.close();}
}));

test('DR bash truncation, saved full output, workspace boundary and cancellation',async()=>fixture(async(root)=>{
  const backend=createBackend(root,{graph:{async close(){}}});
  try {
    const long=await backend.execute('bash',{command:"python3 -c \"print('x'*30000)\""},'long');
    assert.ok(long.content[0].text.length<11000);
    assert.ok(long.details.fullOutputPath);
    const saved=await backend.execute('read',{path:long.details.fullOutputPath,charOffset:20000,charLimit:20},'saved');
    assert.ok(saved.content[0].text.startsWith('x'.repeat(20)));
    await assert.rejects(backend.execute('read',{path:'/etc/passwd'},'outside'),/Outside acquired workspace/);
    const controller=new AbortController();
    const running=backend.execute('bash',{command:'echo $$ > .tool_tmp/pid; sleep 30'},'sleep',controller.signal);
    const settled=running.catch(()=>null);
    const pidFile=resolve(root,'workspace/.tool_tmp/pid');
    let pid;
    for(let i=0;i<100;i++) {
      try{pid=Number(readFileSync(pidFile,'utf8'));break;}catch{}
      await new Promise(r=>setTimeout(r,20));
    }
    assert.ok(pid,'shell started');controller.abort();await settled;
    let alive=true;
    try{alive=!readFileSync(`/proc/${pid}/stat`,'utf8').includes(') Z ');}catch{alive=false;}
    assert.equal(alive,false,'aborted shell must not survive its Pi tool');
  }finally{await backend.close();}
}));

test('DS thinking request and reasoning_content round trip through Pi',async()=>fixture(async(root)=>{
  const server=await serverFor((p,n)=>n===1?{reasoning_content:'Reasoning fixture.',tool_calls:[call(0,'r','read',{path:'x'})]}:{content:'done'});
  let session;
  try{
    ({session}=await createCepSession(root,{...config(server.url),thinking_enabled:true,max_tokens_field:'max_tokens'},
      {execute:async()=>({content:[{type:'text',text:'evidence'}],details:{}})}));
    await session.prompt('q');
    assert.equal(server.requests.length,2);
    assert.deepEqual(server.requests[0].thinking,{type:'enabled'});
    assert.equal(server.requests[0].max_tokens,32000);
    const assistant=server.requests[1].messages.find(m=>m.role==='assistant');
    assert.equal(assistant.reasoning_content,'Reasoning fixture.');
  }finally{session?.dispose();await server.close();}
}));

test('HTTP failure does not trigger a hidden SDK/model retry',async()=>fixture(async(root)=>{
  let calls=0;
  const server=createServer((req,res)=>{calls++;req.resume();res.writeHead(500,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:'fixture failure',type:'server_error'}}));});
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const url=`http://127.0.0.1:${server.address().port}/v1`,restore=installWireLog(root,url);
  let session;
  try{
    const created=await createCepSession(root,config(url),{execute(){throw Error('unexpected tool');}});session=created.session;
    await session.prompt('q');
    assert.equal(calls,1);assert.equal(created.state.decisions,0);
    assert.equal(session.messages.at(-1).stopReason,'error');
    assert.equal(JSON.parse(readFileSync(resolve(root,'api/0001/status.json'))).status,500);
  }finally{session?.dispose();restore();await new Promise(r=>{server.close(r);server.closeAllConnections();});}
}));

test('normal clock excludes failed API once and retains successful calls, tools, caller cancellation',()=>{
  let now=0;const clock=new BudgetClock(10,()=>now);clock.start();
  clock.beginCall();now=3000;clock.finishCall('error');
  assert.equal(clock.snapshot().failed_api_seconds,3);assert.equal(clock.snapshot().normal_seconds,0);
  now=5000; // two seconds of tool execution/queue/local work
  clock.beginCall();now=9000;clock.finishCall('stop');
  assert.equal(clock.snapshot().normal_seconds,6);
  clock.beginCall();now=11000;clock.finishCall('aborted',true);
  assert.equal(clock.snapshot().normal_seconds,8);assert.equal(clock.failedSeconds,3);
  assert.deepEqual(clock.calls.map(x=>x.excluded_from_normal),[true,false,false]);
});

async function delayedBudgetCase(root,{success=false,timeout=false,cancel=false}={}) {
  const server=createServer(async(req,res)=>{
    req.resume();await new Promise(r=>setTimeout(r,250));
    if (res.destroyed) return;
    if (!success) {res.writeHead(500,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:'delayed failure',type:'server_error'}}));return;}
    res.writeHead(200,{'content-type':'text/event-stream'});
    const base={id:'mock-delayed',object:'chat.completion.chunk',created:1,model:'test'};
    res.write('data: '+JSON.stringify({...base,choices:[{index:0,delta:{role:'assistant',content:'answer'},finish_reason:null}]})+'\n\n');
    res.write('data: '+JSON.stringify({...base,choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:10,total_tokens:110}})+'\n\n');
    res.end('data: [DONE]\n\n');
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const url=`http://127.0.0.1:${server.address().port}/v1`,restore=installWireLog(root,url);
  let session,monitor,cancelTimer;
  try{
    const created=await createCepSession(root,{...config(url),max_seconds:0.1,api_timeout_seconds:timeout?0.15:2},
      {execute(){throw Error('unexpected tool');}});session=created.session;
    created.state.clock.start();
    monitor=watchBudget(created.state.clock,()=>{created.state.status='normal_time_limit';session.agent.abort();},5);
    if(cancel)cancelTimer=setTimeout(()=>{created.state.status='interrupted';session.agent.abort();},150);
    await session.prompt('q');
    clearInterval(monitor);
    return {state:created.state,timing:created.state.clock.snapshot()};
  }finally{clearInterval(monitor);clearTimeout(cancelTimer);session?.dispose();restore();await new Promise(r=>{server.close(r);server.closeAllConnections();});}
}

test('delayed HTTP failure exceeding normal limit is excluded after settlement, not prematurely cancelled',async()=>fixture(async(root)=>{
  const {state,timing}=await delayedBudgetCase(root);
  assert.equal(state.status,null);assert.equal(state.decisions,0);
  assert.ok(timing.wall_seconds>=0.23);
  assert.ok(timing.failed_api_seconds>=0.23);
  assert.ok(timing.normal_seconds<0.1);
  assert.equal(state.clock.calls[0].excluded_from_normal,true);
}));

test('delayed successful response counts toward normal limit and cannot become accepted answer',async()=>fixture(async(root)=>{
  const {state,timing}=await delayedBudgetCase(root,{success:true});
  assert.equal(state.status,'normal_time_limit');assert.equal(state.decisions,1);
  assert.ok(timing.normal_seconds>=0.23);assert.equal(timing.failed_api_seconds,0);
}));

test('transport timeout is excluded, caller cancellation is retained',async()=>fixture(async(root)=>{
  const timeoutDir=resolve(root,'timeout'),cancelDir=resolve(root,'cancel');mkdirSync(timeoutDir);mkdirSync(cancelDir);
  const a=await delayedBudgetCase(timeoutDir,{timeout:true});
  assert.equal(a.state.status,null);assert.ok(a.timing.failed_api_seconds>=0.13);
  assert.ok(a.timing.normal_seconds<0.1);
  const b=await delayedBudgetCase(cancelDir,{cancel:true});
  assert.equal(b.state.status,'interrupted');assert.equal(b.timing.failed_api_seconds,0);
  assert.ok(b.timing.normal_seconds>=0.13);
}));

test('connection failure and SDK retry waiting are excluded without sending a new request',async()=>fixture(async(root)=>{
  const reserved=createServer();await new Promise(r=>reserved.listen(0,'127.0.0.1',r));
  const url=`http://127.0.0.1:${reserved.address().port}/v1`;await new Promise(r=>reserved.close(r));
  const restore=installWireLog(root,url);let session,monitor;
  try{
    const created=await createCepSession(root,{...config(url),max_seconds:0.1,api_timeout_seconds:3},
      {execute(){throw Error('unexpected tool');}});session=created.session;
    created.state.clock.start();
    monitor=watchBudget(created.state.clock,()=>{created.state.status='normal_time_limit';session.agent.abort();},5);
    await session.prompt('q');clearInterval(monitor);
    assert.equal(created.state.status,null);
    assert.equal(created.state.decisions,0);
    assert.equal(readdirSync(resolve(root,'api')).length,1);
    assert.ok(created.state.clock.snapshot().normal_seconds<0.1);
    assert.ok(created.state.clock.snapshot().failed_api_seconds>0);
  }finally{clearInterval(monitor);session?.dispose();restore();}
}));

test('bash queue is separated from execution while read stays independent; outputs retain submission order',async()=>fixture(async(root)=>{
  const backend=createBackend(root,{graph:{async close(){}}});
  writeFileSync(resolve(root,'workspace/evidence.txt'),'evidence');
  try{
    await Promise.all([
      backend.execute('bash',{command:'sleep 0.25; printf first'},'first'),
      backend.execute('bash',{command:'printf second'},'second'),
      backend.execute('read',{path:'evidence.txt'},'read'),
    ]);
    const events=readFileSync(resolve(root,'tool_events.jsonl'),'utf8').trim().split('\n').map(JSON.parse).sort((a,b)=>a.index-b.index);
    assert.deepEqual(events.map(e=>e.id),['first','second','read']);
    assert.ok(events[1].queue_seconds>=0.23);
    assert.ok(events[1].execution_seconds<events[1].queue_seconds);
    assert.ok(events[2].queue_seconds<0.02);
    for (const e of events){
      assert.equal(e.timing_complete,true);
      assert.ok(Math.abs(e.seconds-e.queue_seconds-e.execution_seconds-e.adapter_overhead_seconds)<0.005);
      assert.ok(backend.store.readJson(e.result_ref).ok);
    }
    assert.equal(backend.store.readJson(events[1].result_ref).result.content[0].text,'second');
  }finally{await backend.close();}
}));

test('graph backend with no worker timing is reported as unknown, not zero queue',async()=>fixture(async(root)=>{
  const backend=createBackend(root,{graph:{async execute(){return {text:'fixture',details:{}}},async close(){}}});
  try{
    await backend.execute('graphsearch',{queries:['q']},'unknown');
    const [e]=readFileSync(resolve(root,'tool_events.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(e.queue_seconds,null);assert.equal(e.execution_seconds,null);assert.equal(e.timing_complete,false);
  }finally{await backend.close();}
}));

test('timing summary deduplicates overlapping tool and queue intervals',()=>{
  const event=(submitted,started,ended)=>({clock_domain_verified:true,timing_complete:true,
    submitted_monotonic:submitted,started_monotonic:started,ended_monotonic:ended,
    returned_monotonic:ended,queue_seconds:started-submitted,execution_seconds:ended-started});
  const a=summarizeToolTiming([event(0,0,2),event(0,2,3),event(0,0,1)]);
  assert.equal(a.per_call_execution_seconds_known_sum,4);
  assert.equal(a.per_call_queue_seconds_known_sum,2);
  assert.equal(a.tool_wall_seconds_union,3);
  assert.equal(a.exclusive_known_queue_wall_seconds,0);
  assert.equal(summarizeToolTiming([event(0,2,3)]).exclusive_known_queue_wall_seconds,2);
});

test('compact case restores full trajectory, tool bodies, exact compressed context and API after cleanup',async()=>fixture(async(root)=>{
  const backend=createBackend(root,{graph:{async close(){}}});
  const text='EVIDENCE_RETAINED_AFTER_CONTEXT_CLEAR '.repeat(100);
  writeFileSync(resolve(root,'workspace/evidence.txt'),text);
  const server=await serverFor((p,n)=>n===1?{tool_calls:[call(0,'a','read',{path:'evidence.txt'}),call(1,'b','read',{path:'evidence.txt'})]}
    :n===2?{tool_calls:[call(0,'c','read',{path:'evidence.txt'})]}:{content:'final answer'});
  let session,restore=()=>{},recorder;
  try{
    const created=await createCepSession(root,{...config(server.url),compaction:{threshold:0,keep:1}},backend);
    session=created.session;recorder=created.recorder;
    restore=installWireLog(root,server.url,{store:recorder.store,currentContext:()=>recorder.contexts.at(-1)?.index});
    await session.prompt('Question stays intact');
    await restore.settled();
    const original=structuredClone(session.messages);
    assert.equal(server.requests.length,3);
    assert.deepEqual(server.requests[2].messages.filter(m=>m.role==='tool').map(m=>m.content),['[cleared]','[cleared]',text]);
    assert.ok(original.filter(m=>m.role==='toolResult').every(m=>m.content[0].text===text));
    assert.equal(recorder.finish(original).messages,original.length);
    session.dispose();await backend.close();restore();
    compactCase(root,recorder.store);
    assert.deepEqual(readTrajectory(root),original);
    assert.equal(readTool(root,'a').result.result.content[0].text,text);
    const context=readContext(root,3);
    assert.equal(context.boundary_advanced,true);
    assert.equal(context.turn,3);
    assert.deepEqual(context.newly_cleared_tool_ids,['a','b']);
    assert.deepEqual(context.cleared_tool_ids,['a','b']);
    assert.equal(context.before.filter(m=>m.role==='toolResult')[0].content[0].text,text);
    assert.equal(context.after.filter(m=>m.role==='toolResult')[0].content[0].text,'[cleared]');
    const api=readApi(root,3);
    assert.deepEqual(api.request,server.requests[2]);
    assert.match(api.response_sse,/prompt_cache_hit_tokens/);
    assert.equal(api['request.meta.json'].context_projection_index,3);
    assert.equal(readdirSync(root).includes('workspace'),false);
    assert.equal(readdirSync(root).includes('pi-session'),false);
    assert.equal(readdirSync(root).includes('conversation.json'),false);
    assert.equal(readdirSync(root).includes('objects'),false);
    assert.equal(JSON.parse(readFileSync(resolve(root,'STORAGE_VERIFICATION.json'))).status,'PASS');
    const lifecycle=readFileSync(resolve(root,'events.jsonl'),'utf8');
    assert.equal(lifecycle.includes(text),false);assert.equal(lifecycle.includes('"payload"'),false);
  }finally{session?.dispose();await backend.close();await restore.settled?.();restore();await server.close();}
}));

test('failed API evidence and error trajectory survive compact archiving',async()=>fixture(async(root)=>{
  const server=createServer((req,res)=>{req.resume();res.writeHead(500,{'content-type':'application/json'});res.end('{"error":{"message":"retain failure evidence"}}');});
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const url=`http://127.0.0.1:${server.address().port}/v1`;
  let session,restore=()=>{},recorder;
  try{
    const c=await createCepSession(root,config(url),{execute(){throw Error('unexpected tool');}});
    session=c.session;recorder=c.recorder;restore=installWireLog(root,url,{store:recorder.store});
    await session.prompt('failed case');await restore.settled();
    const original=structuredClone(session.messages);recorder.finish(original);session.dispose();restore();
    compactCase(root,recorder.store);
    assert.deepEqual(readTrajectory(root),original);
    assert.equal(readTrajectory(root).at(-1).stopReason,'error');
    const api=readApi(root,1);assert.equal(api['status.json'].status,500);
    assert.equal(api.response_sse,'{"error":{"message":"retain failure evidence"}}');
  }finally{session?.dispose();await restore.settled?.();restore();await new Promise(r=>{server.close(r);server.closeAllConnections();});}
}));

test('content store deduplicates repeated data and restores Unicode/binary partial bytes after archive',async()=>fixture(async(root)=>{
  const store=new CaseStore(root),data=Buffer.from(JSON.stringify({role:'tool',content:'😀'.repeat(50000)}));
  const a=store.putBytes(data),files=readdirSync(resolve(root,'objects')).length,b=store.putBytes(data);
  assert.deepEqual(a,b);assert.equal(readdirSync(resolve(root,'objects')).length,files);
  const partial=Buffer.from([0xF0,0x9F]),p=store.putBytes(partial);
  store.archive();assert.deepEqual(store.readBytes(a),data);assert.deepEqual(store.readBytes(p),partial);
}));

test('production runner finalizes compact outputs only after closing its owned backend',async()=>fixture(async(root)=>{
  const server=await serverFor((p,n)=>n===1?{tool_calls:[call(0,'read-one','read',{path:'evidence.txt'})]}:{content:'final fixture'});
  const cfg=resolve(root,'config.json'),q=resolve(root,'question.txt'),out=resolve(root,'case');
  writeFileSync(cfg,JSON.stringify({...config(server.url),pricing:{currency:'USD',tiers:[{input:.15,cache_read:.003,output:.6}]}}));writeFileSync(q,'case question');
  let closed=false;
  try{
    const summary=await run(cfg,q,out,{backendFactory:folder=>{
      const backend=createBackend(folder,{graph:{start:async()=>{},close:async()=>{closed=true;}}});
      writeFileSync(resolve(folder,'workspace/evidence.txt'),'case evidence');return backend;
    }});
    assert.equal(closed,true);assert.equal(summary.status,'answered');assert.equal(summary.tools,1);
    assert.equal(readdirSync(out).includes('STORAGE_ERROR.json'),false);
    assert.equal(JSON.parse(readFileSync(resolve(out,'STORAGE_VERIFICATION.json'))).status,'PASS');
    assert.equal(readFileSync(resolve(out,'final.txt'),'utf8'),'final fixture');
    assert.equal(readTool(out,'read-one').result.result.content[0].text,'case evidence');
    assert.equal(readTrajectory(out)[0].content[0].text,'case question');
    assert.equal(readApi(out,2).request.messages.at(-1).content,'case evidence');
    const accounting=JSON.parse(readFileSync(resolve(out,'TOKEN_USAGE.json')));
    assert.equal(accounting.physical_requests,2);assert.equal(accounting.usage_complete,true);
    assert.equal(accounting.token_counts.cache_hit,160);assert.equal(accounting.token_counts.cache_miss,40);
    assert.equal(accounting.token_counts.output_total,20);assert.equal(accounting.token_counts.reasoning_subset,null);
    assert.ok(Math.abs(accounting.total_cost-.00001848)<1e-15);
    const {readAccounting}=await import('../extensions/accounting.mjs');
    const restored=readAccounting(out,JSON.parse(readFileSync(cfg)));
    assert.deepEqual(restored,accounting);
  }finally{await server.close();}
}));

test('TREC-COVID sends the ranking prompt and four tool definitions',async()=>fixture(async(root)=>{
  const {taskPrompt}=await import('../extensions/task.mjs');
  const server=await serverFor(()=>({content:'Relevant Documents:\n1. source_id=2'}));
  const options={...config(server.url),task:{type:'trec-covid',subset:'trec-covid'}};
  let session;
  try {
    ({session}=await createCepSession(root,options,{execute:async()=>{throw new Error('Unexpected tool');}}));
    await session.prompt('Retrieve and rank documents for this query.\n\nQUERY:\nQuery text only');
    assert.equal(server.requests.length,1);
    assert.equal(server.requests[0].messages[0].content,taskPrompt(options));
    assert.deepEqual(server.requests[0].tools.map(t=>({name:t.function.name,description:t.function.description,parameters:t.function.parameters})),schemas);
    assert.deepEqual(server.requests[0].messages.at(-1).content,[{type:'text',text:'Retrieve and rank documents for this query.\n\nQUERY:\nQuery text only'}]);
  } finally {session?.dispose();await server.close();}
}));

test('Chat Completions supports all configured models with reasoning and no fixed temperature',async()=>fixture(async(root)=>{
  const profiles={luna:'gpt-6-luna',ds:'deepseek-v4.1-flash',qwen:'qwen3.8-27b',flash:'qwen3.8-flash'};
  for(const [name,file] of Object.entries(profiles)) {
    const profile=JSON.parse(readFileSync(resolve(new URL('../configs/models/'+file+'.json',import.meta.url).pathname)));
    const server=await serverFor(()=>({reasoning_content:'reasoning preserved',content:'final text'}));
    const folder=resolve(root,name);mkdirSync(folder);
    let session;
    try {
      ({session}=await createCepSession(folder,{...profile,api:'openai-completions',base_url:server.url,api_key_env:'CEP_PI_TEST_KEY'}, {execute:async()=>{throw new Error('No tools expected');}}));
      await session.prompt('test');
      assert.equal(server.requests.length,1,JSON.stringify(session.messages));
      const request=server.requests[0];assert.equal(request.model,profile.model);assert.equal('temperature' in request,false);
      if(name==='ds')assert.deepEqual(request.thinking,{type:'enabled'});
      if(name==='qwen') {assert.deepEqual(request.chat_template_kwargs,{enable_thinking:true});assert.equal(request.reasoning_effort,'xhigh');}
      if(name==='flash') {assert.equal(request.chat_template_kwargs,undefined);assert.equal(request.reasoning_effort,'xhigh');}
      assert.ok(session.messages.at(-1).content.some(c=>c.type==='thinking'&&c.thinking==='reasoning preserved'));
    } finally {session?.dispose();await server.close();}
  }
}));

test('provider error followed by usage retains billing tail independently of Pi stopping the stream',async()=>fixture(async(root)=>{
  let requests=0;
  const server=createServer(async(req,res)=>{
    for await(const unused of req){} requests++;
    const frame=x=>'data: '+JSON.stringify(x)+'\n\n';
    res.writeHead(200,{'content-type':'text/event-stream'});
    res.end(frame({id:'overloaded-response',choices:[{index:0,delta:{role:'assistant'},finish_reason:null}]})+
      frame({error:{type:'upstream_error',message:'Our servers are currently overloaded. Please try again later.'}})+
      frame({id:'overloaded-response',choices:[],usage:{prompt_tokens:1200,completion_tokens:0,total_tokens:1200,prompt_tokens_details:{cached_tokens:0}}})+'data: [DONE]\n\n');
  });
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const cfg=resolve(root,'config.json'),q=resolve(root,'question.txt'),out=resolve(root,'case');
  writeFileSync(cfg,JSON.stringify({...config(`http://127.0.0.1:${server.address().port}/v1`),pricing:{currency:'USD',tiers:[{input:.1,cache_read:.01,output:.5}]}}));
  writeFileSync(q,'question');
  try {
    const summary=await run(cfg,q,out,{backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
    assert.equal(requests,1);assert.equal(summary.status,'model_error');assert.equal(summary.decisions,0);
    assert.ok(summary.failed_api_seconds>0);
    const usage=JSON.parse(readFileSync(resolve(out,'TOKEN_USAGE.json')));
    assert.equal(usage.token_counts.input_total,1200);assert.equal(usage.token_counts.output_total,0);
    assert.equal(usage.total_cost,null);assert.ok(Math.abs(usage.partial_or_failed_known_usage_cost-.00012)<1e-15);
    assert.ok(readApi(out,1).response_sse.includes('"total_tokens":1200'));
    assert.equal(readdirSync(out).includes('STORAGE_ERROR.json'),false);
  } finally {await new Promise(r=>{server.close(r);server.closeAllConnections();});}
}));

test('configured request pacing spaces physical calls and remains in normal budget',async()=>fixture(async(root)=>{
  const times=[];
  const server=await serverFor((p,n)=>{times.push(Date.now());return n===1?{tool_calls:[call(0,'x','read',{path:'x'})]}:{content:'done'};});
  let session;
  try {
    const created=await createCepSession(root,{...config(server.url),minimum_request_interval_seconds:.12},{execute:async()=>({content:[{type:'text',text:'ok'}]})});
    session=created.session;created.state.clock.start();await session.prompt('question');
    assert.equal(times.length,2);assert.ok(times[1]-times[0]>=100);
    assert.equal(created.state.requestWaits.length,1);assert.ok(created.state.requestWaits[0].seconds>.08);
    assert.equal(created.state.clock.snapshot().failed_api_seconds,0);
    assert.ok(created.state.clock.snapshot().normal_seconds>=created.state.requestWaits[0].seconds);
  } finally {session?.dispose();await server.close();}
}));

test('real Pi rate-limit retries preserve exact payload/physical usage and exclude waits without extra decisions',async()=>fixture(async(root)=>{
 const payloads=[];
 const server=createServer(async(req,res)=>{
  let text='';for await(const b of req)text+=b;payloads.push(text);
  if(payloads.length<3){res.writeHead(payloads.length===1?503:429,{'content-type':'application/json','retry-after':'60'});res.end(JSON.stringify({error:{code:'model:rate_limited',message:'Too many requests'}}));return;}
  res.writeHead(200,{'content-type':'text/event-stream'});
  const frame=x=>'data: '+JSON.stringify(x)+'\n\n';
  res.end(frame({id:'success-after-limit',choices:[{index:0,delta:{role:'assistant',content:'final'},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:30,total_tokens:130,prompt_cache_hit_tokens:80,prompt_cache_miss_tokens:20,completion_tokens_details:{reasoning_tokens:20}}})+'data: [DONE]\n\n');
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const cfg=resolve(root,'config.json'),q=resolve(root,'question.txt'),out=resolve(root,'case');
 writeFileSync(cfg,JSON.stringify({...config(`http://127.0.0.1:${server.address().port}/v1`),api_timeout_seconds:.15,max_seconds:.5,
  rate_limit_retry:{enabled:true,min_delay_seconds:.25,max_delay_seconds:.3,max_attempts:3,max_elapsed_seconds:3},pricing:{currency:'USD',tiers:[{input:.15,cache_read:.003,output:.6}]}}));writeFileSync(q,'question');
 try {
  const summary=await run(cfg,q,out,{backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
  assert.equal(summary.status,'answered');assert.equal(summary.decisions,1);assert.equal(summary.attempted,1);
  assert.equal(payloads.length,3);assert.equal(new Set(payloads).size,1);
  assert.ok(summary.failed_api_seconds>=.49);assert.ok(summary.normal_seconds<.5);
  assert.equal(summary.rate_limit_retries.length,2);assert.equal(summary.retry_time_exclusions.length,4);
  const u=JSON.parse(readFileSync(resolve(out,'TOKEN_USAGE.json')));
  assert.equal(u.physical_requests,3);assert.equal(u.token_counts.output_total,30);assert.equal(u.token_counts.reasoning_subset,20);
  assert.equal(u.unknown_usage_calls,2);assert.equal(u.total_cost,null);assert.ok(Math.abs(u.successful_generation_cost-.00002124)<1e-15);
  assert.equal(readTrajectory(out).length,2);
  for(const i of [1,2,3]){const a=readApi(out,i);assert.equal(a['request.meta.json'].logical_request_index,1);assert.equal(a['request.meta.json'].physical_attempt,i);}
 }finally{await new Promise(r=>{server.close(r);server.closeAllConnections();});}
}));

test('pre-content SSE overload retries; failed usage tail and successful answer remain separate',async()=>fixture(async(root)=>{
 let requests=0;
 const frame=x=>'data: '+JSON.stringify(x)+'\n\n';
 const server=createServer(async(req,res)=>{
  for await(const b of req){}requests++;res.writeHead(200,{'content-type':'text/event-stream'});
  if(requests===1)res.end(frame({id:'failed-stream',choices:[{delta:{role:'assistant'},finish_reason:null}]})+frame({error:{type:'upstream_error',message:'Our servers are currently overloaded.'}})+frame({id:'failed-stream',choices:[],usage:{prompt_tokens:1200,completion_tokens:0,total_tokens:1200}})+'data: [DONE]\n\n');
  else res.end(frame({id:'successful-stream',choices:[{index:0,delta:{role:'assistant',content:'answer'},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:10,total_tokens:110}})+'data: [DONE]\n\n');
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const cfg=resolve(root,'config.json'),q=resolve(root,'question.txt'),out=resolve(root,'case');
 writeFileSync(cfg,JSON.stringify({...config(`http://127.0.0.1:${server.address().port}/v1`),rate_limit_retry:{enabled:true,min_delay_seconds:.01,max_delay_seconds:.01,max_attempts:2,max_elapsed_seconds:2},pricing:{currency:'USD',tiers:[{input:.1,output:.5}]}}));writeFileSync(q,'question');
 try {
  const summary=await run(cfg,q,out,{backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
  assert.equal(requests,2);assert.equal(summary.status,'answered');assert.equal(summary.decisions,1);
  assert.equal(readTrajectory(out).at(-1).content[0].text,'answer');
  const usage=JSON.parse(readFileSync(resolve(out,'TOKEN_USAGE.json')));
  assert.equal(usage.token_counts.input_total,1300);assert.equal(usage.token_counts.output_total,10);
  assert.ok(Math.abs(usage.partial_or_failed_known_usage_cost-.00012)<1e-15);
  assert.ok(readApi(out,1).response_sse.includes('"total_tokens":1200'));
 }finally{await new Promise(r=>{server.close(r);server.closeAllConnections();});}
}));

test('explicit continuation restores full Pi history and cumulative budget without a new user prompt',async()=>fixture(async(root)=>{
 const server=await serverFor(()=>({content:'resumed final'}));
 const cfg=resolve(root,'config.json'),q=resolve(root,'q.txt'),prior=resolve(root,'prior'),out=resolve(root,'continued');
 const settings=config(server.url);writeFileSync(cfg,JSON.stringify(settings));writeFileSync(q,'resume question');mkdirSync(prior);
 writeFileSync(resolve(prior,'CONFIG.json'),JSON.stringify(settings));writeFileSync(resolve(prior,'question.txt'),'resume question');
 const {CaseRecorder}=await import('../extensions/storage.mjs');const recorder=new CaseRecorder(prior);
 const messages=[{role:'user',content:[{type:'text',text:'resume question'}],timestamp:1},{role:'assistant',content:[],stopReason:'error',timestamp:2}];
 for(const message of messages)recorder.record({type:'message_end',message});
 const complete={status:'model_error',decisions:0,attempted:1,wall_seconds:12,failed_api_seconds:10,normal_seconds:2,api_call_timing:[{outcome:'error',seconds:10,excluded_seconds:10}]};
 writeFileSync(resolve(prior,'COMPLETE.json'),JSON.stringify(complete));writeFileSync(resolve(prior,'TOKEN_USAGE.json'),'{}');writeFileSync(resolve(prior,'context.jsonl'),'');
 const ref=recorder.store.putBytes(Buffer.from('{"ids":[]}'));writeFileSync(resolve(prior,'retrieval_and_artifacts.json'),JSON.stringify({'state.json':ref}));recorder.store.archive();
 try{
  const summary=await run(cfg,q,out,{resumeFrom:prior,backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
  assert.equal(summary.status,'answered',JSON.stringify(readTrajectory(out).at(-1)));assert.equal(summary.attempted,2);assert.equal(summary.decisions,1);
  assert.ok(summary.normal_seconds>=2);assert.equal(summary.failed_api_seconds,10);
  assert.equal(readTrajectory(out).filter(m=>m.role==='user').length,1);assert.equal(server.requests.length,1);
  assert.equal(server.requests[0].messages[0].content,prompt);
 }finally{await server.close();}
}));

test('explicit interrupted empty request continuation preserves history and cumulative budget',async()=>fixture(async(root)=>{
 const server=await serverFor(()=>({content:'resumed final'}));
 const cfg=resolve(root,'config.json'),q=resolve(root,'q.txt'),prior=resolve(root,'prior'),out=resolve(root,'continued');
 const settings=config(server.url);writeFileSync(cfg,JSON.stringify(settings));writeFileSync(q,'resume question');mkdirSync(prior);
 writeFileSync(resolve(prior,'CONFIG.json'),JSON.stringify(settings));writeFileSync(resolve(prior,'question.txt'),'resume question');
 const {CaseRecorder}=await import('../extensions/storage.mjs');const recorder=new CaseRecorder(prior);
 const messages=[{role:'user',content:[{type:'text',text:'resume question'}],timestamp:1},{role:'assistant',content:[],stopReason:'aborted',timestamp:2}];
 for(const message of messages)recorder.record({type:'message_end',message});
 const complete={status:'interrupted',decisions:0,attempted:1,wall_seconds:12,failed_api_seconds:10,normal_seconds:2,api_call_timing:[{outcome:'error',seconds:10,excluded_seconds:10}]};
 writeFileSync(resolve(prior,'COMPLETE.json'),JSON.stringify(complete));writeFileSync(resolve(prior,'TOKEN_USAGE.json'),'{}');writeFileSync(resolve(prior,'context.jsonl'),'');
 const ref=recorder.store.putBytes(Buffer.from('{"ids":[]}'));writeFileSync(resolve(prior,'retrieval_and_artifacts.json'),JSON.stringify({'state.json':ref}));recorder.store.archive();
 try{
  const summary=await run(cfg,q,out,{resumeFrom:prior,backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
  assert.equal(summary.status,'answered',JSON.stringify(readTrajectory(out).at(-1)));assert.equal(summary.attempted,2);assert.equal(summary.decisions,1);
  assert.ok(summary.normal_seconds>=2);assert.equal(summary.failed_api_seconds,10);
  assert.equal(readTrajectory(out).filter(m=>m.role==='user').length,1);assert.equal(server.requests.length,1);
  assert.equal(server.requests[0].messages[0].content,prompt);
 }finally{await server.close();}
}));

test('explicit interrupted before first tool resumes without nonexistent graph state',async()=>fixture(async(root)=>{
 const server=await serverFor(()=>({content:'resumed final'}));
 const cfg=resolve(root,'config.json'),q=resolve(root,'q.txt'),prior=resolve(root,'prior'),out=resolve(root,'continued');
 const settings=config(server.url);writeFileSync(cfg,JSON.stringify(settings));writeFileSync(q,'resume question');mkdirSync(prior);
 writeFileSync(resolve(prior,'CONFIG.json'),JSON.stringify(settings));writeFileSync(resolve(prior,'question.txt'),'resume question');
 const {CaseRecorder}=await import('../extensions/storage.mjs');const recorder=new CaseRecorder(prior);
 const messages=[{role:'user',content:[{type:'text',text:'resume question'}],timestamp:1},{role:'assistant',content:[],stopReason:'aborted',timestamp:2}];
 for(const message of messages)recorder.record({type:'message_end',message});
 const complete={status:'interrupted',tools:0,decisions:0,attempted:1,wall_seconds:12,failed_api_seconds:10,normal_seconds:2,api_call_timing:[{outcome:'error',seconds:10,excluded_seconds:10}]};
 writeFileSync(resolve(prior,'COMPLETE.json'),JSON.stringify(complete));writeFileSync(resolve(prior,'TOKEN_USAGE.json'),'{}');writeFileSync(resolve(prior,'context.jsonl'),'');
 const ref=recorder.store.putBytes(Buffer.from('{"ids":[]}'));writeFileSync(resolve(prior,'retrieval_and_artifacts.json'),JSON.stringify({}));recorder.store.archive();
 try{
  const summary=await run(cfg,q,out,{resumeFrom:prior,backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
  assert.equal(summary.status,'answered',JSON.stringify(readTrajectory(out).at(-1)));assert.equal(summary.attempted,2);assert.equal(summary.decisions,1);
  assert.ok(summary.normal_seconds>=2);assert.equal(summary.failed_api_seconds,10);
  assert.equal(readTrajectory(out).filter(m=>m.role==='user').length,1);assert.equal(server.requests.length,1);
  assert.equal(server.requests[0].messages[0].content,prompt);
 }finally{await server.close();}
}));

test('midstream reasoning and tool SSE overload retries; failed usage tail and successful answer remain separate',async()=>fixture(async(root)=>{
 let requests=0;
 const frame=x=>'data: '+JSON.stringify(x)+'\n\n';
 const server=createServer(async(req,res)=>{
  for await(const b of req){}requests++;res.writeHead(200,{'content-type':'text/event-stream'});
  if(requests===1)res.end(frame({id:'failed-stream',choices:[{delta:{role:'assistant',reasoning_content:'partial plan',tool_calls:[{index:0,id:'must-not-execute',type:'function',function:{name:'bash',arguments:'{"command":"echo should-not-run"}'}}]},finish_reason:null}]})+frame({error:{type:'upstream_error',message:'Our servers are currently overloaded.'}})+frame({id:'failed-stream',choices:[],usage:{prompt_tokens:1200,completion_tokens:0,total_tokens:1200}})+'data: [DONE]\n\n');
  else res.end(frame({id:'successful-stream',choices:[{index:0,delta:{role:'assistant',content:'answer'},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:10,total_tokens:110}})+'data: [DONE]\n\n');
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const cfg=resolve(root,'config.json'),q=resolve(root,'question.txt'),out=resolve(root,'case');
 writeFileSync(cfg,JSON.stringify({...config(`http://127.0.0.1:${server.address().port}/v1`),rate_limit_retry:{enabled:true,min_delay_seconds:.01,max_delay_seconds:.01,max_attempts:2,max_elapsed_seconds:2},pricing:{currency:'USD',tiers:[{input:.1,output:.5}]}}));writeFileSync(q,'question');
 try {
  const summary=await run(cfg,q,out,{backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
  assert.equal(requests,2);assert.equal(summary.status,'answered');assert.equal(summary.decisions,1);
  assert.equal(readTrajectory(out).at(-1).content[0].text,'answer');
  const usage=JSON.parse(readFileSync(resolve(out,'TOKEN_USAGE.json')));
  assert.equal(usage.token_counts.input_total,1300);assert.equal(usage.token_counts.output_total,10);
  assert.ok(Math.abs(usage.partial_or_failed_known_usage_cost-.00012)<1e-15);
  assert.ok(readApi(out,1).response_sse.includes('"total_tokens":1200'));
 }finally{await new Promise(r=>{server.close(r);server.closeAllConnections();});}
}));

test('explicit interrupted partial unexecuted request continuation preserves history and cumulative budget',async()=>fixture(async(root)=>{
 const server=await serverFor(()=>({content:'resumed final'}));
 const cfg=resolve(root,'config.json'),q=resolve(root,'q.txt'),prior=resolve(root,'prior'),out=resolve(root,'continued');
 const settings=config(server.url);writeFileSync(cfg,JSON.stringify(settings));writeFileSync(q,'resume question');mkdirSync(prior);
 writeFileSync(resolve(prior,'CONFIG.json'),JSON.stringify(settings));writeFileSync(resolve(prior,'question.txt'),'resume question');
 const {CaseRecorder}=await import('../extensions/storage.mjs');const recorder=new CaseRecorder(prior);
 const messages=[{role:'user',content:[{type:'text',text:'resume question'}],timestamp:1},{role:'assistant',content:[{type:'thinking',thinking:'partial'},{type:'toolCall',id:'pending-unexecuted',name:'bash',arguments:{command:'echo x'}}],stopReason:'aborted',timestamp:2}];
 for(const message of messages)recorder.record({type:'message_end',message});
 const complete={status:'interrupted',decisions:0,attempted:1,wall_seconds:12,failed_api_seconds:10,normal_seconds:2,api_call_timing:[{outcome:'error',seconds:10,excluded_seconds:10}]};
 writeFileSync(resolve(prior,'COMPLETE.json'),JSON.stringify(complete));writeFileSync(resolve(prior,'TOKEN_USAGE.json'),'{}');writeFileSync(resolve(prior,'context.jsonl'),'');
 const ref=recorder.store.putBytes(Buffer.from('{"ids":[]}'));writeFileSync(resolve(prior,'retrieval_and_artifacts.json'),JSON.stringify({'state.json':ref}));recorder.store.archive();
 try{
  const summary=await run(cfg,q,out,{resumeFrom:prior,backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
  assert.equal(summary.status,'answered',JSON.stringify(readTrajectory(out).at(-1)));assert.equal(summary.attempted,2);assert.equal(summary.decisions,1);
  assert.ok(summary.normal_seconds>=2);assert.equal(summary.failed_api_seconds,10);
  assert.equal(readTrajectory(out).filter(m=>m.role==='user').length,1);assert.equal(server.requests.length,1);
  assert.equal(server.requests[0].messages[0].content,prompt);
 }finally{await server.close();}
}));

test('explicit interrupted uncommitted initial graph failure retains error and empty committed state',async()=>fixture(async(root)=>{
 const server=await serverFor(()=>({content:'resumed final'}));
 const cfg=resolve(root,'config.json'),q=resolve(root,'q.txt'),prior=resolve(root,'prior'),out=resolve(root,'continued');
 const settings=config(server.url);writeFileSync(cfg,JSON.stringify(settings));writeFileSync(q,'resume question');mkdirSync(prior);
 writeFileSync(resolve(prior,'CONFIG.json'),JSON.stringify(settings));writeFileSync(resolve(prior,'question.txt'),'resume question');
 const {CaseRecorder}=await import('../extensions/storage.mjs');const recorder=new CaseRecorder(prior);
 const messages=[{role:'user',content:[{type:'text',text:'resume question'}],timestamp:1},{role:'assistant',content:[{type:'toolCall',id:'failed-graph',name:'graphsearch',arguments:{queries:['test']}}],stopReason:'toolUse',timestamp:2},{role:'toolResult',toolCallId:'failed-graph',toolName:'graphsearch',isError:true,content:[{type:'text',text:'Graph bridge exited'}],timestamp:3},{role:'assistant',content:[],stopReason:'aborted',timestamp:4}];
 for(const message of messages)recorder.record({type:'message_end',message});
 const complete={status:'interrupted',tools:1,decisions:1,attempted:2,wall_seconds:12,failed_api_seconds:10,normal_seconds:2,api_call_timing:[{outcome:'error',seconds:10,excluded_seconds:10}]};
 writeFileSync(resolve(prior,'COMPLETE.json'),JSON.stringify(complete));writeFileSync(resolve(prior,'TOKEN_USAGE.json'),'{}');writeFileSync(resolve(prior,'context.jsonl'),'');
 const ref=recorder.store.putBytes(Buffer.from('{"ids":[]}'));writeFileSync(resolve(prior,'retrieval_and_artifacts.json'),JSON.stringify({}));recorder.store.archive();
 try{
  const summary=await run(cfg,q,out,{resumeFrom:prior,backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});
  assert.equal(summary.status,'answered',JSON.stringify(readTrajectory(out).at(-1)));assert.equal(summary.attempted,3);assert.equal(summary.decisions,2);
  assert.ok(summary.normal_seconds>=2);assert.equal(summary.failed_api_seconds,10);
  assert.equal(readTrajectory(out).filter(m=>m.role==='user').length,1);assert.equal(server.requests.length,1);
  assert.equal(server.requests[0].messages[0].content,prompt);
 }finally{await server.close();}
}));

test('physical SSE timeout retries identical payload without delivering unfinished tool deltas',async()=>fixture(async(root)=>{
 const bodies=[],frame=x=>'data: '+JSON.stringify(x)+'\n\n';
 const server=createServer(async(req,res)=>{let body='';for await(const b of req)body+=b;bodies.push(body);res.writeHead(200,{'content-type':'text/event-stream'});
  if(bodies.length===1){res.write(frame({id:'timed-out-stream',choices:[{index:0,delta:{role:'assistant',reasoning_content:'partial',tool_calls:[{index:0,id:'not-executed',type:'function',function:{name:'bash',arguments:'{"command":"'}}]},finish_reason:null}]}));}
  else res.end(frame({id:'success-after-timeout',choices:[{index:0,delta:{role:'assistant',content:'answer'},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:10,total_tokens:110}})+'data: [DONE]\n\n');
 });await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const cfg=resolve(root,'config.json'),q=resolve(root,'question.txt'),out=resolve(root,'case');
 writeFileSync(cfg,JSON.stringify({...config(`http://127.0.0.1:${server.address().port}/v1`),api_timeout_seconds:.15,rate_limit_retry:{enabled:true,min_delay_seconds:.01,max_delay_seconds:.01,max_attempts:2,max_elapsed_seconds:2}}));writeFileSync(q,'question');
 try{const summary=await run(cfg,q,out,{backendFactory:folder=>createBackend(folder,{graph:{start:async()=>{},close:async()=>{}}})});assert.equal(summary.status,'answered');assert.equal(summary.decisions,1);assert.equal(summary.tools,0);assert.equal(bodies.length,2);assert.equal(bodies[0],bodies[1]);assert.ok(summary.failed_api_seconds>=.15);assert.ok(readApi(out,1)['ERROR.json'].error.includes('Timeout'));assert.equal(readTrajectory(out).length,2);assert.equal(summary.rate_limit_retries[0].kind,'gateway_timeout');}finally{await new Promise(r=>{server.close(r);server.closeAllConnections();});}
}));
