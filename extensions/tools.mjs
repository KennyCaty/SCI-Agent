import {spawn,execFile} from 'node:child_process';
import {createInterface} from 'node:readline';
import {mkdirSync, writeFileSync, appendFileSync, realpathSync, existsSync, openSync, closeSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {randomUUID} from 'node:crypto';
import {ROOT, DR_RUNTIME, NATIVE, SCOPED_SHELL, PI, DEPLOYMENT, json} from './paths.mjs';
import {CaseStore} from './storage.mjs';
const {createBashToolDefinition,createLocalBashOperations} = await import(pathToFileURL(resolve(PI,'dist/core/tools/bash.js')));
const monotonic = () => Number(process.hrtime.bigint()) / 1e9;

function unionSeconds(intervals) {
  const sorted=intervals.filter(([a,b])=>Number.isFinite(a)&&Number.isFinite(b)&&b>=a).sort((a,b)=>a[0]-b[0]);
  let sum=0,start=null,end=null;
  for(const [a,b] of sorted){
    if(start===null){start=a;end=b;}else if(a<=end){end=Math.max(end,b);}else{sum+=end-start;start=a;end=b;}
  }
  return sum+(start===null?0:end-start);
}
export function summarizeToolTiming(events) {
  const queue=[],active=[],all=[];
  for(const e of events){
    all.push([e.submitted_monotonic,e.returned_monotonic]);
    if(e.clock_domain_verified===true){
      if(typeof e.queue_seconds==='number')queue.push([e.received_monotonic ?? e.submitted_monotonic,e.started_monotonic]);
      active.push([e.started_monotonic,e.ended_monotonic]);
    }
  }
  const knownSum=k=>events.reduce((n,e)=>n+(typeof e[k]==='number'?e[k]:0),0);
  return {calls:events.length,unknown_or_partial_timing_calls:events.filter(e=>!e.timing_complete).length,
    per_call_queue_seconds_known_sum:knownSum('queue_seconds'),
    per_call_execution_seconds_known_sum:knownSum('execution_seconds'),
    tool_wall_seconds_union:unionSeconds(all),known_queue_wall_seconds_union:unionSeconds(queue),
    known_execution_wall_seconds_union:unionSeconds(active),
    exclusive_known_queue_wall_seconds:Math.max(0,unionSeconds([...queue,...active])-unionSeconds(active)),
    note:'Per-call sums may overlap. Exclusive known queue is queue time without another observed tool executing; internal waits remain unmeasured.'};
}

// Stop only the process group owned by this question.
function terminate(child) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  try { process.kill(-child.pid, 'SIGKILL'); } catch (e) { if (e.code !== 'ESRCH') throw e; }
}
function childDone(child) {
  return new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('close', code => code === 0 ? resolve() : reject(new Error(`Tool child exited ${code}`)));
  });
}

export class GraphBridge {
  constructor(folder, python, script = null, env = {}) {
    this.folder = folder; this.python = python; this.script = script;
    this.pending = new Map(); this.child = null; this.closed = false;
    this.env = env;
  }
  async start() {
    if (this.closed) throw new Error('Graph bridge closed');
    if (this.ready) return this.ready;
    const log = openSync(resolve(this.folder, 'graph.stderr.log'), 'a', 0o600);
    this.child = spawn(this.python, ['-u', ...(this.script ? [this.script] : ['-m', 'sci.runtime.retrieval_server']), this.folder], {
      detached: true, stdio: ['pipe', 'pipe', log],
      env: {...process.env, ...this.env},
    });
    closeSync(log);
    this.ready = new Promise((resolveReady, reject) => {
      const fail = error => {
        reject(error);
        for (const p of this.pending.values()) {
          p.onTiming?.({...p.partial, timing_complete:false, execution_seconds:null,
            queue_seconds:p.partial.started_monotonic === undefined ? null : p.partial.started_monotonic-p.partial.received_monotonic});
          p.reject(error);
        }
        this.pending.clear();
      };
      this.child.on('error', fail);
      this.child.on('close', code => fail(new Error(`Graph bridge exited ${code}; see graph.stderr.log`)));
      this.child.stdin.on('error', fail);
      createInterface({input: this.child.stdout}).on('line', line => {
        try {
          const message = JSON.parse(line);
          if (message.ready) { this.pid = message.pid; this.embedding = message.embedding; resolveReady(); return; }
          const p = this.pending.get(message.id);
          if (!p) throw new Error('Unexpected graph response ID');
          if (message.event) { Object.assign(p.partial,message); return; }
          const responseReceived = monotonic(), timing = message.timing;
          if (timing) {
            p.onTiming?.({...timing, initialization_seconds:p.initialization,
              ipc_seconds:Math.max(0,responseReceived-p.sent-timing.queue_seconds-timing.execution_seconds),
              clock_domain_verified:timing.received_monotonic>=p.sent-0.05 && timing.ended_monotonic<=responseReceived+0.05,
              timing_complete:true});
          }
          this.pending.delete(message.id);
          message.ok ? p.resolve(message.result) : p.reject(new Error(message.error));
        } catch (e) { fail(e); terminate(this.child); }
      });
    });
    return this.ready;
  }
  async execute(name, args, id, signal, onTiming) {
    const abort = () => { if (this.child) terminate(this.child); };
    signal?.throwIfAborted(); signal?.addEventListener('abort', abort, {once: true});
    try {
      const beforeReady=monotonic(), wasReady=this.pid !== undefined;
      await this.start(); signal?.throwIfAborted();
      const initialization=wasReady ? 0 : monotonic()-beforeReady;
      return await new Promise((resolve, reject) => {
        this.pending.set(id, {resolve, reject, onTiming, initialization, sent:monotonic(),partial:{}});
        this.child.stdin.write(JSON.stringify({id, name, args}) + '\n');
      });
    } finally { signal?.removeEventListener('abort', abort); }
  }
  async close() {
    this.closed = true;
    if (!this.child || this.child.exitCode !== null || this.child.signalCode !== null) return;
    const done = new Promise(resolve => this.child.once('close', resolve));
    terminate(this.child); await done;
  }
}

export function createBackend(folder, {python = DEPLOYMENT.python ?? 'python3', graph, embedding, tool_processing = 'dr', bash_strip_rg_head_tail = false, task, workspace_naming, workspace_path_manifest} = {}) {
  if (!['dr','dci'].includes(tool_processing)) throw new Error('Invalid tool_processing');
  if(workspace_naming && workspace_naming!=='id')throw new Error('Only numeric source-ID workspaces are supported');
  if(bash_strip_rg_head_tail)throw new Error('bash_strip_rg_head_tail is not supported');
  const store=new CaseStore(folder);
  const workspace = resolve(folder, 'workspace');
  mkdirSync(resolve(workspace, '.tool_tmp'), {recursive: true});
  const embeddingFile = resolve(folder, 'EMBEDDING_CONFIG.json');
  writeFileSync(embeddingFile, JSON.stringify(embedding ?? {profile:'256'}, null, 2) + '\n');
  const bridge = graph ?? new GraphBridge(folder, python, undefined, {CEP_PI_EMBEDDING_CONFIG:embeddingFile,
    CEP_PI_TASK_TYPE:task?.type??'qa'});
  const children = new Set();
  const activeBash = new Set();
  const outputIndex = resolve(workspace,'.tool_tmp/native_outputs.json');
  const outputs = () => existsSync(outputIndex) ? json(outputIndex) : [];
  const quote = s => "'" + s.replaceAll("'", "'\\''") + "'";
  // Run Pi bash in this process so cancellation reaches its detached shell.
  const bash = createBashToolDefinition(workspace, {
    operations:createLocalBashOperations(),
    spawnHook:ctx => ({...ctx,command:[python,
      SCOPED_SHELL,
      '--workspace',workspace,...outputs().flatMap(p=>['--output-file',p]),
      '--','/bin/bash','--noprofile','--norc','-c',ctx.command].map(quote).join(' ')}),
  });
  // Serialize bash writes to the saved-output allowlist.
  let bashTail = Promise.resolve();
  let nextIndex=0, round=null;
  const events=[];
  async function run(argv, signal, log) {
    signal?.throwIfAborted();
    const fd = openSync(log, 'a', 0o600);
    const child = spawn(process.execPath, argv, {detached: true, stdio: ['ignore', fd, fd]});
    closeSync(fd); children.add(child);
    const abort = () => terminate(child);
    signal?.addEventListener('abort', abort, {once: true});
    try { await childDone(child); } finally { children.delete(child); signal?.removeEventListener('abort', abort); }
  }
  async function native(name, args, signal, timing) {
    signal?.throwIfAborted();
    timing.started_monotonic=monotonic();
    timing.queue_seconds=timing.started_monotonic-timing.submitted_monotonic;
    timing.execution_started=true;
    timing.clock_domain_verified=true;
    try {
    const io = resolve(folder, 'tool_io', randomUUID()); mkdirSync(io, {recursive: true});
    const input = resolve(io, 'input.json'), output = resolve(io, 'output.json');
    writeFileSync(input, JSON.stringify(args));
    if (name === 'read') {
      await run(tool_processing === 'dci' ? [NATIVE,workspace,'read',input,output] : [resolve(DR_RUNTIME,'dr_read.mjs'),workspace,input,output,NATIVE],signal,resolve(io,'stderr.log'));
    } else {
      const abort = new AbortController(), relay=()=>abort.abort();
      signal?.throwIfAborted(); signal?.addEventListener('abort',relay,{once:true}); activeBash.add(abort);
      try {
        let effectiveArgs=args;

        const result = await bash.execute(randomUUID(),effectiveArgs,abort.signal);
        if (result.details?.fullOutputPath) writeFileSync(outputIndex,JSON.stringify([...new Set([...outputs(),realpathSync(result.details.fullOutputPath)])]));
        writeFileSync(output,JSON.stringify({ok:true,result}));
      } finally {activeBash.delete(abort);signal?.removeEventListener('abort',relay);}
    }
    let record = json(output);
    if (name === 'bash' && record.ok && tool_processing === 'dr') {
      const postStarted=monotonic();
      const post = resolve(io, 'dr_input.json'), result = resolve(io, 'dr_output.json');
      writeFileSync(post, JSON.stringify({result: record, command: args.command}));
      await run([resolve(DR_RUNTIME, 'dr_bash_postprocess.mjs'), workspace, post, result], signal, resolve(io, 'stderr.log'));
      const processed = json(result);
      if (!processed.ok) throw new Error(processed.error);
      record = processed.result;
      timing.postprocessing_seconds=monotonic()-postStarted;
    }
    if (!record.ok) throw new Error(record.error);
    return record.result;
    } finally {
      timing.ended_monotonic=monotonic();
      timing.execution_seconds=timing.ended_monotonic-timing.started_monotonic;
      timing.timing_complete=true;
    }
  }
  return {
    bridge,
    store,
    beginTurn(value) { round=value; },
    timingSummary() {return summarizeToolTiming(events);},
    async execute(name, args, id, signal) {
      const submitted=monotonic(), index=++nextIndex;
      const event={index,round,id,name,tool:name,args,submitted_monotonic:submitted,
        submitted_at:new Date().toISOString(),queue_seconds:null,execution_seconds:null,
        initialization_seconds:0,ipc_seconds:0,postprocessing_seconds:0,timing_complete:false,
        execution_scope:'active_tool_wall_including_internal_waits',unmeasured_internal_queue_seconds:null};
      let result,error;
      try {
        if (name === 'graphsearch' || name === 'graphexpand') {
          const graphResult = await bridge.execute(name, args, id, signal, timing=>Object.assign(event,timing));
          result={content: [{type: 'text', text: graphResult.text}], details: graphResult.details};
        } else if (name === 'read') {
          result=await native(name,args,signal,event);
        } else if (name === 'bash') {
          const task = bashTail.then(() => native(name,args,signal,event));
          bashTail=task.catch(()=>{});
          result=await task;
        } else {
          throw new Error('Unknown tool');
        }
        return result;
      } catch(e) {error=String(e);throw e;
      } finally {
        event.seconds=monotonic()-submitted;event.returned_monotonic=monotonic();event.tool_error=!!error || !!result?.isError;
        if(error)event.error=error;
        if (['read','bash'].includes(name) && !event.execution_started && signal?.aborted) {
          event.queue_seconds=event.seconds;event.execution_seconds=0;event.cancelled_before_execution=true;event.timing_complete=true;
          event.started_monotonic=event.returned_monotonic;event.ended_monotonic=event.returned_monotonic;event.clock_domain_verified=true;
        }
        const accounted=[event.queue_seconds,event.execution_seconds,event.initialization_seconds,event.ipc_seconds];
        event.adapter_overhead_seconds=accounted.every(x=>typeof x==='number')
          ? Math.max(0,event.seconds-accounted.reduce((a,b)=>a+b,0)) : null;
        event.result_ref=store.putJson(error?{ok:false,error}: {ok:true,result});
        appendFileSync(resolve(folder,'tool_events.jsonl'),JSON.stringify(event)+'\n');
        events.push(event);
      }
    },
    async close() { for (const a of activeBash) a.abort(); for (const child of children) terminate(child); await bridge.close(); await bashTail; },
  };
}
