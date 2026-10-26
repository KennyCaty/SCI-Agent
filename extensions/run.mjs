import {resolve} from 'node:path';
import {readFileSync, mkdirSync, writeFileSync, readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {json, ROOT, PI, MAP_RUNTIME, DR_RUNTIME} from './paths.mjs';
import {createBackend} from './tools.mjs';
import {createCepSession} from './session.mjs';
import {installWireLog} from './wire-log.mjs';
import {watchBudget} from './budget.mjs';
import {compactCase} from './storage.mjs';
import {validatePricing,writeAccounting} from './accounting.mjs';
import {retryPolicy} from './rate-limit.mjs';
import {restoreContinuation} from './resume.mjs';
import {taskPrompt} from './task.mjs';
import {applyRuntimeOptions,parseRuntimeArgs} from './runtime-options.mjs';

export async function run(configFile, questionFile, folder, {backendFactory=createBackend,resumeFrom=null,...runtimeOptions}={}) {
  const config = applyRuntimeOptions(json(configFile),runtimeOptions);
  validatePricing(config.pricing);
  retryPolicy(config.rate_limit_retry);
  if (!Number.isInteger(config.max_turns) || config.max_turns < 1 || !(config.max_seconds > 0)) throw new Error('Invalid budgets');
  folder = resolve(folder);
  mkdirSync(folder, {recursive:false});
  const question = readFileSync(questionFile,'utf8');
  if (!question.trim()) throw new Error('Empty question');
  writeFileSync(resolve(folder,'question.txt'),question);
  writeFileSync(resolve(folder,'CONFIG.json'),JSON.stringify(config,null,2)+'\n');
  const resumed=resumeFrom?restoreContinuation(resumeFrom,folder,config,question):null;
  const sha = p => createHash('sha256').update(readFileSync(p)).digest('hex');
  const sources = [resolve(ROOT,'prompts/qa.txt'),resolve(ROOT,'configs/tools.json'),
    resolve(MAP_RUNTIME,'expand_map.py'),...['dr_read.mjs','dr_bash_postprocess.mjs','dr_dci_truncate.cjs'].map(p=>resolve(DR_RUNTIME,p)),
    resolve(PI,'dist/core/sdk.js'),...['bash.js','read.js','truncate.js'].map(p=>resolve(PI,'dist/core/tools',p)),resolve(PI,'../agent/dist/agent-loop.js'),
    resolve(ROOT,'configs/retrieval/embedding_profiles.json'),
    ...readdirSync(MAP_RUNTIME).filter(p=>p.endsWith('.py')).map(p=>resolve(MAP_RUNTIME,p)),
    ...readdirSync(resolve(ROOT,'extensions')).filter(p=>p.endsWith('.mjs')).map(p=>resolve(ROOT,'extensions',p)),
    ...['runtime','benchmark'].flatMap(d=>readdirSync(resolve(ROOT,'src/sci',d)).filter(p=>p.endsWith('.py')).map(p=>resolve(ROOT,'src/sci',d,p))),
    ...['cli.py','paths.py','__main__.py'].map(p=>resolve(ROOT,'src/sci',p)),
    ...(config.task?.type==='trec-covid'?[resolve(ROOT,'prompts/trec_covid.txt')]:[]),
    ...(process.env.CEP_PI_DEPLOYMENT_CONFIG?[process.env.CEP_PI_DEPLOYMENT_CONFIG]:[])];
  writeFileSync(resolve(folder,'SOURCE_HASHES.json'),JSON.stringify(Object.fromEntries(sources.map(p=>[p,sha(p)])),null,2)+'\n');
  const backend = backendFactory(folder,config);
  if(resumed)backend.resumeContext=resumed.context;
  let session, state, recorder, timer, restoreFetch=()=>{}, interrupted=false;
  const stop = () => { interrupted=true; if(state)state.status='interrupted';session?.agent.abort(); void backend.close(); };
  process.once('SIGINT',stop); process.once('SIGTERM',stop);
  try {
    ({session,state,recorder} = await createCepSession(folder,config,backend));
    if(resumed){
      session.agent.state.messages=resumed.messages;
      session.agent.state.systemPrompt=taskPrompt(config);
      for(const message of resumed.messages)recorder.record({type:'message_end',message});
      state.decisions=resumed.complete.decisions;state.attempted=resumed.complete.attempted;
      state.clock.failedSeconds=resumed.complete.failed_api_seconds;
      state.clock.calls=[...resumed.complete.api_call_timing];
    }
    await backend.bridge.start();
    if (interrupted) throw new Error('Interrupted before generation');
    restoreFetch = installWireLog(folder,config.base_url ?? process.env[config.base_url_env],
      {api:config.api,store:recorder.store,expectedFirstRequest:resumed?.expectedRequest,initialRetryState:resumed?.initialRetryState,currentContext:()=>recorder.contexts.at(-1)?.index,
        retry:config.rate_limit_retry,apiTimeoutSeconds:config.api_timeout_seconds??300,
        onRetry:record=>state.rateLimitRetries.push(record),onRetryExcluded:record=>state.clock.excludeRetrySpan(record)});
    state.clock.start();
    if(resumed)state.clock.started-=resumed.complete.wall_seconds*1000;
    timer = watchBudget(state.clock,() => {
        state.status='normal_time_limit'; session.agent.abort(); void backend.close();
    });
    if(resumed)await session.agent.continue();else await session.prompt(question);
    clearInterval(timer);
    const messages = session.messages;
    const last = messages.findLast(m=>m.role==='assistant');
    const timing = state.clock.snapshot(), normal = timing.normal_seconds;
    const answered = !interrupted && !state.status && normal < config.max_seconds && last?.stopReason === 'stop'
      && !last.content.some(c=>c.type==='toolCall');
    const status = interrupted ? 'interrupted' : state.status ?? (answered ? 'answered' : 'model_error');
    if (answered) writeFileSync(resolve(folder,'final.txt'),last.content.filter(c=>c.type==='text').map(c=>c.text).join(''));
    const coverage=recorder.finish(messages);
    const summary = {status,decisions:state.decisions,attempted:state.attempted,...timing,
      embedding:backend.bridge.embedding,resumed_from:resumed?.parent??null,
      api_call_timing:state.clock.calls,tools:messages.filter(m=>m.role==='toolResult').length,
      request_pacing:state.requestWaits,
      rate_limit_retries:state.rateLimitRetries,retry_time_exclusions:state.clock.retryExclusions,
      tool_timing:backend.timingSummary(),
      trajectory_coverage:coverage,
      usages:messages.filter(m=>m.role==='assistant').map(m=>({stopReason:m.stopReason,usage:m.usage})),
      pricing:'Authoritative raw-usage accounting: TOKEN_USAGE.json. Pi usage/cost is diagnostic only. No judge run.',token_accounting_file:'TOKEN_USAGE.json'};
    writeFileSync(resolve(folder,'COMPLETE.json'),JSON.stringify(summary,null,2)+'\n');
    return summary;
  } catch (e) {
    writeFileSync(resolve(folder,'ERROR.json'),JSON.stringify({error:e.message,interrupted},null,2)+'\n');
    throw e;
  } finally {
    clearInterval(timer); session?.dispose(); await backend.close();
    await restoreFetch.settled?.();restoreFetch();
    try{writeAccounting(folder,config);}
    catch(e){writeFileSync(resolve(folder,'ACCOUNTING_ERROR.json'),JSON.stringify({error:e.message,raw_usage_retained:true})+'\n');}
    if(recorder){
      try{recorder.finish(session.messages);compactCase(folder,recorder.store);}
      catch(e){writeFileSync(resolve(folder,'STORAGE_ERROR.json'),JSON.stringify({error:e.message,cleanup_skipped:true})+'\n');}
    }
    process.removeListener('SIGINT',stop); process.removeListener('SIGTERM',stop);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const {positionals,...options}=parseRuntimeArgs(process.argv.slice(2));
    console.log(JSON.stringify(await run(...positionals,options),null,2));
  } catch(e) {console.error(e.message);console.error('Usage: sci run CONFIG.json QUESTION.txt NEW_OUTPUT_DIRECTORY [--resume PREVIOUS_CASE] [--request-interval-seconds N] [--api openai-responses|openai-completions]');process.exitCode=1;}
}
