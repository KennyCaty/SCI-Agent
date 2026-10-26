import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {mkdirSync} from 'node:fs';
import {PI,schemas} from './paths.mjs';
import {cepExtension, assertContract} from './extension.mjs';
import {BudgetClock} from './budget.mjs';
import {CaseStore,CaseRecorder} from './storage.mjs';
import {setTimeout as delay} from 'node:timers/promises';
import {taskPrompt} from './task.mjs';
import {retryPolicy} from './rate-limit.mjs';
import {applyRuntimeOptions} from './runtime-options.mjs';
const sdk = await import(pathToFileURL(resolve(PI, 'dist/index.js')));

export async function createCepSession(folder, config, backend) {
  config=applyRuntimeOptions(config);
  const limitRetry=retryPolicy(config.rate_limit_retry);
  const systemPrompt=taskPrompt(config);
  const toolSchemas=schemas;
  const recorder=new CaseRecorder(folder,backend.store ?? new CaseStore(folder));
  backend.recordContext=(before,after,info)=>recorder.context(before,after,info);
  const workspace = resolve(folder, 'workspace'); mkdirSync(workspace, {recursive:true});
  const authStorage = sdk.AuthStorage.inMemory();
  const modelRegistry = sdk.ModelRegistry.inMemory(authStorage);
  const key = process.env[config.api_key_env];
  if (!key) throw new Error(`Missing environment variable ${config.api_key_env}`);
  // The OpenAI provider preserves reasoning items and IDs during serialization.
  const api=config.api??'openai-completions';
  const providerId=api==='openai-responses'?'openai':'cep-configured';
  authStorage.setRuntimeApiKey(providerId, key);
  const baseUrl = config.base_url ?? process.env[config.base_url_env];
  if (!baseUrl) throw new Error('Missing provider base URL');
  modelRegistry.registerProvider(providerId, {
    baseUrl, api, apiKey:config.api_key_env,
    models: [{id:config.model, name:config.model, reasoning:true, input:['text'],
      contextWindow:config.context_window, maxTokens:config.max_output_tokens,
      cost:config.cost_per_million ?? {input:0, output:0, cacheRead:0, cacheWrite:0},
      compat:{supportsDeveloperRole:false, supportsReasoningEffort:true,
        supportsStore:false, maxTokensField:config.max_tokens_field ?? 'max_completion_tokens',
        ...(config.compat ?? {})}}],
  });
  const fixedSettings = {
    compaction:{enabled:false}, runtimeContextManagement:{level:'level0'},
    retry:{enabled:false},
  };
  const settingsManager = sdk.SettingsManager.inMemory(fixedSettings);
  const loader = new sdk.DefaultResourceLoader({
    cwd:workspace, agentDir:resolve(folder, '.pi-agent'), settingsManager,
    noExtensions:true, noSkills:true, noPromptTemplates:true, noThemes:true,
    agentsFilesOverride:() => ({agentsFiles:[]}),
    appendSystemPromptOverride:() => [], systemPrompt,
    extensionFactories:[cepExtension(backend, config.compaction, systemPrompt,toolSchemas)],
  });
  // Reapply overrides because resource loading reloads settings from storage.
  const reload = loader.reload.bind(loader);
  loader.reload = async () => { await reload(); settingsManager.applyOverrides(fixedSettings); };
  await loader.reload();
  const {session} = await sdk.createAgentSession({
    cwd:workspace, agentDir:resolve(folder,'.pi-agent'), authStorage, modelRegistry,
    model:modelRegistry.find(providerId,config.model), thinkingLevel:config.thinking_level ?? 'high',
    tools:[], resourceLoader:loader, settingsManager,
    sessionManager:sdk.SessionManager.create(workspace, resolve(folder,'pi-session')),
  });
  session.agent.toolExecution = 'parallel';
  session.subscribe(event=>{
    // Expose plaintext reasoning from DS/vLLM; retain its signature for replay.
    if(api==='openai-responses'&&event.type==='message_end'&&event.message.role==='assistant'){
      for(const block of event.message.content)if(block.type==='thinking'&&!block.thinking&&block.thinkingSignature){
        try{
          const item=JSON.parse(block.thinkingSignature);
          if(Array.isArray(item.content))block.thinking=item.content.filter(p=>p.type==='reasoning_text'&&typeof p.text==='string').map(p=>p.text).join('\n\n');
        }catch{/* Opaque or unknown signatures stay unchanged. */}
      }
    }
    recorder.record(event);
  });
  const clock = new BudgetClock(config.max_seconds);
  const state = {decisions:0, attempted:0, clock, status:null,requestWaits:[],rateLimitRetries:[],lastRequestStarted:null,lastRequestFinished:null};
  Object.defineProperty(state,'failedSeconds',{get:()=>clock.failedSeconds});
  const budgetStop = () => { if (!state.status) state.status='normal_time_limit'; session.agent.abort(); };
  const nativeStream = session.agent.streamFn;
  session.agent.streamFn = async (model, context, options) => {
    if (state.decisions >= config.max_turns) { state.status='turn_limit'; throw new Error('CEP decision budget exhausted'); }
    if (clock.exhausted()) { budgetStop(); throw new Error('CEP normal-time budget exhausted'); }
    const spacing=config.minimum_request_interval_seconds??0;
    if(!Number.isFinite(spacing)||spacing<0)throw new Error('Invalid request interval');
    const last=state.lastRequestFinished??state.lastRequestStarted;
    const wait=last===null?0:Math.max(0,spacing*1000-(Date.now()-last));
    if(wait>0) {
      const started=Date.now();
      try {await delay(wait,undefined,{signal:options?.signal});}
      finally {state.requestWaits.push({before_attempt:state.attempted+1,seconds:(Date.now()-started)/1000,scope:'local request pacing; included in normal budget'});}
      if(clock.exhausted()){budgetStop();throw new Error('CEP normal-time budget exhausted during request pacing');}
    }
    state.lastRequestStarted=Date.now();
    state.attempted++; clock.beginCall();
    const callerSignal = options?.signal;
    const seconds = config.api_timeout_seconds ?? 300;
    if (!Number.isFinite(seconds) || seconds <= 0) {clock.finishCall('error',true);throw new Error('Invalid API timeout');}
    // Each HTTP retry has its own timeout, excluding Retry-After waits.
    const timeout = limitRetry.enabled ? null : AbortSignal.timeout(Math.ceil(seconds*1000));
    const signal = timeout ? (callerSignal ? AbortSignal.any([callerSignal,timeout]) : timeout) : callerSignal;
    try {
      const stream = await nativeStream(model, context, {...options, signal, maxTokens:config.max_output_tokens});
      stream.result().then(message => {
        state.lastRequestFinished=Date.now();
        clock.finishCall(message.stopReason,!!callerSignal?.aborted);
        if (!['error','aborted'].includes(message.stopReason)) state.decisions++;
        if (clock.exhausted()) budgetStop();
      });
      return stream;
    } catch (error) {state.lastRequestFinished=Date.now();clock.finishCall('error',!!callerSignal?.aborted);if(clock.exhausted())budgetStop();throw error;}
  };
  const nativePayload = session.agent.onPayload;
  session.agent.onPayload = async (payload, model) => {
    const p = await nativePayload(payload, model);
    p.parallel_tool_calls = true;
    if(api==='openai-responses'){
      if(config.responses_system_role)for(const item of p.input)if(['system','developer'].includes(item.role))item.role=config.responses_system_role;
      if(config.reasoning_effort===null)delete p.reasoning;
      else p.reasoning={...p.reasoning,effort:config.reasoning_effort??'high'};
      if(config.compat?.thinkingFormat==='qwen-chat-template')p.chat_template_kwargs={...p.chat_template_kwargs,enable_thinking:config.reasoning_effort!=='none'&&config.thinking_level!=='off'};
      if(backend.resumeContext?.prompt_cache_key)p.prompt_cache_key=backend.resumeContext.prompt_cache_key;
      delete p.reasoning_effort;delete p.stream_options;
    }else{
      p.stream_options={include_usage:true};
      if(config.reasoning_effort===null)delete p.reasoning_effort;
      else p.reasoning_effort=config.reasoning_effort??'high';
      if(config.thinking_enabled)p.thinking={type:'enabled'};
    }
    // Check here so extension event handling cannot swallow a contract error.
    assertContract(p,systemPrompt,toolSchemas);
    return p;
  };
  return {session, state, loader, settingsManager, recorder};
}
