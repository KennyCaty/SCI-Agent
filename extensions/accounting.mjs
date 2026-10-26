import {readFileSync,existsSync,readdirSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {CaseStore} from './storage.mjs';
const number=(v,name)=>{if(!Number.isSafeInteger(v)||v<0)throw new Error(`Invalid ${name}`);return v;};
export function normalizeUsage(u) {
  if(!u||typeof u!=='object')throw new Error('Missing provider usage');
  const input=number(u.prompt_tokens??u.input_tokens,'input tokens'),output=number(u.completion_tokens??u.output_tokens,'completion tokens');
  if(u.total_tokens!==undefined&&number(u.total_tokens,'total tokens')!==input+output)throw new Error('Usage total mismatch');
  const details=u.prompt_tokens_details??u.input_tokens_details??{};
  const reasoning=u.completion_tokens_details?.reasoning_tokens??u.output_tokens_details?.reasoning_tokens??u.reasoning_tokens??null;
  if(reasoning!==null&&(number(reasoning,'reasoning tokens')>output))throw new Error('Reasoning is not a completion subset');
  const writes=number(details.cache_creation_tokens??details.cache_write_tokens??u.cache_creation_input_tokens??0,'cache write tokens');
  if(details.cache_creation_tokens!==undefined&&details.cache_write_tokens!==undefined&&details.cache_creation_tokens!==details.cache_write_tokens)throw new Error('Conflicting cache write fields');
  const reportedHit=u.prompt_cache_hit_tokens??details.cached_tokens??u.cache_read_input_tokens;
  const reportedMiss=u.prompt_cache_miss_tokens;
  // Chat cache-write counts overlap cached_tokens; Responses counts are disjoint.
  let hit=reportedHit===undefined?(reportedMiss===undefined?0:input-number(reportedMiss,'cache miss tokens')-writes):number(reportedHit,'cache hit tokens');
  if(u.prompt_tokens!==undefined&&u.prompt_cache_hit_tokens===undefined&&details.cache_creation_tokens===undefined&&details.cache_write_tokens!==undefined&&details.cached_tokens!==undefined)hit-=writes;
  const miss=input-hit-writes;
  if(hit<0||miss<0)throw new Error('Cache token partitions overlap/exceed input');
  if(reportedMiss!==undefined&&number(reportedMiss,'cache miss tokens')!==miss)throw new Error('Cache hit/miss mismatch');
  return {input_total:input,cache_hit:hit,cache_miss:miss,cache_write:writes,
    cache_fields_missing:reportedHit===undefined&&reportedMiss===undefined,
    missing_cache_policy:'assume_zero_hit_when_not_reported; retained as estimate',
    output_total:output,reasoning_subset:reasoning,non_reasoning_output:reasoning===null?null:output-reasoning,
    reasoning_added_to_billing:false};
}
export function validatePricing(pricing) {
  if(!pricing||pricing.mode==='unpriced_self_hosted')return;
  if(!pricing.currency||!Array.isArray(pricing.tiers)||!pricing.tiers.length)throw new Error('Invalid pricing config');
  let previous=-1;
  for(const [i,tier] of pricing.tiers.entries()) {
    const limit=tier.up_to_input_tokens??Infinity;
    if(!(limit>previous)||(!Number.isFinite(limit)&&i!==pricing.tiers.length-1))throw new Error('Invalid pricing tier thresholds');
    for(const field of ['input','output','cache_read','cache_write'])if(tier[field]!==undefined&&(!Number.isFinite(tier[field])||tier[field]<0))throw new Error(`Invalid price ${field}`);
    if(tier.input===undefined||tier.output===undefined)throw new Error('Missing input/output price');
    previous=limit;
  }
}
export function costOf(tokens,pricing) {
  validatePricing(pricing);
  if(!pricing||pricing.mode==='unpriced_self_hosted')return null;
  const tier=pricing.tiers.find(t=>tokens.input_total<=(t.up_to_input_tokens??Infinity));
  if(!tier)return null;
  const parts=[['cache_miss','input'],['cache_hit','cache_read'],['cache_write','cache_write'],['output_total','output']];
  if(parts.some(([n,p])=>tokens[n]>0&&tier[p]===undefined))return null;
  return parts.reduce((sum,[n,p])=>sum+tokens[n]*(tier[p]??0),0)/1e6;
}
export function extractUsage(text) {
  let last=null,id=null,finishReason=null,providerError=null;const issues=[];
  for(const block of text.split(/\r?\n\r?\n/)) {
    const data=block.split(/\r?\n/).filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n');
    if(!data||data==='[DONE]')continue;
    try {
      const event=JSON.parse(data);const message=event.response??event;if(message.id)id=message.id;
      if(event.type==='response.completed'&&message.status==='completed')finishReason=message.output?.some(x=>x.type==='function_call')?'tool_calls':'stop';
      if(event.type==='response.incomplete')finishReason='length';
      if(message.error)providerError=message.error;
      for(const choice of message.choices??[])if(choice.finish_reason)finishReason=choice.finish_reason;
      const u=message.usage;
      if(u&&((u.prompt_tokens??u.input_tokens)!==undefined)&&((u.completion_tokens??u.output_tokens)!==undefined))last=u;
    } catch {issues.push('Unparseable SSE frame retained in raw response');}
  }
  // Some endpoints return JSON even with stream=true.
  if(!last)try{
    const event=JSON.parse(text),body=event.response??event;
    id=body.id??id;if(body.usage)last=body.usage;
    if(body.error&&(event.response||body.object==='response'||(body.status&&body.output)))providerError=body.error;
    finishReason=body.status==='completed'?(body.output?.some(x=>x.type==='function_call')?'tool_calls':'stop'):body.choices?.[0]?.finish_reason??null;
  }catch{}
  return {raw_usage:last,response_id:id,finish_reason:finishReason,provider_error:providerError,issues};
}
export function summarizeCalls(calls,pricing) {
  validatePricing(pricing);
  const records=[],seen=new Map();let duplicate=0;
  for(const call of calls) {
    const record={...call,normalized:null,cost:null};
    if(call.raw_usage) {
      try {record.normalized=normalizeUsage(call.raw_usage);record.cost=costOf(record.normalized,pricing);}
      catch(error){record.accounting_error=error.message;}
      if(call.response_id) {
        if(seen.has(call.response_id)) {
          duplicate++;record.duplicate_response=true;
          if(JSON.stringify(seen.get(call.response_id))!==JSON.stringify(call.raw_usage))record.accounting_error='Conflicting usage for repeated response ID';
        } else seen.set(call.response_id,call.raw_usage);
      }
    }
    records.push(record);
  }
  const unique=records.filter(r=>!r.duplicate_response),known=unique.filter(r=>r.normalized);
  const sum=key=>known.reduce((s,r)=>s+r.normalized[key],0);
  const reasoningKnown=known.filter(r=>r.normalized.reasoning_subset!==null);
  const unknownUsage=unique.filter(r=>!r.normalized).length;
  const complete=unique.length>0&&unknownUsage===0&&unique.every(r=>r.final_response)&&!records.some(r=>r.accounting_error);
  const priced=known.filter(r=>r.cost!==null),estimated=priced.reduce((s,r)=>s+r.cost,0);
  return {version:'raw-provider-usage-v1',currency:pricing?.currency??null,pricing:pricing??null,
    billed_output_rule:'completion_tokens already includes reasoning; never add reasoning again',
    token_counts:{input_total:sum('input_total'),cache_hit:sum('cache_hit'),cache_miss:sum('cache_miss'),cache_write:sum('cache_write'),output_total:sum('output_total'),
      reasoning_subset:reasoningKnown.length===known.length&&known.length?sum('reasoning_subset'):null,
      non_reasoning_output:reasoningKnown.length===known.length&&known.length?sum('non_reasoning_output'):null,
      known_reasoning_subtotal:reasoningKnown.reduce((s,r)=>s+r.normalized.reasoning_subset,0),reasoning_unknown_calls:known.length-reasoningKnown.length,
      missing_cache_field_calls:known.filter(r=>r.normalized.cache_fields_missing).length},
    physical_requests:records.length,duplicate_response_ids:duplicate,unknown_usage_calls:unknownUsage,usage_complete:complete,
    known_usage_cost:priced.length?estimated:null,
    successful_generation_cost:priced.some(r=>r.final_response)?priced.filter(r=>r.final_response).reduce((s,r)=>s+r.cost,0):null,
    partial_or_failed_known_usage_cost:priced.some(r=>!r.final_response)?priced.filter(r=>!r.final_response).reduce((s,r)=>s+r.cost,0):null,
    total_cost:complete&&priced.length===known.length?estimated:null,
    cost_is_estimate:true,unreported_failure_billing:'unknown, not zero',calls:records};
}
export function readAccounting(folder,config) {
  const api=resolve(folder,'api'),calls=[];
  if(existsSync(api))for(const index of readdirSync(api).sort()) {
    const dest=resolve(api,index),raw=resolve(dest,'response.sse'),statusFile=resolve(dest,'status.json');
    const status=existsSync(statusFile)?JSON.parse(readFileSync(statusFile)).status:null;
    const parsed=existsSync(raw)?extractUsage(readFileSync(raw,'utf8')):{raw_usage:null,response_id:null,issues:[]};
    calls.push({index:Number(index),...parsed,http_status:status,
      final_response:status>=200&&status<300&&!!parsed.finish_reason&&!parsed.provider_error&&existsSync(resolve(dest,'END.json'))&&!existsSync(resolve(dest,'ERROR.json'))&&!existsSync(resolve(dest,'CANCELLED.json'))});
  }
  else if(existsSync(resolve(folder,'api.jsonl'))) {
    const store=new CaseStore(folder);
    for(const line of readFileSync(resolve(folder,'api.jsonl'),'utf8').split('\n').filter(x=>x.trim())) {
      const row=JSON.parse(line),status=row['status.json']?.status??null;
      const parsed=row.response_ref?extractUsage(store.readBytes(row.response_ref).toString('utf8')):{raw_usage:null,response_id:null,issues:[]};
      calls.push({index:row.index,...parsed,http_status:status,final_response:status>=200&&status<300&&!!parsed.finish_reason&&!parsed.provider_error&&!!row['END.json']&&!row['ERROR.json']&&!row['CANCELLED.json']});
    }
  }
  return summarizeCalls(calls,config.pricing);
}
export function writeAccounting(folder,config) {
  const report=readAccounting(folder,config);
  writeFileSync(resolve(folder,'TOKEN_USAGE.json'),JSON.stringify(report,null,2)+'\n');return report;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href) {
  try {
    const [folder,prices]=process.argv.slice(2);
    if(!folder||process.argv.length>4)throw new Error('Usage: accounting.mjs CASE_DIRECTORY [PRICING.json]');
    const config=JSON.parse(readFileSync(resolve(folder,'CONFIG.json')));
    if(prices){const value=JSON.parse(readFileSync(prices));config.pricing=value.pricing??value;}
    console.log(JSON.stringify(readAccounting(folder,config),null,2));
  }catch(error){console.error(error.message);process.exitCode=1;}
}
