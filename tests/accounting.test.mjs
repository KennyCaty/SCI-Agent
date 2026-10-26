import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeUsage,costOf,extractUsage,summarizeCalls,validatePricing} from '../extensions/accounting.mjs';
const ds={currency:'USD',tiers:[{input:.15,cache_read:.003,output:.60}]};
const raw={prompt_tokens:100,completion_tokens:30,total_tokens:130,prompt_cache_hit_tokens:80,prompt_cache_miss_tokens:20,completion_tokens_details:{reasoning_tokens:20}};
const call=(id,u=raw,final_response=true)=>({response_id:id,raw_usage:u,final_response});
test('DS completion includes reasoning exactly once and cache hit/miss partition input',()=>{
 const tokens=normalizeUsage(raw);assert.equal(tokens.reasoning_subset,20);assert.equal(tokens.non_reasoning_output,10);
 assert.equal(tokens.output_total,30);assert.equal(tokens.cache_hit+tokens.cache_miss+tokens.cache_write,tokens.input_total);
 assert.ok(Math.abs(costOf(tokens,ds)-.00002124)<1e-15);
});
test('final cumulative SSE usage overwrites intermediate/duplicate usage instead of summing',()=>{
 const frame=u=>'data: '+JSON.stringify({id:'r1',usage:u,choices:[{finish_reason:'stop'}]})+'\n\n';
 const parsed=extractUsage(frame({...raw,completion_tokens:10,total_tokens:110})+frame(raw)+frame(raw)+'data: [DONE]\n\n');
 assert.deepEqual(parsed.raw_usage,raw);assert.equal(parsed.finish_reason,'stop');
 const report=summarizeCalls([call('r1'),call('r1')],ds);
 assert.equal(report.token_counts.output_total,30);assert.equal(report.duplicate_response_ids,1);assert.equal(report.total_cost,costOf(normalizeUsage(raw),ds));
});
test('reasoning or cache fields absent remain explicit unknown/assumption, self-hosted cost is not zero',()=>{
 const tokens=normalizeUsage({prompt_tokens:50,completion_tokens:10});
 assert.equal(tokens.reasoning_subset,null);assert.equal(tokens.non_reasoning_output,null);assert.equal(tokens.cache_fields_missing,true);
 const report=summarizeCalls([call('qwen',{prompt_tokens:50,completion_tokens:10})],{mode:'unpriced_self_hosted'});
 assert.equal(report.total_cost,null);assert.equal(report.known_usage_cost,null);assert.equal(report.token_counts.reasoning_subset,null);
});
test('partial or failed usage kept separately; unreported billing is unknown',()=>{
 const report=summarizeCalls([call('one'),call('two',raw,false),call(null,null,false)],ds);
 assert.equal(report.token_counts.output_total,60);assert.equal(report.unknown_usage_calls,1);assert.equal(report.total_cost,null);
 assert.equal(report.successful_generation_cost,costOf(normalizeUsage(raw),ds));
 assert.equal(report.partial_or_failed_known_usage_cost,report.successful_generation_cost);
});
test('input tiers and overlapping or disjoint cache-write counts',()=>{
 const pricing={currency:'USD',tiers:[{up_to_input_tokens:272000,input:.1,cache_read:.01,cache_write:.125,output:.5},{input:.2,cache_read:.02,cache_write:.25,output:.75}]};
 const legacy=normalizeUsage({prompt_tokens:100,completion_tokens:20,prompt_tokens_details:{cached_tokens:50,cache_write_tokens:10}});
 assert.equal(legacy.cache_hit,40);assert.equal(legacy.cache_miss,50);assert.equal(legacy.cache_write,10);
 const explicit=normalizeUsage({prompt_tokens:100,completion_tokens:20,prompt_tokens_details:{cached_tokens:40,cache_creation_tokens:10}});
 assert.deepEqual({...legacy},{...explicit});
 assert.equal(costOf(normalizeUsage({prompt_tokens:272001,completion_tokens:1}),pricing),(272001*.2+.75)/1e6);
});
test('inconsistent usage is flagged instead of silently charged; conflicting IDs invalidate total',()=>{
 assert.throws(()=>normalizeUsage({...raw,total_tokens:150}),/total mismatch/);
 assert.throws(()=>normalizeUsage({...raw,completion_tokens_details:{reasoning_tokens:40}}),/completion subset/);
 assert.throws(()=>normalizeUsage({...raw,prompt_cache_miss_tokens:30}),/mismatch/);
 assert.throws(()=>validatePricing({currency:'USD',tiers:[{input:-1,output:1}]}),/Invalid price/);
 const report=summarizeCalls([call('one'),call('one',{...raw,prompt_cache_hit_tokens:70,prompt_cache_miss_tokens:30})],ds);
 assert.equal(report.total_cost,null);assert.match(report.calls[1].accounting_error,/Conflicting/);
});

test('Responses and Chat cache accounting',()=>{
 const tokens=normalizeUsage({input_tokens:100,output_tokens:20,total_tokens:120,input_tokens_details:{cached_tokens:40,cache_write_tokens:10},output_tokens_details:{reasoning_tokens:15}});
 assert.equal(tokens.cache_hit,40);assert.equal(tokens.cache_write,10);assert.equal(tokens.cache_miss,50);
 assert.equal(tokens.non_reasoning_output,5);assert.equal(tokens.reasoning_added_to_billing,false);
 const allWrite=normalizeUsage({input_tokens:35950,input_tokens_details:{cached_tokens:0,cache_write_tokens:35947},output_tokens:247,output_tokens_details:{reasoning_tokens:188}});
 assert.equal(allWrite.cache_hit,0);assert.equal(allWrite.cache_write,35947);assert.equal(allWrite.cache_miss,3);
 const old=normalizeUsage({prompt_tokens:100,completion_tokens:20,prompt_tokens_details:{cached_tokens:40,cache_write_tokens:10}});
 assert.equal(old.cache_hit,30);assert.equal(old.cache_miss,60);
});
test('Responses nested SSE uses final cumulative usage and retains failed usage',()=>{
 const usage={input_tokens:100,output_tokens:20,total_tokens:120,input_tokens_details:{cached_tokens:40,cache_write_tokens:10},output_tokens_details:{reasoning_tokens:15}};
 const response={id:'resp_1',status:'completed',output:[{type:'function_call'}],usage};
 const frame=(type,response)=>'event: '+type+'\ndata: '+JSON.stringify({type,response})+'\n\n';
 const parsed=extractUsage(frame('response.created',{...response,usage:null,status:'in_progress'})+frame('response.completed',response)+frame('response.completed',response));
 assert.deepEqual(parsed.raw_usage,usage);assert.equal(parsed.response_id,'resp_1');assert.equal(parsed.finish_reason,'tool_calls');
 const rate={currency:'USD',tiers:[{input:.1,cache_read:.01,cache_write:.125,output:.5}]};
 const report=summarizeCalls([{...parsed,final_response:true}],rate);
 assert.equal(report.token_counts.output_total,20);assert.ok(Math.abs(report.total_cost-.00001665)<1e-15);
 const error={code:'rate_limit_exceeded',message:'rate limit; see account/billing'};
 const failed=extractUsage(frame('response.failed',{id:'resp_failed',status:'failed',error,usage}));
 assert.deepEqual(failed.provider_error,error);assert.equal(failed.finish_reason,null);assert.deepEqual(failed.raw_usage,usage);
 assert.equal(summarizeCalls([{...failed,final_response:false}],rate).total_cost,null);
 assert.equal(extractUsage(JSON.stringify(response)).finish_reason,'tool_calls');
});
