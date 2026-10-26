import test from 'node:test';
import assert from 'node:assert/strict';
import {applyRuntimeOptions,parseRuntimeArgs} from '../extensions/runtime-options.mjs';

test('batch flags and positional concurrency preserve zero and config defaults',()=>{
 assert.deepEqual(parseRuntimeArgs(['run-dir','--concurrency','3','--request-interval-seconds','5'],{batch:true}),{root:'run-dir',concurrency:3,requestIntervalSeconds:5});
 assert.deepEqual(parseRuntimeArgs(['run-dir','3','--request-interval-seconds','0'],{batch:true}),{root:'run-dir',concurrency:3,requestIntervalSeconds:0});
 const config={minimum_request_interval_seconds:5,api:'openai-responses'};
 assert.equal(applyRuntimeOptions(config).minimum_request_interval_seconds,5);
 assert.equal(applyRuntimeOptions(config,{requestIntervalSeconds:0}).minimum_request_interval_seconds,0);
 assert.equal(config.minimum_request_interval_seconds,5);
 assert.deepEqual(parseRuntimeArgs(['config','question','out','--resume','prior','--request-interval-seconds','.5']),{positionals:['config','question','out'],resumeFrom:'prior',requestIntervalSeconds:.5});
});
test('bad pacing, unknown protocol and conflicting concurrency fail before any run starts',()=>{
 for(const value of ['-1','NaN','Infinity',''])assert.throws(()=>parseRuntimeArgs(['run-dir','--request-interval-seconds='+value],{batch:true}));
 for(const value of [NaN,Infinity,-1,'5'])assert.throws(()=>applyRuntimeOptions({}, {requestIntervalSeconds:value}));
 assert.throws(()=>parseRuntimeArgs(['run-dir','2','--concurrency','3'],{batch:true}),/once/);
 assert.throws(()=>parseRuntimeArgs(['run-dir','--concurrency','0'],{batch:true}),/Concurrency/);
 assert.throws(()=>parseRuntimeArgs(['c','q','o','--api','typo']),/Unsupported API/);
 assert.throws(()=>parseRuntimeArgs(['c','q','o','--request-interval-second','5']));
});
