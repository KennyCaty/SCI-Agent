import {mkdtempSync,writeFileSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import assert from 'node:assert/strict';
import {createBackend} from '../extensions/tools.mjs';
import {CaseRecorder,compactCase} from '../extensions/storage.mjs';
import {readTool} from '../extensions/inspect.mjs';
const folder=mkdtempSync(resolve(tmpdir(),'cep-pi-retrieval-'));
writeFileSync(resolve(folder,'question.txt'),'What is photosynthesis?');
const backend=createBackend(folder);
const recorder=new CaseRecorder(folder,backend.store);
const abort=new AbortController();
const timeout=setTimeout(()=>{abort.abort();void backend.close();},240000);
try {
  await backend.bridge.start();
  const [search]=await Promise.all([
    backend.execute('graphsearch',{queries:['What is photosynthesis?']},'search',abort.signal),
    backend.execute('graphsearch',{queries:['What is photosynthesis?']},'queued-search',abort.signal),
  ]);
  const queued=readFileSync(resolve(folder,'tool_events.jsonl'),'utf8').trim().split('\n').map(JSON.parse).find(e=>e.id==='queued-search');
  assert.ok(queued.queue_seconds>0.01,'second graph call should wait for owner-thread dispatch');
  assert.ok(queued.execution_seconds>0);
  assert.equal(queued.timing_complete,true);
  const state=JSON.parse(readFileSync(resolve(folder,'state.json'),'utf8'));
  assert.ok(state.ids.length>0);
  assert.ok(state.focused_preview_events.length>0);
  assert.ok(state.focused_preview_events.every(e=>e.encoded_window_count===0 && e.selection==='top2_indexed_segments_lexical_bm25_window240_no_window_embedding'));
  const block=state.blocks.at(-1),sid=block.routes[0].source_id;
  const expansion=await backend.execute('graphexpand',{source_id:sid,query:'How does photosynthesis use light?'},'expand',abort.signal);
  const after=JSON.parse(readFileSync(resolve(folder,'state.json'),'utf8'));
  const previews=Object.values(after.focused_previews).filter(p=>p.selection==='top2_indexed_segments_lexical_bm25_window600_no_window_embedding');
  assert.ok(previews.length>0);
  assert.ok(previews.every(p=>p.encoded_window_count===0 && p.preview_query_embedding_calls===0));
  const selected=previews.find(p=>p.snippet?.length>0);
  const read=await backend.execute('read',{path:`.cep/s/${selected.source_id}.txt`,charOffset:selected.charOffset,charLimit:selected.charLimit},'read',abort.signal);
  assert.ok(read.content[0].text.startsWith(selected.snippet));
  const bash=await backend.execute('bash',{command:'head -n 2 .cep/catalogue.tsv'},'bash',abort.signal);
  assert.ok(bash.content[0].text.includes('source_id'));
  assert.ok(readFileSync(resolve(folder,'workspace/.cep/MAP.txt'),'utf8').includes('source_id='));
  const calls=readFileSync(resolve(folder,'cuda_calls.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
  assert.ok(calls.some(c=>c.method==='pagerank' && c.iterations===24 && c.device.startsWith('cuda')));
  assert.ok(calls.filter(c=>c.method==='encode').every(c=>c.input_kind==='query' && c.rows===1));
  await backend.close();recorder.finish([]);compactCase(folder,backend.store);
  assert.equal(readTool(folder,'expand').result.result.content[0].text,expansion.content[0].text);
  const stored=JSON.parse(readFileSync(resolve(folder,'retrieval_and_artifacts.json'),'utf8'));
  assert.deepEqual(backend.store.readJson(stored['state.json']),after);
  console.log(JSON.stringify({status:'PASS',model_api_calls:0,tools:['graphsearch','graphexpand','read','bash'],
    embedding:backend.bridge.embedding,
    graph_pid:backend.bridge.pid,search_sources:state.ids.length,expanded_sources:after.ids.length,
    queued_graph_seconds:queued.queue_seconds,queued_graph_execution_seconds:queued.execution_seconds,
    archived_tool_and_retrieval_roundtrip:true,
    search_body_window_encodings:0,expand_previews:previews.length,expand_extra_preview_encodings:0,raw_offset_match:true,
    search_map_chars:search.content[0].text.length,expand_map_chars:expansion.content[0].text.length},null,2));
}catch(e){
  console.error(e);
  try{console.error(readFileSync(resolve(folder,'graph.stderr.log'),'utf8').slice(-5000));}catch{}
  process.exitCode=1;
}finally{clearTimeout(timeout);await backend.close();rmSync(folder,{recursive:true,force:true});}
