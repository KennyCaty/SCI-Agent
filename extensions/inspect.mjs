import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {CaseStore} from './storage.mjs';

const rows=(folder,name)=>readFileSync(resolve(folder,name),'utf8').split('\n').filter(Boolean).map(JSON.parse);
export function readTrajectory(folder) {
  const store=new CaseStore(folder);return rows(folder,'trajectory.jsonl').map(r=>store.readJson(r.message_ref));
}
export function readTool(folder,id) {
  const store=new CaseStore(folder),event=rows(folder,'tool_events.jsonl').find(e=>e.id===id || String(e.index)===String(id));
  if(!event)throw Error('Tool call not found');return {event,result:store.readJson(event.result_ref)};
}
export function readContext(folder,index) {
  const store=new CaseStore(folder),record=rows(folder,'context.jsonl').find(r=>r.index===Number(index));
  if(!record)throw Error('Context projection not found');
  return {...record,before:store.readJson(record.before_ref),after:store.readJson(record.after_ref)};
}
export function readApi(folder,index) {
  const store=new CaseStore(folder),record=rows(folder,'api.jsonl').find(r=>r.index===Number(index));
  if(!record)throw Error('API call not found');
  return {...record,request:JSON.parse(store.readBytes(record['request.ref.json']).toString('utf8')),
    response_sse:record.response_ref?store.readBytes(record.response_ref).toString('utf8'):null};
}
export function readArtifact(folder,path) {
  const refs=JSON.parse(readFileSync(resolve(folder,'retrieval_and_artifacts.json'))),ref=refs[path];
  if(!ref)throw Error('Artifact not found');return {path,content:new CaseStore(folder).readBytes(ref).toString('utf8')};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const [folder,kind='--trajectory',id]=process.argv.slice(2);
  try{
    if(!folder)throw Error('Usage: sci inspect CASE [--trajectory | --tools | --tool ID | --contexts | --context N | --api N | --artifact PATH]');
    const value=kind==='--trajectory'?readTrajectory(folder):kind==='--tool'?readTool(folder,id)
      :kind==='--context'?readContext(folder,id):kind==='--api'?readApi(folder,id)
      :kind==='--artifact'?readArtifact(folder,id):kind==='--tools'?rows(folder,'tool_events.jsonl')
      :kind==='--contexts'?rows(folder,'context.jsonl').map(({before_ref,after_ref,...info})=>info):null;
    if(value===null)throw Error('Unknown inspection option');console.log(JSON.stringify(value,null,2));
  }catch(e){console.error(e.message);process.exitCode=1;}
}
