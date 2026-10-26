import {mkdirSync,existsSync,readFileSync,writeFileSync,renameSync,rmSync,readdirSync,appendFileSync,openSync,readSync,closeSync,chmodSync} from 'node:fs';
import {resolve,relative} from 'node:path';
import {createHash} from 'node:crypto';
import {gzipSync,gunzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';

const sha=b=>createHash('sha256').update(b).digest('hex');
const CHUNK=65536;
export class CaseStore {
  constructor(folder) {this.folder=resolve(folder);this.objects=resolve(folder,'objects');this.references=[];}
  putBytes(input) {
    const bytes=Buffer.from(input),refs=[];
    mkdirSync(this.objects,{recursive:true});
    // Split at message boundaries to share repeated chunks between contexts.
    const text=bytes.toString('utf8');
    const pieces=Buffer.from(text).equals(bytes)
      ? text.split(/(?=\{"(?:role|type|id)"\s*:)/).map(s=>Buffer.from(s)) : [bytes];
    for(const piece of pieces)for(let pos=0;pos<piece.length;pos+=CHUNK){
      const chunk=piece.subarray(pos,pos+CHUNK),id=sha(chunk),file=resolve(this.objects,id+'.gz');
      if(!existsSync(file)){const tmp=file+'.tmp';writeFileSync(tmp,gzipSync(chunk),{mode:0o600});renameSync(tmp,file);}
      refs.push(id);
    }
    const ref={format:'sha256-chunks-gzip-v1',sha256:sha(bytes),bytes:bytes.length,chunks:refs};
    this.references.push(ref);return ref;
  }
  putJson(value) {
    const raw=Buffer.from(JSON.stringify(value)),normalized=JSON.parse(raw);
    const encode=x=>typeof x==='string'&&Buffer.byteLength(x)>512?['s',this.putBytes(Buffer.from(x))]
      :Array.isArray(x)?['a',x.map(encode)]
      :x&&typeof x==='object'?['o',Object.entries(x).map(([k,v])=>[k,encode(v)])]:['p',x];
    return {format:'json-tree-gzip-v1',sha256:sha(raw),tree_ref:this.putBytes(Buffer.from(JSON.stringify(encode(normalized))))};
  }
  readBytes(ref) {
    if(ref?.format!=='sha256-chunks-gzip-v1')throw Error('Unknown case object format');
    const buffers=ref.chunks.map(id=>{
      if(!/^[a-f0-9]{64}$/.test(id))throw Error('Invalid chunk ID');
      const path=resolve(this.objects,id+'.gz');
      let packed;
      if(existsSync(path))packed=readFileSync(path);
      else {
        this.index ??= JSON.parse(readFileSync(resolve(this.folder,'chunk_index.json')));
        const entry=this.index[id];if(!entry)throw Error('Case chunk missing from archive index');
        packed=Buffer.alloc(entry.size);const fd=openSync(resolve(this.folder,'objects.tar'),'r');
        try{let n=0;while(n<packed.length){const bytes=readSync(fd,packed,n,packed.length-n,entry.offset+n);if(!bytes)throw Error('Truncated case archive');n+=bytes;}}finally{closeSync(fd);}
      }
      const chunk=gunzipSync(packed);if(sha(chunk)!==id)throw Error('Case chunk hash mismatch');return chunk;
    });
    const bytes=Buffer.concat(buffers);
    if(bytes.length!==ref.bytes||sha(bytes)!==ref.sha256)throw Error('Case object hash mismatch');
    return bytes;
  }
  readJson(ref) {
    if(ref?.format!=='json-tree-gzip-v1')return JSON.parse(this.readBytes(ref).toString('utf8'));
    const decode=n=>n[0]==='s'?this.readBytes(n[1]).toString('utf8'):n[0]==='a'?n[1].map(decode)
      :n[0]==='o'?Object.fromEntries(n[1].map(([k,v])=>[k,decode(v)])):n[1];
    const value=decode(JSON.parse(this.readBytes(ref.tree_ref).toString('utf8')));
    if(sha(Buffer.from(JSON.stringify(value)))!==ref.sha256)throw Error('JSON case object hash mismatch');
    return value;
  }
  archive() {
    if(!existsSync(this.objects)){
      writeFileSync(resolve(this.folder,'STORAGE_VERIFICATION.json'),JSON.stringify({status:'PASS',format:'cep23-2-pi/case-v2',chunks:0,byte_hashes_verified:true})+'\n');return;
    }
    const archive=resolve(this.folder,'objects.tar'),temp=archive+'.tmp';
    execFileSync('tar',['-cf',temp,'-C',this.folder,'objects']);chmodSync(temp,0o600);renameSync(temp,archive);
    // Verify compressed chunks before deleting loose objects.
    const ids=readdirSync(this.objects).filter(x=>x.endsWith('.gz'));
    const verification=JSON.parse(execFileSync('python3',['-c',
      "import sys,json,tarfile,gzip,hashlib\nexpected=set(json.load(sys.stdin));seen=set();index={}\nwith tarfile.open(sys.argv[1]) as t:\n for m in t:\n  if not m.isfile():continue\n  name=m.name.split('/')[-1];b=gzip.decompress(t.extractfile(m).read());assert hashlib.sha256(b).hexdigest()==name[:-3];seen.add(name);index[name[:-3]]={'offset':m.offset_data,'size':m.size}\nassert seen==expected\nwith open(sys.argv[2],'w') as f:json.dump(index,f,separators=(',',':'))\nh=hashlib.sha256()\nwith open(sys.argv[1],'rb') as f:\n for b in iter(lambda:f.read(1048576),b''):h.update(b)\nprint(json.dumps({'archive_sha256':h.hexdigest()}))",
      archive,resolve(this.folder,'chunk_index.json')],{input:JSON.stringify(ids)}).toString());
    rmSync(this.objects,{recursive:true});
    writeFileSync(resolve(this.folder,'STORAGE_VERIFICATION.json'),JSON.stringify({status:'PASS',
      format:'cep23-2-pi/case-v2',chunks:ids.length,...verification,
      byte_hashes_verified:true},null,2)+'\n');
  }
}

export class CaseRecorder {
  constructor(folder,store=new CaseStore(folder)) {
    this.folder=folder;this.store=store;this.messages=[];this.contexts=[];
    writeFileSync(resolve(folder,'OUTPUT_FORMAT.json'),JSON.stringify({format:'cep23-2-pi/case-v2',
      trajectory:'trajectory.jsonl',tools:'tool_events.jsonl',contexts:'context.jsonl',objects:'objects.tar'})+'\n');
  }
  record(event) {
    const metadata={type:event.type,at:new Date().toISOString()};
    if(event.type==='message_update'||event.type==='tool_execution_update')return;
    if(event.type==='message_end'){
      const index=this.messages.length+1,ref=this.store.putJson(event.message);
      this.messages.push(ref);
      appendFileSync(resolve(this.folder,'trajectory.jsonl'),JSON.stringify({index,role:event.message.role,
        toolCallId:event.message.toolCallId,stopReason:event.message.stopReason,message_ref:ref})+'\n');
      metadata.message_index=index;
    }
    for(const key of ['turnIndex','toolCallId','toolName','isError','requestIndex'])if(event[key]!==undefined)metadata[key]=event[key];
    appendFileSync(resolve(this.folder,'events.jsonl'),JSON.stringify(metadata)+'\n');
  }
  context(before,after,info) {
    const record={index:this.contexts.length+1,at:new Date().toISOString(),...info,
      before_ref:this.store.putJson(before),after_ref:this.store.putJson(after)};
    this.contexts.push(record);
    appendFileSync(resolve(this.folder,'context.jsonl'),JSON.stringify(record)+'\n');
  }
  finish(messages) {
    const saved=this.messages.map(ref=>this.store.readJson(ref));
    if(JSON.stringify(saved)!==JSON.stringify(messages))throw Error('Persisted full trajectory differs from Pi history; refusing cleanup');
    return {messages:saved.length,contexts:this.contexts.length,triggered:this.contexts.filter(x=>x.boundary_advanced).length};
  }
}

export function compactCase(folder,store) {
  const marker=JSON.parse(readFileSync(resolve(folder,'OUTPUT_FORMAT.json')));
  if(marker.format!=='cep23-2-pi/case-v2')throw Error('Refusing to compact an unowned output format');
  const evidence={};
  const api=resolve(folder,'api');
  if(existsSync(api))for(const id of readdirSync(api).sort()){
    const dest=resolve(api,id),record={index:Number(id)};
    for(const name of ['request.ref.json','request.meta.json','status.json','END.json','ERROR.json','CANCELLED.json']){
      const file=resolve(dest,name);if(existsSync(file))record[name]=JSON.parse(readFileSync(file));
    }
    const response=resolve(dest,'response.sse');
    if(existsSync(response))record.response_ref=store.putBytes(readFileSync(response));
    appendFileSync(resolve(folder,'api.jsonl'),JSON.stringify(record)+'\n');
  }
  const keepFile=path=>{if(existsSync(path))evidence[relative(folder,path)]=store.putBytes(readFileSync(path));};
  keepFile(resolve(folder,'state.json'));
  const graphEvents=resolve(folder,'events');
  if(existsSync(graphEvents))for(const revision of readdirSync(graphEvents)){
    for(const name of ['raw_batch.json','result.json','state.json'])keepFile(resolve(graphEvents,revision,name));
  }
  const workspace=resolve(folder,'workspace');
  const outputs=resolve(workspace,'.tool_tmp/native_outputs.json');
  if(existsSync(outputs))for(const path of JSON.parse(readFileSync(outputs))){
    if(/^\/tmp\/pi-bash-[^/]+$/.test(path)&&existsSync(path))evidence[path]=store.putBytes(readFileSync(path));
  }
  function walk(path){if(!existsSync(path))return;for(const entry of readdirSync(path,{withFileTypes:true})){
    const file=resolve(path,entry.name),rel=relative(workspace,file);
    // Corpus and Map files can be rebuilt; retain generated files and full bash output.
    if(rel==='.cep'||rel==='.tool_tmp/native_outputs.json')continue;
    if(entry.isDirectory())walk(file);else if(entry.isFile())keepFile(file);
  }}
  walk(workspace);
  const io=resolve(folder,'tool_io');
  if(existsSync(io))for(const dir of readdirSync(io)){
    const log=resolve(io,dir,'stderr.log');if(existsSync(log)&&readFileSync(log).length)keepFile(log);
  }
  writeFileSync(resolve(folder,'retrieval_and_artifacts.json'),JSON.stringify(evidence)+'\n');
  store.archive();
  // Compact only after the archive is verified and tool processes have closed.
  for(const name of ['workspace','events','tool_io','tool_results','pi-session','api'])rmSync(resolve(folder,name),{recursive:true,force:true});
  for(const name of ['MAP.txt','MAP.json','session.lock','conversation.json','requests.jsonl','tool_events.json','tool_timing.jsonl','tool_timing_summary.json'])rmSync(resolve(folder,name),{force:true});
}
