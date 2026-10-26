import {readFileSync,realpathSync,writeFileSync,existsSync} from 'node:fs';
import {resolve,sep} from 'node:path';
import {pathToFileURL} from 'node:url';
import {PI, SCOPED_SHELL, DEPLOYMENT} from './paths.mjs';
const {createReadToolDefinition} = await import(pathToFileURL(resolve(PI,'dist/core/tools/read.js')));
const {createBashToolDefinition,createLocalBashOperations} = await import(pathToFileURL(resolve(PI,'dist/core/tools/bash.js')));
const [workspace,operation,input,output]=process.argv.slice(2),args=JSON.parse(readFileSync(input,'utf8'));
const outputsPath=resolve(workspace,'.tool_tmp/native_outputs.json');
const outputs=existsSync(outputsPath)?JSON.parse(readFileSync(outputsPath,'utf8')):[];
const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
let tool;
if(operation==='read'){
 const p=realpathSync(resolve(workspace,args.path));
 if(!p.startsWith(realpathSync(workspace)+sep)&&!outputs.includes(p))throw new Error('Outside acquired workspace');
 tool=createReadToolDefinition(workspace);
}else if(operation==='bash'){
 tool=createBashToolDefinition(workspace,{operations:createLocalBashOperations(),spawnHook:ctx=>{
  const a=[DEPLOYMENT.python ?? 'python3',SCOPED_SHELL,'--workspace',workspace];
  for(const p of outputs)a.push('--output-file',p);
  a.push('--','/bin/bash','--noprofile','--norc','-c',ctx.command);
  return {...ctx,command:a.map(quote).join(' ')};
 }});
}else throw new Error('Unknown native operation');
try{
 const result=await tool.execute('cep16-tool',args);
 if(result.details?.fullOutputPath){outputs.push(realpathSync(result.details.fullOutputPath));writeFileSync(outputsPath,JSON.stringify(outputs));}
 writeFileSync(output,JSON.stringify({ok:true,result}));
}catch(e){writeFileSync(output,JSON.stringify({ok:false,error:String(e)}));}
