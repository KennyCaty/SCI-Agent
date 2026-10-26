// Stdout carries JSONL responses for the Python runner.
import {createInterface} from 'node:readline';
import {format} from 'node:util';
const send = value => process.stdout.write(JSON.stringify(value)+'\n');
console.log = (...args) => process.stderr.write(format(...args)+'\n');
const {run} = await import('./run.mjs');
const input = createInterface({input:process.stdin});
for await (const line of input) {
  let request;
  try {
    request=JSON.parse(line);
    if(request.type!=='prompt'||request.id!==1)throw Error('Expected one prompt request');
    const result=await run(request.config,request.question,request.output,request.options??{});
    send({id:request.id,type:'response',success:true,result});
  } catch(error) {
    send({id:request?.id??null,type:'response',success:false,error:error.message});
    process.exitCode=1;
  }
  input.close();
  break;
}
