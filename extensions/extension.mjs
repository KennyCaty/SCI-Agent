import {prompt, schemas} from './paths.mjs';

// Keep at least 12 tool results, including the whole group at the cutoff.
export class ToolHistoryCompactor {
  constructor(threshold = 200000, keep = 12) {
    this.threshold = threshold; this.keep = keep; this.boundary = 0;
    this.lastInput = null; this.lastContent = 0;
  }
  observe(message) {
    const u = message.usage;
    if (u) this.lastInput = u.input + u.cacheRead + u.cacheWrite;
    const calls = message.content.filter(c => c.type === 'toolCall');
    const text = message.content.filter(c => c.type === 'text').map(c => c.text).join('');
    this.lastContent = Math.floor((text.length + calls.reduce((n,c) => n + JSON.stringify(c.arguments).length, 0))/4) + 12 + calls.length*8;
  }
  project(messages) {
    const previousBoundary=this.boundary;
    const last = messages.findLastIndex(m => m.role === 'assistant');
    const pending = messages.slice(last + 1).filter(m => m.role === 'toolResult');
    const textLength = m => m.content.filter(c => c.type === 'text').map(c => c.text).join('\n').length;
    const estimate = this.lastInput === null ? Math.floor(JSON.stringify(messages).length/4)
      : this.lastInput + this.lastContent + pending.reduce((n,m) => n + Math.floor(textLength(m)/4) + 15, 0);
    const indices = messages.flatMap((m,i) => m.role === 'toolResult' ? [i] : []);
    if (estimate >= this.threshold && indices.length > this.keep) {
      const idx = indices.at(-this.keep), callID = messages[idx].toolCallId;
      let boundary = idx;
      for (let i=idx-1; i>=0; i--) {
        if (messages[i].role === 'assistant' && messages[i].content.some(c => c.type === 'toolCall' && c.id === callID)) {
          boundary = i+1; break;
        }
      }
      this.boundary = Math.max(this.boundary, boundary);
    }
    const result=messages.map((m,i) => i < this.boundary && m.role === 'toolResult'
      ? {...m, content: [{type:'text', text:'[cleared]'}]} : m);
    this.lastProjection={estimated_next_input_tokens:estimate,threshold:this.threshold,keep_tool_results:this.keep,
      boundary_before:previousBoundary,boundary_after:this.boundary,boundary_advanced:this.boundary>previousBoundary,
      threshold_exceeded:estimate>=this.threshold,last_model_input_tokens:this.lastInput,
      newly_cleared_tool_ids:messages.filter((m,i)=>i>=previousBoundary&&i<this.boundary&&m.role==='toolResult').map(m=>m.toolCallId),
      cleared_tool_ids:messages.filter((m,i)=>i<this.boundary&&m.role==='toolResult').map(m=>m.toolCallId),
      retained_tool_ids:messages.filter((m,i)=>i>=this.boundary&&m.role==='toolResult').map(m=>m.toolCallId)};
    return result;
  }
}

export function cepExtension(backend, {threshold = 200000, keep = 12} = {}, systemPrompt=prompt, toolSchemas=schemas) {
  return pi => {
    // Number contexts here because turn_start may arrive after context creation.
    let turn=0;
    for (const spec of toolSchemas) {
      pi.registerTool({...spec, label: spec.name,
        execute: (id, args, signal) => backend.execute(spec.name, args, id, signal)});
    }
    // Remove the Pi date/cwd footer to preserve the configured system prompt.
    pi.on('before_agent_start', () => ({systemPrompt}));
    const compact = new ToolHistoryCompactor(threshold, keep);
    if(backend.resumeContext){
      compact.boundary=backend.resumeContext.boundary_after;
      if(backend.resumeContext.last_successful_message)compact.observe(backend.resumeContext.last_successful_message);
    }
    pi.on('context', event => {
      turn++;backend.beginTurn?.(turn);
      const messages=compact.project(event.messages);
      backend.recordContext?.(event.messages,messages,{turn,...compact.lastProjection});
      return {messages};
    });
    pi.on('turn_end', event => {
      if (event.message.role === 'assistant' && !['error','aborted'].includes(event.message.stopReason)) compact.observe(event.message);
    });
  };
}

export function assertContract(payload, expectedPrompt=prompt, expectedSchemas=schemas) {
  const system = (payload.messages??payload.input)?.filter(m => ['system','developer'].includes(m.role));
  if (!system || system.length !== 1 || system[0].content !== expectedPrompt) {
    throw new Error('CEP prompt isolation failed: outgoing system prompt differs');
  }
  const actual = payload.tools?.map(t => ({name:(t.function??t).name, description:(t.function??t).description, parameters:(t.function??t).parameters}));
  const stable = x => JSON.stringify(x, function(k,v) {
    return v && typeof v === 'object' && !Array.isArray(v)
      ? Object.fromEntries(Object.entries(v).sort(([a],[b]) => a.localeCompare(b))) : v;
  });
  if (stable(actual) !== stable(expectedSchemas)) throw new Error('CEP tool schema/description drift');
}
