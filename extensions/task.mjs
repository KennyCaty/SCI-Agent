import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {prompt,ROOT} from './paths.mjs';
export function taskPrompt(config={}) {
  if(config.workspace_naming && config.workspace_naming!=='id')throw new Error('Only numeric source-ID workspaces are supported');
  if (!config.task || config.task.type==='qa') return prompt;
  if(config.task.type==='trec-covid') return readFileSync(resolve(ROOT,'prompts/trec_covid.txt'),'utf8');
  throw new Error('Supported tasks: qa, trec-covid');
}
