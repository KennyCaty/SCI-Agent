import {fileURLToPath} from 'node:url';
import {resolve} from 'node:path';
import {readFileSync} from 'node:fs';
export const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const DEPLOYMENT = process.env.CEP_PI_DEPLOYMENT_CONFIG
  ? JSON.parse(readFileSync(process.env.CEP_PI_DEPLOYMENT_CONFIG,'utf8')) : {};
export const PROJECT = DEPLOYMENT.project_root ?? ROOT;
export const PI = DEPLOYMENT.pi_root ?? resolve(ROOT, 'vendor/pi-mono/packages/coding-agent');
export const NATIVE = resolve(ROOT, 'extensions/native_tools.mjs');
export const MAP_RUNTIME = resolve(ROOT, 'src/sci/retrieval');
export const DR_RUNTIME = resolve(ROOT, 'extensions/native');
export const SCOPED_SHELL = resolve(ROOT, 'src/sci/runtime/scoped_shell.py');
export const prompt = readFileSync(resolve(ROOT, 'prompts/qa.txt'), 'utf8');
export const schemas = JSON.parse(readFileSync(resolve(ROOT, 'configs/tools.json'), 'utf8'));
export const json = p => JSON.parse(readFileSync(p, 'utf8'));
