import {execFileSync} from 'node:child_process';
import {cpSync, chmodSync, mkdirSync, readdirSync, rmSync} from 'node:fs';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const vendor = resolve(root, 'vendor/pi-mono/packages');
const compiler = resolve(root, 'node_modules/@typescript/native-preview/bin/tsgo.js');
for (const name of ['tui', 'ai', 'agent', 'coding-agent']) {
  const cwd = resolve(vendor, name);
  rmSync(resolve(cwd, 'dist'), {recursive:true, force:true});
  console.log(`Building ${name}`);
  execFileSync(process.execPath, [compiler, '-p', 'tsconfig.build.json'], {cwd, stdio:'inherit'});
}
const agent = resolve(vendor, 'coding-agent');
const theme = 'modes/interactive/theme';
mkdirSync(resolve(agent, 'dist', theme), {recursive:true});
for (const name of readdirSync(resolve(agent, 'src', theme))) {
  if (name.endsWith('.json')) cpSync(resolve(agent, 'src', theme, name), resolve(agent, 'dist', theme, name));
}
const html = 'core/export-html';
mkdirSync(resolve(agent, 'dist', html), {recursive:true});
for (const name of ['template.html', 'template.css', 'template.js', 'vendor']) {
  cpSync(resolve(agent, 'src', html, name), resolve(agent, 'dist', html, name), {recursive:true});
}
chmodSync(resolve(agent, 'dist/cli.js'), 0o755);
