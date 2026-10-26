// DR-DCI text reads scoped to the acquired workspace.
import {createRequire} from 'node:module';
import {readFileSync, realpathSync, writeFileSync, existsSync} from 'node:fs';
import {resolve, sep, extname} from 'node:path';
import {spawnSync} from 'node:child_process';

const require = createRequire(import.meta.url);
const {truncateHead, formatSize, DEFAULT_MAX_BYTES} = require('./dr_dci_truncate.cjs');
const [workspaceArg, inputFile, outputFile, piNative] = process.argv.slice(2);
const DEFAULT_WINDOW = 4096;

try {
  const workspace = realpathSync(workspaceArg);
  const args = JSON.parse(readFileSync(inputFile, 'utf8'));
  const path = realpathSync(resolve(workspace, args.path));
  const outputIndex = resolve(workspace, '.tool_tmp/native_outputs.json');
  const allowedOutputs = existsSync(outputIndex) ? JSON.parse(readFileSync(outputIndex, 'utf8')) : [];
  if (!path.startsWith(workspace + sep) && !allowedOutputs.includes(path)) {
    throw new Error('Outside acquired workspace');
  }

  if (['.jpg', '.jpeg', '.png', '.gif', '.webp'].includes(extname(path).toLowerCase())) {
    const child = spawnSync(process.execPath, [piNative, workspace, 'read', inputFile, outputFile]);
    if (child.status !== 0) throw new Error(child.stderr.toString() || 'Pi image read failed');
    process.exit(0);
  }

  const buffer = readFileSync(path);
  const text = buffer.toString('utf8');
  let outputText, truncation;
  if (args.byteOffset !== undefined || args.byteLimit !== undefined) {
    const start = Math.max(0, Math.floor(args.byteOffset ?? 0));
    if (start >= buffer.length) throw new Error(`byteOffset ${start} is beyond end of file (${buffer.length} bytes total)`);
    const count = Math.max(1, Math.min(DEFAULT_MAX_BYTES, Math.floor(args.byteLimit ?? DEFAULT_WINDOW)));
    const end = Math.min(start + count, buffer.length);
    outputText = buffer.subarray(start, end).toString('utf8');
    if (end < buffer.length) outputText += `\n\n[Showing bytes ${start}-${end} of ${buffer.length}. Use byteOffset=${end} to continue.]`;
  } else if (args.charOffset !== undefined || args.charLimit !== undefined) {
    // DR-DCI character offsets use UTF-16 code units.
    const start = Math.max(0, Math.floor(args.charOffset ?? 0));
    if (start >= text.length) throw new Error(`charOffset ${start} is beyond end of file (${text.length} chars total)`);
    const count = Math.max(1, Math.min(DEFAULT_MAX_BYTES, Math.floor(args.charLimit ?? DEFAULT_WINDOW)));
    const end = Math.min(start + count, text.length);
    outputText = text.slice(start, end);
    if (end < text.length) outputText += `\n\n[Showing chars ${start}-${end} of ${text.length}. Use charOffset=${end} to continue.]`;
  } else {
    const allLines = text.split('\n');
    const totalLines = allLines.length;
    const start = args.offset ? Math.max(0, args.offset - 1) : 0;
    if (start >= totalLines) throw new Error(`Offset ${args.offset} is beyond end of file (${totalLines} lines total)`);
    let selected, selectedLines;
    if (args.limit !== undefined) {
      const end = Math.min(start + args.limit, totalLines);
      selected = allLines.slice(start, end).join('\n');
      selectedLines = end - start;
    } else {
      selected = allLines.slice(start).join('\n');
    }
    truncation = truncateHead(selected);
    if (truncation.firstLineExceedsLimit) {
      const lineStart = start === 0 ? 0 : allLines.slice(0, start).join('\n').length + 1;
      const end = Math.min(lineStart + Math.min(DEFAULT_MAX_BYTES, DEFAULT_WINDOW), text.length);
      outputText = text.slice(lineStart, end);
      if (end < text.length) outputText += `\n\n[chars ${lineStart}-${end}/${text.length}; next charOffset=${end}]`;
    } else if (truncation.truncated) {
      const endLine = start + truncation.outputLines;
      const next = endLine + 1;
      outputText = truncation.content;
      if (truncation.truncatedBy === 'lines') {
        outputText += `\n\n[Showing lines ${start + 1}-${endLine} of ${totalLines}. Use offset=${next} to continue.]`;
      } else {
        outputText += `\n\n[Showing lines ${start + 1}-${endLine} of ${totalLines} (${formatSize(DEFAULT_MAX_BYTES)} limit). Use offset=${next} to continue.]`;
      }
    } else if (selectedLines !== undefined && start + selectedLines < totalLines) {
      const remaining = totalLines - (start + selectedLines);
      outputText = `${truncation.content}\n\n[${remaining} more lines in file. Use offset=${start + selectedLines + 1} to continue.]`;
    } else {
      outputText = truncation.content;
    }
  }
  writeFileSync(outputFile, JSON.stringify({ok: true, result: {
    content: [{type: 'text', text: outputText}],
    details: truncation ? {truncation} : undefined,
  }}));
} catch (error) {
  writeFileSync(outputFile, JSON.stringify({ok: false, error: String(error)}));
}
