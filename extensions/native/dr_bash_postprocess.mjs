import {createRequire} from 'node:module';
import {randomUUID} from 'node:crypto';
import {existsSync, readFileSync, realpathSync, writeFileSync} from 'node:fs';
import {resolve, sep} from 'node:path';

process.env.DCI_BASH_MAX_LINE_CHARS = '2000';
process.env.DCI_BASH_LONG_MATCH_SNIPPET_CHARS = '1500';
const require = createRequire(import.meta.url);
const {clampLongLines, truncateTail, formatSize} = require('./dr_dci_truncate.cjs');
const [workspaceArg, inputFile, outputFile] = process.argv.slice(2);

try {
  const workspace = realpathSync(workspaceArg);
  const {result, command} = JSON.parse(readFileSync(inputFile, 'utf8'));
  const toolResult = result.result;
  const oldDetails = toolResult.details || {};
  const oldFullPath = oldDetails.fullOutputPath;
  if (oldDetails.truncation?.truncated && !oldFullPath) {
    throw new Error('Pi truncated bash output without saving the full output');
  }
  if (oldFullPath && !existsSync(oldFullPath)) {
    throw new Error('Pi full bash output file is missing');
  }

  const raw = oldFullPath
    ? readFileSync(oldFullPath, 'utf8')
    : toolResult.content.filter(item => item.type === 'text').map(item => item.text).join('\n');
  const clamped = clampLongLines(raw, {command});
  const truncation = truncateTail(clamped.content);
  let fullPath = oldFullPath;
  if ((clamped.clamped || truncation.truncated) && !fullPath) {
    fullPath = resolve(workspace, '.tool_tmp', `dr-bash-${randomUUID()}.log`);
    writeFileSync(fullPath, raw);
  }
  if (fullPath) {
    const safeRoot = realpathSync(resolve(workspace, '.tool_tmp')) + sep;
    const originalPiFile = oldFullPath && oldFullPath.startsWith('/tmp/pi-bash-');
    if (!originalPiFile && !resolve(fullPath).startsWith(safeRoot)) {
      throw new Error('Unexpected full bash output path');
    }
  }

  let text = truncation.content || '(no output)';
  if (clamped.clamped) {
    text += `\n\n[${clamped.clampedLines} long line(s) clipped; full=${fullPath}]`;
  }
  if (truncation.truncated) {
    const start = truncation.totalLines - truncation.outputLines + 1;
    const end = truncation.totalLines;
    if (truncation.lastLinePartial) {
      const lastLine = raw.split('\n').at(-1) || '';
      text += `\n\n[Showing last ${formatSize(truncation.outputBytes)} of line ${end} (line is ${formatSize(Buffer.byteLength(lastLine, 'utf8'))}). Full output: ${fullPath}]`;
    } else if (truncation.truncatedBy === 'lines') {
      text += `\n\n[Showing lines ${start}-${end} of ${truncation.totalLines}. Full output: ${fullPath}]`;
    } else {
      text += `\n\n[Showing lines ${start}-${end} of ${truncation.totalLines} (${formatSize(10 * 1024)} limit). Full output: ${fullPath}]`;
    }
  }
  toolResult.content = [{type: 'text', text}];
  toolResult.details = {
    ...oldDetails,
    truncation: truncation.truncated ? truncation : undefined,
    fullOutputPath: fullPath,
    drBashOutput: {
      maxBytes: 10 * 1024,
      maxLines: 2000,
      maxLineChars: 2000,
      matchSnippetChars: 1500,
      clippedLines: clamped.clampedLines,
      structuredMatchLines: clamped.structuredMatchLines,
    },
  };
  writeFileSync(outputFile, JSON.stringify({ok: true, result}));
} catch (error) {
  writeFileSync(outputFile, JSON.stringify({ok: false, error: String(error)}));
}
