"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BASH_LONG_MATCH_READ_WINDOW_CHARS = exports.BASH_LONG_MATCH_SNIPPET_CHARS = exports.BASH_MAX_LINE_LENGTH = exports.GREP_MAX_LINE_LENGTH = exports.DEFAULT_MAX_BYTES = exports.DEFAULT_MAX_LINES = void 0;
exports.isLegacyTruncationMode = isLegacyTruncationMode;
exports.formatSize = formatSize;
exports.clampLongLines = clampLongLines;
exports.truncateHead = truncateHead;
exports.truncateTail = truncateTail;
exports.truncateLine = truncateLine;
exports.DEFAULT_MAX_LINES = 2000;
exports.DEFAULT_MAX_BYTES = 10 * 1024; // 10KB
exports.GREP_MAX_LINE_LENGTH = 500; // Max chars per grep match line
const DEFAULT_LONG_MATCH_SNIPPET_CHARS = 1000;
const DEFAULT_LONG_MATCH_READ_WINDOW_CHARS = 1600;
function readPositiveIntegerEnv(name, fallback) {
    const raw = process.env[name];
    if (!raw)
        return fallback;
    const parsed = Number.parseInt(raw, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}
exports.BASH_MAX_LINE_LENGTH = readPositiveIntegerEnv("DCI_BASH_MAX_LINE_CHARS", 1500);
exports.BASH_LONG_MATCH_SNIPPET_CHARS = readPositiveIntegerEnv("DCI_BASH_LONG_MATCH_SNIPPET_CHARS", DEFAULT_LONG_MATCH_SNIPPET_CHARS);
exports.BASH_LONG_MATCH_READ_WINDOW_CHARS = readPositiveIntegerEnv("DCI_BASH_LONG_MATCH_READ_WINDOW_CHARS", DEFAULT_LONG_MATCH_READ_WINDOW_CHARS);
function isLegacyTruncationMode() {
    return process.env.DCI_TRUNCATION_MODE === "legacy";
}
function formatSize(bytes) {
    if (bytes < 1024) {
        return `${bytes}B`;
    }
    else if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)}KB`;
    }
    else {
        return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
    }
}
function parseMatchLine(line) {
    const separator = /:(\d+):/g;
    let match = separator.exec(line);
    while (match !== null) {
        const path = line.slice(0, match.index);
        if (!path) {
            match = separator.exec(line);
            continue;
        }
        const lineNumber = Number.parseInt(match[1] ?? "", 10);
        if (!Number.isFinite(lineNumber) || lineNumber <= 0) {
            match = separator.exec(line);
            continue;
        }
        const text = line.slice(match.index + match[0].length);
        return { path, lineNumber, text };
    }
    return undefined;
}
function unescapeShellQuoted(value) {
    return value
        .replace(/\\(["'\\])/g, "$1")
        .replace(/\\b/g, "")
        .replace(/\\s/g, " ")
        .replace(/\\\//g, "/");
}
function extractLikelySearchTerms(command) {
    if (!command)
        return [];
    const terms = [];
    const quoted = /(["'])((?:\\.|(?!\1).)+)\1/g;
    let match = quoted.exec(command);
    while (match !== null) {
        const raw = unescapeShellQuoted(match[2] ?? "").trim();
        if (!raw) {
            match = quoted.exec(command);
            continue;
        }
        for (const part of raw.split("|")) {
            const cleaned = part
                .replace(/\(\?:?/g, "")
                .replace(/[()[\]{}^$+*?.]/g, " ")
                .replace(/\\/g, "")
                .replace(/\s+/g, " ")
                .trim();
            if (cleaned.length >= 2)
                terms.push(cleaned);
        }
        if (terms.length > 0)
            break;
        match = quoted.exec(command);
    }
    return Array.from(new Set(terms)).slice(0, 16);
}
function findLikelyMatchOffset(text, command) {
    const lower = text.toLowerCase();
    for (const term of extractLikelySearchTerms(command)) {
        const idx = lower.indexOf(term.toLowerCase());
        if (idx >= 0)
            return { offset: idx, term };
    }
    return { offset: 0 };
}
function jsonString(value) {
    return JSON.stringify(value);
}
function clampStart(value) {
    return Math.max(0, Math.floor(value));
}
function buildLongMatchReplacement(line, command, maxChars) {
    const parsed = parseMatchLine(line);
    if (!parsed) {
        const omitted = line.length - maxChars;
        const headChars = Math.ceil(maxChars / 2);
        const tailChars = Math.floor(maxChars / 2);
        return {
            text: `${line.slice(0, headChars)}... [line truncated, ${omitted} chars omitted] ...${line.slice(-tailChars)}`,
            structured: false,
        };
    }
    const snippetChars = Math.max(80, exports.BASH_LONG_MATCH_SNIPPET_CHARS);
    const readWindowChars = Math.max(snippetChars, exports.BASH_LONG_MATCH_READ_WINDOW_CHARS);
    const likelyMatch = findLikelyMatchOffset(parsed.text, command);
    const snippetStart = clampStart(likelyMatch.offset - Math.floor(snippetChars / 2));
    const snippetEnd = Math.min(parsed.text.length, snippetStart + snippetChars);
    const readStart = clampStart(likelyMatch.offset - Math.floor(readWindowChars / 4));
    const snippetPrefix = snippetStart > 0 ? "..." : "";
    const snippetSuffix = snippetEnd < parsed.text.length ? "..." : "";
    const snippet = `${snippetPrefix}${parsed.text.slice(snippetStart, snippetEnd)}${snippetSuffix}`;
    const term = likelyMatch.term ? `term=${JSON.stringify(likelyMatch.term)}; ` : "";
    if (parsed.lineNumber === 1) {
        return {
            text: `${parsed.path}:${parsed.lineNumber}: ${snippet} [long line clipped; ${term}lineChars=${parsed.text.length}; read={"path":${jsonString(parsed.path)},"charOffset":${readStart},"charLimit":${readWindowChars}}]`,
            structured: true,
        };
    }
    else {
        return {
            text: `${parsed.path}:${parsed.lineNumber}: ${snippet} [long line clipped; ${term}lineChars=${parsed.text.length}; read={"path":${jsonString(parsed.path)},"offset":${parsed.lineNumber},"limit":20}]`,
            structured: true,
        };
    }
}
function clampLongLines(content, optionsOrMaxChars = exports.BASH_MAX_LINE_LENGTH) {
    const options = typeof optionsOrMaxChars === "number" ? { maxChars: optionsOrMaxChars } : optionsOrMaxChars;
    const maxChars = options.maxChars ?? exports.BASH_MAX_LINE_LENGTH;
    if (isLegacyTruncationMode() || process.env.DCI_DISABLE_BASH_LINE_CLAMP === "1") {
        return { content, clamped: false, clampedLines: 0, maxChars, structuredMatchLines: 0 };
    }
    if (maxChars < 20) {
        return { content, clamped: false, clampedLines: 0, maxChars, structuredMatchLines: 0 };
    }
    let clamped = false;
    let clampedLines = 0;
    let structuredMatchLines = 0;
    const lines = content.split("\n");
    const output = lines.map((line) => {
        if (line.length <= maxChars)
            return line;
        clamped = true;
        clampedLines++;
        const replacement = buildLongMatchReplacement(line, options.command, maxChars);
        if (replacement.structured)
            structuredMatchLines++;
        return replacement.text;
    });
    return {
        content: output.join("\n"),
        clamped,
        clampedLines,
        maxChars,
        structuredMatchLines,
    };
}
function truncateHead(content, options = {}) {
    const maxLines = options.maxLines ?? exports.DEFAULT_MAX_LINES;
    const maxBytes = options.maxBytes ?? exports.DEFAULT_MAX_BYTES;
    const totalBytes = Buffer.byteLength(content, "utf-8");
    const lines = content.split("\n");
    const totalLines = lines.length;
    if (totalLines <= maxLines && totalBytes <= maxBytes) {
        return {
            content,
            truncated: false,
            truncatedBy: null,
            totalLines,
            totalBytes,
            outputLines: totalLines,
            outputBytes: totalBytes,
            lastLinePartial: false,
            firstLineExceedsLimit: false,
            maxLines,
            maxBytes,
        };
    }
    const firstLineBytes = Buffer.byteLength(lines[0], "utf-8");
    if (firstLineBytes > maxBytes) {
        return {
            content: "",
            truncated: true,
            truncatedBy: "bytes",
            totalLines,
            totalBytes,
            outputLines: 0,
            outputBytes: 0,
            lastLinePartial: false,
            firstLineExceedsLimit: true,
            maxLines,
            maxBytes,
        };
    }
    const outputLinesArr = [];
    let outputBytesCount = 0;
    let truncatedBy = "lines";
    for (let i = 0; i < lines.length && i < maxLines; i++) {
        const line = lines[i];
        const lineBytes = Buffer.byteLength(line, "utf-8") + (i > 0 ? 1 : 0); // +1 for newline
        if (outputBytesCount + lineBytes > maxBytes) {
            truncatedBy = "bytes";
            break;
        }
        outputLinesArr.push(line);
        outputBytesCount += lineBytes;
    }
    if (outputLinesArr.length >= maxLines && outputBytesCount <= maxBytes) {
        truncatedBy = "lines";
    }
    const outputContent = outputLinesArr.join("\n");
    const finalOutputBytes = Buffer.byteLength(outputContent, "utf-8");
    return {
        content: outputContent,
        truncated: true,
        truncatedBy,
        totalLines,
        totalBytes,
        outputLines: outputLinesArr.length,
        outputBytes: finalOutputBytes,
        lastLinePartial: false,
        firstLineExceedsLimit: false,
        maxLines,
        maxBytes,
    };
}
// Keep the tail on a UTF-8 boundary when one line exceeds the byte limit.
function truncateTail(content, options = {}) {
    const maxLines = options.maxLines ?? exports.DEFAULT_MAX_LINES;
    const maxBytes = options.maxBytes ?? exports.DEFAULT_MAX_BYTES;
    const totalBytes = Buffer.byteLength(content, "utf-8");
    const lines = content.split("\n");
    const totalLines = lines.length;
    if (totalLines <= maxLines && totalBytes <= maxBytes) {
        return {
            content,
            truncated: false,
            truncatedBy: null,
            totalLines,
            totalBytes,
            outputLines: totalLines,
            outputBytes: totalBytes,
            lastLinePartial: false,
            firstLineExceedsLimit: false,
            maxLines,
            maxBytes,
        };
    }
    const outputLinesArr = [];
    let outputBytesCount = 0;
    let truncatedBy = "lines";
    let lastLinePartial = false;
    for (let i = lines.length - 1; i >= 0 && outputLinesArr.length < maxLines; i--) {
        const line = lines[i];
        const lineBytes = Buffer.byteLength(line, "utf-8") + (outputLinesArr.length > 0 ? 1 : 0); // +1 for newline
        if (outputBytesCount + lineBytes > maxBytes) {
            truncatedBy = "bytes";
            const retainedBytes = Buffer.byteLength(outputLinesArr.join("\n"), "utf-8");
            if (outputLinesArr.length === 0 || retainedBytes === 0) {
                const truncatedLine = truncateStringToBytesFromEnd(line, maxBytes - retainedBytes);
                outputLinesArr.unshift(truncatedLine);
                outputBytesCount = Buffer.byteLength(outputLinesArr.join("\n"), "utf-8");
                lastLinePartial = true;
            }
            break;
        }
        outputLinesArr.unshift(line);
        outputBytesCount += lineBytes;
    }
    if (outputLinesArr.length >= maxLines && outputBytesCount <= maxBytes) {
        truncatedBy = "lines";
    }
    const outputContent = outputLinesArr.join("\n");
    const finalOutputBytes = Buffer.byteLength(outputContent, "utf-8");
    return {
        content: outputContent,
        truncated: true,
        truncatedBy,
        totalLines,
        totalBytes,
        outputLines: outputLinesArr.length,
        outputBytes: finalOutputBytes,
        lastLinePartial,
        firstLineExceedsLimit: false,
        maxLines,
        maxBytes,
    };
}
function truncateStringToBytesFromEnd(str, maxBytes) {
    const buf = Buffer.from(str, "utf-8");
    if (buf.length <= maxBytes) {
        return str;
    }
    let start = buf.length - maxBytes;
    while (start < buf.length && (buf[start] & 0xc0) === 0x80) {
        start++;
    }
    return buf.slice(start).toString("utf-8");
}
function truncateLine(line, maxChars = exports.GREP_MAX_LINE_LENGTH) {
    if (line.length <= maxChars) {
        return { text: line, wasTruncated: false };
    }
    return { text: `${line.slice(0, maxChars)}... [truncated]`, wasTruncated: true };
}
