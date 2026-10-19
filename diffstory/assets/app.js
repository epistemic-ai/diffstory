'use strict';
(() => {
  // The report is inert data, never repository code to execute. All prose/source is escaped.
  const reportDataElement = document.getElementById('report-data');
  let parsedReport = JSON.parse(reportDataElement.textContent);
  if (!parsedReport || typeof parsedReport !== 'object' || Array.isArray(parsedReport)) {
    throw new Error('Unsupported embedded report schema');
  }
  const owns = field => Object.prototype.hasOwnProperty.call(parsedReport, field);
  if (owns('storage_version') || owns('artifact_kind') || (owns('root') && owns('tables'))) {
    throw new Error('Unsupported shared report envelope');
  }
  const reportSchema = parsedReport.schema;
  const flatMode = reportSchema === 'diffstory.report.v2';
  const compilerMode = reportSchema === 'diffstory.report.v1';
  if (!flatMode && !compilerMode) throw new Error('Unsupported embedded report schema');
  if (compilerMode && (owns('generation') || owns('narrative_v2'))) {
    throw new Error('Unsupported generated fields in compiler report');
  }
  if (flatMode && (
    owns('narrative_v2')
    || parsedReport.generation?.version !== 'diffstory.generation.walkthrough.v3'
  )) {
    throw new Error('Unsupported walkthrough generation version');
  }
  // Project flat saved data for display. Source text stays in one store;
  // definitions and original diff rows resolve lazily.
  function projectWalkthrough(report) {
    const packet = report.packet;
    const sources = new Map(packet.sources.map(source => [source.id, source]));
    const lineCache = new Map();
    function lines(id) {
      if (!lineCache.has(id)) {
        const text = sources.get(id).text;
        const value = text ? text.split(/\r\n|\r|\n/) : [];
        if (value.length && /[\r\n]$/.test(text)) value.pop();
        lineCache.set(id, value);
      }
      return lineCache.get(id);
    }
    function storedRowText(row, baseSourceId, headSourceId) {
      const sourceId = row.tag === 'add' ? headSourceId : baseSourceId;
      const number = row.tag === 'add' ? row.new : row.old;
      return lines(sourceId)[number - sources.get(sourceId).start_line];
    }
    function sourceURL(source, ref) {
      const repository = (source.side === 'head' && packet.meta.head_repository) || packet.meta.repository;
      if (!/^[\w.-]+\/[\w.-]+$/.test(repository || '') || !/^[0-9a-f]{7,64}$/.test(source.revision)) return null;
      return `https://github.com/${repository}/blob/${source.revision}/${source.path.split('/').map(encodeURIComponent).join('/')}#L${ref.start}-L${ref.end}`;
    }
    function definition(ref) {
      if (!ref) return null;
      const source = sources.get(ref.source_id);
      return {source_id: source.id, path: source.path, side: source.side, start: ref.start, end: ref.end,
        name: ref.name || 'Cited source', url: ref.url || sourceURL(source, ref),
        get source() { return lines(source.id).slice(ref.start - source.start_line, ref.end - source.start_line + 1).join('\n'); }};
    }
    const rawFiles = packet.diffs.map(diff => ({...diff, hunks: diff.hunks.map(hunk => ({...hunk,
      rows: hunk.rows.map(row => ({...row, get text() {
        return storedRowText(row, diff.base_source, diff.head_source);
      }}))
    }))}));
    const changes = packet.changes.map(change => ({...change,
      before: definition(change.before), after: definition(change.after),
      get fileDiffs() {
        return rawFiles.filter(diff => (change.before && diff.base_source === change.before.source_id)
          || (change.after && diff.head_source === change.after.source_id));
      },
      get functionHunks() {
        return (change.comparison_hunks || []).map(hunk => ({...hunk,
          rows: hunk.rows.map(row => ({...row, get text() {
            return storedRowText(row, change.before?.source_id, change.after?.source_id);
          }}))
        }));
      },
      get hunks() {
        return rawFiles.flatMap(diff => diff.hunks.map(hunk => {
          const rows = hunk.rows.filter(row => (change.before && diff.base_source === change.before.source_id && row.old >= change.before.start && row.old <= change.before.end)
            || (change.after && diff.head_source === change.after.source_id && row.new >= change.after.start && row.new <= change.after.end));
          const old = rows.filter(row => row.old !== null), head = rows.filter(row => row.new !== null);
          return {...hunk, old_start: old[0]?.old || hunk.old_start, new_start: head[0]?.new || hunk.new_start,
            old_count: old.length, new_count: head.length, rows};
        }).filter(hunk => hunk.rows.length));
      }
    }));
    const groups = report.story.sections.map((section, index) => ({
      id: `walkthrough-${index + 1}`, title: section.title, number: index + 1,
      path: '', theme: 'walkthrough', prerequisites: [], test_links: [],
      next_id: index + 1 < report.story.sections.length ? `walkthrough-${index + 2}` : null,
      change_ids: [...new Set(section.passages.flatMap(passage => passage.change_ids))],
      narrative: {intent: section.summary, why_now: '', transition: '', passages: section.passages}
    }));
    const byKind = {};
    for (const change of changes) byKind[change.kind] = (byKind[change.kind] || 0) + 1;
    const rows = packet.diffs.flatMap(diff => diff.hunks.flatMap(hunk => hunk.rows));
    const tests = (packet.analysis?.tests || []).map(test => ({...definition(test.source), id: test.id}));
    return {schema: 'diffstory.report.v1', meta: packet.meta, changes, groups, raw_files: rawFiles,
      generation: report.generation, document: {preamble: report.story.preamble, lead: report.story.introduction, closing: report.story.closing},
      stats: {groups: groups.length, units: changes.length, by_kind: byKind,
        identical_ast_moves: byKind.moved || 0, supplied_paths: new Set(packet.sources.map(source => source.path)).size,
        supplied_additions: rows.filter(row => row.tag === 'add').length,
        supplied_deletions: rows.filter(row => row.tag === 'delete').length, test_definitions: tests.length, test_runs: 0},
      tests, edges: [], symbol_edges: packet.analysis?.symbol_edges || [], cycles: [], unresolved: packet.analysis?.unresolved || [],
      source_evidence: {file_evidence: packet.file_evidence || [], fragments: packet.sources.map(source => ({key: source.id, side: source.side, path: source.path, scope: source.scope, parse_status: source.parse_status}))},
      warnings: packet.warnings, notes: packet.notes,
      method: {matching: 'Compiler symbol matching and original coordinates are retained.',
        narration: 'Source-linked model prose is unverified. Structural validation checks references and coverage.'},
      flatDefinition: definition};
  }
  const R = flatMode ? projectWalkthrough(parsedReport) : parsedReport;
  const generation = R.generation;
  const flatStory = flatMode ? parsedReport.story : null;
  const flatPassages = new Map();
  if (flatMode) {
    for (const group of R.groups) {
      for (const passage of group.narrative.passages || []) {
        if (typeof passage.id === 'string') flatPassages.set(passage.id, {passage, group});
      }
    }
  }
  parsedReport = null;
  const $ = (selector, parent = document) => parent.querySelector(selector);
  const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
  const esc = value => String(value ?? '').replace(/[&<>"'\r]/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '\r': '&#13;'
  }[c]));
  const basename = path => String(path || '').split('/').pop();
  // Explicit backticks are preferred; report-backed code tokens are the rendering fallback.
  const codeTerms = new Set(['None', 'True', 'False', 'null', 'undefined']);
  const identifier = /^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*$/;
  const pathLike = /^(?:[A-Za-z0-9_.-]+\/)+[A-Za-z0-9_.-]+$|^[A-Za-z0-9_.-]+\.[A-Za-z0-9]+$/;
  const commonCalls = new Set([
    'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 'int',
    'len', 'list', 'map', 'max', 'min', 'open', 'print', 'range', 'set',
    'str', 'sum', 'tuple', 'type', 'zip'
  ]);
  function addCodeTerm(value, kind = 'name') {
    if (typeof value !== 'string') return;
    const term = value.trim();
    if (!term || term.length > 160 || /[\r\n`]/.test(term)) return;
    if (identifier.test(term) || pathLike.test(term)) {
      if (kind !== 'call' || !commonCalls.has(term)) codeTerms.add(term);
    }
  }
  for (const change of R.changes) {
    for (const source of [change.before, change.after]) {
      if (!source) continue;
      addCodeTerm(source.path, 'path');
      addCodeTerm(basename(source.path), 'path');
      addCodeTerm(source.name);
      for (const binding of source.local_bindings || []) {
        if (binding.length <= 2) addCodeTerm(binding, 'binding');
      }
      for (const call of source.calls || []) addCodeTerm(call.name, 'call');
      for (const ref of source.references || []) {
        if (ref.name.length <= 2) addCodeTerm(ref.name, 'binding');
      }
    }
  }
  for (const file of R.raw_files || []) {
    addCodeTerm(file.path, 'path');
    addCodeTerm(basename(file.path), 'path');
  }
  for (const field of ['base_ref', 'head_ref', 'base_branch', 'head_branch']) {
    addCodeTerm(R.meta[field], 'path');
  }
  for (const group of R.groups) addCodeTerm(group.path, 'path');
  for (const test of R.tests || []) {
    addCodeTerm(test.path, 'path');
    addCodeTerm(test.name);
  }
  const escapedCodeTerms = [...codeTerms]
    .sort((a, b) => b.length - a.length)
    .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pathPattern = String.raw`(?:[A-Za-z0-9_.-]+\/)+[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.(?:py|pyi|js|jsx|ts|tsx|json|md|toml|yaml|yml|sh|html|css|go|rs|java|kt|c|h|cpp|hpp|sql|txt|xml)`;
  const codeTermPattern = new RegExp(
    `(^|[^A-Za-z0-9_$])(${escapedCodeTerms.join('|')}|--[A-Za-z][A-Za-z0-9-]*|${pathPattern})(?=$|[^A-Za-z0-9_$])`,
    'g'
  );
  function inlineText(value) {
    codeTermPattern.lastIndex = 0;
    let result = '', cursor = 0;
    for (const match of value.matchAll(codeTermPattern)) {
      const start = match.index + match[1].length;
      result += esc(value.slice(cursor, start)) + `<code>${esc(match[2])}</code>`;
      cursor = start + match[2].length;
    }
    return result + esc(value.slice(cursor));
  }
  // Only code spans and emphasis are supported. An annotation cannot supply arbitrary HTML.
  const inline = value => String(value ?? '').split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g)
    .map(part => {
      if (part.startsWith('`') && part.endsWith('`')) {
        return `<code>${esc(part.slice(1, -1))}</code>`;
      }
      if (part.startsWith('**') && part.endsWith('**')) {
        return `<strong>${inline(part.slice(2, -2))}</strong>`;
      }
      return inlineText(part);
    }).join('');
  const plainInline = value => String(value ?? '')
    .replace(/`([^`\n]+)`/g, '$1')
    .replace(/\*\*([^*\n]+)\*\*/g, '$1');
  const prose = value => String(value ?? '').split(/\n\s*\n/).filter(Boolean)
    .map(p => `<p>${inline(p)}</p>`).join('');
  const G = new Map(R.groups.map(group => [group.id, group]));
  const C = new Map(R.changes.map(c => [c.id, c]));
  function passageAnchor(id) {
    const value = String(id ?? '');
    let encoded = '';
    for (let index = 0; index < value.length; index += 1) {
      encoded += value.charCodeAt(index).toString(16).padStart(4, '0');
    }
    return `passage-${encoded}`;
  }
  const safeURL = value => {
    try { const u = new URL(value); return u.protocol === 'https:' && u.hostname === 'github.com' ? u.href : null; }
    catch (_) { return null; }
  };
  const link = (url, label, cls = '') => safeURL(url)
    ? `<a href="${esc(safeURL(url))}" class="${cls}" target="_blank" rel="noopener noreferrer">${esc(label)} ↗</a>` : '';
  let activeSection = null;
  let blockCounter = 0;
  let flatCitationCount = 0;
  const blocks = new Map();
  const anchors = new Set();
  const PREVIEW_LIMIT = 34;
  const PREVIEW_HEAD = 19;
  const PREVIEW_TAIL = 5;
  const CHUNK = 160;

  function classification(c) {
    const fileStatus = {
      added: ['Added to file', 'move'],
      removed: ['Removed from file', 'removed'],
      observed_head: ['Addition unresolved', ''],
      observed_base: ['Removal unresolved', '']
    };
    if (fileStatus[c.kind]) return fileStatus[c.kind];
    if ((c.after || c.before)?.is_test) return ['Test source', ''];
    const kinds = {
      moved: ['Moved · same AST', 'move'],
      moved_renamed: ['Moved + renamed', 'rename'],
      renamed: ['Renamed', 'rename'],
      moved_modified: ['Moved + edited', 'edited'],
      modified: ['Edited', 'edited'],
      source_only: ['Source-only edit', ''],
      wiring: ['Wiring', ''],
      text: ['Source change', '']
    };
    return kinds[c.kind] || [c.label || c.kind, ''];
  }

  // A deliberately small lexer: colors aid reading, not semantic classification.
  function highlight(line) {
    const re = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#[^\n]*|\b(?:async|await|def|class|return|from|import|as|if|elif|else|for|in|not|and|or|is|None|True|False|with|raise|try|except|finally|pass|lambda|yield|while|break|continue|assert)\b|\b\d+(?:\.\d+)?\b)/g;
    let result = '', previous = 0;
    for (const match of line.matchAll(re)) {
      result += esc(line.slice(previous, match.index));
      const t = match[0];
      const cls = t[0] === '#' ? 'comment' : /["']/.test(t[0]) ? 'string' : /^\d/.test(t) ? 'number' : 'keyword';
      result += `<span class="token-${cls}">${esc(t)}</span>`;
      previous = match.index + t.length;
    }
    return result + esc(line.slice(previous));
  }
  function definitionRows(source) {
    return source ? source.source.split('\n').map((text, i) => ({type: 'line', tag: 'context', new: source.start + i, old: null, text})) : [];
  }
  function diffRows(hunks) {
    const rows = [];
    for (const h of hunks || []) {
      rows.push({type: 'hunk', text: `@@ −${h.old_start},${h.old_count} +${h.new_start},${h.new_count} @@`});
      for (const row of h.rows) rows.push({type: 'line', ...row});
    }
    return rows;
  }
  function focusRanges(spec, side) {
    const focus = spec.focus;
    if (!focus) return [];
    const selected = focus[side];
    if (Array.isArray(selected)) return selected;
    return Number.isInteger(focus.start) && Number.isInteger(focus.end)
      ? [focus] : [];
  }
  function lineInRanges(line, ranges, margin = 0) {
    return Number.isInteger(line) && ranges.some(range =>
      line >= range.start - margin && line <= range.end + margin
    );
  }
  function citedDiffRows(rows, spec) {
    const before = focusRanges(spec, 'before');
    const after = focusRanges(spec, 'after');
    if (!before.length && !after.length) return null;
    const included = new Set();
    const targets = [];
    let hunkStart = 0;
    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index];
      if (row.type === 'hunk') { hunkStart = index + 1; continue; }
      if (row.type !== 'line') continue;
      const cited = row.tag === 'delete' ? lineInRanges(row.old, before)
        : row.tag === 'add' ? lineInRanges(row.new, after)
          : lineInRanges(row.old, before) || lineInRanges(row.new, after);
      if (!cited || row.tag === 'context') continue;
      targets.push(index);
      const start = Math.max(hunkStart, index - 2);
      const end = Math.min(rows.length - 1, index + 2);
      for (let current = start; current <= end; current += 1) {
        if (rows[current]?.type === 'line') included.add(current);
      }
    }
    if (!targets.length) return null;
    for (const index of targets) included.add(index);
    const byHunk = [];
    let heading = null, lineIndexes = [], groupStart = 0;
    const flush = end => {
      if (!lineIndexes.length) return;
      byHunk.push({heading, lineIndexes, start: groupStart, end});
      lineIndexes = [];
    };
    rows.forEach((row, index) => {
      if (row.type === 'hunk') { flush(index); heading = row; groupStart = index + 1; }
      else if (included.has(index)) lineIndexes.push(index);
    });
    flush(rows.length);
    const output = [];
    let priorEnd = 0;
    for (const hunk of byHunk) {
      if (hunk.start > priorEnd) {
        const omitted = countLines(rows.slice(priorEnd, hunk.start));
        if (omitted) output.push({type: 'gap', text: `… ${omitted} diff lines outside the cited ranges omitted`});
      }
      if (hunk.heading) output.push(hunk.heading);
      if (hunk.lineIndexes[0] > hunk.start) {
        const omitted = countLines(rows.slice(hunk.start, hunk.lineIndexes[0]));
        if (omitted) output.push({type: 'gap', text: `… ${omitted} diff lines before cited edits omitted`});
      }
      let previous = null;
      for (const index of hunk.lineIndexes) {
        if (previous !== null && index > previous + 1) {
          const omitted = countLines(rows.slice(previous + 1, index));
          if (omitted) output.push({type: 'gap', text: `… ${omitted} diff lines between cited edits omitted`});
        }
        output.push(rows[index]);
        previous = index;
      }
      const last = hunk.lineIndexes.at(-1);
      const end = hunk.end;
      const after = countLines(rows.slice(last + 1, end));
      if (after) output.push({type: 'gap', text: `… ${after} diff lines after cited edits omitted`});
      priorEnd = end;
    }
    const finalOmitted = countLines(rows.slice(priorEnd));
    if (finalOmitted) output.push({type: 'gap', text: `… ${finalOmitted} additional diff lines omitted`});
    return output;
  }
  function definitionPreview(rows, ranges) {
    const selected = [];
    rows.forEach((row, index) => {
      if (row.type === 'line' && lineInRanges(row.new, ranges, 2)) selected.push(index);
    });
    if (!selected.length) return null;
    const output = [];
    const first = selected[0], last = selected.at(-1);
    const before = countLines(rows.slice(0, first));
    if (before) output.push({type: 'gap', text: `… ${before} lines before this cited definition omitted`});
    let previous = null;
    for (const index of selected) {
      if (previous !== null && index > previous + 1) {
        const omitted = countLines(rows.slice(previous + 1, index));
        output.push({type: 'gap', text: `… ${omitted} lines between cited ranges omitted`});
      }
      output.push(rows[index]);
      previous = index;
    }
    const after = countLines(rows.slice(last + 1));
    if (after) output.push({type: 'gap', text: `… ${after} lines after this cited definition omitted`});
    return output;
  }
  function allRows(spec) {
    if (spec.cache[spec.view]) return spec.cache[spec.view];
    let rows = [];
    if (spec.raw) rows = diffRows(spec.raw.hunks);
    else if (spec.test) rows = definitionRows(spec.test);
    else {
      const ordered = spec.view === 'diff' && spec.changes.length > 1
        ? [...spec.changes].sort((left, right) =>
          (left.after || left.before).start - (right.after || right.before).start
        ) : spec.changes;
      const seenDiffRows = new Set();
      for (const c of ordered) {
        if (spec.view === 'comparison' && !Array.isArray(c.comparison_hunks)) continue;
        if (spec.view === 'diff' || spec.view === 'comparison') {
          const r = diffRows(spec.view === 'comparison' ? c.functionHunks : c.hunks);
          const source = c.after || c.before;
          const unavailable = c.fileDiffs?.some(diff => ['unavailable', 'unbound'].includes(diff.patch_status));
          const message = spec.view === 'diff' && unavailable
            ? 'The original file patch is unavailable here. Inspect the supplied definitions or the separate function comparison.'
            : `${source.name}: no textual edits in this comparison.`;
          for (const row of (r.length ? r : [{type: 'note', text: message}])) {
            const key = row.type === 'line'
              ? JSON.stringify([row.type, row.tag, row.old, row.new, row.text])
              : JSON.stringify([row.type, row.text]);
            if (spec.changes.length === 1 || !seenDiffRows.has(key)) rows.push(row);
            seenDiffRows.add(key);
          }
        } else if (spec.view === 'before' || spec.view === 'after') {
          const source = c[spec.view];
          if (!source) continue;
          const prior = rows.filter(r => r.type === 'line').at(-1);
          if (prior && prior.new + 1 !== source.start) rows.push({type: 'gap', text: '… between declarations'});
          rows.push(...definitionRows(source));
        } else {
          const source = c.after || c.before;
          const prior = rows.filter(r => r.type === 'line').at(-1);
          if (prior && prior.new + 1 !== source.start) rows.push({type: 'gap', text: '… between declarations'});
          rows.push(...definitionRows(source));
        }
      }
    }
    spec.cache[spec.view] = rows;
    return rows;
  }
  const countLines = rows => rows.filter(r => r.type === 'line').length;
  function foldedFocusRows(rows, label) {
    if (countLines(rows) <= PREVIEW_LIMIT) return rows;
    const head = rows.slice(0, PREVIEW_HEAD);
    const tail = rows.slice(-PREVIEW_TAIL);
    const omitted = countLines(rows) - countLines(head) - countLines(tail);
    return [...head, {type: 'gap', text: `… ${omitted} cited ${label} lines folded; load the full view to inspect them`}, ...tail];
  }
  function previewRows(spec) {
    const rows = allRows(spec);
    if (spec.expanded) return rows.slice(0, spec.loaded);
    if (spec.view === 'diff' || spec.view === 'comparison') {
      const focused = citedDiffRows(rows, spec);
      if (focused) return foldedFocusRows(focused, 'diff');
    } else {
      const side = spec.view === 'before' ? 'before' : 'after';
      const focused = definitionPreview(rows, focusRanges(spec, side));
      if (focused) return foldedFocusRows(focused, 'definition');
    }
    if (rows.length <= PREVIEW_LIMIT) return rows;
    const shown = rows.slice(0, PREVIEW_HEAD);
    const tail = rows.slice(-PREVIEW_TAIL);
    const omitted = countLines(rows) - countLines(shown) - countLines(tail);
    return [...shown, {type: 'gap', text: `… ${omitted} lines folded`}, ...tail];
  }
  function rowsHTML(rows, isDiff) {
    return rows.map(r => {
      if (r.type === 'gap') return `<div class="source-gap">${esc(r.text)}</div>`;
      if (r.type === 'hunk') return `<div class="hunk-label">${esc(r.text)}</div>`;
      if (r.type === 'note') return `<div class="code-note">${esc(r.text)}</div>`;
      const gutters = isDiff
        ? `<span class="line-no dual">${r.old ?? ''}</span><span class="line-no dual">${r.new ?? ''}</span><span class="code-sign">${r.tag === 'add' ? '+' : r.tag === 'delete' ? '−' : ' '}</span>`
        : `<span class="line-no">${r.new ?? r.old ?? ''}</span>`;
      return `<div class="code-line ${isDiff ? esc(r.tag) : ''}">${gutters}<span class="code-text">${highlight(r.text)}</span></div>`;
    }).join('');
  }
  function fileEvidenceForChangeSide(change, side) {
    const entries = R.source_evidence?.file_evidence || [];
    const other = side === 'base' ? 'head' : 'base';
    const ownPath = change[side === 'base' ? 'before' : 'after']?.path;
    if (ownPath) {
      const entry = entries.find(item => item[side]?.path === ownPath);
      if (entry) return entry[side];
    }
    const otherPath = change[side === 'base' ? 'after' : 'before']?.path;
    if (otherPath) {
      const entry = entries.find(item => item[other]?.path === otherPath);
      if (entry) return entry[side] || null;
    }
    return null;
  }
  function materialCaveats(changes) {
    const caveats = new Map();
    const sourceIssues = new Map();
    const add = (key, text, change = null) => {
      let caveat = caveats.get(key);
      if (!caveat) {
        caveat = {text, reasons: []};
        caveats.set(key, caveat);
      }
      const reason = String(change?.basis || '').trim();
      if (reason && !caveat.reasons.includes(reason)) caveat.reasons.push(reason);
    };
    for (const change of changes) {
      const before = change.before, after = change.after;
      const path = before?.path || after?.path || '';
      if (change.kind === 'observed_head') {
        const name = after?.name || 'This declaration';
        add(`unresolved-head:${path}:${name}`,
          `Cannot confirm whether \`${name}\` was added to this file.`, change);
      } else if (change.kind === 'observed_base') {
        const name = before?.name || 'This declaration';
        add(`unresolved-base:${path}:${name}`,
          `Cannot confirm whether \`${name}\` was removed from this file.`, change);
      } else if (change.kind === 'moved_modified') {
        const name = after?.name || before?.name || 'This declaration';
        add(`edited-move:${before?.path || ''}:${after?.path || ''}:${name}`,
          `\`${name}\` may have moved and changed. Review both versions to confirm.`, change);
      }
      for (const side of ['base', 'head']) {
        const evidence = fileEvidenceForChangeSide(change, side);
        const source = change[side === 'base' ? 'before' : 'after'];
        const other = change[side === 'base' ? 'after' : 'before'];
        const sourcePath = evidence?.path || source?.path || other?.path;
        if (!sourcePath) continue;
        const fragments = (R.source_evidence?.fragments || []).filter(fragment =>
          fragment.side === side && fragment.path === sourcePath);
        const parseFailed = fragments.some(fragment => fragment.parse_status === 'failed')
          || (evidence?.parse_statuses || []).includes('failed')
          || source?.parse_status === 'failed';
        const unavailable = evidence?.state === 'unavailable';
        const partial = evidence?.state === 'supplied' && evidence.coverage === 'partial';
        if (!unavailable && !partial && !parseFailed) continue;
        const key = `${side}:${sourcePath}`;
        let issue = sourceIssues.get(key);
        if (!issue) {
          issue = {side, path: sourcePath, unavailable: false, partial: false, parseFailed: false};
          sourceIssues.set(key, issue);
        }
        issue.unavailable ||= unavailable;
        issue.partial ||= partial;
        issue.parseFailed ||= parseFailed;
      }
    }
    for (const issue of sourceIssues.values()) {
      const version = issue.side === 'base' ? 'earlier' : 'later';
      let detail;
      if (issue.unavailable) {
        detail = `The ${version} version of \`${issue.path}\` was not available for comparison. This does not mean the file was absent.`;
      } else if (issue.partial && issue.parseFailed) {
        detail = `Only part of the ${version} version of \`${issue.path}\` is available. The tool could not read the supplied excerpt as Python, so function comparison may be unreliable. Code outside the excerpt was not checked.`;
      } else if (issue.partial) {
        detail = `Only part of the ${version} version of \`${issue.path}\` is available. Code outside the excerpt was not checked.`;
      } else {
        detail = `The tool could not read the ${version} version of \`${issue.path}\` as Python, so function comparison may be unreliable.`;
      }
      add(`source:${issue.side}:${issue.path}`, detail);
    }
    return [...caveats.values()];
  }
  function caveatsHTML(changes) {
    return materialCaveats(changes).map(caveat => {
      const details = caveat.reasons.length
        ? `<details class="code-caveat-detail"><summary>Why this is uncertain</summary><ul>${caveat.reasons.map(reason => `<li>${inline(reason)}</li>`).join('')}</ul></details>`
        : '';
      return `<div class="code-caveat"><p>${inline(caveat.text)}</p>${details}</div>`;
    }).join('');
  }

  function blockHTML(changeIds, options = {}) {
    const cs = (changeIds || []).map(id => C.get(id)).filter(Boolean);
    const first = cs[0];
    const source = options.test || (first && (first.after || first.before));
    const primaryPath = options.raw?.path || source?.path || '';
    const spec = {
      id: `block-${++blockCounter}`, changes: cs, test: options.test || null,
      raw: options.raw || null, focus: options.focus || null, label: options.label || '',
      related: options.related || false,
      sourceOnly: options.sourceOnly || false,
      view: options.view || (options.test ? 'definition' : 'diff'),
      expanded: false, loaded: 0, materialized: false, cache: {}, primaryPath,
      baseURL: options.raw?.before_url || first?.before?.url,
      headURL: options.raw?.after_url || options.test?.url || first?.after?.url,
    };
    blocks.set(spec.id, spec);
    const fileStatusChange = cs.find(change => ['added', 'removed', 'observed_head', 'observed_base'].includes(change.kind));
    const badge = options.raw ? ['Source hunk', ''] : options.test ? ['Test source', ''] : spec.related ? [`${cs.length} related changes`, ''] : fileStatusChange ? classification(fileStatusChange) : cs.length > 1 ? [`${cs.length} moved declarations`, 'move'] : classification(first);
    const name = options.label || (cs.length > 1 ? 'graph constants' : source?.name === 'module imports / context' ? 'imports' : source?.name === 'file context' ? '' : source?.name || '');
    spec.badge = badge;
    spec.name = name;
    const symbol = name ? `<span class="symbol-name">· ${esc(name)}</span>` : '';
    const estimateSide = spec.view === 'before' ? 'before' : 'after';
    const focusedEstimate = spec.view === 'diff' ? null
      : focusRanges(spec, estimateSide).reduce((total, range) => total + range.end - range.start + 5, 0) || null;
    const sourceEstimate = options.raw
      ? options.raw.hunks.reduce((total, hunk) => total + hunk.rows.length + 1, 0)
      : options.test
        ? options.test.end - options.test.start + 1
        : cs.reduce((total, change) => {
          const source = spec.view === 'before' || spec.view === 'after'
            ? change[spec.view] : change.after || change.before;
          if (!source) return total;
          const size = spec.view === 'diff'
            ? change.hunks.reduce((hunkTotal, hunk) => hunkTotal + hunk.rows.length + 1, 0)
            : source.end - source.start + 1;
          return total + size;
        }, 0);
    const estimated = Math.min(PREVIEW_LIMIT, focusedEstimate ?? sourceEstimate);
    const beforeAnchors = cs.filter(c => !anchors.has(c.id)).map(c => { anchors.add(c.id); return `<span id="unit-${c.id}" class="change-anchor"></span>`; }).join('');
    const caveats = caveatsHTML(cs);
    return `${beforeAnchors}<figure class="code-block" id="${spec.id}" data-block="${spec.id}" data-view="${esc(spec.view)}" data-changes="${cs.map(c => c.id).join(' ')}"><figcaption class="code-header"><span class="filename" title="${esc(primaryPath)}">${esc(basename(primaryPath))}${symbol}</span><span class="classification ${badge[1]}">${esc(badge[0])}</span></figcaption>${caveats}<div class="code-body"><div class="lazy-shell" style="min-height:${Math.max(58, Math.min(430, estimated * 22))}px"><button data-load-preview="${spec.id}">Load ${spec.view === 'diff' ? 'diff' : 'code'} ↓</button></div></div><div class="code-footer"></div></figure>`;
  }
  function blockFooter(spec, shown, total) {
    const first = spec.changes[0];
    const side = spec.view === 'before' ? 'before' : 'after';
    const a = spec.test || (spec.view === 'before' || spec.view === 'after'
      ? spec.changes.find(change => change[spec.view])?.[spec.view]
      : first?.after || first?.before);
    const b = first?.before;
    let location = spec.changes.length > 1
      ? (spec.view === 'diff' ? 'Paired diffs' : `${spec.view === 'before' ? 'Before' : 'After'} definitions`)
      : spec.view === 'diff' ? 'Paired diff' : a ? `L${focusRanges(spec, side)[0]?.start || a.start}–${focusRanges(spec, side).at(-1)?.end || a.end}` : 'Source';
    if (spec.view === 'comparison') {
      location = 'Function comparison · generated';
    } else if (!flatMode && !spec.raw && spec.view === 'diff') {
      location = 'Source comparison · generated';
    } else if (spec.view === 'diff') {
      const diffs = spec.raw ? [spec.raw] : spec.changes.flatMap(change => change.fileDiffs || []);
      const origins = new Set(diffs.map(diff => diff.diff_origin));
      location = diffs.some(diff => ['unavailable', 'unbound'].includes(diff.patch_status))
        ? 'Original patch unavailable'
        : origins.size === 1 && origins.has('github') ? (spec.raw ? 'Original GitHub patch' : 'GitHub patch excerpt')
          : origins.size === 1 && origins.has('git') ? (spec.raw ? 'Original Git patch' : 'Git patch excerpt') : 'Original patch excerpts';
    }
    if (first?.before && first?.after && first.before.path !== first.after.path) location = `${basename(b.path)} → ${basename(first.after.path)} · ${location}`;
    let modeButton = '';
    if (!spec.raw && !spec.test && !spec.sourceOnly && spec.changes.length > 0) {
      if (flatMode) {
        const hasBefore = spec.changes.some(change => change.before);
        const hasAfter = spec.changes.some(change => change.after);
        const beforeButton = hasBefore ? `<button class="text-button${spec.view === 'before' ? ' is-active' : ''}" data-view="${spec.id}" data-side="before" aria-pressed="${spec.view === 'before'}">Before definition</button>` : '';
        const afterButton = hasAfter ? `<button class="text-button${spec.view === 'after' ? ' is-active' : ''}" data-view="${spec.id}" data-side="after" aria-pressed="${spec.view === 'after'}">After definition</button>` : '';
        const comparison = spec.changes.some(change => Array.isArray(change.comparison_hunks))
          ? `<button class="text-button${spec.view === 'comparison' ? ' is-active' : ''}" data-view="${spec.id}" data-side="comparison" aria-pressed="${spec.view === 'comparison'}">Compare function versions</button>` : '';
        modeButton = `${beforeButton}${afterButton}<button class="text-button${spec.view === 'diff' ? ' is-active' : ''}" data-view="${spec.id}" data-side="diff" aria-pressed="${spec.view === 'diff'}">View diff</button>${comparison}`;
      } else {
        modeButton = `<button class="text-button" data-switch="${spec.id}">${spec.view === 'diff' ? 'Read definition' : 'View diff'}</button>`;
      }
    }
    const collapse = spec.expanded ? `<button class="text-button" data-collapse="${spec.id}">Collapse</button>` : '';
    const base = spec.baseURL ? link(spec.baseURL, 'Base', 'code-source-link') : '';
    const head = spec.headURL ? link(spec.headURL, 'Source', 'code-source-link') : '';
    return `<span class="code-location">${esc(location)}</span>${modeButton}${collapse}${base}${head}`;
  }
  function paintBlockHeader(spec, element) {
    const active = spec.view === 'before' || spec.view === 'after'
      ? spec.changes.find(change => change[spec.view])?.[spec.view]
      : spec.test || spec.changes[0]?.after || spec.changes[0]?.before;
    const path = spec.raw?.path || active?.path || spec.primaryPath;
    const filename = $('.filename', element);
    filename.title = path || '';
    filename.innerHTML = `${esc(basename(path))}${spec.name ? `<span class="symbol-name">· ${esc(spec.name)}</span>` : ''}`;
    element.dataset.view = spec.view;
  }
  function paintBlock(spec) {
    const element = document.getElementById(spec.id);
    if (!element) return;
    spec.materialized = true;
    paintBlockHeader(spec, element);
    const full = allRows(spec), shown = previewRows(spec);
    const total = countLines(full), visible = countLines(shown);
    const isDiff = spec.view === 'diff' || spec.view === 'comparison';
    let load = '';
    if (!spec.expanded && visible < total) {
      const label = spec.view === 'before' || spec.view === 'after'
        ? `Load full ${spec.view} definition`
        : spec.focus && !isDiff ? 'Read full definition' : isDiff ? 'Load full diff' : 'Load full code';
      load = `<button class="load-bar" data-expand="${spec.id}"><span aria-hidden="true">↓</span> ${label} <span class="load-count">· ${total - visible} more lines${total > CHUNK ? ' · loaded in chunks' : ''}</span></button>`;
    } else if (spec.expanded && spec.loaded < full.length) {
      load = `<button class="load-bar" data-expand="${spec.id}">Load next ${Math.min(CHUNK, full.length - spec.loaded)} rows <span class="load-count">· ${total - visible} lines remaining</span></button>`;
    }
    const activePath = $('.filename', element)?.title || spec.primaryPath;
    const viewLabel = spec.view === 'comparison' ? 'function comparison' : isDiff ? 'diff' : spec.view === 'before' ? 'before definition' : spec.view === 'after' ? 'after definition' : 'code';
    const emptyMessage = spec.raw && ['unavailable', 'unbound'].includes(spec.raw.patch_status)
      ? 'Original patch rows are unavailable for this file.' : 'No line changes in this view.';
    $('.code-body', element).innerHTML = `<div class="code-region" tabindex="0" role="region" aria-label="${esc(basename(activePath))} ${viewLabel}"><div class="code-lines">${shown.length ? rowsHTML(shown, isDiff) : `<div class="code-note">${esc(emptyMessage)}</div>`}</div></div>${load}`;
    $('.code-footer', element).innerHTML = blockFooter(spec, visible, total);
    element.dataset.loaded = 'true';
  }
  const observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting || entry.target.closest('details:not([open])')) continue;
      const spec = blocks.get(entry.target.dataset.block);
      if (spec && !spec.materialized) paintBlock(spec);
      observer.unobserve(entry.target);
    }
  }, {rootMargin: '500px 0px'}) : null;
  function observeBlocks(parent = document) {
    $$('[data-block]', parent).forEach(el => {
      const spec = blocks.get(el.dataset.block);
      if (!spec || spec.materialized) return;
      if (observer) observer.observe(el);
      else if (!el.closest('details:not([open])')) paintBlock(spec);
    });
  }

  function defaultPassages(g) {
    // Full-PR reports without authored passages still alternate explanation and evidence.
    const cs = g.change_ids.map(id => C.get(id));
    return cs.map((c, i) => {
      const a = c.after || c.before;
      let text = i === 0 ? g.narrative.intent : `Next, inspect \`${a.name}\` in \`${basename(a.path)}\`.`;
      if (c.kind === 'moved') text += ' The declaration has an identical AST at its new location; its surrounding bindings still deserve review.';
      else if (c.kind === 'moved_renamed') text += ` \`${c.before.name}\` becomes \`${c.after.name}\`. The comparison excludes the declaration name and leading docstring, not internal identifiers or literals.`;
      else if (c.kind === 'moved_modified') text += ' This pair contains edits, not just a relocation. Compare its definition and paired diff.';
      return {text, change_ids: [c.id]};
    });
  }
  function passageFocus(p, changes) {
    if (!flatMode || !Array.isArray(p.citations)) return p.focus || null;
    const focus = {before: [], after: []};
    const refs = p.citations.map(ref => R.flatDefinition(ref));
    for (const change of changes) {
      for (const side of ['before', 'after']) {
        const source = change[side];
        if (!source) continue;
        for (const ref of refs) {
          if (ref.source_id !== source.source_id || ref.side !== (side === 'before' ? 'base' : 'head')) continue;
          focus[side].push({start: ref.start, end: ref.end});
        }
      }
    }
    return focus.before.length || focus.after.length ? focus : null;
  }
  function passageBlocks(p) {
    const ids = p.change_ids || [];
    if (!flatMode) {
      const changes = ids.map(id => C.get(id)).filter(Boolean);
      if (changes.length > 1 && changes.every(change => change.kind === 'moved')) {
        return blockHTML(ids, {view: 'diff', label: p.label, focus: p.focus});
      }
      return ids.map(id => blockHTML([id], {view: 'diff', focus: p.focus, label: p.label})).join('');
    }
    const groups = new Map();
    for (const id of ids) {
      const change = C.get(id);
      if (!change) continue;
      const key = `${change.before?.path || ''}\u0000${change.after?.path || ''}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(change);
    }
    return [...groups.values()].map(changes => {
      const label = changes.length > 1 ? `${changes.length} related changes` : '';
      return blockHTML(changes.map(change => change.id), {
        view: 'diff',
        focus: passageFocus(p, changes),
        label,
        related: changes.length > 1,
      });
    }).join('');
  }
  function passageHTML(p) {
    const anchor = flatMode && typeof p.id === 'string'
      ? ` id="${passageAnchor(p.id)}"` : '';
    const code = passageBlocks(p);
    let citations = '';
    if (flatMode && p.citations.length) {
      const sources = p.citations.map(ref => {
        const source = R.flatDefinition(ref);
        const id = `flat-citation-${++flatCitationCount}`;
        C.set(id, {id, kind: 'text', label: 'Cited source', basis: `Pinned ${source.side} revision · original lines ${ref.start}–${ref.end}`,
          before: source.side === 'base' ? source : null, after: source.side === 'head' ? source : null, hunks: []});
        return blockHTML([id], {view: 'definition', sourceOnly: true});
      }).join('');
      citations = `<details class="section-extra citation-sources"><summary>Cited source <span>· ${p.citations.length}</span></summary>${sources}</details>`;
    }
    return `<div class="passage"${anchor}><div class="prose">${prose(p.text)}</div>${code}${citations}</div>`;
  }
  function sectionHTML(g) {
    const n = g.narrative;
    const passages = Array.isArray(n.passages) && n.passages.length ? n.passages : defaultPassages(g);
    const used = new Set(passages.flatMap(p => p.change_ids));
    const extra = g.change_ids.filter(id => !used.has(id));
    const context = [n.intent, n.why_now].filter(value => typeof value === 'string' && value.trim());
    const contextHTML = context.length
      ? `<div class="section-context">${context.map(value => `<p>${inline(value)}</p>`).join('')}</div>`
      : '';
    return `<section class="story-section" id="${g.id}" data-story-section="${g.id}" aria-labelledby="heading-${g.id}"><header class="section-heading"><span class="section-number">${String(g.number).padStart(2, '0')}</span><h2 id="heading-${g.id}">${inline(g.title)}<a class="section-anchor" href="#${g.id}" aria-label="Link to ${esc(plainInline(g.title))}">#</a></h2></header>${contextHTML}${passages.map(passageHTML).join('')}${n.transition ? `<div class="section-end">${inline(n.transition)}</div>` : ''}${extra.length ? `<details class="section-extra" data-extra="${g.id}"><summary>Supporting changes <span>· ${extra.length}</span></summary><div class="extra-body"></div></details>` : ''}</section>`;
  }
  // Accept one branch with the expected marker; split at its first arrow.
  function decisionBranch(line, marker) {
    const match = /^\s*([├└])─\s+([^→]+)→(.+)$/.exec(line);
    if (!match || match[1] !== marker) return null;
    const condition = match[2].trim(), outcome = match[3].trim();
    if (!condition || !outcome || condition.length > 120 || outcome.length > 120) return null;
    return { condition, outcome };
  }
  // Recognize the compact three-line decision format; other sketches stay text.
  function decisionFlowchart(value) {
    const lines = value.trim().split('\n');
    if (lines.length !== 3) return null;
    const label = lines[0].trim();
    if (!label || label.length > 36) return null;
    const paths = [decisionBranch(lines[1], '├'), decisionBranch(lines[2], '└')];
    return paths.every(Boolean) ? { label, paths } : null;
  }
  function sketchHTML(value) {
    const flowchart = decisionFlowchart(value);
    if (flowchart) {
      const branchesHTML = flowchart.paths.map(path => `
        <div class="flowchart-path">
          <div class="flowchart-condition">${esc(path.condition)}</div>
          <span class="flowchart-down" aria-hidden="true">↓</span>
          <div class="flowchart-outcome">${esc(path.outcome)}</div>
        </div>`).join('');
      return `<figure class="preamble-flowchart">
        <figcaption>Decision flow chart</figcaption>
        <div class="flowchart-body">
          <div class="flowchart-decision">
            <svg viewBox="0 0 240 110" aria-hidden="true"><polygon points="120,2 238,55 120,108 2,55"/></svg>
            <span>${esc(flowchart.label)}</span>
          </div>
          <svg class="flowchart-connectors" viewBox="0 0 100 42" preserveAspectRatio="none" aria-hidden="true">
            <path d="M50 0 V16 H25 V38 M50 16 H75 V38"/>
            <path d="M22 35 L25 41 L28 35 M72 35 L75 41 L78 35"/>
          </svg>
          <div class="flowchart-paths">${branchesHTML}</div>
        </div>
      </figure>`;
    }
    return `<figure class="preamble-sketch"><figcaption>Conceptual sketch</figcaption><pre><code>${esc(value)}</code></pre></figure>`;
  }
  function preambleHTML() {
    const preamble = R.document?.preamble;
    if (typeof preamble !== 'string' || !preamble.trim()) return '';
    const content = preamble.replace(/\r\n?/g, '\n');
    const sketchFence = /(^|\n)```text[ \t]*\n([\s\S]*?)\n```(?=\n|$)/g;
    const paragraphs = value => value.trim().split(/\n\s*\n/).filter(Boolean)
      .map(part => `<p>${inline(part.trim())}</p>`).join('');
    let body = '', cursor = 0;
    for (const match of content.matchAll(sketchFence)) {
      body += paragraphs(content.slice(cursor, match.index));
      body += sketchHTML(match[2]);
      cursor = match.index + match[0].length;
    }
    body += paragraphs(content.slice(cursor));
    return `<section class="preamble" aria-labelledby="preamble-heading"><h2 id="preamble-heading">Before the code</h2><div class="preamble-body">${body}</div></section>`;
  }
  function passageSentences(value) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (!text) return [];
    const sentences = [];
    let start = 0;
    for (let index = 0; index < text.length; index++) {
      if (!'.!?'.includes(text[index])) continue;
      let end = index + 1;
      while (end < text.length && '.!?'.includes(text[end])) end++;
      if (end < text.length && !/\s/u.test(text[end])) continue;
      const sentence = text.slice(start, end).trim();
      if (sentence) sentences.push(sentence);
      while (end < text.length && /\s/u.test(text[end])) end++;
      start = end;
      index = end - 1;
    }
    const rest = text.slice(start).trim();
    if (rest) sentences.push(rest);
    return sentences.length ? sentences : [text];
  }
  function passageSentence(record) {
    const text = String(record.passage.text || '').replace(/\s+/g, ' ').trim();
    const sentence = passageSentences(text)[0] || text;
    return sentence.length > 150 ? `${sentence.slice(0, 147).trimEnd()}…` : sentence;
  }
  function passageLinks(ids, className = 'passage-links') {
    const targets = [...new Set(Array.isArray(ids) ? ids : [])]
      .map(id => ({id, record: flatPassages.get(id)}))
      .filter(target => target.record);
    const sectionCounts = new Map();
    for (const {record} of targets) {
      sectionCounts.set(record.group, (sectionCounts.get(record.group) || 0) + 1);
    }
    const links = targets.map(({id, record}) => {
      const title = record.group.title;
      const paragraphIndex = (record.group.narrative.passages || [])
        .findIndex(passage => passage.id === id) + 1;
      const label = sectionCounts.get(record.group) > 1 && paragraphIndex > 0
        ? `${title} · paragraph ${paragraphIndex}` : title;
      return `<a href="#${passageAnchor(id)}" title="${esc(passageSentence(record))}">${inline(label)}</a>`;
    }).filter(Boolean);
    return links.length
      ? `<span class="${className}">${links.join('')}</span>` : '';
  }
  function walkthroughOpeningHTML() {
    if (!flatMode || !flatStory) return '';
    let summary = '';
    const quick = flatStory.quick_summary;
    if (quick && typeof quick.before === 'string' && typeof quick.after === 'string') {
      summary = `<section class="quick-summary" aria-labelledby="quick-summary-heading"><h2 id="quick-summary-heading">Before and after</h2><dl class="summary-pair"><div><dt>Before</dt><dd>${inline(quick.before)}</dd></div><div><dt>After</dt><dd>${inline(quick.after)}</dd></div></dl><p class="summary-evidence"><span class="evidence-label">Evidence</span>${passageLinks(quick.passage_ids)}</p></section>`;
    }
    let impact = '';
    if (Object.prototype.hasOwnProperty.call(flatStory, 'impact')) {
      impact = flatStory.impact && typeof flatStory.impact.text === 'string'
        ? `<section class="walkthrough-impact"><h2>Practical impact</h2><p>${inline(flatStory.impact.text)}</p>${passageLinks(flatStory.impact.passage_ids, 'passage-links impact-links')}</section>`
        : flatStory.impact === null
          ? '<section class="walkthrough-impact"><h2>Practical impact</h2><p>Practical impact is not established by the supplied evidence.</p></section>'
          : '';
    }
    const route = Array.isArray(flatStory.route) && flatStory.route.length
      ? `<nav class="reading-route" aria-label="Reading route"><h2>Reading route</h2>${flatStory.route.map(item => `<p><span class="route-label">${esc(item.kind[0].toUpperCase() + item.kind.slice(1))}</span>${passageLinks(item.passage_ids, 'passage-links route-links')}</p>`).join('')}</nav>`
      : '';
    const checks = Array.isArray(flatStory.author_reported_checks)
      ? flatStory.author_reported_checks : [];
    const authorBody = checks.length
      ? `<ul>${checks.map(check => `<li>${inline(check)}</li>`).join('')}</ul>`
      : '<p>No results were extracted from the captured description.</p>';
    const analyzerRuns = Number.isInteger(R.stats.test_runs)
      ? `${R.stats.test_runs} test runs verified by Diffstory.`
      : 'Test-run results are not recorded by Diffstory.';
    const staticStatus = R.tests?.length
      ? ' Static test sources are references for review; they do not show that a test ran.'
      : ' No individual test definitions are listed separately. Test code may still be available in the cited source.';
    const authorLabel = checks.length
      ? 'Author-reported checks · not independently verified'
      : 'Author-reported checks · not independently verified · No results were extracted';
    const verification = `<section class="verification-status" aria-label="Verification status"><p><strong>Analyzer verification</strong> ${esc(analyzerRuns)}${esc(staticStatus)}</p><details class="author-reported"><summary>${esc(authorLabel)}</summary>${authorBody}</details></section>`;
    return `<section class="walkthrough-opening" aria-label="Change summary">${summary}${impact}${route}${verification}</section>`;
  }
  function reviewChecksHTML() {
    if (!flatMode || !flatStory) return '';
    const checks = flatStory.review_checks;
    if (!checks.length) {
      return '<section class="review-checks"><h2>Review checks</h2><p>No open review checks were recorded in this story.</p></section>';
    }
    const testById = new Map((R.tests || []).map(test => [test.id, test]));
    const rows = checks.map(check => {
      const tests = (check.test_ids || []).map(id => testById.get(id)).filter(Boolean);
      const testHTML = tests.length
        ? `<details class="review-test-source"><summary>Static test source · ${tests.map(test => esc(test.name || basename(test.path))).join(', ')} <span>(source reference, not execution proof)</span></summary>${tests.map(test => blockHTML([], {test})).join('')}</details>`
        : (check.test_ids || []).length
          ? '<p class="review-source-note">Local source for a referenced static test is not included.</p>' : '';
      return `<article class="review-check"><p>${inline(check.text)}</p><p class="review-evidence"><span class="evidence-label">Related passages</span>${passageLinks(check.passage_ids, 'passage-links review-links')}</p>${testHTML}</article>`;
    }).join('');
    return `<section class="review-checks" aria-labelledby="review-checks-heading"><h2 id="review-checks-heading">Open review checks</h2><p class="review-source-note">These are questions for source review. Static test references show code that can be inspected; they do not show execution.</p>${rows}</section>`;
  }
  function buildHeader() {
    const m = R.meta, doc = R.document || {};
    const title = m.title || 'Read the change, one idea at a time';
    document.title = `Diffstory · ${m.number ? '#' + m.number : plainInline(title)} · Reader`;
    const lead = doc.lead || 'Read the change in one continuous thread: the idea, its code, then what follows from it. Source stays close to the explanation.';
    const provider = generation?.provider;
    const model = generation?.model;
    const generationNotice = generation
      ? `<p class="generation-notice" role="status"><strong>Model-generated narration · unverified.</strong> Provider: ${esc(provider)} · model: ${esc(model)}. Check every claim against the linked source.</p>` : '';
    $('#documentHeader').innerHTML = `<div class="document-kicker">${link(m.url, m.repository || 'Local repository') || esc(m.repository || 'Local repository')}<span class="kicker-dot">/</span><span>${m.number ? 'PR ' + m.number : m.input === 'synthetic example' ? 'SYNTHETIC EXAMPLE' : 'COMMITTED COMPARISON'}</span></div><h1>${inline(title)}</h1><p class="lead">${inline(lead)}</p>${generationNotice}<div class="document-meta"><span>${R.groups.length} sections</span><span class="sep">·</span><span>${R.stats.supplied_paths} source paths</span><span class="sep">·</span><span class="revisions">${esc((m.base_sha || 'base').slice(0, 7))} → ${esc((m.head_sha || 'head').slice(0, 7))}</span></div><p class="document-scope">${m.input === 'synthetic example' ? 'Original synthetic example. No private repository source.' : m.scope === 'selected excerpts' ? 'Selected source excerpts, not the complete PR.' : 'Committed, changed-file snapshot.'} <a href="#evidence">Evidence & scope</a></p><div class="header-rule"></div>`;
  }
  function buildFooter() {
    const closing = R.document?.closing || 'The definitions, their callers, and the available tests are now in one reading path. Structural matches and test references guide review; they do not substitute for execution evidence.';
    $('#documentFooter').innerHTML = `<div class="closing"><span class="closing-label">The shape of the change</span>${inline(closing)}</div>${reviewChecksHTML()}<details id="source-hunks" class="appendix" data-appendix="raw"><summary>Original source hunks</summary><div class="appendix-content"></div></details><details id="evidence" class="appendix" data-appendix="evidence"><summary>Evidence, revisions & scope</summary><div class="appendix-content"></div></details><div class="bottom-line"><span>Diffstory 0.3 · by Epistemic AI · stored locally</span><a href="#top">Back to top ↑</a></div>`;
  }
  function normalizedSearch(value) {
    return plainInline(String(value ?? '')).replace(/\s+/g, ' ').trim().toLowerCase();
  }
  function passageSearchSentence(passage, query) {
    const text = String(passage.text || '').replace(/\s+/g, ' ').trim();
    const result = passageSentences(text).find(sentence => normalizedSearch(sentence).includes(query)) || text;
    const plain = plainInline(result).replace(/\s+/g, ' ').trim();
    if (plain.length <= 240) return result;

    const queryText = normalizedSearch(query);
    let source = result;
    let matchIndex = source.toLowerCase().indexOf(queryText);
    if (matchIndex < 0) {
      source = plain;
      matchIndex = source.toLowerCase().indexOf(queryText);
    }
    if (matchIndex < 0) return `${plain.slice(0, 237).trimEnd()}…`;

    const budget = Math.max(240, queryText.length + 96);
    const context = Math.floor((budget - queryText.length) / 2);
    let start = Math.max(0, matchIndex - context);
    let end = Math.min(source.length, matchIndex + queryText.length + context);
    if (source === result) {
      for (const match of source.matchAll(/`[^`\n]+`/g)) {
        const spanStart = match.index;
        const spanEnd = spanStart + match[0].length;
        if (spanStart < start && spanEnd > start) start = spanStart;
        if (spanStart < end && spanEnd > end) end = spanEnd;
      }
    }
    while (start > 0 && !/\s/u.test(source[start - 1])) start--;
    while (end < source.length && !/\s/u.test(source[end])) end++;
    const excerpt = source.slice(start, end).trim();
    return `${start > 0 ? '…' : ''}${excerpt}${end < source.length ? '…' : ''}`;
  }
  function contents() {
    const q = normalizedSearch($('#search').value);
    $('#sectionCount').textContent = `${R.groups.length} sections`;
    const rows = [];
    for (const g of R.groups) {
      const sectionText = [g.title, g.path, g.theme, g.narrative.intent, ...g.change_ids.map(id => {
        const c = C.get(id); return [c.before?.name, c.after?.name, c.before?.path, c.after?.path].join(' ');
      })].join(' ');
      const sectionMatches = !q || normalizedSearch(sectionText).includes(q);
      const passageMatches = flatMode && q
        ? (g.narrative.passages || []).filter(p => normalizedSearch(p.text).includes(q)).slice(0, 2)
        : [];
      if (sectionMatches || passageMatches.length) {
        rows.push(`<a class="contents-item" href="#${g.id}" data-section="${g.id}" ${activeSection === g.id ? 'aria-current="location"' : ''}><span class="contents-number">${String(g.number).padStart(2, '0')}</span><span>${inline(g.title)}</span></a>`);
      }
      for (const passage of passageMatches) {
        const sentence = passageSearchSentence(passage, q);
        rows.push(`<a class="contents-item passage-result" href="#${passageAnchor(passage.id)}" data-passage-result="true"><span class="contents-number" aria-hidden="true">↳</span><span>${inline(sentence)}</span></a>`);
      }
    }
    $('#contentsList').innerHTML = rows.join('') || '<p class="empty">No matching section, symbol, or passage.</p>';
  }
  function ensureExtra(details) {
    if (details.dataset.ready) return;
    const g = G.get(details.dataset.extra);
    const used = new Set((g.narrative.passages || []).flatMap(p => p.change_ids));
    $('.extra-body', details).innerHTML = g.change_ids.filter(id => !used.has(id)).map(id => blockHTML([id])).join('');
    details.dataset.ready = 'true';
    observeBlocks(details);
  }
  function ensureAppendix(details) {
    if (details.dataset.ready) return;
    const body = $('.appendix-content', details);
    if (details.dataset.appendix === 'raw') {
      const files = new Map();
      R.raw_files.forEach(f => { if (!files.has(f.path)) files.set(f.path, []); files.get(f.path).push(f); });
      body.innerHTML = `<p>Git/GitHub patches keep their original line grouping. An unavailable original patch is never replaced with a generated one. Open a file to inspect its diff.</p>${[...files.entries()].map(([path, fs], i) => `<details class="raw-file" data-raw="${i}"><summary>${esc(path)} <span class="region-label">· ${fs.length} region${fs.length === 1 ? '' : 's'}</span></summary><div class="raw-body"></div></details>`).join('')}`;
    } else {
      const m = R.meta;
      body.innerHTML = `<dl><dt>Repository</dt><dd>${link(m.url, m.repository || 'Local Git') || esc(m.repository || 'Local Git')}</dd><dt>Base revision</dt><dd class="mono">${esc(m.base_sha)}</dd><dt>Head revision</dt><dd class="mono">${esc(m.head_sha)}</dd><dt>Scope</dt><dd>${esc(m.scope)}</dd><dt>Captured</dt><dd>${esc(m.captured_at || 'Not recorded')}</dd><dt>Supplied source</dt><dd>${R.stats.supplied_paths} paths · ${R.stats.units} analysis units</dd><dt>Test runs</dt><dd>0 runs verified by this analyzer</dd></dl><h3>What the matches establish</h3><p>An identical AST is structural evidence, not a proof that a move preserves behavior. Changed global bindings, import paths, initialization order and external consumers still need review. An edited pair is not automatically a confirmed behavioral change.</p><h3>Provenance</h3><p>${esc(m.validation || 'No execution evidence was attached.')}</p><p>Narrative paragraphs are authored interpretations linked to immutable source ranges. They cannot override the compiler’s source, structural classifications or test status.</p>${R.notes.length ? `<h3>Analysis notes</h3><ul>${R.notes.map(note => `<li>${esc(note)}</li>`).join('')}</ul>` : ''}${R.warnings.length ? `<h3>Limits of this input</h3><ul>${R.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>` : ''}<details><summary>Compiler methods</summary><dl>${Object.entries(R.method).map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl></details>${m.description ? `<details><summary>PR description, as captured</summary><pre>${esc(m.description)}</pre></details>` : ''}<h3>Local by design</h3><p>No external fonts, libraries, analytics or network requests are needed. The HTML embeds the source snapshot; keep it private for a private repository. Diffs are rendered lazily from that embedded data, not fetched from a server. Clicking a source link opens GitHub.</p>`;
    }
    details.dataset.ready = 'true';
  }
  function ensureRaw(details) {
    if (details.dataset.ready) return;
    const paths = [...new Set(R.raw_files.map(f => f.path))];
    const files = R.raw_files.filter(f => f.path === paths[Number(details.dataset.raw)]);
    $('.raw-body', details).innerHTML = files.map(f => {
      const original = typeof f.original_patch === 'string'
        ? `<details class="original-patch"><summary>Original patch text</summary><pre>${esc(f.original_patch || 'No line changes in the supplied patch.')}</pre></details>`
        : f.patch_status === 'unavailable' ? '<p class="code-note">The original patch was not supplied. No replacement diff has been generated.</p>' : '';
      return blockHTML([], {raw: f, view: 'diff'}) + original;
    }).join('');
    details.dataset.ready = 'true'; observeBlocks(details);
  }
  document.addEventListener('toggle', event => {
    const d = event.target;
    if (d.tagName !== 'DETAILS' || !d.open) return;
    if (d.dataset.extra) ensureExtra(d);
    else if (d.dataset.appendix) ensureAppendix(d);
    else if ('raw' in d.dataset) ensureRaw(d);
    observeBlocks(d);
  }, true);

  function closePopovers(returnFocus = false) {
    let trigger = null;
    if (!$('#contentsPanel').hidden) trigger = $('#contentsBtn');
    if (!$('#morePanel').hidden) trigger = $('#moreBtn');
    $('#contentsPanel').hidden = true; $('#morePanel').hidden = true;
    $('#contentsBtn').setAttribute('aria-expanded', 'false'); $('#moreBtn').setAttribute('aria-expanded', 'false');
    if (returnFocus && trigger) trigger.focus();
  }
  function openPopover(name, focusSearch = false) {
    const panel = $('#' + name + 'Panel'), wasOpen = !panel.hidden;
    closePopovers();
    if (wasOpen && !focusSearch) return;
    panel.hidden = false; $('#' + name + 'Btn').setAttribute('aria-expanded', 'true');
    if (name === 'contents') { contents(); if (focusSearch) $('#search').focus(); }
  }
  function navigateTo(hash, push = true) {
    let id;
    try { id = decodeURIComponent(hash.replace(/^#/, '')); } catch (_) { return; }
    const target = document.getElementById(id);
    if (!target) return;
    if (target.tagName === 'DETAILS') {
      target.open = true;
      if (target.dataset.appendix) ensureAppendix(target);
    }
    let parent = target.parentElement;
    while (parent) { if (parent.tagName === 'DETAILS') parent.open = true; parent = parent.parentElement; }
    closePopovers();
    if (push) { try { history.pushState(null, '', '#' + id); } catch (_) { location.hash = id; } }
    target.scrollIntoView({behavior: 'auto', block: 'start'});
    updateScroll(); observeBlocks(target);
  }
  function updateScroll() {
    const sections = $$('.story-section');
    const start = $('#story').offsetTop;
    const end = $('#documentFooter').offsetTop;
    const percent = end > start ? Math.max(0, Math.min(100, (window.scrollY + 140 - start) / (end - start) * 100)) : 0;
    $('#readingProgress').style.width = percent + '%';
    let current = null;
    for (const s of sections) { if (s.getBoundingClientRect().top <= 180) current = s.dataset.storySection; else break; }
    if (current !== activeSection) {
      activeSection = current;
      const g = G.get(current);
      $('#readingPosition').textContent = g ? `${String(g.number).padStart(2, '0')} / ${R.groups.length}  ·  ${g.title}` : '';
      $$('.contents-item').forEach(el => { if (el.dataset.section === current) el.setAttribute('aria-current', 'location'); else el.removeAttribute('aria-current'); });
    }
  }
  let scrollScheduled = false;
  window.addEventListener('scroll', () => {
    if (scrollScheduled) return;
    scrollScheduled = true;
    requestAnimationFrame(() => { updateScroll(); scrollScheduled = false; });
  }, {passive: true});
  window.addEventListener('resize', updateScroll);
  window.addEventListener('hashchange', () => navigateTo(location.hash, false));
  window.addEventListener('popstate', () => navigateTo(location.hash || '#top', false));

  function embeddedReportJSONParts(scriptJSON) {
    const replacements = new Map([
      ['\\u0026', '&'], ['\\u003c', '<'], ['\\u003e', '>'],
      ['\\u2028', '\u2028'], ['\\u2029', '\u2029']
    ]);
    let inString = false, escaped = false, copiedThrough = 0, parts = null;
    for (let index = 0; index < scriptJSON.length; index += 1) {
      const character = scriptJSON[index];
      if (!inString) {
        if (character === '"') inString = true;
        continue;
      }
      if (escaped) { escaped = false; continue; }
      if (character === '"') { inString = false; continue; }
      if (character !== '\\') continue;
      const token = scriptJSON.slice(index, index + 6);
      if (replacements.has(token)) {
        parts ||= [];
        parts.push(scriptJSON.slice(copiedThrough, index), replacements.get(token));
        index += 5; copiedThrough = index + 1;
      } else escaped = true;
    }
    if (!parts) return [scriptJSON];
    parts.push(scriptJSON.slice(copiedThrough));
    return parts;
  }
  function download(name, value, rawJSON = false) {
    const parts = rawJSON ? embeddedReportJSONParts(value) : [JSON.stringify(value, null, 2)];
    const blob = new Blob(parts, {type: 'application/json'});
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
  }
  document.addEventListener('click', event => {
    const anchor = event.target.closest('a[href^="#"]');
    if (anchor) { event.preventDefault(); navigateTo(anchor.getAttribute('href')); return; }
    const button = event.target.closest('button');
    if (!button) { if (!event.target.closest('.popover')) closePopovers(); return; }
    if (button.id === 'contentsBtn') { openPopover('contents'); return; }
    if (button.id === 'moreBtn') { openPopover('more'); return; }
    if (button.dataset.loadPreview) { paintBlock(blocks.get(button.dataset.loadPreview)); return; }
    if (button.dataset.expand) {
      const s = blocks.get(button.dataset.expand);
      s.loaded = Math.min(allRows(s).length, (s.expanded ? s.loaded : 0) + CHUNK);
      s.expanded = true; paintBlock(s); return;
    }
    if (button.dataset.collapse) {
      const s = blocks.get(button.dataset.collapse), element = document.getElementById(s.id);
      const above = element.getBoundingClientRect().top < 60;
      s.expanded = false; s.loaded = 0; paintBlock(s);
      if (above) element.scrollIntoView({block: 'start'});
      return;
    }
    if (button.dataset.view) {
      const s = blocks.get(button.dataset.view), element = document.getElementById(s.id);
      const above = element.getBoundingClientRect().top < 60;
      s.view = button.dataset.side; s.expanded = false; s.loaded = 0;
      paintBlock(s); if (above) element.scrollIntoView({block: 'start'}); return;
    }
    if (button.dataset.switch) {
      const s = blocks.get(button.dataset.switch);
      const element = document.getElementById(s.id), above = element.getBoundingClientRect().top < 60;
      s.view = s.view === 'diff' ? 'definition' : 'diff'; s.expanded = false; s.loaded = 0;
      paintBlock(s); if (above) element.scrollIntoView({block: 'start'}); return;
    }
    if (button.id === 'collapseBtn') {
      const active = activeSection;
      for (const s of blocks.values()) if (s.materialized && s.expanded) { s.expanded = false; s.loaded = 0; paintBlock(s); }
      closePopovers(); if (active) navigateTo('#' + active, false); return;
    }
    if (button.id === 'downloadReport') {
      if (flatMode) download('diffstory-report.json', reportDataElement.textContent, true);
      else download('diffstory-report.json', R);
      closePopovers(); return;
    }
  });
  $('#search').addEventListener('input', contents);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') { closePopovers(true); return; }
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName) || event.target.isContentEditable || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === '/') { event.preventDefault(); openPopover('contents', true); }
    // Deliberately leave arrow keys, Page Down, Space and browser Find alone.
  });

  buildHeader();
  $('#story').innerHTML = `${preambleHTML()}${walkthroughOpeningHTML()}${R.groups.length ? R.groups.map(sectionHTML).join('') : '<div class="prose"><p>No changed units in this committed comparison.</p></div>'}`;
  buildFooter(); contents(); observeBlocks(); updateScroll();
  if (location.hash) requestAnimationFrame(() => navigateTo(location.hash, false));
})();
