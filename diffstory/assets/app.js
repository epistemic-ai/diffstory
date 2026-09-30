'use strict';
(() => {
  // The report is inert data, never repository code to execute. All prose/source is escaped.
  const R = JSON.parse(document.getElementById('report-data').textContent);
  const $ = (selector, parent = document) => parent.querySelector(selector);
  const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
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
        return `<strong>${inlineText(part.slice(2, -2))}</strong>`;
      }
      return inlineText(part);
    }).join('');
  const plainInline = value => String(value ?? '')
    .replace(/`([^`\n]+)`/g, '$1')
    .replace(/\*\*([^*\n]+)\*\*/g, '$1');
  const prose = value => String(value ?? '').split(/\n\s*\n/).filter(Boolean)
    .map(p => `<p>${inline(p)}</p>`).join('');
  const G = new Map(R.groups.map(g => [g.id, g]));
  const C = new Map(R.changes.map(c => [c.id, c]));
  const safeURL = value => {
    try { const u = new URL(value); return u.protocol === 'https:' && u.hostname === 'github.com' ? u.href : null; }
    catch (_) { return null; }
  };
  const link = (url, label, cls = '') => safeURL(url)
    ? `<a href="${esc(safeURL(url))}" class="${cls}" target="_blank" rel="noopener noreferrer">${esc(label)} ↗</a>` : '';
  let activeSection = null;
  let blockCounter = 0;
  const blocks = new Map();
  const anchors = new Set();
  const PREVIEW_LIMIT = 34;
  const PREVIEW_HEAD = 19;
  const PREVIEW_TAIL = 5;
  const CHUNK = 160;

  function classification(c) {
    if ((c.after || c.before)?.is_test) return ['Test source', ''];
    const kinds = {
      moved: ['Moved · same AST', 'move'],
      moved_renamed: ['Moved + renamed', 'rename'],
      renamed: ['Renamed', 'rename'],
      moved_modified: ['Moved + edited', 'edited'],
      modified: ['Edited', 'edited'],
      source_only: ['Source-only edit', ''],
      observed_head: ['Head excerpt', ''],
      observed_base: ['Base excerpt', ''],
      added: ['Added', 'move'],
      removed: ['Removed', 'removed'],
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
  function allRows(spec) {
    if (spec.cache[spec.view]) return spec.cache[spec.view];
    let rows = [];
    if (spec.raw) rows = diffRows(spec.raw.hunks);
    else if (spec.test) rows = definitionRows(spec.test);
    else {
      for (const c of spec.changes) {
        const source = c.after || c.before;
        if (spec.view === 'diff') {
          const r = diffRows(c.hunks);
          rows.push(...(r.length ? r : [{type: 'note', text: `${source.name}: no textual edits in this paired definition.`}]));
        } else {
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
  function previewRows(spec) {
    const rows = allRows(spec);
    if (spec.expanded) return rows.slice(0, spec.loaded);
    if (spec.focus && spec.view === 'definition') {
      const selected = rows.filter(r => r.type === 'line' && r.new >= spec.focus.start && r.new <= spec.focus.end);
      if (selected.length) {
        const nAbove = rows.filter(r => r.type === 'line' && r.new < spec.focus.start).length;
        const nBelow = rows.filter(r => r.type === 'line' && r.new > spec.focus.end).length;
        return [...(nAbove ? [{type: 'gap', text: `… ${nAbove} lines above this excerpt`}] : []), ...(selected.length <= PREVIEW_LIMIT ? selected : [...selected.slice(0, PREVIEW_HEAD), {type: 'gap', text: `… ${selected.length - PREVIEW_HEAD - PREVIEW_TAIL} lines folded inside this excerpt`}, ...selected.slice(-PREVIEW_TAIL)]),
          ...(nBelow ? [{type: 'gap', text: `… ${nBelow} lines below this excerpt`}] : [])];
      }
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

  function blockHTML(changeIds, options = {}) {
    const cs = (changeIds || []).map(id => C.get(id)).filter(Boolean);
    const first = cs[0];
    const source = options.test || (first && (first.after || first.before));
    const primaryPath = options.raw?.path || source?.path || '';
    const spec = {
      id: `block-${++blockCounter}`, changes: cs, test: options.test || null,
      raw: options.raw || null, focus: options.focus || null, label: options.label || '',
      view: options.view || (options.test ? 'definition' : 'diff'),
      expanded: false, loaded: 0, materialized: false, cache: {}, primaryPath,
      baseURL: options.raw?.before_url || first?.before?.url,
      headURL: options.raw?.after_url || options.test?.url || first?.after?.url,
    };
    blocks.set(spec.id, spec);
    let badge = options.raw ? ['Source hunk', ''] : options.test ? ['Test source', ''] : cs.length > 1 ? [`${cs.length} moved declarations`, 'move'] : classification(first);
    const name = options.label || (cs.length > 1 ? 'graph constants' : source?.name === 'module imports / context' ? 'imports' : source?.name === 'file context' ? '' : source?.name || '');
    const symbol = name ? `<span class="symbol-name">· ${esc(name)}</span>` : '';
    const focusedEstimate = spec.focus && spec.view === 'definition'
      ? spec.focus.end - spec.focus.start + 3
      : null;
    const sourceEstimate = options.raw
      ? options.raw.hunks.reduce((total, hunk) => total + hunk.rows.length + 1, 0)
      : options.test
        ? options.test.end - options.test.start + 1
        : cs.reduce((total, change) => {
          const source = change.after || change.before;
          const size = spec.view === 'diff'
            ? change.hunks.reduce((hunkTotal, hunk) => hunkTotal + hunk.rows.length + 1, 0)
            : source.end - source.start + 1;
          return total + size;
        }, 0);
    const estimated = Math.min(PREVIEW_LIMIT, focusedEstimate ?? sourceEstimate);
    const beforeAnchors = cs.filter(c => !anchors.has(c.id)).map(c => { anchors.add(c.id); return `<span id="unit-${c.id}" class="change-anchor"></span>`; }).join('');
    return `${beforeAnchors}<figure class="code-block" id="${spec.id}" data-block="${spec.id}" data-changes="${cs.map(c => c.id).join(' ')}"><figcaption class="code-header"><span class="filename" title="${esc(primaryPath)}">${esc(basename(primaryPath))}${symbol}</span><span class="classification ${badge[1]}">${esc(badge[0])}</span></figcaption><div class="code-body"><div class="lazy-shell" style="min-height:${Math.max(58, Math.min(430, estimated * 22))}px"><button data-load-preview="${spec.id}">Load ${spec.view === 'diff' ? 'diff' : 'code'} ↓</button></div></div><div class="code-footer"></div></figure>`;
  }
  function blockFooter(spec, shown, total) {
    const first = spec.changes[0];
    const a = spec.test || first?.after || first?.before;
    const b = first?.before;
    let location = spec.changes.length > 1
      ? (spec.view === 'diff' ? 'Paired diffs' : 'Definitions')
      : spec.view === 'diff' ? 'Paired diff' : a ? `L${spec.focus && !spec.expanded ? spec.focus.start : a.start}–${spec.focus && !spec.expanded ? spec.focus.end : (spec.changes.at(-1)?.after || spec.changes.at(-1)?.before || a).end}` : 'Source';
    if (first?.before && first?.after && first.before.path !== first.after.path) location = `${basename(b.path)} → ${basename(first.after.path)} · ${location}`;
    const modeButton = !spec.raw && !spec.test && spec.changes.length > 0
      ? `<button class="text-button" data-switch="${spec.id}">${spec.view === 'diff' ? 'Read definition' : 'View diff'}</button>` : '';
    const collapse = spec.expanded ? `<button class="text-button" data-collapse="${spec.id}">Collapse</button>` : '';
    const base = spec.baseURL ? link(spec.baseURL, 'Base', 'code-source-link') : '';
    const head = spec.headURL ? link(spec.headURL, 'Source', 'code-source-link') : '';
    return `<span class="code-location">${esc(location)}</span>${modeButton}${collapse}${base}${head}`;
  }
  function paintBlock(spec) {
    const element = document.getElementById(spec.id);
    if (!element) return;
    spec.materialized = true;
    const full = allRows(spec), shown = previewRows(spec);
    const total = countLines(full), visible = countLines(shown);
    const isDiff = spec.view === 'diff';
    let load = '';
    if (!spec.expanded && visible < total) {
      const label = spec.focus && !isDiff ? 'Read full definition' : isDiff ? 'Load full diff' : 'Load full code';
      load = `<button class="load-bar" data-expand="${spec.id}"><span aria-hidden="true">↓</span> ${label} <span class="load-count">· ${total - visible} more lines${total > CHUNK ? ' · loaded in chunks' : ''}</span></button>`;
    } else if (spec.expanded && spec.loaded < full.length) {
      load = `<button class="load-bar" data-expand="${spec.id}">Load next ${Math.min(CHUNK, full.length - spec.loaded)} rows <span class="load-count">· ${total - visible} lines remaining</span></button>`;
    }
    $('.code-body', element).innerHTML = `<div class="code-region" tabindex="0" role="region" aria-label="${esc(basename(spec.primaryPath))} ${isDiff ? 'diff' : 'code'}"><div class="code-lines">${shown.length ? rowsHTML(shown, isDiff) : '<div class="code-note">No textual edits between these paired definitions. Module paths and bindings still matter.</div>'}</div></div>${load}`;
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
      else if (c.kind === 'observed_head') text += ' Only the head-side definition is supplied here; a missing counterpart is not proof of new behavior.';
      return {text, change_ids: [c.id]};
    });
  }
  function passageHTML(p) {
    // Lead with the paired edits; a focused definition remains available in the block toggle.
    const changeIds = p.change_ids;
    const code = changeIds.length > 1 && changeIds.every(id => C.get(id)?.kind === 'moved')
      ? blockHTML(changeIds, {view: 'diff', label: p.label, focus: p.focus})
      : changeIds.map(id => blockHTML([id], {view: 'diff', focus: p.focus, label: p.label})).join('');
    return `<div class="passage"><div class="prose">${prose(p.text)}</div>${code}</div>`;
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
  function buildHeader() {
    const m = R.meta, doc = R.document || {};
    const title = m.title || 'Read the change, one idea at a time';
    document.title = `Diffstory · ${m.number ? '#' + m.number : plainInline(title)} · Reader`;
    const lead = doc.lead || 'Read the change in one continuous thread: the idea, its code, then what follows from it. Source stays close to the explanation.';
    const generationNotice = R.generation
      ? `<p class="generation-notice" role="status"><strong>Model-generated narration · unverified.</strong> Provider: ${esc(R.generation.provider)} · model: ${esc(R.generation.model)}. Check every claim against the linked source.</p>` : '';
    $('#documentHeader').innerHTML = `<div class="document-kicker">${link(m.url, m.repository || 'Local repository') || esc(m.repository || 'Local repository')}<span class="kicker-dot">/</span><span>${m.number ? 'PR ' + m.number : m.input === 'synthetic example' ? 'SYNTHETIC EXAMPLE' : 'COMMITTED COMPARISON'}</span></div><h1>${inline(title)}</h1><p class="lead">${inline(lead)}</p>${generationNotice}<div class="document-meta"><span>${R.groups.length} sections</span><span class="sep">·</span><span>${R.stats.supplied_paths} source paths</span><span class="sep">·</span><span class="revisions">${esc((m.base_sha || 'base').slice(0, 7))} → ${esc((m.head_sha || 'head').slice(0, 7))}</span></div><p class="document-scope">${m.input === 'synthetic example' ? 'Original synthetic example. No private repository source.' : m.scope === 'selected excerpts' ? 'Selected source excerpts, not the complete PR.' : 'Committed, changed-file snapshot.'} <a href="#evidence">Evidence & scope</a></p><div class="header-rule"></div>`;
  }
  function buildFooter() {
    const closing = R.document?.closing || 'The definitions, their callers, and the available tests are now in one reading path. Structural matches and test references guide review; they do not substitute for execution evidence.';
    $('#documentFooter').innerHTML = `<div class="closing"><span class="closing-label">The shape of the change</span>${inline(closing)}</div><details id="source-hunks" class="appendix" data-appendix="raw"><summary>Original source hunks</summary><div class="appendix-content"></div></details><details id="evidence" class="appendix" data-appendix="evidence"><summary>Evidence, revisions & scope</summary><div class="appendix-content"></div></details><div class="bottom-line"><span>Diffstory 0.3 · by Epistemic AI · stored locally</span><a href="#top">Back to top ↑</a></div>`;
  }
  function contents() {
    const q = $('#search').value.trim().toLowerCase();
    $('#sectionCount').textContent = `${R.groups.length} sections`;
    const filtered = R.groups.filter(g => [g.title, g.path, g.theme, g.narrative.intent, ...g.change_ids.map(id => {
      const c = C.get(id); return [c.before?.name, c.after?.name, c.before?.path, c.after?.path].join(' ');
    })].join(' ').toLowerCase().includes(q));
    $('#contentsList').innerHTML = filtered.map(g => `<a class="contents-item" href="#${g.id}" data-section="${g.id}" ${activeSection === g.id ? 'aria-current="location"' : ''}><span class="contents-number">${String(g.number).padStart(2, '0')}</span><span>${inline(g.title)}</span></a>`).join('') || '<p class="empty">No matching section or symbol.</p>';
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
      body.innerHTML = `<p>${R.meta.scope === 'selected excerpts' ? 'These are diffs of the supplied excerpts, not the full GitHub patch.' : 'Hunks computed from the supplied before/after file snapshots.'} Open a file to load its hunks. Nothing below changes the reading order above.</p>${[...files.entries()].map(([path, fs], i) => `<details class="raw-file" data-raw="${i}"><summary>${esc(path)} <span class="region-label">· ${fs.length} region${fs.length === 1 ? '' : 's'}</span></summary><div class="raw-body"></div></details>`).join('')}`;
    } else {
      const m = R.meta;
      body.innerHTML = `<dl><dt>Repository</dt><dd>${link(m.url, m.repository || 'Local Git') || esc(m.repository || 'Local Git')}</dd><dt>Base revision</dt><dd class="mono">${esc(m.base_sha)}</dd><dt>Head revision</dt><dd class="mono">${esc(m.head_sha)}</dd><dt>Scope</dt><dd>${esc(m.scope)}</dd><dt>Captured</dt><dd>${esc(m.captured_at || 'Not recorded')}</dd><dt>Supplied source</dt><dd>${R.stats.supplied_paths} paths · ${R.stats.units} analysis units</dd><dt>Test runs</dt><dd>0 runs verified by this analyzer</dd></dl><h3>What the matches establish</h3><p>An identical AST is structural evidence, not a proof that a move preserves behavior. Changed global bindings, import paths, initialization order and external consumers still need review. An edited pair is not automatically a confirmed behavioral change.</p><h3>Provenance</h3><p>${esc(m.validation || 'No execution evidence was attached.')}</p><p>Narrative paragraphs are authored interpretations linked to immutable source ranges. They cannot override the compiler’s source, structural classifications or test status.</p>${R.warnings.length ? `<h3>Limits of this input</h3><ul>${R.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>` : ''}<details><summary>Compiler methods</summary><dl>${Object.entries(R.method).map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl></details>${m.description ? `<details><summary>PR description, as captured</summary><pre>${esc(m.description)}</pre></details>` : ''}<h3>Local by design</h3><p>No external fonts, libraries, analytics or network requests are needed. The HTML embeds the source snapshot; keep it private for a private repository. Diffs are rendered lazily from that embedded data, not fetched from a server. Clicking a source link opens GitHub.</p>`;
    }
    details.dataset.ready = 'true';
  }
  function ensureRaw(details) {
    if (details.dataset.ready) return;
    const paths = [...new Set(R.raw_files.map(f => f.path))];
    const files = R.raw_files.filter(f => f.path === paths[Number(details.dataset.raw)]);
    $('.raw-body', details).innerHTML = files.map(f => blockHTML([], {raw: f, view: 'diff'})).join('');
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

  function download(name, value) {
    const blob = new Blob([JSON.stringify(value, null, 2)], {type: 'application/json'});
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
    if (button.id === 'downloadReport') { download('diffstory-report.json', R); closePopovers(); return; }
  });
  $('#search').addEventListener('input', contents);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') { closePopovers(true); return; }
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName) || event.target.isContentEditable || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === '/') { event.preventDefault(); openPopover('contents', true); }
    // Deliberately leave arrow keys, Page Down, Space and browser Find alone.
  });

  buildHeader();
  $('#story').innerHTML = R.groups.length ? R.groups.map(sectionHTML).join('') : '<div class="prose"><p>No changed units in this committed comparison.</p></div>';
  buildFooter(); contents(); observeBlocks(); updateScroll();
  if (location.hash) requestAnimationFrame(() => navigateTo(location.hash, false));
})();
