'use strict';
(() => {
  // The report is inert data, never repository code to execute. All prose/source is escaped.
  const SHARED_TABLES = [
    'results', 'plans', 'evidence_refs', 'source_targets', 'resolved_citations',
    'urls', 'runs', 'invocations', 'requests', 'producers', 'receipts', 'readers',
    'inherited_provenance'
  ];
  const testMetricsEnabled = window.__DIFFSTORY_TEST_METRICS__ === true;
  const readerMetrics = {
    embeddedJSONBytes: null,
    jsonParseMs: null,
    indexBuildMs: null,
    sectionProjectionRenderMs: 0,
    sectionProjectionCount: 0,
    contextProjectionRenderMs: 0,
    contextProjectionCount: 0,
    citationProjectionRenderMs: 0,
    citationProjectionCount: 0,
    sourceDecodeRenderMs: 0,
    sourceDecodeRenderCount: 0,
    resultCount: 0,
    referenceCount: 0,
    readerResultCount: 0,
    readerReferenceCount: 0,
    storageResultRecordCount: null,
    storageReferenceCount: null,
    maxOwnedSectionProjections: 0,
    heapAfterParseBytes: null,
    heapAfterIndexBytes: null,
    heapAfterRenderBytes: null
  };
  const metricNow = () => testMetricsEnabled ? performance.now() : 0;
  const heapBytes = () => {
    const bytes = performance.memory?.usedJSHeapSize;
    return Number.isFinite(bytes) ? Math.round(bytes) : null;
  };
  function publishReaderMetrics() {
    if (!testMetricsEnabled) return;
    window.__DIFFSTORY_READER_METRICS__ = {
      ...readerMetrics,
      heapCurrentBytes: heapBytes()
    };
  }
  const reportDataElement = document.getElementById('report-data');
  let embeddedJSON = reportDataElement.textContent;
  if (testMetricsEnabled) {
    readerMetrics.embeddedJSONBytes = new TextEncoder().encode(embeddedJSON).byteLength;
  }
  const parseStarted = metricNow();
  let parsedEnvelope = JSON.parse(embeddedJSON);
  if (testMetricsEnabled) {
    readerMetrics.jsonParseMs = Number((metricNow() - parseStarted).toFixed(3));
    readerMetrics.heapAfterParseBytes = heapBytes();
  }
  const indexBuildStarted = metricNow();
  const sharedMode = parsedEnvelope?.storage_version === 'diffstory.report-storage.v1';

  function indexSharedEnvelope(envelope) {
    if (
      envelope.artifact_kind !== 'report'
      || !envelope.root || typeof envelope.root !== 'object'
      || Array.isArray(envelope.root)
      || envelope.root.shape !== 'report'
      || !envelope.root.base_report || typeof envelope.root.base_report !== 'object'
      || Array.isArray(envelope.root.base_report)
      || !envelope.root.generation || typeof envelope.root.generation !== 'object'
      || Array.isArray(envelope.root.generation)
      || !envelope.tables || typeof envelope.tables !== 'object'
      || Array.isArray(envelope.tables)
    ) throw new Error('Invalid shared report envelope');
    const tableNames = Object.keys(envelope.tables).sort();
    if (tableNames.join('\u0000') !== [...SHARED_TABLES].sort().join('\u0000')) {
      throw new Error('Invalid shared report table set');
    }
    const indexes = Object.create(null);
    for (const table of SHARED_TABLES) {
      const rows = envelope.tables[table];
      if (!Array.isArray(rows)) throw new Error(`Invalid shared ${table} table`);
      const index = new Map();
      for (const entry of rows) {
        if (
          !entry || typeof entry !== 'object' || Array.isArray(entry)
          || typeof entry.id !== 'string' || typeof entry.kind !== 'string'
          || !entry.record || typeof entry.record !== 'object' || Array.isArray(entry.record)
          || index.has(entry.id)
        ) throw new Error(`Invalid or duplicate shared ${table} row`);
        index.set(entry.id, entry);
      }
      indexes[table] = index;
    }
    return {root: envelope.root, indexes};
  }

  if (parsedEnvelope?.storage_version && !sharedMode) {
    throw new Error('Unsupported embedded report storage version');
  }
  const sharedGraph = sharedMode ? indexSharedEnvelope(parsedEnvelope) : null;
  const R = sharedMode ? sharedGraph.root.base_report : parsedEnvelope;
  const generation = sharedMode ? sharedGraph.root.generation : R.generation;
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

  // The backend seals this presentation against source and generation. The
  // shared reader below resolves only the storage schema's declared pointers.
  // Compiler metadata is never walked for objects that happen to contain $ref.
  const refKey = ref => {
    if (ref.kind === 'statement') return JSON.stringify([
      ref.kind, ref.producer_item_key, ref.result_digest,
      ref.statement_path.field, ref.statement_path.index, ref.statement_digest
    ]);
    if (ref.kind === 'source_slice') return JSON.stringify([
      ref.kind, refKey(ref.source), ref.byte_start, ref.byte_end, ref.content_digest
    ]);
    return JSON.stringify([ref.span_id, ref.unit_key, ref.group_key, ref.side,
      ref.path, ref.byte_start, ref.byte_end, ref.line_start, ref.line_end, ref.content_digest]);
  };
  const claimKey = ref => JSON.stringify([ref.unit_key, ref.observation_id]);

  function copyJSON(value) {
    if (Array.isArray(value)) return value.map(copyJSON);
    if (!value || typeof value !== 'object') return value;
    const output = {};
    for (const key of Object.keys(value)) output[key] = copyJSON(value[key]);
    return output;
  }

  function requiredArray(record, field, label = field) {
    if (!Array.isArray(record?.[field])) throw new Error(`Invalid shared ${label} array`);
    return record[field];
  }

  function optionalArray(record, field, label = field) {
    if (!Object.prototype.hasOwnProperty.call(record || {}, field)) return [];
    return requiredArray(record, field, label);
  }

  function setStatementDefaults(statement) {
    for (const field of ['claims', 'citations', 'basis', 'uncertainty']) {
      if (!Object.prototype.hasOwnProperty.call(statement, field)) statement[field] = [];
      else if (!Array.isArray(statement[field])) throw new Error(`Invalid shared statement ${field}`);
    }
    for (const field of ['effect', 'author_context_digest']) {
      if (!Object.prototype.hasOwnProperty.call(statement, field)) statement[field] = null;
    }
    return statement;
  }

  function countStatementReferences(statement) {
    return optionalArray(statement, 'citations').length
      + optionalArray(statement, 'basis').length;
  }

  function countRecordReferences(table, entry) {
    const record = entry.record;
    let count = 0;
    const countStatements = (field, definitions = false) => {
      for (const value of optionalArray(record, field)) {
        count += countStatementReferences(definitions ? value.definition : value);
      }
    };
    if (table === 'results') {
      if (entry.kind === 'detail') {
        for (const family of ['explanations', 'observations', 'questions']) {
          for (const item of optionalArray(record, family)) count += optionalArray(item, 'citations').length;
        }
      } else if (entry.kind === 'planning_card') {
        countStatements('abstraction');
        countStatements('terminology', true);
      } else if (entry.kind === 'section') {
        if (record.title) count += countStatementReferences(record.title);
        for (const field of ['orientation', 'synthesis', 'transition']) countStatements(field);
        countStatements('terminology', true);
      } else if (entry.kind === 'document') {
        for (const field of ['preamble', 'lead', 'closing']) countStatements(field);
        countStatements('terminology', true);
      }
    } else if (table === 'evidence_refs' && entry.kind === 'source_slice_ref') {
      count += 1;
    } else if (table === 'resolved_citations') {
      count += 3;
    } else if (table === 'producers') {
      count += 2 + optionalArray(record, 'requests').length;
    } else if (table === 'readers') {
      count += 1;
      const sectionField = record.shape === 'presentation' ? 'sections' : 'steps';
      for (const section of optionalArray(record, sectionField)) {
        count += 1 + optionalArray(section, 'details').length;
      }
      count += optionalArray(record, 'context_details').length;
      count += optionalArray(record, 'citations').length;
      count += optionalArray(record, 'support_provenance').length;
    } else if (table === 'inherited_provenance') {
      count += 3;
    }
    return count;
  }

  function countStorageReferences(graph) {
    let count = 1; // root.reader
    const generation = graph.root.generation;
    for (const field of ['run', 'plan', 'document']) if (generation[field]) count += 1;
    for (const field of ['details', 'planning', 'sections', 'producers', 'receipts']) {
      count += optionalArray(generation, field).length;
    }
    for (const table of SHARED_TABLES) {
      for (const entry of graph.indexes[table].values()) count += countRecordReferences(table, entry);
    }
    return count;
  }

  function createSharedReader(graph) {
    function entryFor(table, pointer, allowedKinds, label) {
      const kinds = Array.isArray(allowedKinds) ? allowedKinds : [allowedKinds];
      if (
        !pointer || typeof pointer !== 'object' || Array.isArray(pointer)
        || Object.keys(pointer).length !== 2
        || typeof pointer.$ref !== 'string' || typeof pointer.kind !== 'string'
        || !kinds.includes(pointer.kind)
      ) throw new Error(`Invalid shared ${label} selector`);
      const entry = graph.indexes[table]?.get(pointer.$ref);
      if (!entry || entry.kind !== pointer.kind || !kinds.includes(entry.kind)) {
        throw new Error(`Missing or wrong-kind shared ${label} selector`);
      }
      return entry;
    }

    function evidence(pointer, allowedKinds) {
      const entry = entryFor('evidence_refs', pointer, allowedKinds, 'evidence reference');
      if (entry.kind === 'source_slice_ref') {
        const value = copyJSON(entry.record);
        value.source = evidence(entry.record.source, 'span_ref');
        return value;
      }
      return entry.record;
    }

    function prepareStatement(value) {
      setStatementDefaults(value);
      return value;
    }

    function materializeStatement(raw) {
      return prepareStatement(copyJSON(raw));
    }

    function materializeResult(pointer, expectedKind) {
      const entry = entryFor('results', pointer, expectedKind, 'result');
      const value = copyJSON(entry.record);
      if (entry.kind === 'detail') {
        for (const family of ['explanations', 'observations', 'questions']) {
          for (const item of requiredArray(value, family, `detail ${family}`)) {
            requiredArray(item, 'citations', 'detail citations');
          }
        }
        return value;
      }

      if (!Object.prototype.hasOwnProperty.call(value, 'terminology')) value.terminology = [];
      else if (!Array.isArray(value.terminology)) throw new Error('Invalid shared result terminology');
      const visitStatements = field => {
        for (const statement of requiredArray(value, field, `result ${field}`)) {
          prepareStatement(statement);
        }
      };
      if (entry.kind === 'planning_card') {
        visitStatements('abstraction');
      } else if (entry.kind === 'section') {
        value.title = prepareStatement(value.title);
        for (const field of ['orientation', 'synthesis', 'transition']) visitStatements(field);
      } else if (entry.kind === 'document') {
        for (const field of ['preamble', 'lead', 'closing']) visitStatements(field);
      }
      for (const term of value.terminology) term.definition = prepareStatement(term.definition);
      return value;
    }

    const readerEntry = entryFor('readers', graph.root.reader, 'reader', 'reader root');
    const readerRecord = readerEntry.record;
    if (readerRecord.shape !== 'presentation') throw new Error('Shared report has no presentation reader');
    const sections = requiredArray(readerRecord, 'sections', 'reader sections');
    const contextDetails = requiredArray(readerRecord, 'context_details', 'reader context details');
    const citationPointers = requiredArray(readerRecord, 'citations', 'reader citations');
    const provenancePointers = requiredArray(readerRecord, 'support_provenance', 'reader provenance');

    function rawInheritedStatement(provenance) {
      const reference = provenance.reference;
      const resultEntry = entryFor(
        'results', provenance.record.result,
        ['planning_card', 'section', 'document'], 'inherited result'
      );
      const path = reference.statement_path;
      const field = path?.field;
      const index = path?.index === undefined ? 0 : path.index;
      if (typeof index !== 'number' || !Number.isInteger(index) || index < 0) {
        throw new Error('Invalid shared inherited statement index');
      }
      const allowed = {
        planning_card: ['abstraction', 'terminology'],
        section: ['title', 'orientation', 'synthesis', 'transition', 'terminology'],
        document: ['preamble', 'lead', 'closing', 'terminology']
      }[resultEntry.kind];
      if (!path || !allowed.includes(field)) throw new Error('Invalid shared inherited statement path');
      if (field === 'title') {
        if (resultEntry.kind !== 'section' || index !== 0) throw new Error('Invalid shared title selector');
        return resultEntry.record.title;
      }
      if (field === 'terminology') {
        const terms = optionalArray(resultEntry.record, 'terminology');
        if (!terms[index] || !terms[index].definition) throw new Error('Missing shared inherited term definition');
        return terms[index].definition;
      }
      const statements = requiredArray(resultEntry.record, field, 'inherited statement family');
      if (!statements[index]) throw new Error('Missing shared inherited statement');
      return statements[index];
    }

    function materializeInherited(provenance) {
      const value = copyJSON(provenance.record);
      value.reference = provenance.reference;
      value.statement = materializeStatement(rawInheritedStatement(provenance));
      const receipt = entryFor('receipts', provenance.record.receipt, 'receipt', 'inherited receipt');
      value.receipt = receipt.record;
      delete value.result;
      return value;
    }

    if (testMetricsEnabled) {
      const readerResultCount = 1 + sections.length + contextDetails.length
        + sections.reduce((total, section) => total + requiredArray(section, 'details', 'reader section details').length, 0);
      const readerReferenceCount = 1 + 1 + sections.reduce(
        (total, section) => total + 1 + requiredArray(section, 'details', 'reader section details').length,
        contextDetails.length + citationPointers.length + provenancePointers.length
      );
      readerMetrics.resultCount = readerResultCount;
      readerMetrics.referenceCount = readerReferenceCount;
      readerMetrics.readerResultCount = readerResultCount;
      readerMetrics.readerReferenceCount = readerReferenceCount;
      readerMetrics.storageResultRecordCount = graph.indexes.results.size;
      readerMetrics.storageReferenceCount = countStorageReferences(graph);
    }

    return {
      shared: true,
      record: readerRecord,
      document: () => materializeResult(readerRecord.document, 'document'),
      sectionRows: () => sections,
      sectionProjection: section => ({
        group_id: section.group_id,
        group_key: section.group_key,
        number: section.number,
        next_id: section.next_id,
        prose: materializeResult(section.prose, 'section'),
        details: requiredArray(section, 'details', 'reader section details')
          .map(pointer => materializeResult(pointer, 'detail'))
      }),
      contextRows: () => contextDetails,
      materializeDetail: pointer => materializeResult(pointer, 'detail'),
      *detailRecords() {
        for (const section of sections) {
          for (const pointer of requiredArray(section, 'details', 'reader section details')) {
            yield entryFor('results', pointer, 'detail', 'detail').record;
          }
        }
        for (const pointer of contextDetails) yield entryFor('results', pointer, 'detail', 'context detail').record;
      },
      *citationValues() {
        for (const pointer of citationPointers) {
          const entry = entryFor('resolved_citations', pointer, 'resolved_citation', 'citation');
          const value = copyJSON(entry.record);
          value.reference = evidence(entry.record.reference, ['span_ref', 'source_slice_ref']);
          const target = entryFor('source_targets', entry.record.source_target, 'source_target', 'citation source target');
          Object.assign(value, copyJSON(target.record));
          const url = entryFor('urls', entry.record.url, 'url', 'citation URL');
          value.url = url.record.value;
          yield value;
        }
      },
      *provenanceEntries() {
        for (const pointer of provenancePointers) {
          const entry = entryFor('inherited_provenance', pointer, 'inherited_provenance', 'provenance');
          yield {
            pointer,
            record: entry.record,
            reference: evidence(entry.record.reference, 'statement_ref')
          };
        }
      },
      evidence,
      rawInheritedStatement,
      materializeInherited,
      sectionCount: sections.length,
      contextCount: contextDetails.length,
      citationCount: citationPointers.length
    };
  }

  function createExpandedReader(value) {
    return {
      shared: false,
      document: () => value.document,
      sectionRows: () => value.sections,
      sectionProjection: section => section,
      contextRows: () => value.context_details,
      materializeDetail: detail => detail,
      *detailRecords() {
        for (const section of value.sections) yield* section.details;
        yield* value.context_details;
      },
      *citationValues() { yield* value.citations; },
      *provenanceEntries() {
        for (const record of value.support_provenance) yield {record, reference: record.reference};
      },
      evidence: ref => ref,
      rawInheritedStatement: provenance => provenance.record.statement,
      materializeInherited: provenance => provenance.record,
      sectionCount: value.sections.length,
      contextCount: value.context_details.length,
      citationCount: value.citations.length
    };
  }

  // Legacy annotations never enter the typed branch by a label or matching field name.
  const V2 = !sharedMode && generation?.version === 'diffstory.generation.v2'
    && R.narrative_v2?.version === 'diffstory.presentation.v2' ? R.narrative_v2 : null;
  const Reader = V2 ? createExpandedReader(V2) : sharedGraph ? createSharedReader(sharedGraph) : null;
  const typedDocument = Reader ? Reader.document() : null;
  parsedEnvelope = null;
  embeddedJSON = null;
  const citations = new Map(), observations = new Map(), inherited = new Map();
  const citationCatalog = [];
  const unitSources = new Map(), inheritedEndpoints = new Map(), fragmentData = new Map();
  const fragments = new Map((Reader ? R.source_evidence.fragments : []).map(f => [f.key, f]));
  function citationForLogical(logical) {
    const citation = citations.get(refKey(logical));
    if (!citation) throw new Error('Missing shared resolved citation');
    return citation;
  }
  function directEndpoints(statement) {
    const found = new Map();
    const addLogical = logical => {
      const citation = citationForLogical(logical);
      found.set(citation.key, citation);
    };
    const add = ref => addLogical(Reader.evidence(ref, ['span_ref', 'source_slice_ref']));
    for (const ref of optionalArray(statement, 'citations')) add(ref);
    for (const ref of optionalArray(statement, 'claims')) {
      const observation = observations.get(claimKey(ref));
      if (!observation) throw new Error('Missing shared claim observation');
      for (const citation of optionalArray(observation, 'citations')) add(citation);
    }
    for (const ref of optionalArray(statement, 'basis')) {
      const logical = Reader.evidence(ref, ['source_slice_ref', 'statement_ref']);
      if (logical.kind === 'source_slice') addLogical(logical);
    }
    return found;
  }
  if (Reader) {
    for (const citation of Reader.citationValues()) {
      citationCatalog.push(citation);
      citations.set(refKey(citation.reference), citation);
      const unit = (citation.reference.source || citation.reference).unit_key;
      if (!unitSources.has(unit)) unitSources.set(unit, []);
      unitSources.get(unit).push(citation);
    }
    for (const detail of Reader.detailRecords()) {
      for (const observation of requiredArray(detail, 'observations', 'detail observations')) {
        observations.set(claimKey({unit_key: detail.unit_key, observation_id: observation.id}), observation);
      }
    }
    for (const provenance of Reader.provenanceEntries()) inherited.set(refKey(provenance.reference), provenance);
    // One iterative DAG walk, shared by all displayed statements. Keep every branch.
    for (const root of inherited.keys()) {
      const stack = [[root, false]];
      while (stack.length) {
        const [key, ready] = stack.pop();
        if (inheritedEndpoints.has(key)) continue;
        const provenance = inherited.get(key);
        if (!provenance) throw new Error('Missing shared inherited statement');
        const statement = Reader.rawInheritedStatement(provenance);
        const children = optionalArray(statement, 'basis').map(ref => Reader.evidence(ref, ['source_slice_ref', 'statement_ref']))
          .filter(ref => ref.kind === 'statement').map(refKey);
        if (!ready) {
          stack.push([key, true]);
          for (const child of children) if (!inheritedEndpoints.has(child)) stack.push([child, false]);
        } else {
          const endpoints = directEndpoints(statement);
          for (const child of children) {
            const endpointsById = inheritedEndpoints.get(child);
            if (!endpointsById) throw new Error('Missing shared inherited endpoint branch');
            for (const [id, citation] of endpointsById) endpoints.set(id, citation);
          }
          inheritedEndpoints.set(key, endpoints);
        }
      }
    }
  }
  readerMetrics.indexBuildMs = testMetricsEnabled ? Number((metricNow() - indexBuildStarted).toFixed(3)) : null;
  if (testMetricsEnabled) {
    if (Reader && !Reader.shared) {
      const sectionRows = Reader.sectionRows();
      const detailCount = sectionRows.reduce((total, section) => total + section.details.length, 0)
        + Reader.contextCount;
      readerMetrics.resultCount = 1 + sectionRows.length + detailCount;
      readerMetrics.referenceCount = 2 + sectionRows.reduce(
        (total, section) => total + 1 + section.details.length,
        Reader.contextCount + Reader.citationCount + inherited.size
      );
      readerMetrics.readerResultCount = readerMetrics.resultCount;
      readerMetrics.readerReferenceCount = readerMetrics.referenceCount;
    }
    readerMetrics.heapAfterIndexBytes = heapBytes();
  }
  publishReaderMetrics();
  const sideLabel = side => side === 'base' ? 'Base' : 'Head';
  function citationLabel(c) {
    return `${sideLabel(c.side)} · ${c.path} · L${c.line_start}–${c.line_end} · bytes ${c.byte_start}–${c.byte_end} (end excluded)${c.byte_start === c.byte_end ? ' · Empty supplied fragment' : ''}`;
  }
  function sourceLinks(values, inheritedSource = false) {
    if (!values.length) return '';
    return `<ul class="v2-source-links">${values.map(c => `<li><a href="#source-${c.key}" data-citation="${c.key}" data-consumption="${c.consumption}">${inheritedSource ? 'Inherited · ' : ''}${c.consumption === 'source_slice' ? 'Exact source slice' : 'Direct source'} · ${esc(citationLabel(c))}</a></li>`).join('')}</ul>`;
  }
  const refsHTML = refs => sourceLinks(refs.map(ref => {
    const logical = Reader.evidence(ref, ['span_ref', 'source_slice_ref']);
    return citationForLogical(logical);
  }));
  const kindLabels = {
    implementation_claim: 'Source-linked interpretation · unverified',
    author_motivation: 'Author-reported motivation · not implementation evidence',
    editorial: 'Reading guidance · not implementation evidence',
    uncertainty: 'Uncertainty · not implementation evidence'
  };
  const typedText = (text, field) => `<div class="v2-text" data-field="${field}">${inline(text)}</div>`;
  function statementBody(s) {
    const effect = s.effect ?? null;
    return `${typedText(s.text, 'text')}${effect !== null ? `<div class="v2-field"><span class="v2-label">Supported effect · unverified</span>${typedText(effect, 'effect')}</div>` : ''}${(s.uncertainty || []).map(text => `<div class="v2-field"><span class="v2-label">Uncertainty</span>${typedText(text, 'uncertainty')}</div>`).join('')}`;
  }
  function statementSupport(s) {
    const claims = (s.claims || []).map(ref => {
      const o = observations.get(claimKey(ref));
      if (!o) throw new Error('Missing shared claim observation');
      return `<div class="v2-claim"><span class="v2-label">Supporting observation</span>${typedText(o.text, 'observation')}${refsHTML(o.citations)}</div>`;
    }).join('');
    const basis = (s.basis || []).map(ref => {
      const logical = Reader.evidence(ref, ['source_slice_ref', 'statement_ref']);
      if (logical.kind === 'source_slice') {
        return sourceLinks([citationForLogical(logical)]);
      }
      const key = refKey(logical), entry = inherited.get(key);
      if (!entry) throw new Error('Missing shared inherited statement');
      const provenance = Reader.materializeInherited(entry);
      const endpoints = inheritedEndpoints.get(key);
      if (!endpoints) throw new Error('Missing shared inherited endpoint branch');
      return `<div class="v2-inherited" data-inherited="${provenance.key}"><span class="v2-label">Inherited statement · ${esc(kindLabels[provenance.statement.kind])}</span>${statementBody(provenance.statement)}${sourceLinks([...endpoints.values()], true)}${!endpoints.size ? '<p class="v2-label">No implementation source endpoints in this inherited statement.</p>' : ''}</div>`;
    }).join('');
    return `${refsHTML(s.citations || [])}${claims}${basis}`;
  }
  function statementHTML(s) {
    return `<div class="v2-statement" data-kind="${s.kind}"><span class="v2-label">${kindLabels[s.kind]}</span>${statementBody(s)}${statementSupport(s)}</div>`;
  }
  const statementsHTML = statements => statements.map(statementHTML).join('');
  function termsHTML(terms) {
    const values = terms || [];
    return values.length ? `<div class="v2-terms"><h3>Terms used here</h3>${values.map(t => `<div class="v2-term"><h4>${inline(t.term)}</h4>${statementHTML(t.definition)}</div>`).join('')}</div>` : '';
  }
  function detailHTML(d) {
    const explanations = d.explanations.map(e => `<div class="v2-explanation">${['before', 'after', 'effect', 'uncertainty'].map(field => `<div class="v2-field"><span class="v2-label">${{before: 'Before', after: 'After', effect: 'Effect · interpretation', uncertainty: 'Uncertainty'}[field]}</span>${typedText(e[field], field)}</div>`).join('')}${refsHTML(e.citations)}${e.limitations.map(l => `<div class="v2-limitation"><span class="v2-label">Evidence limitation · ${esc(sideLabel(l.side))} · ${esc(l.path)}</span>${typedText(l.reason.replace(/_/g, ' '), 'reason')}<span class="v2-label">${esc(l.code.replace(/_/g, ' '))} · This limit does not establish absence or behavior.</span></div>`).join('')}</div>`).join('');
    const observed = d.observations.map(o => `<div class="v2-observation" id="observation-${d.unit_key}-${o.id}"><span class="v2-label">Source-linked observation · unverified</span>${typedText(o.text, 'observation')}${refsHTML(o.citations)}</div>`).join('');
    const questions = d.questions.map(q => `<div class="v2-question"><span class="v2-label">Open question · not implementation evidence</span>${typedText(q.text, 'question')}${refsHTML(q.citations)}</div>`).join('');
    const empty = !d.explanations.length && !d.observations.length && !d.questions.length;
    return `<article class="v2-detail prose" data-detail="${d.unit_key}">${empty ? `<p class="v2-context-ack">Supplied context was retained; no change explanation was required for this evidence.</p>${sourceLinks(unitSources.get(d.unit_key) || [])}` : explanations + observed + questions}</article>`;
  }
  function typedSectionHTML(section) {
    const g = G.get(section.group_id), p = section.prose;
    if (!g) throw new Error('Missing compiler group for shared reader section');
    const prerequisites = g.prerequisites.length ? `<p class="v2-label">Compiler prerequisites: ${g.prerequisites.map(id => `<a href="#${id}">${inline(G.get(id).title)}</a>`).join(', ')}</p>` : '';
    return `<section class="story-section" id="${g.id}" data-story-section="${g.id}" aria-labelledby="heading-${g.id}"><header class="section-heading"><span class="section-number">${String(section.number).padStart(2, '0')}</span><h2 id="heading-${g.id}">${inline(p.title.text)}<a class="section-anchor" href="#${g.id}" aria-label="Link to ${esc(plainInline(p.title.text))}">#</a></h2></header><div class="prose v2-title-support"><span class="v2-label">Heading · ${kindLabels[p.title.kind]}</span>${(p.title.effect ?? null) !== null ? typedText(p.title.effect, 'effect') : ''}${(p.title.uncertainty || []).map(t => typedText(t, 'uncertainty')).join('')}${statementSupport(p.title)}${prerequisites}</div><div class="section-context">${statementsHTML(p.orientation)}</div>${section.details.map(detailHTML).join('')}<div class="prose">${statementsHTML(p.synthesis)}${termsHTML(p.terminology)}</div>${g.change_ids.map(id => blockHTML([id], {view: 'diff'})).join('')}<div class="section-end">${statementsHTML(p.transition)}</div></section>`;
  }
  function measuredSectionHTML(section) {
    const started = metricNow();
    const html = typedSectionHTML(section);
    if (testMetricsEnabled) {
      readerMetrics.sectionProjectionRenderMs += metricNow() - started;
      readerMetrics.sectionProjectionCount += 1;
    }
    return html;
  }
  function sourceBlockHTML(c) {
    const id = `block-${++blockCounter}`;
    blocks.set(id, {id, citation: c, expanded: false, loaded: 0, materialized: false});
    return `<details class="v2-source" id="source-${c.key}"><summary>${c.consumption === 'source_slice' ? 'Exact source slice' : 'Direct source'} · ${esc(citationLabel(c))}</summary><figure class="code-block exact-code" id="${id}" data-block="${id}" data-citation="${c.key}" data-fragment="${esc(c.fragment_key)}" data-side="${c.side}" data-byte-start="${c.byte_start}" data-byte-end="${c.byte_end}"><figcaption class="code-header">${esc(sideLabel(c.side))} · ${esc(c.path)}</figcaption><div class="code-body"><div class="lazy-shell"><button data-load-preview="${id}">Load cited source ↓</button></div></div><div class="code-footer">${link(c.url, `${sideLabel(c.side)} source`, 'code-source-link')}</div></figure></details>`;
  }
  function typedStoryHTML() {
    const sections = Reader.sectionRows().map(section => measuredSectionHTML(Reader.sectionProjection(section))).join('');
    const context = Reader.contextCount
      ? `<section class="v2-context"><div class="prose"><h2>Retained source context</h2></div>${[...Reader.contextRows()].map(detail => detailHTML(Reader.materializeDetail(detail))).join('')}</section>`
      : '';
    const citationsHTML = citationCatalog.map(sourceBlockHTML).join('');
    return `<section class="preamble"><h2>Before the code</h2><div class="preamble-body">${statementsHTML(typedDocument.preamble)}${termsHTML(typedDocument.terminology)}</div></section>${sections}${context}<section class="v2-catalog"><div class="prose"><h2>Source references</h2><p>Each excerpt uses the named revision and exact byte range. Open it to inspect the source, then load the full supplied fragment if needed.</p></div>${citationsHTML}</section>`;
  }
  function appendSharedTypedStory() {
    const story = $('#story');
    story.insertAdjacentHTML('beforeend', `<section class="preamble"><h2>Before the code</h2><div class="preamble-body">${statementsHTML(typedDocument.preamble)}${termsHTML(typedDocument.terminology)}</div></section>`);
    for (const row of Reader.sectionRows()) {
      const started = metricNow();
      let projection = Reader.sectionProjection(row);
      const html = typedSectionHTML(projection);
      story.insertAdjacentHTML('beforeend', html);
      projection = null;
      if (testMetricsEnabled) {
        readerMetrics.sectionProjectionRenderMs += metricNow() - started;
        readerMetrics.sectionProjectionCount += 1;
        readerMetrics.maxOwnedSectionProjections = Math.max(readerMetrics.maxOwnedSectionProjections, 1);
      }
    }
    if (Reader.contextCount) {
      const context = document.createElement('section');
      context.className = 'v2-context';
      const heading = document.createElement('div');
      heading.className = 'prose';
      heading.innerHTML = '<h2>Retained source context</h2>';
      context.append(heading);
      for (const pointer of Reader.contextRows()) {
        const started = metricNow();
        let projection = Reader.materializeDetail(pointer);
        context.insertAdjacentHTML('beforeend', detailHTML(projection));
        projection = null;
        if (testMetricsEnabled) {
          readerMetrics.contextProjectionRenderMs += metricNow() - started;
          readerMetrics.contextProjectionCount += 1;
        }
      }
      story.append(context);
    }
    const catalog = document.createElement('section');
    catalog.className = 'v2-catalog';
    const introduction = document.createElement('div');
    introduction.className = 'prose';
    introduction.innerHTML = '<h2>Source references</h2><p>Each excerpt uses the named revision and exact byte range. Open it to inspect the source, then load the full supplied fragment if needed.</p>';
    catalog.append(introduction);
    for (const citation of citationCatalog) {
      const started = metricNow();
      catalog.insertAdjacentHTML('beforeend', sourceBlockHTML(citation));
      if (testMetricsEnabled) {
        readerMetrics.citationProjectionRenderMs += metricNow() - started;
        readerMetrics.citationProjectionCount += 1;
      }
    }
    story.append(catalog);
  }
  function sourceData(c) {
    if (!fragmentData.has(c.fragment_key)) {
      const fragment = fragments.get(c.fragment_key);
      const bytes = new TextEncoder().encode(fragment.text), starts = [0];
      // CRLF is one delimiter. LS and PS are ordinary source characters.
      for (let i = 0; i < bytes.length; i++) {
        if (bytes[i] === 13) { if (bytes[i + 1] === 10) i++; starts.push(i + 1); }
        else if (bytes[i] === 10) starts.push(i + 1);
      }
      if (starts.at(-1) === bytes.length) starts.pop();
      fragmentData.set(c.fragment_key, {fragment, bytes, starts});
    }
    return fragmentData.get(c.fragment_key);
  }
  function exactRows(c, full = false) {
    const {fragment, bytes, starts} = sourceData(c);
    const start = full ? 0 : c.byte_start, end = full ? bytes.length : c.byte_end;
    let lo = 0, hi = starts.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (starts[mid] <= start) lo = mid + 1; else hi = mid; }
    const rows = [], decoder = new TextDecoder('utf-8', {fatal: true, ignoreBOM: true});
    for (let i = Math.max(0, lo - 1); i < starts.length && starts[i] < end; i++) {
      rows.push({type: 'line', new: fragment.start_line + i,
        text: decoder.decode(bytes.subarray(Math.max(start, starts[i]), Math.min(end, starts[i + 1] ?? bytes.length)))});
    }
    return rows;
  }
  function paintSourceBlock(spec) {
    const started = metricNow();
    const element = document.getElementById(spec.id), c = spec.citation;
    const selected = exactRows(c, spec.expanded);
    const shown = spec.expanded ? selected.slice(0, spec.loaded) : selected.length <= PREVIEW_LIMIT ? selected
      : [...selected.slice(0, PREVIEW_HEAD), {type: 'gap', text: `… ${selected.length - PREVIEW_HEAD - PREVIEW_TAIL} cited lines folded`}, ...selected.slice(-PREVIEW_TAIL)];
    const empty = sourceData(c).bytes.length === 0;
    const load = empty ? '' : spec.expanded ? (spec.loaded < selected.length ? `<button class="load-bar" data-expand="${spec.id}">Load next ${Math.min(CHUNK, selected.length - spec.loaded)} source lines</button>` : '')
      : `<button class="load-bar" data-expand="${spec.id}">Read full supplied ${sideLabel(c.side).toLowerCase()} source · loaded in chunks</button>`;
    $('.code-body', element).innerHTML = `${empty ? '<p class="code-note v2-empty-source">Empty supplied fragment (0 bytes). This does not establish whether the whole file or its counterpart is empty.</p>' : ''}<div class="code-region" tabindex="0" role="region" aria-label="${esc(sideLabel(c.side) + ' ' + c.path)}"><div class="code-lines">${rowsHTML(shown, false)}</div></div>${load}`;
    $('.code-footer', element).innerHTML = `<span class="code-location">${esc(citationLabel(c))}</span>${spec.expanded ? `<button class="text-button" data-collapse="${spec.id}">Return to cited excerpt</button>` : ''}${link(c.url, `${sideLabel(c.side)} source`, 'code-source-link')}`;
    spec.materialized = true; element.dataset.loaded = 'true';
    element.dataset.sourceView = spec.expanded ? 'full' : 'excerpt';
    if (testMetricsEnabled) {
      readerMetrics.sourceDecodeRenderMs += metricNow() - started;
      readerMetrics.sourceDecodeRenderCount += 1;
      publishReaderMetrics();
    }
  }

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
  function allRows(spec) {
    if (spec.citation) return exactRows(spec.citation, true);
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
    const fileStatusChange = cs.find(change => ['added', 'removed', 'observed_head', 'observed_base'].includes(change.kind));
    let badge = options.raw ? ['Source hunk', ''] : fileStatusChange ? classification(fileStatusChange) : options.test ? ['Test source', ''] : cs.length > 1 ? [`${cs.length} moved declarations`, 'move'] : classification(first);
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
    const basis = cs.map(change => `<div class="code-note code-basis">${esc(change.basis)}</div>`).join('');
    return `${beforeAnchors}<figure class="code-block" id="${spec.id}" data-block="${spec.id}" data-changes="${cs.map(c => c.id).join(' ')}"><figcaption class="code-header"><span class="filename" title="${esc(primaryPath)}">${esc(basename(primaryPath))}${symbol}</span><span class="classification ${badge[1]}">${esc(badge[0])}</span></figcaption>${basis}<div class="code-body"><div class="lazy-shell" style="min-height:${Math.max(58, Math.min(430, estimated * 22))}px"><button data-load-preview="${spec.id}">Load ${spec.view === 'diff' ? 'diff' : 'code'} ↓</button></div></div><div class="code-footer"></div></figure>`;
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
    if (spec.citation) { paintSourceBlock(spec); return; }
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
  function buildHeader() {
    const m = R.meta, doc = R.document || {};
    const title = m.title || 'Read the change, one idea at a time';
    document.title = `Diffstory · ${m.number ? '#' + m.number : plainInline(title)} · Reader`;
    const lead = doc.lead || 'Read the change in one continuous thread: the idea, its code, then what follows from it. Source stays close to the explanation.';
    const provider = Reader ? generation.request_configuration.name : generation?.provider;
    const model = Reader ? generation.request_configuration.model : generation?.model;
    const generationNotice = generation
      ? `<p class="generation-notice" role="status"><strong>Model-generated narration · unverified.</strong> Provider: ${esc(provider)} · model: ${esc(model)}. Check every claim against the linked source.</p>` : '';
    $('#documentHeader').innerHTML = `<div class="document-kicker">${link(m.url, m.repository || 'Local repository') || esc(m.repository || 'Local repository')}<span class="kicker-dot">/</span><span>${m.number ? 'PR ' + m.number : m.input === 'synthetic example' ? 'SYNTHETIC EXAMPLE' : 'COMMITTED COMPARISON'}</span></div><h1>${inline(title)}</h1>${Reader ? `<div class="lead">${statementsHTML(typedDocument.lead)}</div>` : `<p class="lead">${inline(lead)}</p>`}${generationNotice}<div class="document-meta"><span>${R.groups.length} sections</span><span class="sep">·</span><span>${R.stats.supplied_paths} source paths</span><span class="sep">·</span><span class="revisions">${esc((m.base_sha || 'base').slice(0, 7))} → ${esc((m.head_sha || 'head').slice(0, 7))}</span></div><p class="document-scope">${m.input === 'synthetic example' ? 'Original synthetic example. No private repository source.' : m.scope === 'selected excerpts' ? 'Selected source excerpts, not the complete PR.' : 'Committed, changed-file snapshot.'} <a href="#evidence">Evidence & scope</a></p><div class="header-rule"></div>`;
  }
  function buildFooter() {
    const closing = R.document?.closing || 'The definitions, their callers, and the available tests are now in one reading path. Structural matches and test references guide review; they do not substitute for execution evidence.';
    $('#documentFooter').innerHTML = `<div class="closing"><span class="closing-label">The shape of the change</span>${Reader ? statementsHTML(typedDocument.closing) : inline(closing)}</div><details id="source-hunks" class="appendix" data-appendix="raw"><summary>Original source hunks</summary><div class="appendix-content"></div></details><details id="evidence" class="appendix" data-appendix="evidence"><summary>Evidence, revisions & scope</summary><div class="appendix-content"></div></details><div class="bottom-line"><span>Diffstory 0.3 · by Epistemic AI · stored locally</span><a href="#top">Back to top ↑</a></div>`;
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
      body.innerHTML = `<dl><dt>Repository</dt><dd>${link(m.url, m.repository || 'Local Git') || esc(m.repository || 'Local Git')}</dd><dt>Base revision</dt><dd class="mono">${esc(m.base_sha)}</dd><dt>Head revision</dt><dd class="mono">${esc(m.head_sha)}</dd><dt>Scope</dt><dd>${esc(m.scope)}</dd><dt>Captured</dt><dd>${esc(m.captured_at || 'Not recorded')}</dd><dt>Supplied source</dt><dd>${R.stats.supplied_paths} paths · ${R.stats.units} analysis units</dd><dt>Test runs</dt><dd>0 runs verified by this analyzer</dd></dl><h3>What the matches establish</h3><p>An identical AST is structural evidence, not a proof that a move preserves behavior. Changed global bindings, import paths, initialization order and external consumers still need review. An edited pair is not automatically a confirmed behavioral change.</p><h3>Provenance</h3><p>${esc(m.validation || 'No execution evidence was attached.')}</p><p>Narrative paragraphs are authored interpretations linked to immutable source ranges. They cannot override the compiler’s source, structural classifications or test status.</p>${R.notes.length ? `<h3>Analysis notes</h3><ul>${R.notes.map(note => `<li>${esc(note)}</li>`).join('')}</ul>` : ''}${R.warnings.length ? `<h3>Limits of this input</h3><ul>${R.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>` : ''}<details><summary>Compiler methods</summary><dl>${Object.entries(R.method).map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl></details>${m.description ? `<details><summary>PR description, as captured</summary><pre>${esc(m.description)}</pre></details>` : ''}<h3>Local by design</h3><p>No external fonts, libraries, analytics or network requests are needed. The HTML embeds the source snapshot; keep it private for a private repository. Diffs are rendered lazily from that embedded data, not fetched from a server. Clicking a source link opens GitHub.</p>`;
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

  function download(name, value, rawJSON = false) {
    const blob = new Blob([rawJSON ? value : JSON.stringify(value, null, 2)], {type: 'application/json'});
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
    if (button.id === 'downloadReport') {
      if (sharedMode) download('diffstory-report.json', reportDataElement.textContent, true);
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
  if (Reader?.shared) appendSharedTypedStory();
  else $('#story').innerHTML = Reader ? typedStoryHTML() : `${preambleHTML()}${R.groups.length ? R.groups.map(sectionHTML).join('') : '<div class="prose"><p>No changed units in this committed comparison.</p></div>'}`;
  buildFooter(); contents(); observeBlocks(); updateScroll();
  if (testMetricsEnabled) {
    readerMetrics.heapAfterRenderBytes = heapBytes();
    publishReaderMetrics();
  }
  if (location.hash) requestAnimationFrame(() => navigateTo(location.hash, false));
})();
