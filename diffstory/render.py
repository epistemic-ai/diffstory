"""Single-file report renderer; no server, npm, CDN, or network dependency."""
from __future__ import annotations
import json
import re
from pathlib import Path
from .analysis import SCHEMA, validate_generated_report, validate_passages


def validate_report(report: dict) -> None:
    """Check fields used as HTML attributes/numbers; prose is escaped by the UI."""
    if report.get('schema') != SCHEMA: raise ValueError('Expected a diffstory report')
    for name in ('groups','changes','tests','edges','raw_files','warnings'):
        if not isinstance(report.get(name),list): raise ValueError(f'Report {name} must be a list')
    ids = re.compile(r'[a-f0-9]{16}')
    numeric = {'number','start','end','old_start','new_start','old_count','new_count','old','new',
               'changed_files','additions','deletions','source_bytes'}
    def walk(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if item is not None and (key=='id' or key.endswith('_id') or key in {'from','to'}):
                    if not isinstance(item,str) or not ids.fullmatch(item):raise ValueError(f'Invalid report identifier: {key}')
                if item is not None and key in numeric:
                    if type(item) is not int or item<0:raise ValueError(f'Invalid report number: {key}')
                if key=='tag' and item not in {'context','add','delete'}:raise ValueError('Invalid diff row tag')
                walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    walk(report)
    for key,value in report['stats'].items():
        if key=='by_kind':
            if not isinstance(value,dict) or any(type(n) is not int or n<0 for n in value.values()):raise ValueError('Invalid classification counts')
        elif type(value) is not int or value<0:raise ValueError(f'Invalid statistic: {key}')
    groups={g['id'] for g in report['groups']};changes={c['id'] for c in report['changes']}
    if len(groups)!=len(report['groups']) or len(changes)!=len(report['changes']):raise ValueError('Duplicate report identifiers')
    changes_by_id = {c['id']: c for c in report['changes']}
    if 'document' in report:
        d = report['document']
        if not isinstance(d, dict) or any(k not in {'lead', 'closing'} or not isinstance(v, str) or len(v) > 6000 for k, v in d.items()):
            raise ValueError('Invalid document narrative')
    for g in report['groups']:
        if 'passages' in g.get('narrative', {}):
            validate_passages(g['narrative']['passages'], g, changes_by_id)
        if not set(g['change_ids'])<=changes:raise ValueError('Unknown group change ID')
        if not set(g['prerequisites'])<=groups:raise ValueError('Unknown prerequisite group ID')
        if g.get('next_id') is not None and g['next_id'] not in groups:raise ValueError('Unknown next group ID')
    if 'generation' in report:
        validate_generated_report(report)


def render(report: dict) -> str:
    validate_report(report)
    assets = Path(__file__).with_name('assets')
    payload = json.dumps(report, ensure_ascii=False, separators=(',', ':'))
    # application/json script elements can be closed by an unescaped </script>.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return (assets.joinpath('app.html').read_text(encoding='utf-8').replace('/*__CSS__*/', assets.joinpath('app.css').read_text(encoding='utf-8'))
            .replace('/*__JS__*/', assets.joinpath('app.js').read_text(encoding='utf-8')).replace('__BRAND_MARK__', assets.joinpath('epistemic-mark.svg').read_text(encoding='utf-8')).replace('__REPORT_JSON__', payload))
