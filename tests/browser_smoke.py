"""Literate reader browser tests. Optional dev dependencies: Playwright + Chromium.

Uses set_content because some managed environments disallow file:// navigation.
The offline HTML has no runtime network or framework dependency.
"""
import json
import os
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffstory.analysis import compile_snapshot
from diffstory.render import render


def main():
    html = (ROOT / 'examples/demo.html').read_text(encoding='utf-8')
    report = json.loads((ROOT / 'examples/demo.report.json').read_text())
    (ROOT / 'docs/reader').mkdir(parents=True, exist_ok=True)
    change = lambda name: next(c for c in report['changes'] if (c.get('after') or c.get('before'))['name'] == name)
    parsing_index = next(i for i, g in enumerate(report['groups']) if g['theme'] == 'result parsing')
    parsing_group = report['groups'][parsing_index]
    assertions = 0
    def check(value, message):
        nonlocal assertions
        assert value, message
        assertions += 1

    with sync_playwright() as p:
        options = {'headless': True}
        executable = os.environ.get('DIFFSTORY_CHROMIUM')
        if executable: options['executable_path'] = executable
        browser = p.chromium.launch(**options)
        page = browser.new_page(viewport={'width': 1440, 'height': 1050}, accept_downloads=True)
        page.set_default_timeout(6500)
        errors = []; requests = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: requests.append(r.url))
        page.set_content(html, wait_until='domcontentloaded')
        page.wait_for_timeout(150)
        check(page.title() == f'Diffstory · {report["meta"]["title"]} · Reader', 'title')
        check(page.locator('.story-section').count() == 10, 'all sections present simultaneously')
        check(page.locator('[role="tablist"],.sidebar,.navfooter').count() == 0, 'no dashboard, tabs or pager')
        check(page.locator('#contentsPanel').is_hidden() and page.locator('#morePanel').is_hidden(), 'quiet initial header')
        check(page.locator('.code-block[data-loaded]').count() < page.locator('.code-block').count(), 'offscreen code is deferred')
        check(page.locator('#story .passage').count() > 15, 'multiple prose-code passages')
        check(page.locator('#story .passage').evaluate_all('(nodes)=>nodes.every(n=>n.querySelector(".prose") && n.querySelector(".code-block"))'), 'each passage binds prose to code')
        check(page.locator('[data-audit][open]').count() == 0, 'audit is opt-in')
        check('Original synthetic example' in page.locator('.document-scope').inner_text(), 'snapshot boundary visible')
        check(page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'desktop no document overflow')
        check(page.locator('.brandmark svg').count() == 1, 'real Epistemic AI mark embedded')
        check(page.locator('.edition').inner_text() == 'EPISTEMIC AI', 'Epistemic AI publisher label')
        check(page.evaluate('getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()') == '#2563eb', 'brand palette')
        page.screenshot(path=str(ROOT / 'docs/reader/desktop.png'), full_page=False)

        # Find is a contents filter, never a replacement of the document.
        page.keyboard.press('/')
        check(page.locator('#contentsPanel').is_visible(), 'keyboard contents')
        check(page.locator('#search').evaluate('(el)=>el===document.activeElement'), 'search focus')
        page.locator('#search').fill('parse_tags')
        check(page.locator('.contents-item').count() == 1, 'symbol search')
        check(page.locator('.story-section').count() == 10, 'search leaves whole document intact')
        page.locator('.contents-item').click(no_wait_after=True)
        page.wait_for_timeout(150)
        check(page.locator('#contentsPanel').is_hidden(), 'contents closes after jump')
        check(parsing_group['id'] in page.evaluate('location.hash'), 'section deep link')
        check(f'{parsing_index + 1:02d} / 10' in page.locator('#readingPosition').inner_text(), 'reading position tracks scroll')
        page.screenshot(path=str(ROOT / 'docs/reader/parsing.png'), full_page=False)

        section = page.locator('.story-section').nth(parsing_index)
        code = section.locator(f'[data-changes~="{change("parse_tags")["id"]}"]')
        code.scroll_into_view_if_needed()
        page.wait_for_function('(id)=>document.getElementById(id).dataset.loaded === "true"', arg=code.get_attribute('id'))
        check('return tags' in code.inner_text(), 'actual implementation readable')
        code.locator('[data-switch]').click(no_wait_after=True)
        check(code.locator('.code-line.delete').count() > 0 and code.locator('.code-line.add').count() > 0, 'inline paired diff')
        check(page.locator('.story-section').count() == 10, 'changing code view does not navigate')
        code.locator('[data-switch]').click(no_wait_after=True)
        check('return tags' in code.inner_text(), 'return to definition')

        large = page.locator('.story-section').first.locator('.code-block').first
        large.scroll_into_view_if_needed(); page.wait_for_timeout(100)
        initial = large.locator('.code-line').count()
        check(initial <= 24, 'long diff uses bounded preview')
        check('Load full diff' in large.inner_text(), 'explicit in-place diff expansion')
        large.locator('[data-expand]').click(no_wait_after=True)
        expected = sum(len(h['rows']) for h in change('construct_query')['hunks'])
        check(large.locator('.code-line').count() == expected, 'full diff expands without lost lines')
        check(large.locator('.code-region').evaluate('(el)=>el.scrollHeight<=el.clientHeight+1'), 'no nested vertical code scrollbar')
        large.locator('[data-collapse]').click(no_wait_after=True)
        check(large.locator('.code-line').count() == initial, 'collapse restores source-focused excerpt')
        page.locator('#contentsBtn').click(no_wait_after=True)
        page.keyboard.press('Escape')
        check(page.locator('#contentsPanel').is_hidden(), 'escape closes contents')
        check(page.locator('#contentsBtn').evaluate('(el)=>el===document.activeElement'), 'escape restores focus')

        # Per-section audit and backwards-compatible revision-bound notes.
        audit = section.locator('[data-audit]')
        audit.locator('summary').first.click(no_wait_after=True)
        expect(audit.locator('[data-note]')).to_be_visible()
        check(audit.locator('.audit-columns').count() == 1, 'inline audit content')
        check('No repository tests were executed' in audit.inner_text(), 'static evidence not claimed as execution')
        audit.locator('[data-review]').check()
        note_text = 'Verify page-size bounds.\nLiteral note <not markup>.'
        audit.locator('[data-note]').fill(note_text)
        check('Reviewed' in audit.locator('summary').first.inner_text(), 'review marker')
        page.locator('#moreBtn').click(no_wait_after=True)
        with page.expect_download() as downloaded:
            page.locator('#exportBtn').click(no_wait_after=True)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'review.json'; downloaded.value.save_as(dest)
            data = json.loads(dest.read_text())
            check(data['schema'] == 'diffstory.review.v1', 'export remains backwards compatible')
            check(data['notes'][parsing_group['id']] == note_text, 'notes retained exactly')
            check(data['revision'].endswith(report['meta']['head_sha']), 'revision lock')
            audit.locator('[data-note]').fill('changed')
            page.locator('#importFile').set_input_files(str(dest))
            expect(page.locator('#toast')).to_contain_text('Review imported')
            check(audit.locator('[data-note]').input_value() == note_text, 'import restores notes in place')
            data['revision'] = 'wrong'; dest.write_text(json.dumps(data))
            page.locator('#importFile').set_input_files(str(dest))
            expect(page.locator('#toast')).to_contain_text('different report revision')
            check(audit.locator('[data-note]').input_value() == note_text, 'bad revision leaves review untouched')
        page.locator('#moreBtn').click(no_wait_after=True); page.locator('#auditBtn').click(no_wait_after=True)
        check(page.locator('[data-audit][open]').count() == 10, 'optional all-section review details')
        page.locator('#moreBtn').click(no_wait_after=True); page.locator('#auditBtn').click(no_wait_after=True)
        check(page.locator('[data-audit][open]').count() == 0, 'audit can be decluttered again')

        # Every semantic unit remains reachable, even when not central to the prose.
        for detail in page.locator('[data-extra]').all():
            detail.evaluate('(el)=>el.open=true')
        page.wait_for_timeout(120)
        represented = page.locator('[data-changes]').evaluate_all('(nodes)=>[...new Set(nodes.flatMap(n=>n.dataset.changes.split(" ").filter(Boolean)))]')
        check(set(represented) == {c['id'] for c in report['changes']}, 'all units retained in main or supporting source')
        page.locator('#moreBtn').click(no_wait_after=True); page.locator('#morePanel a[href="#source-hunks"]').click(no_wait_after=True)
        expect(page.locator('#source-hunks')).to_have_attribute('open', '')
        check(page.locator('.raw-file').count() == 8, 'original hunks grouped by eight unique paths')
        page.locator('.raw-file').first.locator('summary').first.click(no_wait_after=True)
        expect(page.locator('.raw-file').first.locator('.code-block').first).to_be_visible()
        check(page.locator('.raw-file').first.locator('.code-block').count() > 0, 'raw hunks load inline')
        page.locator('#moreBtn').click(no_wait_after=True); page.locator('#morePanel a[href="#evidence"]').click(no_wait_after=True)
        check(page.locator('#evidence').get_attribute('open') is not None, 'evidence lives in the same document')
        check('rendered lazily from that embedded data' in page.locator('#evidence').inner_text(), 'offline lazy-rendering disclosure')
        check(report['meta']['head_sha'] in page.locator('#evidence').inner_text(), 'pinned head available')
        check(page.locator('.story-section').count() == 10, 'appendices do not replace narrative')

        # Responsive layout, including expanded source.
        page.set_viewport_size({'width': 390, 'height': 844})
        page.evaluate('scrollTo(0,0)'); page.wait_for_timeout(100)
        page.locator('#toast').wait_for(state='hidden')
        page.screenshot(path=str(ROOT / 'docs/reader/mobile.png'), full_page=False)
        check(page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile top no overflow')
        for index in (0, 1, 3, 8, 9):
            page.locator('.story-section').nth(index).evaluate('(el)=>el.scrollIntoView({block:"start"})')
            page.wait_for_timeout(80)
            check(page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'mobile section {index+1} no overflow')
        page.set_viewport_size({'width': 320, 'height': 700})
        check(page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'narrow 320px no overflow')
        check(not errors, 'JavaScript errors: ' + str(errors))
        check(not requests, 'Unexpected network requests: ' + str(requests))

        # Stress a genuinely oversized diff, not only the small example snapshot.
        source = 'def construct_large_query():\n' + ''.join(f'    value_{i} = {i}\n' for i in range(800)) + '    return value_799\n'
        new = source.replace('value_799 = 799', 'value_799 = 800')
        snapshot = {'schema': 'diffstory.snapshot.v1', 'meta': {'base_sha':'a'*40, 'head_sha':'b'*40, 'changed_files':1, 'scope':'complete'}, 'fragments':[
            {'side':'base','path':'large.py','text':source,'start_line':1,'scope':'complete'},
            {'side':'head','path':'large.py','text':new,'start_line':1,'scope':'complete'}]}
        stress = browser.new_page(viewport={'width':1280,'height':950})
        stress.set_content(render(compile_snapshot(snapshot)), wait_until='domcontentloaded')
        block = stress.locator('.code-block').first
        block.scroll_into_view_if_needed(); stress.wait_for_timeout(100)
        if block.locator('[data-switch]').inner_text() == 'Read definition':
            block.locator('[data-switch]').click(no_wait_after=True)
        check(block.locator('.code-line').count() == 24, 'oversized definition uses bounded preview')
        block.locator('[data-expand]').click(no_wait_after=True)
        check(block.locator('.code-line').count() == 160, 'first expansion bounded to chunk')
        check('remaining' in block.inner_text(), 'remaining source is not silently discarded')
        while block.locator('[data-expand]').count(): block.locator('[data-expand]').click(no_wait_after=True)
        check(block.locator('.code-line').count() == 802, 'all 802 original lines recoverable')
        block.locator('[data-collapse]').click(no_wait_after=True)
        check(block.locator('.code-line').count() == 24, 'huge source collapses back to bounded preview')
        check(stress.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'large source layout bounded')
        stress.close()
        # Oversized paired diff: all changed rows are recovered in bounded batches.
        import re
        snapshot['fragments'][1]['text'] = re.sub(r'= (\d+)', lambda m: '= ' + str(int(m.group(1)) + 1000), source)
        big_report = compile_snapshot(snapshot)
        stress = browser.new_page(viewport={'width': 1280, 'height': 950})
        stress.set_content(render(big_report), wait_until='domcontentloaded')
        block = stress.locator('.code-block').first
        block.scroll_into_view_if_needed(); stress.wait_for_timeout(100)
        check(block.locator('.code-line').count() <= 24, 'oversized diff is initially folded')
        check('Load full diff' in block.inner_text(), 'oversized diff has explicit load action')
        block.locator('[data-expand]').click(no_wait_after=True)
        check(block.locator('.code-line').count() <= 160, 'diff expansion is bounded')
        while block.locator('[data-expand]').count(): block.locator('[data-expand]').click(no_wait_after=True)
        expected = sum(len(h['rows']) for h in big_report['changes'][0]['hunks'])
        check(block.locator('.code-line').count() == expected, 'every diff row can be loaded')
        check(block.locator('.code-line.add').count() == 800, 'all added lines retained')
        check(block.locator('.code-line.delete').count() == 800, 'all removed lines retained')
        stress.close()
        browser.close()
    print(f'Browser reader: {assertions} assertions passed; no page JavaScript errors or network requests in demo.')


if __name__ == '__main__': main()
