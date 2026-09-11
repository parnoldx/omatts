#!/usr/bin/env python3
"""Check the self-contained GitHub Pages site without third-party dependencies."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re

root = Path(__file__).resolve().parent.parent / 'site'
errors = []


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path, self.ids, self.links = path, set(), []
        self.h1 = self.main = 0
        self.lang = self.viewport = self.description = self.title = False

    def handle_starttag(self, tag, pairs):
        a = dict(pairs)
        if 'id' in a:
            if a['id'] in self.ids:
                errors.append(f'{self.path.name}: duplicate ID {a["id"]}')
            self.ids.add(a['id'])
        self.h1 += tag == 'h1'
        self.main += tag == 'main'
        self.title |= tag == 'title'
        self.lang |= tag == 'html' and a.get('lang') == 'en'
        self.viewport |= tag == 'meta' and a.get('name') == 'viewport'
        self.description |= tag == 'meta' and a.get('name') == 'description' and bool(a.get('content'))
        for name in ('href', 'src'):
            if name in a:
                self.links.append(a[name])
        if tag in ('script', 'img', 'iframe') and a.get('src', '').startswith(('https:', 'http:', '//')):
            errors.append(f'{self.path.name}: unexpected external resource {a["src"]}')
        if tag == 'link' and a.get('rel') == 'stylesheet' and a.get('href', '').startswith(('https:', 'http:', '//')):
            errors.append(f'{self.path.name}: unexpected external stylesheet')


pages = {}
for name in ('index.html',):
    if not (root / name).is_file():
        errors.append(f'Missing required page: {name}')
for path in sorted(root.glob('*.html')):
    parser = Page(path)
    parser.feed(path.read_text())
    pages[path] = parser
    if not (parser.h1 == parser.main == 1 and parser.title and parser.lang and parser.viewport and parser.description):
        errors.append(f'{path.name}: missing document structure or metadata (h1={parser.h1}, main={parser.main}, title={parser.title}, lang={parser.lang}, viewport={parser.viewport}, desc={parser.description})')
for path, page in pages.items():
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        if url.path.startswith('/'):
            errors.append(f'{path.name}: root-relative link breaks project Pages URL: {link}')
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path
        if not target.is_relative_to(root):
            errors.append(f'{path.name}: link escapes published site: {link}')
        elif not target.is_file():
            errors.append(f'{path.name}: missing local target: {link}')
        elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
            errors.append(f'{path.name}: missing anchor: {link}')
for path in root.rglob('*'):
    if path.is_symlink():
        errors.append(f'{path.relative_to(root)}: symlinks are not allowed')
    elif path.is_file() and path.suffix not in ('.html', '.css', '.js', '.svg') and path.name != '.nojekyll':
        errors.append(f'{path.relative_to(root)}: unexpected file in publishing directory')
for path in root.rglob('*.css'):
    for target in re.findall(r'url\([\'\"]?([^\)\'\"]+)', path.read_text()):
        if not target.startswith(('data:', '#')) and not (path.parent / target).is_file():
            errors.append(f'{path.name}: missing CSS resource: {target}')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'Site checks passed: {len(pages)} pages; local links, anchors, resources, metadata, and publication boundary valid.')
