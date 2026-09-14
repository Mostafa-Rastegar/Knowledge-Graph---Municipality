# -*- coding: utf-8 -*-
"""Summarise a XeLaTeX log: the things worth acting on, nothing else.

Usage:  python checklog.py build/AUTthesis.log
"""
import io, re, sys, os
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join('build', 'AUTthesis.log')
if not os.path.isfile(path):
    print('no log at', path)
    raise SystemExit(1)

log = io.open(path, encoding='utf-8', errors='replace').read()

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass


def section(title, items, limit=15, colour=''):
    print('%s: %d' % (title, len(items)))
    for it in items[:limit]:
        print('   ', it.strip()[:150])
    if len(items) > limit:
        print('    ... and %d more' % (len(items) - limit))


# Hard errors
errors = re.findall(r'^! (.+)$', log, re.M)
section('ERRORS', errors)

# Missing fonts are fatal for a Persian document even when TeX carries on
fonts = sorted(set(re.findall(r'Font \\?[\w/]+ .*not loadable|Invalid font identifier'
                             r'|cannot find font', log)))
section('FONT PROBLEMS', fonts)

# A missing character is silently dropped from the PDF, so it is the most
# dangerous warning in a Persian document: the source looks perfectly fine.
# TeX hard-wraps the log at ~79 columns, so the message is routinely split
# across lines - flatten before matching, or the count comes out as zero.
flat = log.replace('\n', '')
events = re.findall(r'Missing character: There is no (.) in font ([^;!]*)', flat)
counts = Counter(events)
print('MISSING GLYPHS: %d events, %d distinct' % (len(events), len(counts)))
for (ch, fnt), n in counts.most_common(12):
    print('    U+%04X %r  x%-5d  %s' % (ord(ch), ch, n, fnt.strip()[:46]))

# Unresolved cross-references and citations
undef = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
section('UNDEFINED REFERENCES', undef)
undefc = sorted(set(re.findall(r"Citation `([^']+)' on page", log)))
section('UNDEFINED CITATIONS', undefc)

# Layout: text running into the margin
over = re.findall(r'^(Overfull \\[hv]box \([\d.]+pt too \w+\).*)$', log, re.M)
big = [o for o in over if float(re.search(r'\(([\d.]+)pt', o).group(1)) > 20]
print('OVERFULL BOXES: %d total, %d worse than 20pt' % (len(over), len(big)))
for o in big[:10]:
    print('   ', o.strip()[:150])

# Missing files
missf = sorted(set(re.findall(r"File `([^']+)' not found", log)))
section('FILES NOT FOUND', missf)

pages = re.findall(r'Output written on .*\((\d+) pages?', log)
if pages:
    print('PAGES: %s' % pages[0])

serious = len(errors) + len(fonts) + len(missf) + len(events)
print()
print('VERDICT:', 'clean' if serious == 0 else '%d serious issue(s)' % serious)
