#!/usr/bin/env python3
"""Check probe.s8sc against the clauses it was cut from: same instructions,
header now ending the shader (eos, no next_*, no prefetch, no waits).

  probecheck.py CACHE_DIS.txt PROBE_DIS.txt PROBE_MAP.txt
"""
import re, sys


def split(path):
    t = open(path).read()
    parts = re.split(r'^=== (\S+) stage (\d+) code (\d+)\n', t, flags=re.M)
    return {parts[i]: parts[i + 3] for i in range(1, len(parts), 4)}


def clause_text(body, q):
    m = re.search(rf'^clause_{q}:\n(.*?)(?=^clause_\d+:|\Z)', body, re.M | re.S)
    return m.group(1)


def instrs(text):
    return [l.strip() for l in text.splitlines() if re.match(r'\s+[*+]', l)]


def header(text):
    return text.splitlines()[0]


orig, probe = split(sys.argv[1]), split(sys.argv[2])
bad = 0
for line in open(sys.argv[3]):
    if line.startswith('#'):
        continue
    label, shader, clause = line.split()[:3]
    q = int(clause.split('_')[1])
    src = clause_text(orig[shader], q)
    name = '@' + label
    body = probe[name]
    first = re.search(r'^clause_(\d+):', body, re.M)
    pt = clause_text(body, first.group(1))
    later = re.findall(r'^clause_\d+:', body, re.M)
    h = header(pt).split()
    problems = []
    if instrs(src) != instrs(pt):
        problems.append('instructions differ')
    if 'eos' not in h:
        problems.append('no eos')
    if any(x.startswith('next_') for x in h) or 'ncph' in h:
        problems.append('next/prefetch left: ' + ' '.join(h))
    if any(x.startswith('dwb') for x in h):
        problems.append('wait left')
    # everything after the probe clause must be the zero padding s8_parse_shader adds (not decoded)
    if len(later) != 1:
        problems.append(f'{len(later)} clauses decoded')
    if problems:
        bad += 1
        print(f'BAD {label} ({shader} clause_{q}): {"; ".join(problems)}')
        print('   src hdr:', header(src).strip(), '| probe hdr:', header(pt).strip())
print(f'checked {sum(1 for l in open(sys.argv[3]) if not l.startswith("#"))} probes, {bad} bad')
sys.exit(1 if bad else 0)
