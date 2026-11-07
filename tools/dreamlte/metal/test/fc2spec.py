#!/usr/bin/env python3
"""Function-constant keys from S8 boot logs -> build-shaders.py --spec file.

  fc2spec.py LOG [LOG ...] > spec.txt

The driver logs each distinct key once per process (s8mtl_device.m):
  MTL fc <hash>: pipeline N fs <function>, <len> chars
  MTL fc <hash>+<offset>: <80 characters>
Keys are put back together, checked against their length and hash, and
written as "FUNCTION<TAB>KEY" (sorted, no duplicates). Incomplete or
inconsistent keys (log ring overwritten, truncated) are reported on stderr.
"""
import re, sys

HEAD = re.compile(r'MTL fc ([0-9a-f]{16}): pipeline \d+ (?:vs|fs) (\S+), (\d+) chars( TRUNCATED)?')
PIECE = re.compile(r'MTL fc ([0-9a-f]{16})\+(\d+): (.*)$')


def key_hash(key):
    h = 0xcbf29ce484222325
    for c in key.encode():
        h = ((h ^ c) * 0x100000001b3) & 0xffffffffffffffff
    return h


def main():
    heads, pieces = {}, {}
    for path in sys.argv[1:]:
        for line in open(path, 'rb').read().decode('latin1').splitlines():
            m = HEAD.search(line)
            if m:
                heads[m.group(1)] = (m.group(2), int(m.group(3)), bool(m.group(4)))
                continue
            m = PIECE.search(line)
            if m:
                pieces.setdefault(m.group(1), {})[int(m.group(2))] = m.group(3)
    out, bad = set(), 0
    for h, (fn, n, trunc) in heads.items():
        p = pieces.get(h, {})
        key = ''.join(p[o] for o in sorted(p))
        if trunc or len(key) != n or f'{key_hash(key):016x}' != h:
            bad += 1
            print(f'fc2spec: {h} {fn}: incomplete ({len(key)}/{n} chars{", truncated" if trunc else ""})',
                  file=sys.stderr)
            continue
        out.add((fn, key))
    for fn, key in sorted(out):
        print(f'{fn}\t{key}')
    print(f'fc2spec: {len(out)} keys, {bad} incomplete', file=sys.stderr)


if __name__ == '__main__':
    main()
