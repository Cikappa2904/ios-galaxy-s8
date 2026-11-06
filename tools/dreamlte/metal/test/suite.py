#!/usr/bin/env python3
"""Pipeline suite for the loader replay: CoreAnimation's pipelines as the
phone logged them ("MTL pipeline N: vs V fs F fmt M blend B"), appended to a
probe list (probes.py) as "LABEL VS FS FMT BLEND" lines, and a cache with
the probe shaders plus every QuartzCore shader.

  suite.py PIPELINES.txt PROBE.s8sc QUARTZCORE.s8sc PROBE_LIST.txt OUT.s8sc OUT_LIST.txt
"""
import re, struct, sys


def read_cache(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'S8SC'
    n, = struct.unpack_from('<I', b, 8)
    out = []
    for i in range(n):
        e = 12 + i * 72
        name = b[e:e + 64].split(b'\0')[0].decode()
        off, size = struct.unpack_from('<II', b, e + 64)
        out.append((name, b[off:off + size]))
    return out


def write_cache(path, entries):
    head = 12 + 72 * len(entries)
    index, data, off = b'', b'', head
    for name, blob in entries:
        index += name.encode().ljust(64, b'\0') + struct.pack('<II', off, len(blob))
        data += blob
        off += len(blob)
    open(path, 'wb').write(b'S8SC' + struct.pack('<II', 1, len(entries)) + index + data)


pipes_path, probe_cache, qc_cache, probe_list, out_cache, out_list = sys.argv[1:7]
entries, seen = [], set()
for name, blob in read_cache(probe_cache) + read_cache(qc_cache):
    if name not in seen:
        seen.add(name)
        entries.append((name, blob))
write_cache(out_cache, entries)
lines = open(probe_list).read().splitlines()
n = 0
for line in open(pipes_path):
    m = re.search(r'vs (\S+) fs (\S+) fmt (\d+) blend (\d+)', line)
    if not m:
        continue
    vs, fs, fmt, blend = m.groups()
    if fmt == '115':
        # s8mtl_draw.c has no RGBA16F render target format yet: the pass is
        # skipped there ("MTL pass skipped"), on the phone as here
        print(f'skip {vs}/{fs}: fmt 115 render targets unsupported by the driver')
        continue
    if vs not in seen or fs not in seen:
        print(f'skip {vs}/{fs}: not in the cache')
        continue
    lines.append(f's{n} {vs} {fs} {fmt} {blend}')
    n += 1
open(out_list, 'w').write('\n'.join(lines) + '\n')
print(f'{len(entries)} shaders in {out_cache}, {n} pipelines added to {out_list}')
