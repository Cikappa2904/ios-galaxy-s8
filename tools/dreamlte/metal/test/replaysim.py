#!/usr/bin/env python3
"""Simulate the loader's replay of s8_mali_replay.h (rp_run) on the host.

Applies 'P' pages in order; at each 'S' checks that the chain's job headers
are fresh (exception status 0) and linked as recorded, then does what the
GPU would to them (exception status DONE), so a later chain reusing the
memory must have been refreshed by the table. Optionally writes the image
at given submits for dump-decode.c.

  replaysim.py REPLAY.bin [N:OUT.bin ...]
"""
import struct, sys

b = open(sys.argv[1], 'rb').read()
want = {}
for a in sys.argv[2:]:
    n, path = a.split(':', 1)
    want[int(n)] = path
pages, base, flags, used = {}, None, 0, 0
o, n, bad = 0, 0, 0


def q(addr):
    p = pages.get(addr & ~4095)
    return struct.unpack_from('<Q', p, addr & 4095)[0] if p else None


def d(addr):
    p = pages.get(addr & ~4095)
    return struct.unpack_from('<I', p, addr & 4095)[0] if p else None


while o + 16 <= len(b):
    tag, pad, v = struct.unpack_from('<IIQ', b, o)
    if tag == ord('B'):
        base, flags = v, pad
        o += 16
    elif tag == ord('M'):
        va, pa, ln = struct.unpack_from('<QQQ', b, o + 8)
        print(f'map 0x{va:x} -> 0x{pa:x} +0x{ln:x}')
        o += 32
    elif tag == ord('P'):
        assert v % 4096 == 0
        pages[v] = bytearray(b[o + 16:o + 16 + 4096])
        o += 16 + 4096
    elif tag == ord('S'):
        n += 1
        slot, = struct.unpack_from('<I', b, o + 4)
        jc, probe, nj = struct.unpack_from('<QQI', b, o + 8)
        jobs = struct.unpack_from('<8Q', b, o + 32)[:nj]
        label = b[o + 96:o + 128].split(b'\0')[0].decode()
        problems = []
        j, k = jc, 0
        while j and k < 64:
            if d(j) is None:
                problems.append(f'job 0x{j:x} not in any page')
                break
            if d(j) != 0:
                problems.append(f'job 0x{j:x} exception status 0x{d(j):x} (stale)')
            if k < len(jobs) and jobs[k] != j:
                problems.append(f'job {k} 0x{j:x} != recorded 0x{jobs[k]:x}')
            j = q(j + 0x18)
            k += 1
        if k != len(jobs):
            problems.append(f'{k} jobs linked, {len(jobs)} recorded')
        if problems:
            bad += 1
            print(f'S{n} {label}: ' + '; '.join(problems))
        if n in want:
            lo = min(pages)
            hi = max(pages) + 4096
            img = bytearray(hi - lo)
            for a, p in pages.items():
                img[a - lo:a - lo + 4096] = p
            open(want[n], 'wb').write(img)
            print(f'S{n} {label}: image 0x{lo:x}+0x{hi - lo:x} -> {want[n]} (jc 0x{jc:x})')
        for j in jobs:                         # what the GPU does to them
            struct.pack_into('<I', pages[j & ~4095], j & 4095, 1)
        o += 128
    elif tag == ord('E'):
        used = v
        o += 16
        break
    else:
        print(f'bad tag 0x{tag:x} at {o}')
        bad += 1
        break
print(f'base 0x{base:x} flags {flags} used 0x{used:x}, {len(pages)} pages, {n} submits, '
      f'end at {o} of {len(b)}, {bad} problems')
for a in pages:
    if not (base <= a < base + used):
        print(f'page 0x{a:x} outside the range')
        bad += 1
sys.exit(1 if bad or o != len(b) else 0)
