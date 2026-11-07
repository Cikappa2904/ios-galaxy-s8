#!/usr/bin/env python3
"""Merge S8SC shader caches (build-shaders.py, one per metallib) into one.

  merge-caches.py OUT.s8sc FIRST.s8sc [MORE.s8sc ...]

The driver finds shaders by function name alone, so a name may appear only
once: the first cache wins (QuartzCore's goes first) and later duplicates
are reported and dropped. Each library keeps its own varying table, which
is enough: a pipeline's vertex and fragment functions come from one
library.
"""
import struct, sys


def read(path):
    d = open(path, 'rb').read()
    if d[:4] != b'S8SC' or struct.unpack_from('<I', d, 4)[0] != 1:
        sys.exit(f'{path}: not an S8SC v1 cache')
    n = struct.unpack_from('<I', d, 8)[0]
    out = []
    for i in range(n):
        e = 12 + 72 * i
        name = d[e:e + 64].split(b'\0')[0]
        off, size = struct.unpack_from('<II', d, e + 64)
        out.append((name, d[off:off + size]))
    return out


def main():
    out, *ins = sys.argv[1:]
    seen, good = set(), []
    for path in ins:
        kept = 0
        for name, blob in read(path):
            if name in seen:
                print(f'{path}: {name.decode()} already cached, dropped')
                continue
            seen.add(name)
            good.append((name, blob))
            kept += 1
        print(f'{path}: {kept} shaders')
    head = 12 + 72 * len(good)
    index, blob, off = b'', b'', head
    for name, b in good:
        index += name[:63].ljust(64, b'\0') + struct.pack('<II', off, len(b))
        blob += b
        off += len(b)
    with open(out, 'wb') as fh:
        fh.write(b'S8SC' + struct.pack('<II', 1, len(good)) + index + blob)
    print(f'{out}: {len(good)} shaders, {off} bytes')


if __name__ == '__main__':
    main()
