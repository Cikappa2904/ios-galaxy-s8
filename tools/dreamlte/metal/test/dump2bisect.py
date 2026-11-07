#!/usr/bin/env python3
"""Replay iOS's failing Metal pass in the loader and bisect its fragment shader.

undump.py rebuilt the GPU memory of the pass from the phone's log (MEM.bin
at MEM_BASE, the shader's code put back from the cache: the dump skips its
region). This writes a loader replay table (s8_mali_replay.h format, plus
an 'M' record mapping the IOSurface render target VA onto spare RAM) that
runs the pass's vertex/tiler and fragment chains as on the phone:
  ctl      unchanged: must fault as on the phone;
  cNNNN    the shader ended at clause NNNN (its header patched to end the
           shader, as probes.py): the fault is gone iff the shader got to
           clause NNNN before the faulting instruction.
Cuts: the first clause, ~COARSE evenly spread ones, and every clause start
in FINE_LO..FINE_HI (qwords), branch-free clauses only.

  dump2bisect.py MEM.bin MEM_BASE CACHE_DIS.txt SHADER CODE_GPU VT_JC FRAG_JC \
                 TARGET_VA TARGET_SIZE FINE_LO FINE_HI OUT.h TILER_CTX
"""
import re, struct, sys

sys.path.insert(0, __import__('os').path.dirname(__file__))
from probes import end_clause  # noqa: E402

BASE = 0x8c1000000          # loader range base: the GPU heap of the 1 GiB layout
TARGET_PA = 0x8c4000000     # spare RAM for the render target, inside the range
COARSE = 48


def main():
    (mem_path, mem_base, dis, shader, code_gpu, vt_jc, frag_jc, tva, tsize, fine_lo, fine_hi, out) = sys.argv[1:13]
    mem_base, code_gpu, vt_jc, frag_jc = (int(x, 0) for x in (mem_base, code_gpu, vt_jc, frag_jc))
    tva, tsize, fine_lo, fine_hi = (int(x, 0) for x in (tva, tsize, fine_lo, fine_hi))
    mem = bytearray(open(mem_path, 'rb').read())
    end = TARGET_PA + ((tsize + 0xfffff) & ~0xfffff)
    assert BASE <= mem_base and mem_base + len(mem) <= TARGET_PA

    def q(a):
        return struct.unpack_from('<Q', mem, a - mem_base)[0]

    def d(a):
        return struct.unpack_from('<I', mem, a - mem_base)[0]

    # the shader's clauses, as s8dis disassembles the cached code
    t = open(dis).read()
    parts = re.split(r'^=== (\S+) stage (\d+) code (\d+)\n', t, flags=re.M)
    body = next(parts[i + 3] for i in range(1, len(parts), 4) if parts[i] == shader)
    clauses = []
    ms = list(re.finditer(r'^clause_(\d+):', body, re.M))
    for k, m in enumerate(ms):
        text = body[m.start():ms[k + 1].start() if k + 1 < len(ms) else len(body)]
        clauses.append((int(m.group(1)), bool(re.search(r'\+(BRANCH|JUMP)', text))))
    starts = [c for c, br in clauses if not br]
    last = clauses[-1][0]
    pick = {starts[0]}
    for i in range(1, COARSE + 1):
        target = last * i // (COARSE + 1)
        pick.add(min((c for c in starts if c >= target), default=starts[-1]))
    pick |= {c for c in starts if fine_lo <= c <= fine_hi}
    cuts = sorted(pick)

    # pages the GPU writes in a run (job headers, framebuffer/tiler descriptors): refreshed per variant
    jobs = []
    for jc in (vt_jc, frag_jc):
        j = jc
        while j:
            jobs.append(j)
            j = q(j + 0x18)

    # The dump was taken after the chains ran: put back what the GPU wrote,
    # as s8mtl_draw.c packed it (pan_pack zeroes every field it does not set).
    def setd(a, v):
        struct.pack_into('<I', mem, a - mem_base, v)

    for j in jobs:                      # exception status, first incomplete task, fault pointer
        for k in range(4):
            setd(j + 4 * k, 0)
    fbd = q(frag_jc + 0x28) & ~63       # FRAGMENT_JOB payload: framebuffer pointer
    ctx = int(sys.argv[13], 0)          # TILER_CONTEXT (the fragment's FBD tiler field)
    heap = q(ctx + 24)                  # Tiler Context.Heap (word 6)
    assert heap and q(heap + 8) == BASE, f'heap descriptor 0x{heap:x}: base 0x{q(heap + 8):x}'
    print(f'fbd 0x{fbd:x}, tiler context 0x{ctx:x}: polygon list 0x{q(ctx):x} -> 0, '
          f'heap 0x{heap:x}: bottom 0x{q(heap + 16):x} -> 0x{q(heap + 8):x}')
    for k in list(range(0, 2)) + list(range(8, 16)) + list(range(32, 48)):   # polygon list, weights, state
        setd(ctx + 4 * k, 0)
    struct.pack_into('<Q', mem, heap + 16 - mem_base, q(heap + 8))         # bottom = base
    refresh = sorted({(j - mem_base) & ~4095 for j in jobs} |
                     {(ctx - mem_base) & ~4095, (heap - mem_base) & ~4095, (fbd - mem_base) & ~4095})

    rec = bytearray()
    rec += struct.pack('<IIQ', ord('B'), 1, BASE)                       # page tables only
    rec += struct.pack('<IIQQQ', ord('M'), 0, tva, TARGET_PA, (tsize + 0xfff) & ~0xfff)

    def page(o, data=None):
        rec.extend(struct.pack('<IIQ', ord('P'), 0, mem_base + o) + bytes(data or mem[o:o + 4096]).ljust(4096, b'\0'))

    npages = 0
    for o in range(0, len(mem), 4096):
        if any(mem[o:o + 4096]):
            page(o)
            npages += 1

    def submit(slot, jc, label):
        js = []
        j = jc
        while j and len(js) < 8:
            js.append(j)
            j = q(j + 0x18)
        rec.extend(struct.pack('<IIQQII', ord('S'), slot, jc, 0, len(js), 0))
        rec.extend(b''.join(struct.pack('<Q', a) for a in js).ljust(64, b'\0'))
        rec.extend(label.encode()[:31].ljust(32, b'\0'))

    nsub = 0
    for cut in [None] + cuts:
        label = 'ctl' if cut is None else f'c{cut}'
        patched = None
        if cut is not None:
            a = code_gpu + cut * 16
            o = (a - mem_base) & ~4095
            p = bytearray(mem[o:o + 4096])
            off = (a - mem_base) & 4095
            assert off + 16 <= 4096, 'clause header across a page'
            head, tag = end_clause(bytes(p[off:off + 16]))
            assert (tag >> 3) & 7 in (1, 5), f'clause_{cut}: first quadword not format 0 (tag {tag:#x})'
            p[off:off + 16] = head
            page(o, p)
            patched = o
        for o in refresh:
            page(o)
        submit(1, vt_jc, label + ' vt')
        submit(0, frag_jc, label + ' frag')
        nsub += 2
        if patched is not None:
            page(patched)                                    # restore the shader
    rec += struct.pack('<IIQ', ord('E'), 0, end - BASE)
    with open(out, 'w') as f:
        f.write('/* Generated by tools/dreamlte/metal/test/dump2bisect.py: do not edit. */\n')
        f.write(f'#define S8_REPLAY_SIZE {len(rec)}u\n')
        f.write('static const unsigned char s8_replay[] __attribute__((aligned(16), section(".replay"))) = {\n')
        for o in range(0, len(rec), 16):
            f.write(' ' + ', '.join(f'0x{x:02x}' for x in rec[o:o + 16]) + ',\n')
        f.write('};\n')
    print(f'{out}: {len(rec)} bytes, {npages} pages, {len(cuts)} cuts ({sum(1 for c in cuts if fine_lo <= c <= fine_hi)} '
          f'fine), {nsub} submits; refresh pages {[hex(mem_base + o) for o in refresh]}; range 0x{BASE:x}+0x{end - BASE:x}')
    print('cuts:', ' '.join(map(str, cuts)))


if __name__ == '__main__':
    main()
