#!/usr/bin/env python3
"""G71 instruction probe: one clause per instruction variant, as shaders.

The Mali-G71 (Bifrost v6, first generation) lacks some instructions Mesa's
Bifrost ISA (G52-based) has; one QuartzCore fragment shader faulted with
INSTR_INVALID_ENC (0x51) on the phone. This takes the clauses of every
shader in the QuartzCore cache (as s8dis disassembles the cached code),
covers each instruction variant (unit + mnemonic with modifiers) with the
smallest branch-free clause using it, patches each picked clause's header
to end the shader (flow END, no prefetch, no waits, no next message) and
writes a cache where each probe is @test_vs / @test_fs with that clause as
its code. replay-gen.c (S8_PROBE_LIST) draws once with each; the loader
runs them on the GPU and logs every job's status.

  probes.py CACHE_DIS.txt QUARTZCORE.s8sc TEST.s8sc OUT.s8sc OUT_LIST.txt OUT_MAP.txt
"""
import re, struct, sys

BRANCH = re.compile(r'\+(BRANCH|JUMP|BRANCHZ|BRANCHC|BRANCH_)')
EXTRA = [('fixed_frag_lph_cpf', 275), ('fixed_frag_lph_cpf', 280), ('fixed_frag_lph_cpf', 285)]


def parse_dis(path):
    t = open(path).read()
    parts = re.split(r'^=== (\S+) stage (\d+) code (\d+)\n', t, flags=re.M)
    shaders = {}
    for i in range(1, len(parts), 4):
        name, stage, code, body = parts[i], int(parts[i + 1]), int(parts[i + 2]), parts[i + 3]
        cl = [(int(m.group(1)), m.start()) for m in re.finditer(r'^clause_(\d+):', body, re.M)]
        clauses = []
        for k, (q, pos) in enumerate(cl):
            end_q = cl[k + 1][0] if k + 1 < len(cl) else (code + 15) // 16
            text = body[pos:cl[k + 1][1] if k + 1 < len(cl) else len(body)]
            ops = set()
            for line in text.splitlines():
                m = re.match(r'\s+([*+])([A-Z0-9_.a-z]+)', line)
                if m and m.group(2) != 'NOP':
                    ops.add(m.group(1) + m.group(2))
            clauses.append({'q': q, 'end': end_q, 'ops': ops, 'branch': bool(BRANCH.search(text)), 'text': text})
        shaders[name] = {'stage': stage, 'code': code, 'clauses': clauses}
    return shaders


def read_cache(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'S8SC'
    n, = struct.unpack_from('<I', b, 8)
    out = {}
    for i in range(n):
        e = 12 + i * 72
        name = b[e:e + 64].split(b'\0')[0].decode()
        off, size = struct.unpack_from('<II', b, e + 64)
        out[name] = b[off:off + size]
    return out


def write_cache(path, entries):
    head = 12 + 72 * len(entries)
    index, data, off = b'', b'', head
    for name, blob in entries:
        index += name.encode().ljust(64, b'\0') + struct.pack('<II', off, len(blob))
        data += blob
        off += len(blob)
    open(path, 'wb').write(b'S8SC' + struct.pack('<II', 1, len(entries)) + index + data)


def code_of(blob):
    n, = struct.unpack_from('<I', blob, 12)
    return blob[len(blob) - n:]


def with_code(blob, code):
    n, = struct.unpack_from('<I', blob, 12)
    new = bytearray(blob[:len(blob) - n] + code)
    struct.pack_into('<I', new, 12, len(code))
    return bytes(new)


def end_clause(clause):
    """Header (disassemble.c: bits(w2,19,32) | w3 << 13 of the first quadword):
    flow_control 11-13 = END, next_clause_prefetch 16, dependency_wait 24-31,
    next_message_type 40-44 cleared."""
    c = bytearray(clause)
    w2, w3 = struct.unpack_from('<II', c, 8)
    tag = struct.unpack_from('<I', c, 0)[0] & 0xff
    h = (w2 >> 19) | (w3 << 13)
    h &= (1 << 45) - 1
    for lo, n in ((11, 3), (16, 1), (24, 8), (40, 5)):
        h &= ~(((1 << n) - 1) << lo)
    w2 = (w2 & ((1 << 19) - 1)) | ((h & ((1 << 13) - 1)) << 19)
    w3 = h >> 13
    struct.pack_into('<II', c, 8, w2, w3)
    return bytes(c), tag


def main():
    dis, qc_path, test_path, out_cache, out_list, out_map = sys.argv[1:7]
    shaders = parse_dis(dis)
    qc, test = read_cache(qc_path), read_cache(test_path)
    cands = {}   # op -> [(nops, shader, clause)]
    allops = set()
    for name, s in shaders.items():
        for c in s['clauses']:
            allops |= c['ops']
            if c['branch'] or not c['ops']:
                continue
            for op in c['ops']:
                cands.setdefault(op, []).append((len(c['ops']), name, c))
    picked, covered = [], set()
    for op in sorted(cands, key=lambda o: (len(cands[o]), o)):
        if op in covered:
            continue
        n, name, c = min(cands[op], key=lambda x: (x[0], x[1], x[2]['q']))
        picked.append((name, c))
        covered |= c['ops']
    for name, q in EXTRA:
        c = next(c for c in shaders[name]['clauses'] if c['q'] == q)
        picked.append((name, c))
    branch_only = sorted(allops - covered)
    entries = [(k, v) for k, v in test.items()]
    lines, mapping = [], []
    for i, (name, c) in enumerate(picked):
        code = code_of(qc[name])
        clause = code[c['q'] * 16:c['end'] * 16]
        assert len(clause) == (c['end'] - c['q']) * 16 and len(clause) >= 16
        patched, tag = end_clause(clause)
        frag = shaders[name]['stage'] == 4
        pname = f'@p{i}{"f" if frag else "v"}'
        base = test['@test_fs' if frag else '@test_vs']
        entries.append((pname, with_code(base, patched)))
        label = f'p{i}{"f" if frag else "v"}'
        lines.append(f'{label} {"@test_vs" if frag else pname} {pname if frag else "@test_fs"}')
        mapping.append(f'{label} {name} clause_{c["q"]} qwords {c["end"] - c["q"]} tag 0x{tag:x} ops {" ".join(sorted(c["ops"]))}')
    write_cache(out_cache, entries)
    open(out_list, 'w').write('\n'.join(lines) + '\n')
    open(out_map, 'w').write('\n'.join(mapping) + '\n' +
                             f'# {len(allops)} instruction variants, {len(covered)} probed, '
                             f'{len(branch_only)} only in clauses with branches: {" ".join(branch_only)}\n')
    print(f'{len(shaders)} shaders, {sum(len(s["clauses"]) for s in shaders.values())} clauses, '
          f'{len(allops)} instruction variants; {len(picked)} probes cover {len(covered)}; '
          f'{len(branch_only)} only in branch clauses')


if __name__ == '__main__':
    main()
