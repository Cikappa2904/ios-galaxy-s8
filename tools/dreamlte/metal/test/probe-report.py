#!/usr/bin/env python3
"""Read the loader's replay of the G71 probe/suite table back from a kernel log.

The loader logs per chain (gpu_probe.c rp_run):
  S8 LOADER| R <label> <vt|frag>: irq=.. js=.. us=.. head=.. fault=.. faddr=..
  S8 LOADER|   t=<type> e=<exception> ... [pixel=..] [stop=.. js=..]
A probe 'pNf' runs its clause as the fragment shader (judged on its frag
chain), 'pNv' as the vertex shader (vt chain); 'sN' are CoreAnimation
pipelines (both chains). An instruction variant is OK when some probe
holding it completed; SUSPECT when it only appears in probes that faulted
INSTR_INVALID_ENC (0x51) / INSTR_TYPE_MISMATCH (0x52).

  probe-report.py SWEETLOG.txt PROBE_MAP.txt LIST.txt
"""
import re, sys

EXC = {0x1: 'DONE', 0x8: 'ACTIVE', 0x40: 'CONFIG', 0x42: 'READ', 0x43: 'WRITE', 0x48: 'BUS',
       0x50: 'INVALID_PC', 0x51: 'INVALID_ENC', 0x52: 'TYPE_MISMATCH', 0x53: 'OPERAND', 0x54: 'TLS',
       0x56: 'ALIGN', 0x58: 'DATA_INVALID', 0x59: 'TILE_RANGE', 0x5a: 'ADDR_RANGE', 0x5b: 'IMPRECISE',
       0x60: 'OOM'}


def name(x):
    return EXC.get(x, hex(x)) if x is not None else 'missing'


log = open(sys.argv[1], 'rb').read().decode('latin-1').splitlines()
res = {}
for i, l in enumerate(log):
    m = re.search(r'S8 LOADER\| R (\S+) (vt|frag): irq=0x([0-9a-f]+) js=0x([0-9a-f]+) us=0x([0-9a-f]+)', l)
    if not m:
        continue
    lab, kind = m.group(1), m.group(2)
    irq, js, us = int(m.group(3), 16), int(m.group(4), 16), int(m.group(5), 16)
    exc = []
    if i + 1 < len(log) and 'S8 LOADER|' in log[i + 1] and ' R ' not in log[i + 1]:
        exc = [int(x, 16) for x in re.findall(r'e=0x([0-9a-f]+)', log[i + 1])]
    done = (irq & (1 if kind == 'frag' else 2)) != 0 and js == 0
    # the job that failed: first non-DONE exception, else the slot status
    bad = next((e for e in exc if e != 1), None)
    res.setdefault(lab, {})[kind] = {'done': done, 'js': js, 'us': us, 'exc': exc, 'bad': bad if bad is not None else (None if done else js)}
print(f'{len(res)} labels found in the log')
if not res:
    sys.exit('no "S8 LOADER| R" lines: replay not run or not echoed (kernel without 0142?)')

ops_of = {}
for l in open(sys.argv[2]):
    if l.startswith('#'):
        print(l.strip())
        continue
    f = l.split()
    ops_of[f[0]] = (f[1], f[2], f[f.index('ops') + 1:])
ctl = res.get('ctl', {})
print('control (test shaders):', {k: ('ok' if v['done'] else name(v['bad'])) for k, v in ctl.items()})

ok_ops, bad_ops, other = set(), {}, []
print('\n== probes')
for lab, (shader, clause, ops) in sorted(ops_of.items(), key=lambda x: int(x[0][1:-1])):
    kind = 'frag' if lab.endswith('f') else 'vt'
    r = res.get(lab, {}).get(kind)
    if r is None:
        print(f'{lab:5} {"NOT RUN":13} {shader} {clause}')
        continue
    verdict = 'ok' if r['done'] else name(r['bad'])
    if r['done']:
        ok_ops |= set(ops)
    elif r['bad'] in (0x51, 0x52):
        for o in ops:
            bad_ops.setdefault(o, []).append(lab)
    else:
        other.append(lab)
    if not r['done']:
        print(f'{lab:5} {verdict:13} {shader} {clause} us={r["us"]}: {" ".join(ops)}')
print(f'{sum(1 for l in ops_of if res.get(l, {}).get("frag" if l.endswith("f") else "vt", {}).get("done"))} '
      f'of {len(ops_of)} probes completed; other faults (decoded fine): {" ".join(other) or "none"}')
print('\n== instruction variants that never completed in a probe and faulted INVALID_ENC/TYPE_MISMATCH')
sus = {o: l for o, l in bad_ops.items() if o not in ok_ops}
for o in sorted(sus):
    print(f'SUSPECT {o:45} in {" ".join(sus[o])}')
print(f'{len(sus)} suspects; {len(ok_ops)} variants proven to decode')

print('\n== CoreAnimation pipelines (suite)')
for l in open(sys.argv[3]):
    f = l.split()
    if len(f) != 5:
        continue
    r = res.get(f[0], {})
    vt, fr = r.get('vt'), r.get('frag')
    s = lambda x: 'missing' if x is None else ('ok' if x['done'] else name(x['bad']))
    print(f'{f[0]:4} {f[1]:28} {f[2]:36} fmt {f[3]:3} blend {f[4]}: vt {s(vt):12} frag {s(fr)}')
