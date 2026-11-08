#!/usr/bin/env python3
"""Per-pass pictures of snapshot frames -> one PNG per snapshot, the passes
side by side (left to right), each 45x92 scaled 3x, with a 4-pixel gap.

  mini2png.py LOG OUTPREFIX

Lines (s8_mini in s8mtl_commands.m): "MTL ps SNAP PASS: WxH of ...", then
"MTL ps SNAP PASS ROW PART: <base64, 3 characters a pixel>".
"""
import re, struct, sys, zlib

B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'


def png(path, w, h, rows):
    raw = b''.join(b'\0' + bytes(r) for r in rows)
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def main():
    log, prefix = sys.argv[1:]
    txt = open(log, 'rb').read().decode('latin1')
    heads = {}
    for s, p, w, h in re.findall(r'MTL ps (\d+) (\d+): (\d+)x(\d+) of', txt):
        heads.setdefault(int(s), {})[int(p)] = (int(w), int(h))
    data = {}
    for s, p, row, part, d in re.findall(r'MTL ps (\d+) (\d+) (\d+) (\d+): ([A-Za-z0-9+/]+)', txt):
        data.setdefault((int(s), int(p)), {}).setdefault(int(row), {})[int(part)] = d
    S, G = 3, 4
    for snap, passes in sorted(heads.items()):
        order = sorted(passes)
        w, h = max(v[0] for v in passes.values()), max(v[1] for v in passes.values())
        W = len(order) * (w * S + G)
        rows = [[40, 0, 40] * W for _ in range(h * S)]
        for k, p in enumerate(order):
            pw, ph = passes[p]
            rd = data.get((snap, p), {})
            for y in range(ph):
                line = ''.join(rd.get(y, {}).get(i, '') for i in sorted(rd.get(y, {})))
                for x in range(pw):
                    c = line[3 * x:3 * x + 3]
                    rgb = [B64.index(ch) * 255 // 63 for ch in c] if len(c) == 3 else [255, 0, 255]
                    for dy in range(S):
                        for dx in range(S):
                            X = k * (w * S + G) + x * S + dx
                            rows[y * S + dy][3 * X:3 * X + 3] = rgb
        path = f'{prefix}-snap{snap}.png'
        png(path, W, h * S, rows)
        print(f'{path}: passes {order}')


if __name__ == '__main__':
    main()
