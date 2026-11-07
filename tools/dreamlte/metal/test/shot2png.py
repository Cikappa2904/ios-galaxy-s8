#!/usr/bin/env python3
"""Screen thumbnails from an S8 boot log -> PNG files (no dependencies).

  shot2png.py LOG OUTPREFIX

The driver logs the display composite a few times (s8_shot in
s8mtl_commands.m): "MTL shot K: WxH of ..." then rows as
"MTL shot K ROW PART: <base64, 3 characters a pixel, 6 bits per channel>".
Each shot K becomes OUTPREFIX-K.png, scaled up 4x; missing rows are
magenta.
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
    heads = {int(k): (int(w), int(h)) for k, w, h in re.findall(r'MTL shot (\d+): (\d+)x(\d+) of', txt)}
    parts = {}
    for k, row, part, data in re.findall(r'MTL shot (\d+) (\d+) (\d+): ([A-Za-z0-9+/]+)', txt):
        parts.setdefault(int(k), {}).setdefault(int(row), {})[int(part)] = data
    for k, (w, h) in sorted(heads.items()):
        rows, missing = [], 0
        for y in range(h):
            p = parts.get(k, {}).get(y, {})
            data = ''.join(p[i] for i in sorted(p))
            px = []
            for x in range(w):
                c = data[3 * x:3 * x + 3]
                if len(c) == 3:
                    px += [B64.index(ch) * 255 // 63 for ch in c]
                else:
                    px += [255, 0, 255]
            if len(data) < 3 * w:
                missing += 1
            row = []
            for i in range(0, len(px), 3):
                row += px[i:i + 3] * 4
            rows += [row] * 4
        path = f'{prefix}-{k}.png'
        png(path, w * 4, h * 4, rows)
        print(f'{path}: {w}x{h} (x4), {missing} incomplete rows')


if __name__ == '__main__':
    main()
