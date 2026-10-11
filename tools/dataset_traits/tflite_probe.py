# -*- coding: utf-8 -*-
"""直接解析 .tflite 的 flatbuffer，读实际输出张量形状 —— 不依赖 tensorflow / 注释。

用来核对 App 的 labels 行数、ModelRegistry 的 DIM 常量、模型真实输出维三者是否一致。
用法: python tflite_probe.py <a.tflite> [b.tflite ...]
"""
import struct
import sys


def u32(b, p):
    return struct.unpack_from('<I', b, p)[0]


def i32(b, p):
    return struct.unpack_from('<i', b, p)[0]


def u16(b, p):
    return struct.unpack_from('<H', b, p)[0]


def field(b, tab, i):
    """table 的第 i 个字段位置，缺省返回 0"""
    vt = tab - i32(b, tab)
    if 4 + i * 2 >= u16(b, vt):
        return 0
    off = u16(b, vt + 4 + i * 2)
    return tab + off if off else 0


def vec(b, pos):
    vp = pos + u32(b, pos)
    return vp + 4, u32(b, vp)


def ints(b, pos):
    s, n = vec(b, pos)
    return [i32(b, s + 4 * k) for k in range(n)]


def tables(b, pos):
    s, n = vec(b, pos)
    return [s + 4 * k + u32(b, s + 4 * k) for k in range(n)]


def text(b, pos):
    vp = pos + u32(b, pos)
    n = u32(b, vp)
    return b[vp + 4:vp + 4 + n].decode('utf-8', 'replace')


def probe(path):
    b = open(path, 'rb').read()
    model = u32(b, 0)
    subs = tables(b, field(b, model, 2))
    print(f"\n{path}")
    print(f"  subgraph 数 = {len(subs)}")
    for si, sg in enumerate(subs[:2]):
        tens = tables(b, field(b, sg, 0))
        outs = ints(b, field(b, sg, 2))
        ins = ints(b, field(b, sg, 1))
        print(f"  subgraph[{si}] 张量数={len(tens)} 输入={ins} 输出={outs}")
        for idx in ins + outs:
            t = tens[idx]
            shape = ints(b, field(b, t, 0))
            name = text(b, field(b, t, 3)) if field(b, t, 3) else ''
            kind = 'IN ' if idx in ins else 'OUT'
            print(f"    {kind} [{idx}] shape={shape} name={name}")


for p in sys.argv[1:]:
    try:
        probe(p)
    except Exception as e:
        print(f"{p}: 解析失败 {type(e).__name__}: {e}")
