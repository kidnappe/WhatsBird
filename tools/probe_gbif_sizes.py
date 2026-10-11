# -*- coding: utf-8 -*-
"""查清 GBIF 图片各档位到底多少像素（2026-10-09 挂了一整天的那件事）。

背景：C（部署增广微调 + 升分辨率重抓）要重下数据，`tools/build_dataset_gbif.py` 写死
`identifier.replace("/original.jpg", "/medium.jpg")`、存盘再缩到短边 256。
**但 medium 档到底是多少像素，此前从没验过**——而它决定 C 要不要花 15~25 小时、要不要 ~22GB 磁盘。

做法：取若干 region's dataset.csv 里的 occurrenceKey，回 GBIF API 取 media identifier，
分别请求 /medium.jpg 与 /original.jpg，下载后用 PIL 量尺寸（只量尺寸，不保存到磁盘）。

用法：python tools/probe_gbif_sizes.py [--n 12] [--regions huadong huazhong]
"""
import argparse
import io
import json
import os
import sys
import urllib.request

DATA = r"E:\code\code tools\data"
UA = "ObserveBird-probe/1.0 (contact: local dev)"
TIMEOUT = 30


def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def http_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def dims(raw):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        return im.size, len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--regions", nargs="+", default=["huadong", "huazhong"])
    a = ap.parse_args()

    from PIL import Image
    import csv

    rows_all = []
    for region in a.regions:
        import csv as c2
        with open(os.path.join(DATA, region, "dataset.csv"), encoding="utf-8") as f:
            rs = list(c2.DictReader(f))
        step = max(1, len(rs) // a.n)
        rows_all += [(region, rs[i]) for i in range(0, len(rs), step)][: a.n]
    print(f"抽样 {len(rows_all)} 条 occurrence\n")

    stat = {"medium": [], "original": []}
    for region, r in rows_all:
        key = r.get("occurrence_key") or r.get("gbif_key")
        if not key:
            continue
        sci = r["scientific_name"]
        try:
            j = http_json(f"https://api.gbif.org/v1/occurrence/{key}")
        except Exception as e:
            print(f"  {sci:<26} API 失败 {e}")
            continue
        media = (j.get("media") or [{}])[0]
        ident = media.get("identifier") or ""
        if not ident:
            continue
        line = f"  {sci:<26}"
        for tag, url in (
            ("medium", ident.replace("/original.jpg", "/medium.jpg")),
            ("original", ident),
        ):
            try:
                (w, h), nbytes = dims(http_bytes(url))
                stat[tag].append((w, h, nbytes))
                line += f"  {tag}={w}x{h}({nbytes//1024}KB)"
            except Exception as e:
                line += f"  {tag}=失败({type(e).__name__})"
        print(line, flush=True)

    print("\n=== 汇总 ===")
    for tag, v in stat.items():
        if not v:
            print(f"{tag}: 无样本")
            continue
        shorts = sorted(min(w, h) for w, h, _ in v)
        longs = sorted(max(w, h) for w, h, _ in v)
        kb = sorted(n // 1024 for _, _, n in v)
        n = len(v)
        print(f"{tag:<9} n={n}  短边 中位={shorts[n//2]} 范围 {shorts[0]}~{shorts[-1]}   "
              f"长边 中位={longs[n//2]}   文件 中位={kb[n//2]}KB")

    if stat["medium"]:
        sh = sorted(min(w, h) for w, h, _ in stat["medium"])
        print(f"\n结论：medium 档短边中位 ≈ {sh[len(sh)//2]}px，"
              f"而当前训练库存的是短边 256（build_dataset_gbif.py:285 强制缩放）。")
        print(f"  → 换 medium 约等于把有效像素放大 {sh[len(sh)//2]/256:.1f} 倍"
              f"（按短边比），实际收益还要看裁剪后的实际放大倍数。")


if __name__ == "__main__":
    raise SystemExit(main())