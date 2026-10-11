# -*- coding: utf-8 -*-
"""人审闭环：把 review.json 的判定应用到伪标签产物（jsonl + labels txt）。

规则（2026-10-06 与用户对齐）：
  boxes_bad（用户亲手 ✕ 的框）→ 从标签里删除
  manual_boxes（用户手绘的框）→ 追加为人工标签（score=1.0, src=manual）
  verdict=g（整张坏图）→ 写入 excluded_files.txt，make_train_json 整图剔除
  agent 层（review_agent.json）→ 只提示不动标签
同一源图被多轮样张判过 → 合并（删框取并集、手绘取并集、g 任一即剔）。
索引稳定性：判定里的框下标基于 jsonl 当时的 dets 顺序，本脚本运行前 jsonl 未被
任何人改写过（伪标签跑完后只读），顺序稳定可依赖。

Usage: python apply_review.py
产物：patched jsonl/labels + excluded_files.txt（先备份 .bak_prereview）
"""
import json
import os
import shutil
import sys
import time

OUT = r"E:\code\code tools\data\detector_b\pseudo_label"
DATA = r"E:\code\code tools\data\huabei"
JSONL = os.path.join(OUT, "pseudo_dets.jsonl")
REVIEW = os.path.join(OUT, "review.json")
LABELS = os.path.join(OUT, "labels")
EXCL = os.path.join(OUT, "..", "excluded_files.txt")
MIN_BOX_PX = 4


def main():
    shutil.copy2(JSONL, JSONL + ".bak_prereview")
    entries, order = {}, []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            fn = j["file"]
            if fn not in entries:
                order.append(fn)
            entries[fn] = j  # 后行覆盖（防历史追加重复）
    stems = {os.path.splitext(fn)[0] for fn in entries}
    stem2file = {os.path.splitext(fn)[0]: fn for fn in entries}

    def stems_of(key):
        toks = os.path.splitext(key)[0].split("_")
        return ["_".join(toks[k:]) for k in range(1, len(toks))]

    # verdict 键 → 源图 file（可能多个样张键对应同一源图）
    rv = json.load(open(REVIEW, encoding="utf-8")).get("verdicts", {})
    by_stem = {}
    unmapped = []
    for key, e in rv.items():
        hit = None
        for cand in stems_of(key):
            if cand in stems:
                hit = cand
                break
        if hit is None:
            unmapped.append(key)
            continue
        by_stem.setdefault(hit, []).append(e)

    n_del_box = n_add_box = n_g = n_patched = 0
    excluded = []
    for stem, evs in sorted(by_stem.items()):
        fn = stem2file[stem]
        entry = entries[fn]
        dets = entry.get("dets") or []
        h, w = entry.get("h"), entry.get("w")
        del_idx, manual, is_g = set(), [], False
        for e in evs:
            del_idx.update(e.get("boxes_bad") or [])
            manual += e.get("manual_boxes") or []
            is_g = is_g or e.get("verdict") == "g"
        if is_g:
            excluded.append(fn)
            n_g += 1
            continue
        if not del_idx and not manual:
            continue
        # 先删（按原下标倒序 pop），再补手绘
        kept = [d for i, d in enumerate(dets) if i not in del_idx]
        n_del_box += len(dets) - len(kept)
        for b in manual:
            x1, y1, x2, y2 = (float(c) for c in b)
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                continue
            kept.append({"score": 1.0, "x1": round(x1, 1), "y1": round(y1, 1),
                         "x2": round(x2, 1), "y2": round(y2, 1), "src": "manual",
                         "clf_label": "", "clf_prob": 1.0, "low_conf": False})
            n_add_box += 1
        kept = [d for d in kept
                if d["x2"] - d["x1"] >= MIN_BOX_PX and d["y2"] - d["y1"] >= MIN_BOX_PX]
        entry["dets"] = kept
        with open(os.path.join(LABELS, stem + ".txt"), "w") as f:
            for d in kept:
                cx, cy = (d["x1"] + d["x2"]) / 2 / w, (d["y1"] + d["y2"]) / 2 / h
                f.write(f"0 {cx:.6f} {cy:.6f} {(d['x2'] - d['x1']) / w:.6f} "
                        f"{(d['y2'] - d['y1']) / h:.6f}\n")
        n_patched += 1

    with open(JSONL, "w", encoding="utf-8") as f:
        for fn in order:
            f.write(json.dumps(entries[fn], ensure_ascii=False) + "\n")
    with open(EXCL, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(excluded)))
    print(f"闭环完成：改写 {n_patched} 图（删框 {n_del_box}，补手绘框 {n_add_box}），"
          f"剔坏图 {n_g}，未映射判定 {len(unmapped)}（样张已不存在的旧记录，无行动）")
    print(f"excluded_files.txt: {len(excluded)} 张 → make_train_json 整图剔除")
    return 0


if __name__ == "__main__":
    sys.exit(main())