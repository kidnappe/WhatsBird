# -*- coding: utf-8 -*-
r"""用 141 缺口种真值图实测各家 VLM 的鸟类识别能力（硅基流动 OpenAI 兼容接口）。

基线：cn373 在同分布图上 top1 52.86%（eval_cn373_gap.py，n=700）。
判分口径：学名规范化匹配（lower/去空格）；未匹配的回答单独落盘供人工核查
（VLM 可能输出同物异名，模糊匹配不进去不代表错——report 里有未匹配清单）。

断点续跑：结果按「模型|图」追加进 jsonl，重跑自动跳过已答。

Usage:
  set SILICONFLOW_API_KEY=sk-xxx
  python eval_vlm_gap.py --n 150 --per 2
  python eval_vlm_gap.py --models Qwen/Qwen3-VL-32B-Instruct --n 50   # 单模型补测
"""
import argparse
import base64
import csv
import io
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

GAP_CSV = r"E:\code\code tools\data\national_gap\dataset.csv"
GAP_IMG = r"E:\code\code tools\data\national_gap\images"
OUT = r"E:\code\code tools\data\national\vlm_eval"
DEFAULT_MODELS = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "zai-org/GLM-4.5V",
]
PROMPT = (
    "这是一张在中国拍摄的鸟类照片。拍摄位置约北纬{lat:.1f}度、东经{lon:.1f}度，"
    "月份{month}月。请仔细辨别羽色、体型、喙形等特征识别鸟种。"
    '只输出 JSON：{{"scientific_name": "规范拉丁学名", "confidence": 0到1, '
    '"reason": "一句话依据"}}，不要输出其他内容。'
)
PROMPT_LIST = (
    "这是一张在中国拍摄的鸟类照片。拍摄位置约北纬{lat:.1f}度、东经{lon:.1f}度，"
    "月份{month}月。请从下列候选拉丁学名中选出最可能的鸟种（若都不像，选视觉上"
    '最接近的）。只输出 JSON：{{"scientific_name": "...", "confidence": 0到1, '
    '"reason": "一句话依据"}}。\n候选：{names}'
)


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def encode_image(path, max_side=768):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        r = max_side / max(w, h)
        img = img.resize((int(w * r), int(h * r)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_answer(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"scientific_name": text.strip()[:80], "raw": text.strip()[:200]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"scientific_name": "", "raw": text.strip()[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--per", type=int, default=2, help="每类抽样上限")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true",
                    help="受限模式：prompt 给出 373 候选学名（与自由回忆对照）")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    tag = "_list" if a.list else ""

    names_str = ""
    if a.list:
        sps = [l.split("\t")[0].strip() for l in open(
            r"E:\code\observebird\android\app\src\main\assets\models\labels_birdnet373.txt",
            encoding="utf-8") if l.strip()]
        names_str = ", ".join(sps)

    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        print("请先 set SILICONFLOW_API_KEY=sk-xxx"); return 1
    from openai import OpenAI
    client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=key)

    rows = list(csv.DictReader(open(GAP_CSV, encoding="utf-8")))
    by_sp = defaultdict(list)
    for r in rows:
        p = os.path.join(GAP_IMG, r["file"])
        if os.path.exists(p):
            by_sp[r["scientific_name"]].append((r, p))
    rng = random.Random(7)
    sample = []
    for s in sorted(by_sp):
        rs = by_sp[s][:]; rng.shuffle(rs)
        sample += [(s, r, p) for r, p in rs[:a.per]]
    rng.shuffle(sample); sample = sample[:a.n]
    print(f"评测集: {len(sample)} 张 / {len({s for s,_,_ in sample})} 种")

    os.makedirs(OUT, exist_ok=True)
    done = set()
    for f in os.listdir(OUT):
        if f.endswith(tag + ".jsonl") and (tag or not f.endswith("_list.jsonl")):
            for line in open(os.path.join(OUT, f), encoding="utf-8"):
                try:
                    j = json.loads(line); done.add((j["model"], j["file"]))
                except json.JSONDecodeError:
                    pass

    def ask(job):
        model, s, r, p = job
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(p)}},
                    {"type": "text", "text": (PROMPT_LIST if a.list else PROMPT).format(
                        lat=float(r.get("lat") or 35), lon=float(r.get("lon") or 110),
                        month=int(r.get("month") or 6), names=names_str)},
                ]}],
                temperature=0, max_tokens=300, timeout=60)
            ans = parse_answer(resp.choices[0].message.content or "")
            usage = resp.usage.total_tokens if resp.usage else 0
            return {"model": model, "file": r["file"], "gt": s,
                    "pred": norm(ans.get("scientific_name")),
                    "conf": ans.get("confidence"), "reason": ans.get("reason", ""),
                    "tokens": usage, "t_s": 0}
        except Exception as e:
            time.sleep(2)
            return {"model": model, "file": r["file"], "gt": s, "pred": "",
                    "error": str(e)[:150], "tokens": 0, "t_s": 0}

    for model in models:
        jobs = [(model, s, r, p) for s, r, p in sample if (model, r["file"]) not in done]
        path = os.path.join(OUT, model.replace("/", "_") + tag + ".jsonl")
        print(f"\n== {model}: 待测 {len(jobs)} 张")
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
            for i, rec in enumerate(ex.map(ask, jobs), 1):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                if i % 25 == 0:
                    print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")

    print("\n========== 汇总 ==========")
    for model in models:
        path = os.path.join(OUT, model.replace("/", "_") + tag + ".jsonl")
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        recs = [r for r in recs if r["model"] == model]
        n = len(recs); ok = sum(1 for r in recs if r.get("pred") == norm(r["gt"]))
        unm = [r for r in recs if r.get("pred") and r["pred"] != norm(r["gt"])]
        tok = sum(r.get("tokens", 0) for r in recs)
        err = sum(1 for r in recs if r.get("error"))
        print(f"{model}: top1 {ok}/{n} = {ok/max(n,1)*100:.1f}%  "
              f"未匹配 {len(unm)-0}  错误 {err}  tokens {tok}")
        for r in unm[:5]:
            print(f"    gt={r['gt']}  pred={r['pred']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
