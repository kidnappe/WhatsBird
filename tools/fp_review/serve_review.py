# -*- coding: utf-8 -*-
r"""真 FP 人工复核看图器（E-1，2026-10-10）。

## 要回答的问题

**真实观鸟场景的 FP/图到底是多少？** 现役唯一的数 0.86 是 COCO 口径且含七成漏标假象，
验收线 ≤0.3 也是 COCO 口径（`docs/handoff/root/PRIORITIES-20261009.md` §10.1/§11.1）。
真实值从未被测过——这是 E（检测器难例回流）做不做的唯一判据。

## 为什么抽样而不是全看

20,000 条不可能全看。**对 20,000 条做简单随机抽样**（固定种子），
人只判抽样里的那些 ⇒ **估计量无偏**，直接得到真 FP/图 + Wilson 置信区间。
抽样内部按 `p_not_bird` 降序排，只是为了让人先看最可疑的（提高判读效率，不影响估计）。

## 人看什么

每条给两张图 + 三个信号：

  左：**设备视角** 224×224 —— 端上分类器真正吃的那张（原样，不美化）
  右：**全分辨率外扩 crop** —— 给复核人看清细节用
  ① `p_not_bird`  拒识门判非鸟概率（唯一二元信号）
  ② `clf_top1 / clf_prob / clf_match`  该区分类器 top1 + 是否等于真值
     ⚠️ 五区分类器**只含鸟种**，永远输出一个鸟名，**不能**用它判"是不是鸟"；
     但"高置信喊出别的种"是框歪/真值错的强提示
  ③ `det_score`  检测器置信度

## 键位

  空格 / →  是鸟（真检）      X  不是鸟（误检）      S  存疑（不算入分子）
  ←  上一张                  U  撤销上一张            单击左图 = 1:1 原尺寸

判定实时落 `review_fp.json`（原子写）。`serve_review.py --summary` 可随时打印当前估计。

Usage:
    python tools/fp_review/serve_review.py                 # 起服务，默认抽样 400
    python tools/fp_review/serve_review.py --sample 600
    python tools/fp_review/serve_review.py --summary       # 只打印当前估计，不起服务
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ROOT = r"E:\code\observebird"
DATA = r"E:\code\code tools\data"
SCORED_DIR = os.path.join(ROOT, "data", "fp_review")
REVIEW = os.path.join(SCORED_DIR, "review_fp.json")
FRAC = 2.0
PRE_SHORT, CROP = 256, 224

_lock = threading.Lock()
_cache = {}


def latest_scored():
    fs = sorted(glob.glob(os.path.join(SCORED_DIR, "scored_*.json")))
    if not fs:
        sys.exit("没有 scored_*.json，先跑 score_boxes.py")
    return fs[-1]


def load_scored():
    p = latest_scored()
    if _cache.get("path") != p:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        _cache["path"], _cache["d"] = p, d
    return _cache["d"]


def wilson(k, n, z=1.96):
    """Wilson 区间：小样本下比正态近似稳（n=400 时差别明显）。"""
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def build_sample(n_sample, seed):
    d = load_scored()
    rows = d["rows"]
    rnd = random.Random(seed)
    pool = list(range(len(rows)))
    rnd.shuffle(pool)
    idx = sorted(pool[:n_sample], key=lambda i: -rows[i]["p_not_bird"])  # 抽样内降序=判读效率
    # ⚠️ 必须带上 "box"（[x0,y0,x1,y2] 原图像素坐标）：/img 端点靠它重新裁图。
    #    初版漏了这个键 → KeyError → 连接被直接断开（浏览器表现为图片裂/转圈）。
    #    加 "box" 时也加个断言，别再靠线上报错发现。
    carry = ("region", "file", "sci", "det_score", "p_not_bird",
             "clf_top1", "clf_prob", "clf_match", "box")
    missing = [k for k in carry if k not in rows[0]]
    if missing:
        sys.exit(f"scored JSON 缺字段 {missing}；重新跑 score_boxes.py")
    return [{"sid": i, **{k: rows[i][k] for k in carry}} for i in idx]


def load_verdicts():
    try:
        with open(REVIEW, encoding="utf-8") as f:
            return json.load(f).get("verdicts", {})
    except Exception:
        return {}


def save_verdict(sid, v):
    with _lock:
        try:
            with open(REVIEW, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"verdicts": {}}
        if v is None:
            data.get("verdicts", {}).pop(str(sid), None)
        else:
            data.setdefault("verdicts", {})[str(sid)] = v
        data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(SCORED_DIR, exist_ok=True)
        tmp = REVIEW + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, REVIEW)


def _expand(bx, W, H, f=FRAC):
    x0, y0, x1, y1 = bx
    w, h = x1 - x0, y1 - y0
    nx0, ny0 = max(0, int(x0 - w * f)), max(0, int(y0 - h * f))
    nx1, ny1 = min(W, int(x1 + w * f)), min(H, int(y1 + h * f))
    return nx0, ny0, max(nx1, nx0 + 1), max(ny1, ny0 + 1)


def _find(sample, sid):
    return sample[0] if sample else None


def _row(sample, sid):
    return next((r for r in sample if r["sid"] == sid), None)


def crop_jpeg(sample, sid, mode):
    """mode='dev' 设备视角 224；mode='full' 全分辨率外扩 crop（放大便于肉眼看）。"""
    r = _row(sample, sid)
    if not r:
        return None
    path = os.path.join(DATA, r["region"], "images", r["file"])
    img = cv2.imread(path)
    if img is None:
        return None
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _expand(r["box"], W, H)
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return None
    if mode == "dev":                      # 端上：短边缩到256 → 中心裁224（BGR）
        sh = min(c.shape[:2])
        if sh:
            s = PRE_SHORT / sh
            c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))))
        ch, cw = c.shape[:2]
        t, l = max(0, (ch - CROP) // 2), max(0, (cw - CROP) // 2)
        c = c[t:t + CROP, l:l + CROP]
        if c.shape[0] != CROP or c.shape[1] != CROP:
            c = cv2.copyMakeBorder(c, 0, CROP - c.shape[0], 0, CROP - c.shape[1],
                                   cv2.BORDER_REPLICATE)
        c = cv2.resize(c, (448, 448), interpolation=cv2.INTER_NEAREST)  # 放大好读
    else:                                   # 全分辨率：只放大，不做 256/224
        m = max(c.shape[:2])
        if m < 384:
            s = 384 / m
            c = cv2.resize(c, (round(c.shape[1] * s), round(c.shape[0] * s)),
                           interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(c, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), (0, 200, 255), 2)
    ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else None


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>真 FP 人工复核 · E-1</title>
<style>
 body{background:#0f0f0f;color:#ddd;font-family:system-ui,"Microsoft YaHei",sans-serif;
      margin:0;padding:14px 18px 92px;max-width:1420px;margin-inline:auto}
 h1{font-size:17px;color:#fff;margin:0 0 8px}
 h1 small{color:#888;font-weight:normal;margin-left:8px}
 .cnt{display:flex;gap:14px;font-size:13px;color:#999;flex-wrap:wrap;align-items:center;margin:6px 0 10px}
 .cnt b{font-size:16px}
 .cnt .ok b{color:#4ade80}.cnt .bad b{color:#f87171}.cnt .s b{color:#fbbf24}
 .cnt .prog{background:#14321f;border:1px solid #14532d;border-radius:10px;padding:4px 14px;color:#4ade80}
 .cnt .prog b{font-size:20px}
 .est{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:4px 14px;color:#7dd3fc}
 .est b{font-size:17px;color:#7dd3fc}
 .warn{background:#3a2d08;border:1px solid #78591a;border-radius:10px;padding:4px 14px;color:#fbbf24}
 .warn b{font-size:17px}
 .warn2{background:#3a1212;border:1px solid #7f1d1d;border-radius:10px;padding:4px 14px;color:#f87171;font-size:12.5px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 .card{background:#1c1c1c;border-radius:12px;padding:10px}
 .card .t{font-size:12.5px;color:#aaa;margin-bottom:6px}
 .card img{width:100%;border-radius:8px;display:block;background:#000}
 .dev img{cursor:zoom-in}
 .dev img.z{width:auto;max-width:none;height:auto}
 .sig{margin-top:10px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
 .sg{background:#1c1c1c;border-radius:10px;padding:8px 10px;font-size:12px;color:#888}
 .sg b{display:block;font-size:15px;color:#eee;margin-top:2px}
 .sg.hot b{color:#f87171}.sg.mid b{color:#fbbf24}.sg.good b{color:#4ade80}
 .note{margin-top:8px;font-size:12.5px;color:#999}
 .note .bad{color:#f87171}.note .good{color:#4ade80}
 #vb{margin-top:10px;font-size:17px;font-weight:700;padding:8px 16px;border-radius:10px;display:none}
 #vb.on{display:inline-block}
 #vb.ok{background:#14321f;color:#4ade80}#vb.bad{background:#3a1212;color:#f87171}
 #vb.s{background:#3a2d08;color:#fbbf24}
 #legend{position:fixed;left:0;right:0;bottom:0;background:#191919ee;border-top:1px solid #333;
         padding:9px 18px;font-size:13px;color:#aaa;display:flex;gap:16px;flex-wrap:wrap;
         backdrop-filter:blur(4px)}
 kbd{background:#262626;border:1px solid #444;border-bottom-width:3px;border-radius:6px;
     padding:1px 8px;font-family:Consolas,monospace;font-size:12px;color:#eee}
 .done{font-size:16px;color:#4ade80;padding:40px;text-align:center}
</style></head><body>
<h1>真 FP 人工复核 · E-1 <small id="cur"></small></h1>
<div class="cnt" id="cnt"></div>
<div class="grid">
  <div class="card dev"><div class="t">设备视角 224×224（端上分类器真正吃的那张）· 单击 1:1</div>
    <img id="dev"></div>
  <div class="card"><div class="t">全分辨率外扩 crop（BOX_EXPAND_RATIO=4.0，细节参考）</div>
    <img id="full"></div>
</div>
<div class="sig" id="sig"></div>
<div class="note" id="note"></div>
<div id="vb"></div>
<div id="legend">
  <span><kbd>空格</kbd> 是鸟(真检)</span><span><kbd>X</kbd> 不是鸟(误检)</span>
  <span><kbd>S</kbd> 存疑(这张判不了)</span>
  <span><kbd>→</kbd>/<kbd>←</kbd> 只翻页·<b style="color:#fbbf24">不判定</b></span>
  <span><kbd>U</kbd> 撤销上一张</span>
  <span style="color:#888">方向键已刻意不绑判定，避免翻页时误记一条</span>
</div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const VN={yes:'✅ 是鸟（真检）',no:'❌ 不是鸟（误检）',maybe:'🤔 存疑'};
let S=[],V={},idx=0,undo=[];
async function boot(){
  const st=await(await fetch('/api/state')).json();
  S=st.sample; V=st.verdicts||{};
  idx=S.findIndex(r=>!V[r.sid]); if(idx<0)idx=0;
  render();
}
function stats(){
  const done=S.filter(r=>V[r.sid]);
  const yes=done.filter(r=>V[r.sid]==='yes').length;
  const no =done.filter(r=>V[r.sid]==='no').length;
  const mb =done.filter(r=>V[r.sid]==='maybe').length;
  const J=yes+no+mb;                       // 总判定（含存疑）
  // ⚠️ 真 FP 给**区间**，不给点估计：存疑全当鸟 = 下界，存疑全当非鸟 = 上界。
  //   存疑为 0 时两端重合；存疑越多区间越宽——这才是诚实的表达。
  const lo=J? no/J : 0;
  const hi=J? (no+mb)/J : 0;
  // 点估计 + Wilson 95%CI：**只算判得掉的那些**（不含存疑），供对照
  const n=Math.max(1,yes+no);
  const fp=no/n, z=1.96, d=1+z*z/n;
  const c=(fp+z*z/(2*n))/d, h=z*Math.sqrt(fp*(1-fp)/n+z*z/(4*n*n))/d;
  const mbRate=J? mb/J : 0;
  return {done:S.length,doneN:done.length,yes,no,mb,fp,
          wlo:Math.max(0,c-h),whi:Math.min(1,c+h),lo,hi,n,J,mbRate};
}
function estHtml(t){
  if(!t.J) return `<span class="est">判一条就出估计</span>`;
  const pt = t.n>=20
    ? `点估计 <b>${(t.fp*100).toFixed(2)}%</b>（95%CI ${(t.wlo*100).toFixed(1)}~${(t.whi*100).toFixed(1)}%，n=${t.n}，不含存疑）`
    : `点估计样本还太少（n=${t.n}），先看区间`;
  const narrow = Math.abs(t.hi-t.lo)<0.02;
  return `<span class="est">真 FP <b>${(t.lo*100).toFixed(1)}% ~ ${(t.hi*100).toFixed(1)}%</b>`
    +`<br><span style="font-size:12px">（存疑全当鸟 ↔ 全当非鸟） · ${pt}</span></span>`
    +(t.mbRate>0.10?`<span class="warn">⚠ 存疑率 <b>${(t.mbRate*100).toFixed(1)}%</b> >10%`
      +` —— 样本偏难，此估计不可信；请说明是图本身糊还是框有问题</span>`:'')
    +(!narrow?`<span class="warn2">区间较宽：存疑 ${t.mb} 条还没定</span>`:'');
}
function render(){
  const t=stats();
  $('cnt').innerHTML=
     `<span class="prog">已判 <b>${t.doneN}</b>/${t.done}（抽样 ${t.done} 张）</span>`
    +`<span class="ok">是鸟 <b>${t.yes}</b></span>`
    +`<span class="bad">误检 <b>${t.no}</b></span>`
    +`<span class="s">存疑 <b>${t.mb}</b></span>`
    +estHtml(t);
  const r=S[idx];
  if(!r){$('dev').removeAttribute('src');$('full').removeAttribute('src');
    $('sig').innerHTML='';$('note').textContent='';$('cur').textContent='';
    const v=$('vb');v.className='';v.textContent='这一批都判完了 ✓';v.className='on s';
    $('cnt').innerHTML+=`<span class="est">最终：真 FP ∈ <b>${(t.lo*100).toFixed(1)}% ~ ${(t.hi*100).toFixed(1)}%</b></span>`;
    return;}
  $('cur').textContent=`${idx+1}/${S.length} · ${r.region} · ${r.file}`;
  $('dev').src='/img?sid='+r.sid+'&m=dev';
  $('full').src='/img?sid='+r.sid+'&m=full';
  const hot=r.p_not_bird>=0.5, hot2=r.p_not_bird>=0.2;
  $('sig').innerHTML=
     `<div class="sg ${hot?'hot':hot2?'mid':'good'}">拒识门 P(非鸟)<b>${r.p_not_bird.toFixed(3)}</b></div>`
    +`<div class="sg ${r.clf_match?'good':'hot'}">分类器 top1<b>${esc(r.clf_top1)} ${(r.clf_prob*100).toFixed(0)}%</b></div>`
    +`<div class="sg ${r.clf_match?'good':'mid'}">与真值一致<b>${r.clf_match?'是':'否'}</b></div>`
    +`<div class="sg">检测器分数<b>${r.det_score.toFixed(3)}</b></div>`;
  $('note').innerHTML=`真值（frozen_val）：<b class="good">${esc(r.sci)}</b> `
    +`· 分类器 top1：<span class="${r.clf_match?'good':'bad'}">${esc(r.clf_top1)}</span>`
    +` · <span class="${r.clf_match?'good':'bad'}">${r.clf_match?'一致':'不一致 → 框歪/真值错/拍到别的鸟，重点看'}</span>`
    +(r.clf_prob<0.3?` <span style="color:#fbbf24">（分类器也不确定）</span>`:'');
  renderBadge(r);
  const nx=S[idx+1]; if(nx){const a=new Image();a.src='/img?sid='+nx.sid+'&m=dev';}
}
function renderBadge(r){
  const v=V[r.sid],b=$('vb');
  if(!v){b.className='';b.textContent='';return;}
  b.className='on '+(v==='yes'?'ok':v==='no'?'bad':'s'); b.textContent=VN[v];
}
function judge(kind){
  const r=S[idx]; if(!r)return;
  undo.push({sid:r.sid,prev:V[r.sid],idx});
  V[r.sid]=kind;
  fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sid:r.sid,verdict:kind})});
  const nxt=S.findIndex((x,i)=>i>idx&&!V[x.sid]);
  idx = nxt>=0 ? nxt : Math.min(idx+1,S.length-1);
  render();
}
function doUndo(){
  const u=undo.pop(); if(!u)return;
  if(u.prev===undefined)delete V[u.sid];
  else V[u.sid]=u.prev;
  fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sid:u.sid,verdict:u.prev||null})});
  idx=u.idx; render();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  // ⚠️ 判定只绑 空格 / X / S 三键。
  //   **方向键刻意不绑判定**（2026-10-10 用户要求）：翻页时不可能顺手记一条。
  //   初版把 ArrowRight 也绑成「是鸟」，翻个页就白记一条——已改。
  if(e.code==='Space'){e.preventDefault();judge('yes');}
  else if(e.key==='x'||e.key==='X'){e.preventDefault();judge('no');}
  else if(e.key==='s'||e.key==='S'){e.preventDefault();judge('maybe');}
  else if(e.key==='u'||e.key==='U')doUndo();
  else if(e.key==='ArrowRight'||e.key==='ArrowDown'){e.preventDefault();step(1);}
  else if(e.key==='ArrowLeft'||e.key==='ArrowUp'){e.preventDefault();step(-1);}
});
/** 纯导航：跳到下一张/上一张，**不写任何判定**。优先跳未看过的，其次老实翻一格。 */
function step(dir){
  const nxt=dir>0?S.findIndex((r,i)=>i>idx&&!V[r.sid]):-1;
  if(dir>0&&nxt>=0){idx=nxt;}
  else{idx=Math.max(0,Math.min(S.length-1,idx+dir));}
  render();
}
$('dev').onclick=()=>{$('dev').classList.toggle('z');};
boot();
</script></body></html>"""

PNG1 = None


class Handler(BaseHTTPRequestHandler):
    def _json(self, o, code=200):
        b = json.dumps(o, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _bytes(self, b, ct):
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        # ⚠️ 整段包 try：未捕获异常会让 socketserver 直接断开连接，
        #    浏览器只看到"图片裂掉"，看不到任何原因——初版就是这么查了半天才定位到。
        try:
            self._route(parse_qs(urlparse(self.path).query),
                        urlparse(self.path).path)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _route(self, qs, path):
        st = _state()
        if path == "/api/state":
            self._json({"sample": st["sample"], "verdicts": load_verdicts(),
                        "stats": st["stats"], "meta": st["meta"]})
        elif path == "/img":
            try:
                sid = int((qs.get("sid") or ["-1"])[0])
            except ValueError:
                sid = -1
            mode = (qs.get("m") or ["dev"])[0]
            body = crop_jpeg(st["sample"], sid, mode)
            if not body:
                self._json({"error": f"crop 生成失败 sid={sid} m={mode}"}, 404)
                return
            self._bytes(body, "image/jpeg")
        else:
            b = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    def do_POST(self):
        from urllib.parse import urlparse
        if urlparse(self.path).path != "/api/verdict":
            self._json({"error": "unknown"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            sid = str(body.get("sid"))
            v = body.get("verdict")
            save_verdict(int(sid), v if v in ("yes", "no", "maybe") else None)
            self._json({"ok": True})
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *a):
        pass


def _state(n_sample=400, seed=20261010):
    key = (n_sample, seed)
    if _cache.get("statekey") != key:
        sample = build_sample(n_sample, seed)
        d = load_scored()
        _cache["statekey"] = key
        _cache["sample"] = sample
        _cache["meta"] = {"scored": latest_scored(), "n_total": d["n"],
                          "frac": d.get("frac"), "device_ratio": d.get("device_ratio"),
                          "sample": len(sample), "seed": seed}
    v = load_verdicts()
    yes = sum(1 for r in _cache["sample"] if v.get(str(r["sid"])) == "yes")
    no = sum(1 for r in _cache["sample"] if v.get(str(r["sid"])) == "no")
    maybe = sum(1 for r in _cache["sample"] if v.get(str(r["sid"])) == "maybe")
    n = max(1, yes + no)
    lo, hi = wilson(no / n, n)
    _cache["stats"] = {"yes": yes, "no": no, "maybe": maybe, "fp": no / n,
                       "lo": lo, "hi": hi, "n": yes + no,
                       "done": yes + no + maybe}
    return {"sample": _cache["sample"], "meta": _cache["meta"], "stats": _cache["stats"]}


def print_summary():
    st = _state()
    s, m = st["stats"], st["meta"]
    print(f"素材   : {m['scored']}")
    print(f"全集   : {m['n_total']} 条（frac={m['frac']} → 端上 ratio={m['device_ratio']}）")
    print(f"抽样   : {m['sample']} 张（seed={m['seed']}，简单随机 ⇒ 估计量无偏）")
    print(f"已判   : 是鸟 {s['yes']} / 误检 {s['no']} / 存疑 {s['maybe']}")
    if s["n"]:
        print(f"真 FP  : {s['fp']*100:.2f}%  (95%CI {s['lo']*100:.1f}~{s['hi']*100:.1f}%, n={s['n']})")
        print(f"对照   : 现役文档口径 0.86/图（COCO，含七成漏标假象）；验收线 ≤0.3")
    else:
        print("真 FP  : 还没判，先开服务 http://127.0.0.1:8789")


def main():
    global PNG1
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8789)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--summary", action="store_true", help="只打印当前估计，不起服务")
    a = ap.parse_args()
    _state(a.sample, a.seed)
    if a.summary:
        print_summary()
        return 0
    import base64
    PNG1 = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42u3BAQ0AAADCoPdPbQ8H"
        b"FAAAAAAAAAAAAAAAAAAAAAAA8Gq8AWJfn3IAAAAASUVORK5CYII=")
    print_summary()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"\n看图器: http://127.0.0.1:{a.port}\n判定落盘: {REVIEW}\nCtrl-C 停", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())