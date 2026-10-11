# -*- coding: utf-8 -*-
r"""伪标签人审看图器：quality/ 抽检样张键盘流点选，判定实时落盘 review.json。
打开 http://127.0.0.1:8788 。键位（底部常驻提示条，按图类自动切换语义）：
  空格/→  对/真没鸟        X  框到非鸟 / 有鸟没框(zero)
  B       是鸟但框不准      G  整张坏图
  Z       有鸟没框(zero)    U  撤销上一张     ←  上一张
  点击主图 切 1:1 原尺寸
多鸟图右侧每个框有独立 ✕框错，可只否掉具体框。判定即存（原子写），
主窗口 agent 读 review.json 回溯 jsonl/labels 修标签。

Usage: python review_ui.py [--port 8788]
"""
import base64
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

QUALITY = r"E:\code\code tools\data\detector_b\pseudo_label\quality"
JSONL = os.path.join(QUALITY, "..", "pseudo_dets.jsonl")
IMAGES = r"E:\code\code tools\data\huabei\images"
REVIEW = os.path.join(QUALITY, "..", "review.json")
AGENT_REVIEW = os.path.join(QUALITY, "..", "review_agent.json")
CAT_ORDER = {"low": 0, "mid": 1, "hard": 2, "zero": 3, "high": 4, "sp": 5}

_lock = threading.Lock()
_jsonl_cache = {"mtime": None, "index": {}}


def jsonl_index():
    """stem -> jsonl 行（dets 坐标等）。append 型文件按 mtime 缓存。"""
    mt = os.path.getmtime(JSONL) if os.path.exists(JSONL) else None
    with _lock:
        if mt is not None and _jsonl_cache["mtime"] == mt and _jsonl_cache["index"]:
            return _jsonl_cache["index"]
    idx = {}
    if mt is not None:
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                idx.setdefault(os.path.splitext(j["file"])[0], j)  # 后行覆盖重跑残留
    with _lock:
        _jsonl_cache.update({"mtime": mt, "index": idx})
    return idx


def match_stem(base, idx):
    """从样张文件名（去掉 .jpg）里找回源图 stem：逐个切分位置试探，取第一个
    在 jsonl 里存在的候选。sp/low/hard/zero 新旧命名通吃——stem 自己含下划线，
    任何固定位置的 rsplit 都会截错（实测 330 张只对上 36 的根因）。"""
    toks = base.split("_")
    for k in range(1, len(toks)):
        cand = "_".join(toks[k:])
        if cand in idx:
            return cand
    return None


def list_samples():
    """quality/*.jpg -> [{file, cat, stem, dets}]，按 low→hard→zero→sp 排序。"""
    idx = jsonl_index()
    out = []
    for fn in os.listdir(QUALITY):
        if not fn.lower().endswith(".jpg"):
            continue
        cat = fn.split("_", 1)[0]
        stem = match_stem(fn[:-4], idx)
        out.append({"file": fn, "cat": cat, "stem": stem,
                    "dets": (idx.get(stem) or {}).get("dets", [])})
    out.sort(key=lambda s: (CAT_ORDER.get(s["cat"], 9), s["file"]))
    return out


def load_verdicts():
    try:
        return json.load(open(REVIEW, encoding="utf-8"))
    except Exception:
        return {"verdicts": {}}


def load_agent():
    """agent 第一轮初筛（独立文件，不进用户进度/计数，只在样张上显示丝带）。"""
    try:
        return json.load(open(AGENT_REVIEW, encoding="utf-8")).get("agent_verdicts", {})
    except Exception:
        return {}


def save_verdict(file, v):
    with _lock:
        data = load_verdicts()
        data.setdefault("verdicts", {})[file] = v
        data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = REVIEW + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, REVIEW)


def crop_jpeg(orig, i):
    """原图第 i 个伪框的放大 crop（带 0.6 倍边距上下文 + 框线），JPEG 字节。"""
    e = jsonl_index().get(os.path.splitext(orig)[0])
    if not e or i >= len(e.get("dets", [])):
        return None
    d = e["dets"][i]
    img = cv2.imread(os.path.join(IMAGES, e["file"]))
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (float(d[k]) for k in ("x1", "y1", "x2", "y2"))
    m = max(24, int(0.6 * max(x2 - x1, y2 - y1)))
    l, t = max(0, int(x1) - m), max(0, int(y1) - m)
    r, b = min(W, int(x2) + m), min(H, int(y2) + m)
    crop = img[t:b, l:r].copy()
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    if max(ch, cw) < 256:  # 小框放大到能看清
        s = 256 / max(ch, cw)
        crop = cv2.resize(crop, (round(cw * s), round(ch * s)))
    color = (0, 255, 255) if d.get("src") == "rescan" else \
            ((0, 165, 255) if d.get("low_conf") else (0, 220, 0))
    sc = crop.shape[1] / cw
    cv2.rectangle(crop, (int((x1 - l) * sc), int((y1 - t) * sc)),
                  (int((x2 - l) * sc), int((y2 - t) * sc)), color, 2)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>伪标签人审 · 检测器 B</title>
<style>
  body{background:#111;color:#ddd;font-family:system-ui,"Microsoft YaHei",sans-serif;
       margin:0;padding:16px 20px 96px;max-width:1180px;margin-inline:auto}
  h1{font-size:18px;color:#fff;margin:0 0 10px}
  h1 small{color:#888;font-weight:normal;margin-left:8px}
  .cnt{display:flex;gap:14px;font-size:13px;color:#999;margin:6px 0 10px;flex-wrap:wrap;
       align-items:center}
  .cnt b{font-size:16px}
  .cnt .prog{background:#14321f;border:1px solid #14532d;border-radius:10px;
             padding:4px 14px;color:#4ade80}
  .cnt .prog b{font-size:20px;color:#4ade80}
  .cnt .ok b{color:#4ade80}.cnt .bad b{color:#f87171}.cnt .b b{color:#fbbf24}
  .cnt .z b{color:#22d3ee}.cnt .g b{color:#c084fc}
  .vbadge{display:none;margin-top:8px;padding:6px 18px;border-radius:10px;
          font-size:16px;font-weight:700}
  .vbadge.on{display:inline-block}
  .vbadge.ok{background:#14321f;color:#4ade80}
  .vbadge.x{background:#3a1212;color:#f87171}
  .vbadge.b{background:#3a2d08;color:#fbbf24}
  .vbadge.z{background:#082a3a;color:#22d3ee}
  .vbadge.g{background:#2a1240;color:#c084fc}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}
  .chip{padding:5px 14px;border-radius:16px;border:1px solid #333;background:#1b1b1b;
        cursor:pointer;font-size:13px;color:#999}
  .chip:hover{border-color:#666}
  .chip.on{color:#4ade80;border-color:#4ade80}
  .grid{display:grid;grid-template-columns:minmax(0,58%) minmax(0,42%);gap:12px}
  .imgcard{background:#1d1d1d;border-radius:12px;padding:10px}
  .imgwrap{position:relative;overflow:hidden;border-radius:8px}
  .imgwrap img{width:100%;display:block;cursor:crosshair}
  .imgwrap img.z100{width:auto;max-width:none;min-width:100%;height:auto;max-height:none;
                    cursor:zoom-out}
  .imgwrap:has(img.z100){overflow:auto}
  .drawbox{position:absolute;border:2px dashed #22d3ee;background:rgba(34,211,238,.15);
           pointer-events:none}
  .mbox{position:absolute;border:2px solid #f472b6;cursor:pointer}
  .mbox:hover{background:rgba(244,114,182,.25)}
  .mbox .mx{position:absolute;top:-18px;left:-2px;background:#f472b6;color:#111;
            font-size:10px;padding:0 5px;border-radius:4px;white-space:nowrap}
  #lens{position:absolute;width:220px;height:220px;border:2px solid #22d3ee;border-radius:50%;
        pointer-events:none;display:none;z-index:6;box-shadow:0 6px 20px #000a;
        background-repeat:no-repeat;background-color:#111}
  .imgcard .meta{font-size:12px;color:#888;margin-top:6px}
  .boxes{display:flex;flex-direction:column;gap:10px;max-height:78vh;overflow-y:auto}
  .boxcard{background:#1d1d1d;border-radius:12px;padding:10px}
  .boxcard .bl{display:flex;justify-content:space-between;align-items:center;
               font-size:12.5px;color:#aaa;margin-bottom:6px}
  .boxcard img{width:100%;border-radius:8px;display:block}
  .aribbon{background:#1a1030;border:1px solid #5b21b6;color:#c4b5fd;font-size:12.5px}
  .boxcard.bad{outline:2px solid #f87171}
  .boxcard .tag{font-size:11px;padding:1px 8px;border-radius:10px}
  .tag.low{background:#3a2d08;color:#fbbf24}.tag.rs{background:#082a3a;color:#22d3ee}
  .del{background:#3a1212;color:#f87171;border:1px solid #7f1d1d;border-radius:8px;
       padding:2px 10px;cursor:pointer;font-size:12px;font-family:inherit}
  .del.on{background:#f87171;color:#1a0505}
  .verdict{background:#1d1d1d;border-radius:12px;padding:12px 14px;margin-top:12px;
           font-size:14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .verdict input{flex:1;min-width:200px;background:#141414;border:1px solid #333;
                 color:#ddd;border-radius:8px;padding:7px 10px;font-family:inherit}
  #legend{position:fixed;left:0;right:0;bottom:0;background:#191919ee;
          border-top:1px solid #333;padding:9px 20px;font-size:13px;color:#aaa;
          display:flex;gap:16px;align-items:center;flex-wrap:wrap;backdrop-filter:blur(4px)}
  kbd{background:#262626;border:1px solid #444;border-bottom-width:3px;border-radius:6px;
      padding:1px 8px;font-family:Consolas,monospace;font-size:12px;color:#eee}
  .done{font-size:16px;color:#4ade80;padding:40px;text-align:center}
  .note-saved{color:#4ade80;font-size:12px}
</style></head><body>
<h1>伪标签人审 · 检测器 B <small id="cur"></small></h1>
<div class="cnt" id="cnt"></div>
<div class="chips" id="chips"></div>
<div class="grid">
  <div class="imgcard">
    <div class="imgwrap" id="imgwrap"><img id="main" title="拖拽=手绘画框补漏检 · 单击=切换 1:1 · 悬停=放大镜(M开关)"></div>
    <div id="lens"></div>
    <div id="vbadge" class="vbadge"></div>
    <div class="meta" id="mainmeta"></div>
  </div>
  <div class="boxes" id="boxes"></div>
</div>
<div class="verdict">
  <span>备注(可选)</span><input id="note" placeholder="比如：框住的是叶子 / 有两只只框了一只">
</div>
<div id="legend"></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let S=[],V={},AG={},view=[],idx=0,filter='all',undoStack=[];
let manual=[];        // 当前样的手绘框 [[x1,y1,x2,y2]]（原图像素坐标）
let lensOn=true;      // 放大镜开关（M 键切换）
const CATN={low:'低置信(0.2-0.3)',mid:'中置信(0.3-0.5)',hard:'难例(rescan)',zero:'全帧空图',high:'高置信抽检',sp:'旧物种抽检'};
const VERDICT_TXT={ok:'✅ 对',x:'❌ 框到非鸟',b:'⚠ 框不准',g:'🗑 整张坏图',z:'❌ 有鸟没框(漏检)',
  suspect:'🤔 agent 疑点(未定,供复核)'};

async function boot(){
  S=(await(await fetch('/api/list')).json()).samples;
  const st=await(await fetch('/api/state')).json();
  V=st.verdicts||{};
  AG=st.agent||{};                 // agent 初筛层：只显示丝带，不算进度不挡导航
  buildChips(); rebuild();
}
function buildChips(){
  const fs=[['all','全部'],['low','低置信'],['mid','中置信'],['hard','难例'],['zero','空图'],
            ['high','高置信'],['agent','agent 标记'],['todo','只看没看的']];
  $('chips').innerHTML=fs.map(f=>`<div class="chip ${f[0]===filter?'on':''}" data-f="${f[0]}">${f[1]}</div>`).join('');
  [...$('chips').children].forEach(c=>c.onclick=()=>{filter=c.dataset.f;buildChips();rebuild();});
}
function rebuild(){
  view=S.filter(s=>filter==='all'?true:filter==='todo'?!V[s.file]
    :filter==='agent'?!!AG[s.file]:s.cat===filter);
  idx=view.findIndex(s=>!V[s.file]);
  if(idx<0)idx=0;
  render();
}
function counters(){
  // 只统计当前批（S）：历史轮次的判定留在 review.json 里闭环用，但不进进度/计数
  const inS=S.filter(s=>V[s.file]);
  const n=inS.length;
  const by=v=>inS.filter(s=>V[s.file].verdict===v).length;
  const badBox=inS.reduce((a,s)=>a+((V[s.file].boxes_bad&&V[s.file].boxes_bad.length)||0),0);
  const manBox=inS.reduce((a,s)=>a+((V[s.file].manual_boxes&&V[s.file].manual_boxes.length)||0),0);
  $('cnt').innerHTML=`<span class="prog">本批进度 <b>${n}/${S.length}</b></span>`
    +`<span class="ok">对 <b>${by('ok')}</b></span>`
    +`<span class="bad">框到非鸟 <b>${by('x')}</b>图·<b>${badBox}</b>框</span>`
    +`<span class="b">框不准 <b>${by('b')}</b></span>`
    +`<span class="z">漏检 <b>${by('z')}</b></span>`
    +`<span class="g">坏图 <b>${by('g')}</b></span>`
    +`<span class="z">手绘 <b>${manBox}</b>框</span>`;
}
function legend(){
  const zero=view[idx]&&view[idx].cat==='zero';
  $('legend').innerHTML=(zero
    ?`<span><kbd>空格</kbd> 真没鸟 ✓</span><span><kbd>X</kbd> 有鸟没框(漏检)</span>`
    :`<span><kbd>空格</kbd> 对 ✓</span><span><kbd>X</kbd> 有非鸟框(多框图建议改用✕指认)</span><span><kbd>B</kbd> 框不准</span><span><kbd>G</kbd> 整张坏图</span>`)
    +`<span><kbd>Z</kbd> 有鸟没框(漏检,任意图)</span><span><kbd>拖拽</kbd> 手绘画框(松手即存,点框删除)</span><span><kbd>U</kbd> 撤销</span><span><kbd>←</kbd> 上一张</span>`
    +`<span style="color:#666">多框图：直接点右侧 ✕ 标坏框（不跳页），标完空格确认</span>`;
}
function render(){
  counters();legend();
  const s=view[idx];
  if(!s){$('main').removeAttribute('src');
    $('boxes').innerHTML='<div class="done">这一筛选下都看完了 ✓ 换个筛选或直接关页</div>';
    $('mainmeta').textContent='';$('cur').textContent='';
    const vb0=$('vbadge');vb0.className='vbadge';vb0.textContent='';
    document.getElementById('imgwrap').querySelectorAll('.mbox').forEach(e=>e.remove());
    return;}
  $('cur').textContent=`${CATN[s.cat]} · ${idx+1}/${view.length} · ${s.file}`;
  const v=V[s.file];
  manual=(v&&v.manual_boxes)||[];
  const im=$('main');
  im.classList.remove('z100');
  im.onload=()=>drawManualOverlays();
  im.src='/img?f='+encodeURIComponent(s.file);
  drawManualOverlays();   // 浏览器缓存命中时 onload 可能不触发，兜底画一次
  $('mainmeta').textContent=s.stem?('源图: images/'+s.stem+'.jpg'):'(源图未对上，仅看画好的框)';
  // 右侧：有框图逐框放大卡；zero 图给一句提示
  const ag=AG[s.file];
  const arib=ag?`<div class="boxcard aribbon">🔍 ${VERDICT_TXT[ag.verdict]||esc(ag.verdict)}${ag.note?` · ${esc(ag.note)}`:''} <span style="color:#6b7280">（仅提示；与你的判断冲突时，以你的为准）</span></div>`:'';
  if(s.cat==='zero'){
    $('boxes').innerHTML=arib+'<div class="boxcard"><div class="bl">这张没有任何伪框</div>'
      +'<div style="font-size:13px;color:#bbb;line-height:1.8">看全图：画面里到底有没有鸟？'
      +'<b style="color:#f472b6">最好的做法：直接在左图上把鸟拖拽框出来</b>——手绘框会作为人工标签'
      +'写进训练集，这张图从"漏检废图"变成"能救的样本"，画完按 <kbd>空格</kbd> 确认下一张。'
      +'不想画：有鸟没框到按 <kbd>X</kbd>（只计漏检率，图仍不进训练）；真没鸟按 <kbd>空格</kbd>。</div></div>';
  }else{
    $('boxes').innerHTML=arib+((s.dets&&s.dets.length)?s.dets.map((d,i)=>{
      const bad=v&&v.boxes_bad&&v.boxes_bad.includes(i);
      return `<div class="boxcard ${bad?'bad':''}" id="bc${i}">
        <div class="bl"><span>框${i+1} · p=${d.score.toFixed(2)} ${esc(d.clf_label||'')}
          ${d.low_conf?'<span class="tag low">低置信</span>':''}
          ${d.src==='rescan'?'<span class="tag rs">难例</span>':''}</span>
          <button class="del ${bad?'on':''}" onclick="toggleBox(${i})">✕ 框错</button></div>
        <img loading="lazy" src="/crop?orig=${encodeURIComponent(s.stem+'.jpg')}&i=${i}"
             onclick="toggleBox(${i})" style="cursor:pointer"
             title="点击小图 = 判这个框错 / 取消"></div>`;
    }).join(''):'<div class="boxcard">jsonl 里没找到这图的框记录（断点间隙），只能看主图</div>');
  }
  $('note').value=(v&&v.note)||'';
  renderBadge();
  // 预加载下一张
  const nx=view[idx+1]; if(nx){const im=new Image();im.src='/img?f='+encodeURIComponent(nx.file);}
}
function renderBadge(){
  const s=view[idx];const v=s&&V[s.file];
  const vb=$('vbadge');
  if(!s)return;
  const bad=(v&&v.boxes_bad)||[];
  const badTxt=bad.map(i=>'框'+(i+1)).join('、');
  const nBoxes=(s.dets&&s.dets.length)||0;
  const man=manual.length?`（✍ 手绘${manual.length}框已并入）`:'';
  let cls='',txt='';
  if(v&&v.verdict==='x'){
    cls='x';
    txt=bad.length
      ?`❌ ${badTxt} 到非鸟 — 已标记删除；其余 ${nBoxes-bad.length} 框视为对`
      :`❌ 有框到非鸟（还没指明是哪个——点右侧对应框的 ✕ 指认，或 ← 回来补）`;
  }else if(v&&v.verdict==='ok'&&bad.length){
    cls='x';
    txt=`⚠ ${badTxt} 标记删除，其余 ${nBoxes-bad.length} 框放行（要撤销删框就再点它的 ✕）`;
  }else if(v&&v.verdict){
    cls=v.verdict;
    txt='已判定：'+(VERDICT_TXT[v.verdict]||v.verdict);
  }else if(manual.length){
    cls='ok';
    txt=`✍ 手绘${manual.length}框 · 按空格确认并下一张`;
  }
  if(txt){vb.className='vbadge on '+cls;vb.textContent=txt+man;}
  else{vb.className='vbadge';vb.textContent='';}
}
function drawManualOverlays(){
  const wrap=$('imgwrap'),im=$('main');
  wrap.querySelectorAll('.mbox').forEach(e=>e.remove());
  if(!im.naturalWidth||!manual.length)return;
  const r=im.getBoundingClientRect();
  const sx=r.width/im.naturalWidth,sy=r.height/im.naturalHeight;
  manual.forEach((b,i)=>{
    const d=document.createElement('div');d.className='mbox';
    d.style.left=(b[0]*sx)+'px';d.style.top=(b[1]*sy)+'px';
    d.style.width=Math.max(4,(b[2]-b[0])*sx)+'px';
    d.style.height=Math.max(4,(b[3]-b[1])*sy)+'px';
    d.title='手绘框'+(i+1)+'：点击删除';
    d.innerHTML='<span class="mx">✕ 手绘'+(i+1)+'</span>';
    d.onclick=()=>{manual.splice(i,1);persistManual();drawManualOverlays();};
    wrap.appendChild(d);
  });
}
function persistManual(){
  const s=view[idx];if(!s)return;
  undoStack.push({file:s.file,prev:snapPrev(s.file),idx});   // U 统一撤销：画框/删框也算一步
  const cur=V[s.file]||{};
  cur.manual_boxes=manual;
  if(!cur.verdict)cur.verdict='ok';   // 画了框=画面里有鸟；空图由此翻身进训练集
  save(s.file,cur);
  renderBadge();
}
function snapPrev(file){
  // 深拷贝快照：undoStack 里若存对象引用，后续 toggle 会原地变异历史，
  // U 撤销出来的就是被污染的"历史"（实测恢复出空 boxes_bad 的根因）
  const v=V[file];
  return v?JSON.parse(JSON.stringify(v)):undefined;
}
function toggleBox(i){
  const s=view[idx];
  undoStack.push({file:s.file,prev:snapPrev(s.file),idx});   // U 统一撤销：✕勾选也算一步
  const cur=V[s.file]||{verdict:'x',boxes_bad:[]};
  cur.boxes_bad=cur.boxes_bad||[];
  const p=cur.boxes_bad.indexOf(i);
  if(p>=0)cur.boxes_bad.splice(p,1);else cur.boxes_bad.push(i);
  // 取消勾选后若成"空壳判定"（没勾框、没手绘、没备注）→ 整条删除，这张回到未看
  if(!cur.boxes_bad.length&&!(cur.manual_boxes&&cur.manual_boxes.length)&&!(cur.note&&cur.note.trim())){
    delete V[s.file];
    fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({file:s.file,verdict:null})});
    render();return;
  }
  V[s.file]=cur;save(s.file,cur);render();
}
async function save(file,v){
  V[file]=v;
  try{await fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({file,...v})});}catch(e){}
  counters();
}
function judge(kind){
  const s=view[idx];if(!s)return;
  undoStack.push({file:s.file,prev:snapPrev(s.file),idx});
  const note=$('note').value.trim();
  const boxes_bad=[...document.querySelectorAll('.boxcard.bad')].map(el=>+el.id.slice(2));
  save(s.file,{verdict:kind,boxes_bad,note,manual_boxes:manual,
               ts:new Date().toTimeString().slice(0,8)});
  $('note').value='';
  // 判完跳下一张【未看的】（环形回绕）；全看完了才进完成页
  const n=nextUnseen(idx+1);
  idx = n>=0 ? n : view.length;
  render();
}
function nextUnseen(start){
  for(let k=0;k<view.length;k++){
    const j=(start+k)%view.length;
    if(!V[view[j].file])return j;
  }
  return -1;
}
function undo(){
  const u=undoStack.pop();if(!u)return;
  if(u.prev)save(u.file,u.prev);else{delete V[u.file];
    fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({file:u.file,verdict:null})});}
  idx=u.idx;render();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'&&!['Escape'].includes(e.key))return;
  if(e.code==='Space'){e.preventDefault();judge('ok');}
  else if(e.key==='ArrowRight'){e.preventDefault();          // 纯导航：跳下一张未看的，不判定
    const n=nextUnseen(idx+1);idx=n>=0?n:Math.min(idx+1,view.length-1);render();}
  else if(e.key==='x'||e.key==='X')judge('x');
  else if(e.key==='b'||e.key==='B')judge('b');
  else if(e.key==='g'||e.key==='G')judge('g');
  else if(e.key==='z'||e.key==='Z')judge('z');
  else if(e.key==='u'||e.key==='U')undo();
  else if(e.key==='m'||e.key==='M'){lensOn=!lensOn;if(!lensOn)$('lens').style.display='none';}
  else if(e.key==='ArrowLeft'){idx=Math.max(0,idx-1);render();}   // 上一张：老实退一格，不跳
});
/* 手绘画框：mousedown 起笔，拖出虚线框，松手落框（原图像素坐标）并即存；
   没拖动的纯点击 = 切换 1:1 原尺寸 */
(function initDraw(){
  const im=$('main'),wrap=$('imgwrap');
  let st=null,el=null;
  const Z=2.5,LR=110;                        // 放大镜倍率与半径
  const hideLens=()=>{$('lens').style.display='none';};
  im.addEventListener('mousedown',e=>{
    if(!view[idx])return;
    e.preventDefault();
    hideLens();                              // 画框时放大镜让位
    const r=im.getBoundingClientRect();
    st={x:e.clientX-r.left,y:e.clientY-r.top};el=null;
  });
  im.addEventListener('mousemove',e=>{
    if(!st)return;
    const r=im.getBoundingClientRect();
    const cx=Math.max(0,Math.min(e.clientX-r.left,r.width));
    const cy=Math.max(0,Math.min(e.clientY-r.top,r.height));
    if(!el&&(Math.abs(cx-st.x)>4||Math.abs(cy-st.y)>4)){
      el=document.createElement('div');el.className='drawbox';wrap.appendChild(el);
    }
    if(el){
      el.style.left=Math.min(st.x,cx)+'px';el.style.top=Math.min(st.y,cy)+'px';
      el.style.width=Math.abs(cx-st.x)+'px';el.style.height=Math.abs(cy-st.y)+'px';
    }
  });
  // 放大镜：悬停跟随（画框进行中不显示）
  im.addEventListener('mousemove',e=>{
    if(!lensOn||st||!im.naturalWidth||!im.src)return hideLens();
    const r=im.getBoundingClientRect(),wr=wrap.getBoundingClientRect();
    if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)return hideLens();
    const cx=e.clientX-r.left,cy=e.clientY-r.top,lens=$('lens');
    lens.style.display='block';
    lens.style.backgroundImage=`url("${im.src}")`;
    lens.style.backgroundSize=(r.width*Z)+'px '+(r.height*Z)+'px';
    lens.style.backgroundPosition=(-(cx*Z-LR))+'px '+(-(cy*Z-LR))+'px';
    let lx=Math.max(0,Math.min(e.clientX-wr.left-LR,wr.width-2*LR));
    let ly=Math.max(0,Math.min(e.clientY-wr.top-LR,wr.height-2*LR));
    lens.style.left=lx+'px';lens.style.top=ly+'px';
  });
  im.addEventListener('mouseleave',hideLens);
  window.addEventListener('mouseup',e=>{
    if(!st)return;
    const r=im.getBoundingClientRect();
    const cx=Math.max(0,Math.min(e.clientX-r.left,r.width));
    const cy=Math.max(0,Math.min(e.clientY-r.top,r.height));
    if(!el){                                   // 纯点击 = 切 1:1
      im.classList.toggle('z100');drawManualOverlays();st=null;return;
    }
    el.remove();el=null;
    if(!im.naturalWidth){st=null;return;}
    const sx=im.naturalWidth/r.width,sy=im.naturalHeight/r.height;
    const x1=Math.min(st.x,cx)*sx,y1=Math.min(st.y,cy)*sy;
    const x2=Math.max(st.x,cx)*sx,y2=Math.max(st.y,cy)*sy;
    if(x2-x1>=4&&y2-y1>=4){
      manual.push([Math.round(x1),Math.round(y1),Math.round(x2),Math.round(y2)]);
      persistManual();drawManualOverlays();
    }
    st=null;
  });
  window.addEventListener('resize',drawManualOverlays);
})();
boot();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        if u.path == "/api/list":
            self._json({"samples": list_samples()})
        elif u.path == "/api/state":
            with _lock:
                d = load_verdicts()
                d["agent"] = load_agent()
                self._json(d)
        elif u.path == "/img":
            fn = os.path.basename(urllib.parse.unquote((qs.get("f") or [""])[0]))
            path = os.path.join(QUALITY, fn)
            if os.path.isfile(path):
                self._bytes(open(path, "rb").read(), "image/jpeg")
            else:
                self._json({"error": "no such sample"}, 404)
        elif u.path == "/crop":
            orig = urllib.parse.unquote((qs.get("orig") or [""])[0])
            try:
                i = int((qs.get("i") or ["0"])[0])
            except ValueError:
                i = 0
            body = crop_jpeg(orig, i)
            if body:
                self._bytes(body, "image/jpeg")
            else:
                # 1x1 透明占位：让前端 img 不裂
                self._bytes(base64.b64decode(
                    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
                    b"nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="), "image/png")
        else:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        from urllib.parse import urlparse
        if urlparse(self.path).path != "/api/verdict":
            self._json({"error": "unknown endpoint"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            file = os.path.basename(str(body.get("file") or ""))
            if not file:
                self._json({"error": "missing file"}, 400)
                return
            if body.get("verdict") is None:      # U 撤销到"从未看过"
                with _lock:
                    data = load_verdicts()
                    data.get("verdicts", {}).pop(file, None)
                    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    tmp = REVIEW + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    os.replace(tmp, REVIEW)
                self._json({"ok": True, "removed": file})
                return
            mb = body.get("manual_boxes") or []
            try:
                mb = [[int(round(float(c))) for c in b] for b in mb
                      if isinstance(b, (list, tuple)) and len(b) == 4][:50]
            except Exception:
                mb = []
            save_verdict(file, {"verdict": str(body.get("verdict")),
                                "boxes_bad": body.get("boxes_bad") or [],
                                "manual_boxes": mb,
                                "note": str(body.get("note") or "")[:300],
                                "ts": time.strftime("%H:%M:%S")})
            self._json({"ok": True})
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *a):
        pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"review ui: http://127.0.0.1:{a.port}  （样张 {len(list_samples())} 张）",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
