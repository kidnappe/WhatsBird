# -*- coding: utf-8 -*-
r"""实机测试看图器：五区 dataset.csv 随机抽样 → 大图轮放 + 地面真值标签，
供手机 App 对屏实拍验证（识别 Top-5 / 区域路由 / 拒识门 / 类数显示）。
与 tools\dashboard\review_ui.py（检测器 B 伪标签人审）互不干扰，别混用。

数据落点：E:\code\code tools\data\shiji_test\（images 副本 + manifest.json），
按 AGENTS.md 约定管线产物放 code tools\data，本工具只读它。

用法（一键启动 bat 同目录 start_img_view.bat）：
  python tools\img_view\img_view.py --resample 10          # 每区抽 10 张重采样
  python tools\img_view\img_view.py --port 8789            # 起服务（无 manifest 自动抽 10/区）
  python tools\img_view\img_view.py --resample 10 --port 8789

页面键位：←/→ 切图；H 隐藏/显示真值（盲测用）；点图同 H。
"""
import argparse
import csv
import json
import os
import random
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

TEST_DIR = r"E:\code\code tools\data\shiji_test"
DATA_ROOT = r"E:\code\code tools\data"
REGIONS = ["huabei", "huazhong", "huadong", "dongbei", "huanan_sw"]
REGION_ZH = {
    "huabei": "华北", "huazhong": "华中", "huadong": "华东",
    "dongbei": "东北", "huanan_sw": "华南+西南",
}
SEED = 20261007  # 固定种子：同一天重采样得到同一批，便于复现某张的问题


def resample(per_region: int) -> list:
    """每区随机抽 per_region 张（file 必须真实存在），拷进 TEST_DIR 并写 manifest.json。"""
    os.makedirs(TEST_DIR, exist_ok=True)
    rng = random.Random(SEED)
    items = []
    for region in REGIONS:
        csv_path = os.path.join(DATA_ROOT, region, "dataset.csv")
        img_dir = os.path.join(DATA_ROOT, region, "images")
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rng.shuffle(rows)
        picked = 0
        for row in rows:
            if picked >= per_region:
                break
            src = os.path.join(img_dir, row["file"])
            if not os.path.isfile(src):
                continue
            out = f"{region}_{os.path.basename(row['file'])}"
            dst = os.path.join(TEST_DIR, out)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
            items.append({
                "file": out,
                "region": region,
                "region_zh": REGION_ZH[region],
                "zh": row.get("common_name", ""),
                "sci": row.get("scientific_name", ""),
            })
            picked += 1
    manifest = os.path.join(TEST_DIR, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    return items


def load_manifest() -> list:
    path = os.path.join(TEST_DIR, "manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>ObserveBird 实机测试看图器</title>
<style>
  html,body{height:100%;margin:0;background:#101214;color:#e8eaed;
    font-family:"Microsoft YaHei",system-ui,sans-serif;overflow:hidden}
  #wrap{display:flex;flex-direction:column;height:100%}
  #bar{display:flex;align-items:center;gap:14px;padding:10px 16px;
    background:#16191c;border-bottom:1px solid rgba(255,255,255,.08);flex:none}
  #zh{font-size:26px;font-weight:700}
  #sci{font-style:italic;color:#9aa0a6;font-size:16px}
  .badge{background:#2a3b2f;color:#8fd694;border-radius:999px;
    padding:4px 14px;font-size:16px;font-weight:600}
  #cnt{color:#9aa0a6;font-size:14px;margin-left:auto}
  button{background:#22262a;color:#e8eaed;border:1px solid rgba(255,255,255,.12);
    border-radius:999px;padding:7px 18px;font-size:15px;cursor:pointer}
  button:hover{background:#2c3136}
  #stage{flex:1;display:flex;align-items:center;justify-content:center;min-height:0}
  #pic{max-width:100%;max-height:100%;object-fit:contain;cursor:pointer}
  #hidden-tip{position:fixed;right:14px;bottom:12px;color:#5f6368;font-size:13px}
</style></head><body><div id="wrap">
  <div id="bar">
    <button onclick="step(-1)">&#8592; 上一张</button>
    <span id="zh"></span><span class="badge" id="reg"></span>
    <span id="sci"></span>
    <span id="cnt"></span>
    <button onclick="toggleLabel()">盲测 (H)</button>
    <button onclick="step(1)">下一张 &#8594;</button>
  </div>
  <div id="stage"><img id="pic" onclick="toggleLabel()"></div>
  <div id="hidden-tip">键位：&#8592;/&#8594; 切图 · H 或点图 隐藏/显示真值</div>
</div>
<script>
let list=[],i=0,labelOn=true;
function render(){
  const it=list[i];if(!it)return;
  document.getElementById('pic').src='/img/'+encodeURIComponent(it.file);
  document.getElementById('zh').textContent=labelOn?it.zh:'？？？';
  document.getElementById('reg').textContent=labelOn?(it.region_zh||it.region):'';
  document.getElementById('sci').textContent=labelOn?it.sci:'';
  document.getElementById('cnt').textContent=(i+1)+' / '+list.length;
}
function step(d){i=(i+d+list.length)%list.length;render();}
function toggleLabel(){labelOn=!labelOn;render();}
fetch('/api/list').then(r=>r.json()).then(l=>{list=l;render();});
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight')step(1);
  else if(e.key==='ArrowLeft')step(-1);
  else if(e.key==='h'||e.key==='H')toggleLabel();
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/api/list":
            body = json.dumps(load_manifest(), ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        elif self.path.startswith("/img/"):
            name = os.path.basename(quote(self.path[len("/img/"):], errors="ignore"))
            path = os.path.join(TEST_DIR, name)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    self._send(200, "image/jpeg", f.read())
            else:
                self._send(404, "text/plain", b"not found")
        else:
            self._send(404, "text/plain", b"not found")

    def log_message(self, fmt, *args):  # 静默访问日志，控制台只留启动横幅
        pass


def main():
    ap = argparse.ArgumentParser(description="实机测试看图器")
    ap.add_argument("--port", type=int, default=None,
                    help="起服务的端口；不给且带了 --resample 时只采样不起服务")
    ap.add_argument("--resample", type=int, metavar="N",
                    help="每区随机重抽 N 张（覆盖 manifest）")
    args = ap.parse_args()

    if args.resample or not os.path.isfile(os.path.join(TEST_DIR, "manifest.json")):
        items = resample(args.resample or 10)
        print(f"已重采样 {len(items)} 张 -> {TEST_DIR}")
    if not args.port:
        return

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"实机测试看图器 http://127.0.0.1:{args.port}  "
          f"(数据 {TEST_DIR}，{len(load_manifest())} 张，Ctrl+C 停止)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
