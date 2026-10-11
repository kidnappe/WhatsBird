# -*- coding: utf-8 -*-
r"""B 检测器训练外包包打包器（目标机：天选6 Pro / RTX 5060 8G / Intel HX / 16G）。

包结构（自给自足，全部相对路径，目标机无需本机任何绝对路径）：
  observebird_detb_train_pack/
  ├─ yolox_repo/                 官方训练代码（去 .git/output/__pycache__ 等）
  ├─ weights/yolox_s.pth         COCO 预训练热启权重
  ├─ yolox_bird_exp.py           可移植 exp（data_dir 走 DETB_DATA 环境变量）
  ├─ data/detector_b_data/
  │   ├─ annotations/det_train.json det_val.json
  │   ├─ img/huabei/*.jpg        仅打包 json 引用到的图（3.17 万张，1.16GB）
  │   ├─ img/coco_bird/*.jpg     COCO train2017 bird 子集（3,237 张）
  │   └─ train2017|val2017/...   由 prepare_junctions.py 在目标机创建（免重复拷贝）
  ├─ prepare_junctions.py        目标机跑一次：建 train2017|val2017 目录联结
  ├─ start_train.bat             双击起训（ep50、b8；OOM 退 4，显存富余可 16）
  ├─ resume_train.bat            断点续训（--resume latest）
  ├─ detb_pack_monitor.py        只读训练监控台（纯 stdlib，零依赖）
  ├─ start_monitor.bat           双击开监控台 -> http://127.0.0.1:8787
  └─ README-外包机.md            环境安装（torch cu128 for sm_120!）+ 回传清单

强制校验（缺一 sys.exit(2)，照 make_pack 先例）：json 引用的图 100% 在包内、
bundled yolox_repo 可 import、exp 可编译、标注计数与本机一致、训练产物未泄漏、
监控台可编译且能对打包结果采到数据。

Usage: python make_train_pack.py            # 构建 + 校验 + zip
       python make_train_pack.py --no-zip   # 只构建不压缩
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO = r"E:\code\observebird"
DB = r"E:\code\code tools\data\detector_b"
HUABEI = r"E:\code\code tools\data\huabei\images"
COCO = os.path.join(DB, "coco_bird", "train2017_bird")
YOLOX = r"E:\code\code tools\yolox_repo"
WEIGHTS = r"E:\code\code tools\detectors\yolox\yolox_s.pth"
DIST = os.path.join(REPO, "dist")
STAGE = os.path.join(DIST, "observebird_detb_train_pack")
DATA_ROOT = os.path.join(STAGE, "data", "detector_b_data")
EXP_SRC = os.path.join(REPO, "tools", "detector_b", "yolox_bird_exp.py")
PY = sys.executable


def fail(msg):
    print(f"✗ {msg}")
    sys.exit(2)


EXP_PORTABLE = r'''# -*- coding: utf-8 -*-
"""B 路线训练 exp（外包包便携版）：与主仓 yolox_bird_exp.py 唯一差别是
data_dir 走 DETB_DATA 环境变量（start_train.bat 会设为包内相对路径）。
num_classes=1；ep50；no_aug 最后 10 轮；eval 每 5 轮。"""
import os

from yolox.exp import Exp as MyExp

SIZE = int(os.environ.get("B_SIZE", "512"))


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = f"yolox_bird_{SIZE}"
        # start_train.bat: set DETB_DATA=<包>\data\detector_b_data
        self.data_dir = os.environ.get("DETB_DATA", "detector_b_data")
        self.train_ann = "det_train.json"      # 实际读 <data_dir>/annotations/
        self.val_ann = "det_val.json"
        self.num_classes = 1
        self.input_size = (SIZE, SIZE)
        self.test_size = (SIZE, SIZE)
        self.max_epoch = 50
        self.no_aug_epochs = 10
        self.eval_interval = 5
        self.data_num_workers = 4              # 外包机 16G / HX 多核，可比本机放宽
        self.print_interval = 20
'''


def main():
    no_zip = "--no-zip" in sys.argv
    if os.path.isdir(STAGE):
        print(f"清空旧暂存 {STAGE}")
        shutil.rmtree(STAGE, ignore_errors=True)

    # ---- 1) 标注 ----
    ann_dst = os.path.join(DATA_ROOT, "annotations")
    os.makedirs(ann_dst, exist_ok=True)
    referenced = set()
    for f in ("det_train.json", "det_val.json"):
        shutil.copy2(os.path.join(DB, "annotations", f), os.path.join(ann_dst, f))
        j = json.load(open(os.path.join(ann_dst, f), encoding="utf-8"))
        for im in j["images"]:
            referenced.add(im["file_name"])
    print(f"标注就绪：引用图片 {len(referenced):,} 张")

    # ---- 2) 图片（只拷引用到的）----
    n_hb = n_cc = miss = 0
    for rel in sorted(referenced):
        grp, bare = rel.split("/", 1)
        src = os.path.join(HUABEI, bare) if grp == "huabei" else os.path.join(COCO, bare)
        dst = os.path.join(DATA_ROOT, "img", grp, bare)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            n_hb += grp == "huabei"
            n_cc += grp == "coco_bird"
        else:
            miss += 1
            if miss <= 5:
                print(f"  缺图: {src}")
    if miss:
        fail(f"json 引用的图有 {miss} 张在本机不存在——数据不一致，先修再打包")
    print(f"图片拷贝：huabei {n_hb:,} + coco_bird {n_cc:,}")

    # ---- 3) yolox_repo（去重物）----
    # ⚠️ YOLOX_outputs 必须排除：它是**训练产物**目录，本机跑过的实验会整套留在
    # 里面（实测泄漏过 yolox_bird_512/：71MB latest_ckpt.pth + 一份 ep5 评测崩溃
    # 的旧 train_log.txt）。后果不只是包变大——新监控台按 mtime 找 train_log.txt，
    # 会认到这份**别人的旧日志**上，外包同学开训前就看到一屏"崩溃/停滞"。
    repo_dst = os.path.join(STAGE, "yolox_repo")
    # ⚠️ 这里**不能**用 shutil.ignore_patterns：它按 basename 在**任意层级**匹配，
    # 于是 "datasets" 会连 yolox/data/datasets/ 一起删掉——那是训练必需的模块
    # （mosaicdetection.py、coco.py 都在里面），而 yolox/data/__init__.py 有
    # `from .datasets import *`，缺了它 `import yolox.data` 直接 ModuleNotFoundError，
    # 外包机一开训就死。实测踩过：包内 import yolox.data 失败，而旧校验只测
    # `from yolox.exp import Exp`（那条路径不触发 yolox.data）所以没抓住。
    # 因此改用自定义 ignore：**只有仓库根的那几个目录**才排除。
    ROOT_ONLY = {"datasets", "assets", "docs", "tests", "demo", "tutorials",
                 "output", "build", "YOLOX_outputs"}
    GENERIC = (".git", "__pycache__", ".egg-info")

    def _ignore(dirpath, names):
        drop = set()
        rel = os.path.relpath(dirpath, YOLOX)
        at_root = (rel == ".")
        for n in names:
            if any(g in n for g in GENERIC):
                drop.add(n)
            elif at_root and n in ROOT_ONLY:
                drop.add(n)
        return drop

    shutil.copytree(YOLOX, repo_dst, ignore=_ignore, dirs_exist_ok=True)

    # 打包后立刻验：训练真正会 import 到的入口，缺一个都不许出包
    missing = [p for p in ("yolox/data/datasets/__init__.py",
                           "yolox/data/datasets/mosaicdetection.py",
                           "yolox/data/datasets/coco.py",
                           "yolox/data/datasets/datasets_wrapper.py")
               if not os.path.exists(os.path.join(repo_dst, *p.split("/")))]
    if missing:
        fail(f"包内 yolox_repo 缺关键模块（ignore 规则误伤？）：{missing}")

    # ---- 4) 权重 / exp / 启动器 ----
    os.makedirs(os.path.join(STAGE, "weights"), exist_ok=True)
    shutil.copy2(WEIGHTS, os.path.join(STAGE, "weights", "yolox_s.pth"))
    open(os.path.join(STAGE, "yolox_bird_exp.py"), "w", encoding="utf-8").write(EXP_PORTABLE)

    open(os.path.join(STAGE, "prepare_junctions.py"), "w", encoding="utf-8").write(
        '# -*- coding: utf-8 -*-\n'
        '"""目标机跑一次：为 train2017/val2017 建目录联结（免重复拷贝图片）。"""\n'
        'import os, subprocess\n'
        'ROOT = os.path.dirname(os.path.abspath(__file__))\n'
        'DATA = os.path.join(ROOT, "data", "detector_b_data")\n'
        'for name, target in (("train2017/huabei", "../../img/huabei"),\n'
        '                     ("train2017/coco_bird", "../../img/coco_bird"),\n'
        '                     ("val2017/huabei", "../../img/huabei")):\n'
        '    link = os.path.join(DATA, *name.split("/"))\n'
        '    tgt = os.path.normpath(os.path.join(DATA, name, target))\n'
        '    os.makedirs(os.path.dirname(link), exist_ok=True)\n'
        '    if os.path.isdir(link): continue\n'
        '    subprocess.run(["cmd", "/c", "mklink", "/J", link, tgt], check=True)\n'
        '    print("junction:", link, "->", tgt)\n'
        'print("junctions OK")\n')

    # bat 一律 CRLF + ASCII（项目铁律）
    BATS = {
        "start_train.bat": r'''@echo off
chcp 65001 >nul
title ObserveBird B detector training (outsource, ep50)

rem Usage: start_train.bat [size] [batch]   (defaults 640 8; OOM -> "640 4")
rem This pack is the 640 edition: the main machine's 3060 already runs 512.
set SIZE=640
if not "%~1"=="" set SIZE=%~1
set BATCH=8
if not "%~2"=="" set BATCH=%~2
set PYTHONIOENCODING=utf-8
set B_SIZE=%SIZE%
set DETB_DATA=%~dp0data\detector_b_data
set PYTHONPATH=%~dp0yolox_repo;%PYTHONPATH%

echo ================================================
echo  ObserveBird B detector training (ep50, %SIZE%, batch %BATCH%, fp16)
echo  First run: python prepare_junctions.py BEFORE this.
echo ================================================

cd /d %~dp0yolox_repo
python -u tools\train.py -f %~dp0yolox_bird_exp.py -c %~dp0weights\yolox_s.pth -b %BATCH% --fp16
echo.
echo training exited.
pause
''',
        "resume_train.bat": r'''@echo off
chcp 65001 >nul
title ObserveBird B detector training RESUME

rem Resume from latest checkpoint. Pass explicit ckpt as %1 to override.
set LAST=%~1
if "%LAST%"=="" set LAST=YOLOX_outputs\yolox_bird_640\latest_ckpt.pth
set PYTHONIOENCODING=utf-8
set B_SIZE=640
set DETB_DATA=%~dp0data\detector_b_data
set PYTHONPATH=%~dp0yolox_repo;%PYTHONPATH%

cd /d %~dp0yolox_repo
python -u tools\train.py -f %~dp0yolox_bird_exp.py -c "%LAST%" --resume -b 8 --fp16
echo.
echo training exited.
pause
''',
        # 只读监控台启动器（源码 detb_pack_monitor.py 同目录，见下面的拷贝步骤）
        "start_monitor.bat": r'''@echo off
chcp 65001 >nul
title ObserveBird B detector training monitor (read-only)

rem ============================================
rem  Read-only training monitor for this pack.
rem  Usage:
rem    start_monitor.bat          -> http://127.0.0.1:8787
rem    start_monitor.bat 8790     -> custom port
rem  Set NO_BROWSER=1 to skip auto-opening the page.
rem  Close this window to stop the monitor.
rem  It never touches the training process (read-only).
rem ============================================

set PORT=8787
if not "%~1"=="" set PORT=%~1

set PYTHONIOENCODING=utf-8

echo ================================================
echo  ObserveBird B detector monitor (read-only)
echo  root : %~dp0
echo  url  : http://127.0.0.1:%PORT%
echo  Close this window to stop the monitor.
echo ================================================

rem already running -> do not start a second instance, just open the page
curl -s -o nul --max-time 2 http://127.0.0.1:%PORT%/api/status
if not errorlevel 1 (
    echo already running - opening browser only.
    if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%
    exit /b 0
)

if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%

cd /d %~dp0
python -u detb_pack_monitor.py --root "%~dp0." --port %PORT%
echo.
echo monitor exited (port busy or python missing?).
pause
''',
    }
    for name, txt in BATS.items():
        with open(os.path.join(STAGE, name), "w", encoding="utf-8", newline="\r\n") as f:
            f.write(txt)

    # ---- 4b) 只读监控台（自包含源码，与包同目录）----
    MONITOR_SRC = os.path.join(REPO, "tools", "detector_b", "detb_pack_monitor.py")
    shutil.copy2(MONITOR_SRC, os.path.join(STAGE, "detb_pack_monitor.py"))

    # ---- 5) README ----
    open(os.path.join(STAGE, "README-外包机.md"), "w", encoding="utf-8").write(
        """# B 检测器训练包 · README（天选6 Pro / RTX 5060 8G）

## 0. 三十秒流程
1. 装 Python 3.10~3.12（64 位；python.org 官方版即可，无需 Anaconda）
2. NVIDIA 驱动更新到最新（GeForce 驱动自动更新即可；CUDA 12.8 需要的驱动分支 ≥572）
3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   ⚠️ 5060 是 Blackwell（sm_120），**必须 cu128 版**，装 cu121 及以下会报 "no kernel image"
4. `pip install numpy loguru tqdm thop tabulate psutil tensorboard pycocotools opencv-python scipy -i https://pypi.tuna.tsinghua.edu.cn/simple`
   （**不用** yolox_repo/requirements.txt 整包——里面有导出用的 onnx-simplifier，
   Python 3.12 下可能装挂，训练用不到；也**不需要** pip install -e .（无 MSVC 编译），
   启动脚本靠 PYTHONPATH 喂包内 yolox_repo，主仓 3060 实跑验证的就是这条路）
5. `python prepare_junctions.py`（建目录联结，一次即可）
6. 双击 `start_train.bat`（**本包是 640 版**：主仓 3060 正在跑 512，外包机跑 640 互补，
   明早两台各带一个尺寸回来 bench。默认 ep50 / batch 8 / fp16；OOM 就 `start_train.bat 640 4`）

## 1. 产出（训练完把这一个文件夹整个回传）
`yolox_repo/YOLOX_outputs/yolox_bird_640/`，里面有：
- `latest_ckpt.pth`（每 epoch 更新，断点用它）
- `best_ckpt.pth`（**ep5 首次评测后**出现并随 AP 破纪录更新——回传的主角）
- `train_log.txt`（完整迭代/loss/评测日志）
- `tensorboard/`（可选带上）

## 2. 看进度：双击 `start_monitor.bat`（只读监控台）
浏览器自动打开 http://127.0.0.1:8787，实时显示：epoch/iter 进度、总进度条、
total_loss 曲线、**AP 曲线**、GPU 利用率/显存/温度、剩余时间预估、日志尾部、
回传清单勾选状态。

评测节奏（trainer 的真实行为，监控台按此把 AP 对到轮次上）：
- **ep1~39：每 5 轮评一次**（5, 10, …, 35）
- **ep40~50：每轮都评**（进入 no_aug 阶段后 trainer 把评测间隔改成 1）
- 全程共 **18 次**评测；`best_ckpt.pth` 在 AP 破纪录时更新

几点说明：
- **纯只读**：页面上没有任何停止/启动按钮，**不会碰训练进程**，开训前后随时可开。
- 关掉那个黑色控制台窗口 = 停止监控台，训练不受影响。
- 状态条会自己判：绿色=完成，蓝色=训练中，**红色=崩溃或停滞**（15 分钟无新日志）。
- 崩溃时页面会把报错的最后几行直接贴出来，不用自己去日志里翻。
- 端口被占就换一个：`start_monitor.bat 8790`。
- 只用 Python 标准库，不需要额外装任何东西（不用 tensorboard 也能看 AP）。

## 3. 注意事项
- batch 8 起步：8G 显存跑 @640 fp16 约用 4~5G；不建议加大 batch（见 §4）
- 训练中别关启动窗口；关窗=中断，下次双击 `resume_train.bat` 从 latest_ckpt 续
- `--resume` 用的就是 `latest_ckpt.pth`（iter 级断点，官方续训路径）
- 数据已含人工复核修订（删 20 毒框 / 补 102 手绘框 / 剔 34 坏图），val 是冻结清单，勿改动
- ep50 保持不变（用户拍板）

## 4. 性能预期（640 @ 5060 8G，供核对）
- 0.12~0.16 s/iter（batch 8）→ 全程约 8~11 小时（一夜正好）
- 显存约 4~5G/8G；**不建议 batch 16**（640 下容易撞墙），要提速别动 batch，保证跑完优先
- 明显慢于 0.2s/iter：查电源模式（插电/性能档）、TGP 是否满血、别的程序抢卡

## 5. 环境体检（装完先跑这条自检再开训）
```
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
预期输出类似：`2.7.x+cu128 True NVIDIA GeForce RTX 5060 Laptop GPU (12, 0)`
——`True` 和 `(12, 0)` 两个对了才开训；False 多半是装错 CUDA 轮子（回到第 3 步）。

## 6. 硬件余量
- 磁盘：包解压 ~2.6GB + 训练产物（best/last ckpt 各 ~140MB + 日志）→ 留 5GB 富余
- 内存：workers=4 × batch8 fp16@640，16G 足够（训练中别的程序随便用）
- 不需要：MSVC/CMake（不编译）、onnx/onnx2tf（导出量化留在主仓机器）、联网（装完依赖后可断网）
""")

    # ---- 6) 强制校验 ----
    n_ref = len(referenced)
    shipped = sum(len(fs) for _, _, fs in os.walk(os.path.join(DATA_ROOT, "img")))
    if shipped < n_ref - miss:
        fail(f"包内图片 {shipped} < 引用 {n_ref}")
    # ⚠️ py_compile 默认把 .pyc 写到被测文件旁边的 __pycache__/，而这些校验发生在
    # 压缩**之前** → 打包器自己会往包里塞一份 __pycache__（实测混进过 zip）。
    # 用 cfile 把产物丢进系统临时目录，包内保持干净。
    _tmpdir = tempfile.mkdtemp(prefix="detb_pack_chk_")
    try:
        # ⚠️ 校验姿势要紧：以前只 import `yolox.exp`，那条路径**不触发**
        # yolox.data → 包内 datasets 被 ignore 误删也照样"通过"，直到外包机开训
        # 才 ModuleNotFoundError。现在必须把训练真实用到的链条全 import 一遍：
        #   yolox.data (含 .datasets) → exp 实例化 → YOLOX 的 MosaicDetection
        chk = subprocess.run([PY, "-c",
                              "import sys; sys.path.insert(0, r'%s'); "
                              "import yolox.data; "
                              "from yolox.data.datasets import MosaicDetection; "
                              "from yolox.data import DataLoader, TrainTransform; "
                              "from yolox.exp import Exp; "
                              "e = Exp(); e.data_dir, e.num_classes; "
                              "import py_compile; "
                              "py_compile.compile(r'%s', cfile=r'%s', doraise=True); "
                              "print('bundled yolox full-chain OK')" % (
                                  os.path.join(STAGE, "yolox_repo"),
                                  os.path.join(STAGE, "yolox_bird_exp.py"),
                                  os.path.join(_tmpdir, "exp.pyc"))],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
        if chk.returncode != 0:
            fail(f"包内 yolox/exp 校验失败：{(chk.stderr or chk.stdout or '')[-500:]}")
        print("校验：bundled yolox 全链可导入（含 datasets/MosaicDetection）✓")
    finally:
        shutil.rmtree(_tmpdir, ignore_errors=True)
        # 兜底：万一有别的步骤又把 __pycache__ 落进包，压缩前统一清掉
        for _r, _ds, _ in os.walk(STAGE):
            for _d in list(_ds):
                if _d == "__pycache__":
                    shutil.rmtree(os.path.join(_r, _d), ignore_errors=True)

    # 过采样（COCO tiny/small ×3）会把同一 file_name 在 images 里登记多次——
    # 计数必须按**去重 file_name**对齐 referenced（实测 34,791 含重复 vs 去重 31,723）
    ann_files = set()
    n_annot_entries = 0
    for f in ("det_train.json", "det_val.json"):
        j = json.load(open(os.path.join(ann_dst, f), encoding="utf-8"))
        n_annot_entries += len(j["images"])
        ann_files.update(im["file_name"] for im in j["images"])
    if len(ann_files) != n_ref:
        fail(f"标注去重图 {len(ann_files)} != 引用 {n_ref}")
    print(f"校验：标注去重图 {len(ann_files):,} = 引用 {n_ref:,} ✓"
          f"（登记条目 {n_annot_entries:,} 含过采样重复，属正常）")

    # ---- 6b) 监控台校验（三条：产物未泄漏 / 脚本可编译 / 真能采到状态）----
    leaked = os.path.join(repo_dst, "YOLOX_outputs")
    if os.path.exists(leaked):
        fail(f"训练产物泄漏进包内：{leaked} —— 会让监控台认到别人的旧日志")
    mon_dst = os.path.join(STAGE, "detb_pack_monitor.py")
    if not os.path.exists(mon_dst):
        fail("监控台源码未打进包")
    chk2 = subprocess.run([PY, "-c",
                           "import py_compile; py_compile.compile(r'%s', doraise=True);"
                           "print('monitor compile OK')" % mon_dst],
                          capture_output=True, text=True)
    if chk2.returncode != 0:
        fail(f"监控台不可编译：{chk2.stderr[-300:]}")
    # 端到端：在**打包结果**上跑一次 collect()，确认它能独立定位日志/清单
    chk3 = subprocess.run(
        [PY, "-c",
         "import sys; sys.path.insert(0, r'%s');"
         "import detb_pack_monitor as M;"
         "d = M.collect(r'%s');"
         "assert 'status' in d and 'return_list' in d, d;"
         "print('monitor collect OK:', d['status']['text'])" % (STAGE, STAGE)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if chk3.returncode != 0:
        fail(f"监控台对打包结果采集失败：{(chk3.stderr or '')[-300:]}")
    print(f"校验：监控台可编译 + 可采集 ✓（{chk3.stdout.strip()}）")

    # ---- 7) zip ----
    if no_zip:
        print(f"构建完成（未压缩）：{STAGE}")
        return 0
    zip_path = os.path.join(DIST, "observebird_detb_train_pack_20261006.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"压缩中 -> {zip_path}（jpg 已压缩，用 STORED 提速）...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for root, dirs, files in os.walk(STAGE):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, STAGE))
    mb = os.path.getsize(zip_path) / 1048576
    print(f"包完成：{zip_path}（{mb:,.0f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
