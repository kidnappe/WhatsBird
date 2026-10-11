# -*- coding: utf-8 -*-
"""B 路线训练 exp：bird 单类 YOLOX-s（docs/handoff/detector_b/HANDOFF.md §3 阶段 2）。

数据：make_train_json.py 产出的 det_train/det_val（伪标签华北 + COCO bird 过采样）。
热启：yolox_s.pth（COCO 80 类预训练）——官方 train.py -c 会自动跳过类别头不匹配的键。
尺寸：环境变量 B_SIZE=512/576/640（默认 512）；512 先训，PC bench 后再定要不要 576/640。

用法（在 yolox_repo 根目录）：
  B_SIZE=512 python tools/train.py -f E:\\code\\observebird\\tools\\detector_b\\yolox_bird_exp.py \
      -c E:\\code\\code tools\\detectors\\yolox\\yolox_s.pth -b 8 --fp16 --seed 42
显存现实（6GB 笔记本卡 + 桌面占 ~3GB）：b8 OOM 就退 b4；fp16 必开。
"""
import os

from yolox.exp import Exp as MyExp

SIZE = int(os.environ.get("B_SIZE", "512"))


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33       # YOLOX-s
        self.width = 0.50
        self.exp_name = f"yolox_bird_{SIZE}"

        self.data_dir = r"E:\code\code tools\data\detector_b"
        self.train_ann = "det_train.json"   # 实际读 <data_dir>/annotations/<ann>
        self.val_ann = "det_val.json"
        self.num_classes = 1                # 单类 bird；Kotlin 解码通道=6（BIRD_CLS_OFFSET=5+0）

        self.input_size = (SIZE, SIZE)
        self.test_size = (SIZE, SIZE)

        self.max_epoch = 50
        self.no_aug_epochs = 10             # 末 10 轮关 mosaic/增强，收尾在真分布
        self.eval_interval = 5
        self.data_num_workers = 2           # 15GB RAM 项目惯例 0~2
        self.print_interval = 20
