# 画一张「识别状态示意」合成演示图。
#
# 这张图是**合成演示**，不是 App 实拍：底图是华中数据集样张，
# 检测框 / 物种名 / 置信度按 docs/design/DESIGN.md 的仪表色口径画上去，
# 页面文案里也写明了「合成演示」。物种名不是编的——鹊鸲 Copsychus saularis
# 是对着 android/app/src/main/assets/guide/guide_*.json 核过的（858 条图鉴里能查到）。
#
# 用法：python tools/site/demo_identify.py
import os
from PIL import Image, ImageDraw, ImageFont

SRC = r"E:\code\observebird\docs\design\preview-paper\assets\magrobin.jpg"
OUT = r"E:\code\observebird\site\shots\demo-identify.jpg"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

S = 3  # 3 倍超采样再缩回去，文字和线条才不毛边
W, H = 980 * S, 620 * S

# Paper Almanac token（DESIGN.md §2.2）
PAPER = (247, 244, 236)
CARD = (255, 253, 248)
INK = (31, 42, 34)
INK_SOFT = (85, 99, 90)
MOSS = (58, 85, 64)
SAGE = (233, 240, 230)
OLIVE = (91, 107, 79)
AMBER = (221, 164, 90)
BOX_YELLOW = (255, 214, 10)   # 检测框黄，仅取景页仪表色
LINE = (228, 223, 207)

F_CJK = r"C:\Windows\Fonts\msyh.ttc"
F_CJK_B = r"C:\Windows\Fonts\msyhbd.ttc"
F_SERIF_I = r"C:\Windows\Fonts\georgiai.ttf"


def f(path, size):
    return ImageFont.truetype(path, size * S)


img = Image.new("RGB", (W, H), PAPER)
d = ImageDraw.Draw(img, "RGBA")

# ---- 顶部状态条 ----
d.text((28 * S, 22 * S), "拾鸟 WhatsBird", font=f(F_CJK_B, 17), fill=INK)
chip_txt = "区域 · 华东"
cw = d.textlength(chip_txt, font=f(F_CJK, 14)) + 26 * S
d.rounded_rectangle([W - 28 * S - cw, 20 * S, W - 28 * S, 50 * S], radius=15 * S, fill=SAGE)
d.text((W - 28 * S - cw + 13 * S, 26 * S), chip_txt, font=f(F_CJK, 14), fill=OLIVE)

# ---- 取景窗口（纸框里的圆角窗，取景页是暗的，这里压一层暗色）----
wx0, wy0, wx1, wy1 = 28 * S, 66 * S, W - 28 * S, H - 150 * S
r = 14 * S
d.rounded_rectangle([wx0, wy0, wx1, wy1], radius=r, fill=(26, 32, 28))

photo = Image.open(SRC).convert("RGB")
# 裁成窗口比例后铺满（cover）
pw, ph = wx1 - wx0, wy1 - wy0
scale = max(pw / photo.width, ph / photo.height)
photo = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.LANCZOS)
left = (photo.width - pw) // 2
top = (photo.height - ph) // 2
photo = photo.crop((left, top, left + pw, top + ph))

mask = Image.new("L", (pw, ph), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, pw - 1, ph - 1], radius=r, fill=255)
img.paste(photo, (wx0, wy0), mask)

# ---- 检测框（黄）+ 四角刻度 ----
bx0, by0 = wx0 + int(pw * 0.26), wy0 + int(ph * 0.16)
bx1, by1 = wx0 + int(pw * 0.80), wy0 + int(ph * 0.78)
lw = 3 * S
for (ax, ay, dx, dy) in [(bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1)]:
    d.line([ax, ay, ax + 22 * S * dx, ay], fill=BOX_YELLOW, width=lw)
    d.line([ax, ay, ax, ay + 22 * S * dy], fill=BOX_YELLOW, width=lw)

# ---- 框标签（半透黑底白字）----
lab = "鹊鸲  76%"
lw_txt = d.textlength(lab, font=f(F_CJK_B, 15))
px0, py0 = bx0, by0 - 30 * S
d.rounded_rectangle([px0, py0, px0 + lw_txt + 20 * S, py0 + 24 * S], radius=6 * S, fill=(0, 0, 0, 170))
d.text((px0 + 10 * S, py0 + 3 * S), lab, font=f(F_CJK_B, 15), fill=(255, 255, 255))

# ---- 底部身份卡（纸感：大字中文名 + 学名衬线斜体 + 置信条）----
cx0, cy0 = 28 * S, H - 132 * S
cx1, cy1 = W - 28 * S, H - 28 * S
d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=14 * S, fill=CARD, outline=LINE, width=2 * S)

d.text((cx0 + 22 * S, cy0 + 14 * S), "鹊鸲", font=f(F_CJK_B, 30), fill=INK)
d.text((cx0 + 132 * S, cy0 + 26 * S), "Copsychus saularis", font=f(F_SERIF_I, 17), fill=MOSS)

# 置信度条
bar_x0, bar_y0 = cx0 + 22 * S, cy0 + 74 * S
bar_x1, bar_y1 = cx1 - 150 * S, cy0 + 86 * S
d.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=6 * S, fill=(228, 223, 207))
d.rounded_rectangle([bar_x0, bar_y0, bar_x0 + int((bar_x1 - bar_x0) * 0.76), bar_y1], radius=6 * S, fill=MOSS)
d.text((bar_x1 + 14 * S, cy0 + 68 * S), "76%", font=f(F_CJK_B, 15), fill=MOSS)

# 琥珀小刻度（时间/刻度用琥珀，一色一义）
for i in range(5):
    x = cx0 + 22 * S + i * 26 * S
    d.line([x, cy1 - 22 * S, x, cy1 - 12 * S], fill=AMBER, width=2 * S)

img = img.resize((980, 620), Image.LANCZOS)
img.save(OUT, "JPEG", quality=86, optimize=True, progressive=True)
print("written:", OUT, img.size, os.path.getsize(OUT), "bytes")