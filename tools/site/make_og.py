# -*- coding: utf-8 -*-
r"""生成分享卡 og.png（1200×630），给微信 / 微博 / Slack 等转发时预览用。

版面照 App 的纸感 token 走（docs/design/DESIGN.md §2.2）：纸底 + 苔绿 + 琥珀刻度。
左边标题与一行硬指标，右边摆四张真机截图的缩略图（就是页面里的那四张）。

Usage: python make_og.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

SITE = r"E:\code\observebird\site"
SHOTS = os.path.join(SITE, "shots")
OUT = os.path.join(SITE, "og.png")
W, H = 1200, 630

PAPER = (247, 244, 236)
CARD = (255, 253, 248)
INK = (31, 42, 34)
INK_SOFT = (85, 99, 90)
MOSS = (58, 85, 64)
AMBER = (221, 164, 90)
LINE = (228, 223, 207)

F_CJK = r"C:\Windows\Fonts\msyh.ttc"
F_CJK_B = r"C:\Windows\Fonts\msyhbd.ttc"
F_SERIF_I = r"C:\Windows\Fonts\georgiai.ttf"


def f(path, size):
    return ImageFont.truetype(path, size)


img = Image.new("RGB", (W, H), PAPER)
d = ImageDraw.Draw(img)

# 纸面细颗粒感（轻微噪点，别抢内容）
import random
random.seed(7)
for _ in range(9000):
    x, y = random.randint(0, W - 1), random.randint(0, H - 1)
    v = random.choice([(250, 248, 242), (243, 240, 232)])
    d.point((x, y), fill=v)

# 顶部琥珀刻度条（一色一义：琥珀=刻度）
d.rectangle([0, 0, W, 8], fill=AMBER)

# 左上 App 图标
icon = Image.open(os.path.join(SITE, "icon-192.png")).convert("RGBA")
icon = icon.resize((72, 72), Image.LANCZOS)
img.paste(icon, (72, 72), icon)

d.text((164, 82), "拾鸟 WhatsBird", font=f(F_CJK_B, 40), fill=INK)

d.text((72, 196), "举起手机对准鸟，", font=f(F_CJK_B, 62), fill=INK)
d.text((72, 276), "告诉你它是什么。", font=f(F_CJK_B, 62), fill=INK)
d.text((76, 368), "WhatsBird", font=f(F_SERIF_I, 38), fill=MOSS)
d.text((300, 372), "· 拍照认鸟的 Android 应用", font=f(F_CJK, 28), fill=INK_SOFT)

# 硬指标条
stats = [("384", "种图鉴"), ("6", "套区域模型"), ("0", "账号 / 上传"), ("离线", "全程")]
x = 76
for num, label in stats:
    d.text((x, 436), num, font=f(F_CJK_B, 44), fill=MOSS)
    w = d.textlength(num, font=f(F_CJK_B, 44))
    d.text((x, 490), label, font=f(F_CJK, 22), fill=INK_SOFT)
    x += int(max(w, 96) + 56)
    d.line([x - 34, 446, x - 34, 508], fill=LINE, width=2)

d.text((76, 548), "识别全程离线 · 照片与观测记录只留在本机", font=f(F_CJK, 24), fill=INK_SOFT)

# 右侧四张真机截图
names = ["recognition.jpg", "records.jpg", "guide.jpg", "guide-detail.jpg"]
tw, th = 150, 334
gap = 18
x0 = W - 72 - (tw * 4 + gap * 3)
for i, n in enumerate(names):
    p = os.path.join(SHOTS, n)
    if not os.path.exists(p):
        continue
    im = Image.open(p).convert("RGB")
    s = max(tw / im.width, th / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im = im.crop(((im.width - tw) // 2, (im.height - th) // 2,
                   (im.width - tw) // 2 + tw, (im.height - th) // 2 + th))
    x = x0 + i * (tw + gap)
    img.paste(im, (x, 150))
    d.rectangle([x, 150, x + tw - 1, 150 + th - 1], outline=LINE, width=2)

img.save(OUT, "PNG", optimize=True)
print("写出", OUT, img.size, os.path.getsize(OUT), "bytes")