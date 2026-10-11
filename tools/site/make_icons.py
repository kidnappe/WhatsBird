# -*- coding: utf-8 -*-
r"""把 App 的自适应图标合成成网页用的图标。

素材取自 android/app/src/main/res：
  mipmap-xxxhdpi/ic_launcher_foreground.png  前景（432×432）
  values/colors.xml → ic_launcher_bg          底色（#FBE6CF）

自适应图标在桌面上的画法是「底色 + 前景居中 + 裁成圆角」，
这里照着拼一遍，再补一份「不加圆角」的方形版给 apple-touch-icon。

产出（都在 site/）：
  icon-512.png / icon-192.png / favicon-32.png / apple-touch-icon.png

Usage: python make_icons.py
"""
import os

from PIL import Image, ImageDraw

RES = r"E:\code\observebird\android\app\src\main\res"
FG = os.path.join(RES, "mipmap-xxxhdpi", "ic_launcher_foreground.png")
BG = "#FBE6CF"          # values/colors.xml: ic_launcher_bg
DST = r"E:\code\observebird\site"

os.makedirs(DST, exist_ok=True)
fg = Image.open(FG).convert("RGBA")

# 自适应图标的前景只占中间 ~72/108，安全区外的部分会被裁掉；
# 网页图标不需要那圈留白，按前景实际内容裁到最小非透明包围盒再放大。
bbox = fg.getbbox()
fg = fg.crop(bbox)


def rounded(size, radius_ratio=0.22, square=False):
    scale = size / max(fg.size)
    w = max(1, round(fg.width * scale))
    h = max(1, round(fg.height * scale))
    fg_s = fg.resize((w, h), Image.LANCZOS)

    # 前景在 432 画布里是居中的，裁掉留白后按同样比例补一点边距
    pad = round(size * 0.04)
    inner = size - pad * 2
    s2 = min(inner / w, inner / h)
    fg_s = fg.resize((max(1, round(w * s2)), max(1, round(h * s2))), Image.LANCZOS)

    bg = Image.new("RGBA", (size, size), BG)
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    if square:
        d.rectangle([0, 0, size - 1, size - 1], fill=255)
    else:
        d.rounded_rectangle([0, 0, size - 1, size - 1],
                            radius=round(size * radius_ratio), fill=255)
    bg.paste(fg_s, ((size - fg_s.width) // 2, (size - fg_s.height) // 2), fg_s)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(bg, (0, 0), mask)
    return out


for name, size, square in [("icon-512.png", 512, False),
                           ("icon-192.png", 192, False),
                           ("favicon-32.png", 32, False),
                           ("apple-touch-icon.png", 180, True)]:
    p = os.path.join(DST, name)
    rounded(size, square=square).save(p, "PNG", optimize=True)
    print("%-22s %4d  %5d B" % (name, size, os.path.getsize(p)))
print("图标来自 %s" % FG)