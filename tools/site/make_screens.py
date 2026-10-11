# -*- coding: utf-8 -*-
r"""把真机截图里的图片区换成训练集/开放许可的鸟照，并按模型真实检出框画框。

做两件事：
  1. 换图——UI 文字、按钮、卡片全是真机截图原样，只把「照片区」的内容换掉。
     换哪些图、换成哪个物种，由 site/shots/manifest.json 决定（fetch_photos.py 产出，
     已过「许可干净 + 模型 top-1 认得出」两道筛）。
  2. 画框——取景页那张按 manifest 里 YOLOX 的**真实检出框**画黄框，并标上分类器
     真实 top-1 与概率。不是画着玩的，是本仓库模型跑出来的数。

圆角不猜：每个图片区先从原截图里量出 App 画的圆角半径，再按同一半径贴回去，
所以四角不会出现方形硬边。

Usage: python make_screens.py
"""
import json
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHOTS = r"E:\code\observebird\site\shots"
FRESH = r"E:\code\observebird\.shots\fresh"
MANIFEST = os.path.join(SHOTS, "manifest.json")

BOX_YELLOW = (255, 214, 10)   # DESIGN.md §2.5 检测框黄，仅取景页仪表色

# 各页面的图片区：[x0, y0, x1, y1]，坐标从 uiautomator dump 的 ImageView 节点来
REGIONS = {
    "guide": [("hong-er", (641, 1223, 1130, 1712)),
              ("hongshui", (67, 2065, 561, 2559)),
              ("huixique", (641, 2068, 1128, 2555))],
    "records": [("huixique", (78, 825, 241, 988)),
                ("maque", (78, 1074, 241, 1237)),
                ("huixique", (78, 1323, 241, 1486)),
                ("huixique", (78, 1572, 241, 1735)),
                ("huixique", (78, 1821, 241, 1984)),
                ("huixique", (78, 2070, 241, 2233)),
                ("huixique", (78, 2319, 241, 2482))],
    "detail2": [("hong-er", (65, 170, 1135, 885))],
}
OUT = {"guide": "guide.jpg", "records": "records.jpg", "detail2": "guide-detail.jpg"}


def cover(img, w, h):
    """按 cover 规则缩放裁切到 w×h，返回 (图, 变换系数)。"""
    iw, ih = img.size
    s = max(w / iw, h / ih)
    nw, nh = round(iw * s), round(ih * s)
    im = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return im.crop((left, top, left + w, top + h)), s, left, top


def corner_radius(page, x0, y0, x1, y1):
    """量 App 画的圆角半径：角上最远的「仍是卡片底色」的那个像素距离 d，r ≈ 3.41d。"""
    px = page.load()
    bg = px[x0 - 8, y0 + (y1 - y0) // 2]
    d = 0
    for t in range(2, 90):
        c = px[min(x0 + t, x1 - 1), min(y0 + t, y1 - 1)]
        if sum(abs(c[i] - bg[i]) for i in range(3)) < 90:
            d = t
        else:
            break
    return max(8, round(d * 3.41))


def paste_rounded(page, photo_path, box, radius):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    crop, s, left, top = cover(Image.open(photo_path).convert("RGB"), w, h)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    page.paste(crop, (x0, y0), mask)
    return s, left, top


def camera_window(page):
    """取景页没有语义节点，从像素上找相机窗：连续暗行/暗列的交集。"""
    g = page.convert("L")
    w, h = g.size
    px = g.load()
    step = 4
    rows = [y for y in range(0, h, step)
            if sum(1 for x in range(0, w, 20) if px[x, y] < 110) > (w / 20) * 0.6]
    if not rows:
        return None
    y0, y1 = rows[0], rows[-1]
    x0 = min(x for y in range(y0, y1, step) for x in range(0, w, step) if px[x, y] < 110)
    x1 = max(x for y in range(y0, y1, step) for x in range(0, w, step) if px[x, y] < 110)
    return (x0, y0, x1 + 1, y1 + 1)


def main():
    man = json.load(open(MANIFEST, encoding="utf-8"))

    for page_key, regions in REGIONS.items():
        src = os.path.join(FRESH, page_key + ".png")
        page = Image.open(src).convert("RGB")
        for key, box in regions:
            if key not in man:
                print("  跳过（manifest 里没有 %s）：%s" % (key, page_key))
                continue
            r = corner_radius(page, *box)
            paste_rounded(page, man[key]["path"], box, r)
        out = os.path.join(SHOTS, OUT[page_key])
        page.resize((560, round(page.height * 560 / page.width)), Image.LANCZOS)\
            .save(out, "JPEG", quality=82, optimize=True, progressive=True)
        print("%-9s -> %s（换了 %d 处）" % (page_key, os.path.basename(out), len(regions)))

    # 取景页：换图 + 画模型真实检出框
    src = os.path.join(FRESH, "recognition.png")
    page = Image.open(src).convert("RGB")
    win = camera_window(page)
    if not win:
        print("取景页：没找到相机窗，跳过")
        return
    key = "queshu"
    if key not in man:
        return
    m = man[key]
    r = corner_radius(page, *win)
    s, left, top = paste_rounded(page, m["path"], win, r)
    d = ImageDraw.Draw(page, "RGBA")
    bx0, by0, bx1, by1 = m["det_box"]
    X0 = win[0] + (bx0 - left) * s
    Y0 = win[1] + (by0 - top) * s
    X1 = win[0] + (bx1 - left) * s
    Y1 = win[1] + (by1 - top) * s
    t = m["top5"][0]
    for (ax, ay, dx, dy) in [(X0, Y0, 1, 1), (X1, Y0, -1, 1), (X0, Y1, 1, -1), (X1, Y1, -1, -1)]:
        d.line([ax, ay, ax + 44 * dx, ay], fill=BOX_YELLOW, width=6)
        d.line([ax, ay, ax, ay + 44 * dy], fill=BOX_YELLOW, width=6)
    try:
        from PIL import ImageFont
        f = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 40)
    except Exception:
        f = None
    tag = "%s  %.0f%%" % (t["zh"], t["p"] * 100)
    tw = d.textlength(tag, font=f) if f else 260
    d.rounded_rectangle([X0, Y0 - 62, X0 + tw + 34, Y0 - 12], radius=12, fill=(0, 0, 0, 175))
    d.text((X0 + 16, Y0 - 56), tag, font=f, fill=(255, 255, 255))
    out = os.path.join(SHOTS, "recognition.jpg")
    page.resize((560, round(page.height * 560 / page.width)), Image.LANCZOS)\
        .save(out, "JPEG", quality=82, optimize=True, progressive=True)
    print("recognition -> recognition.jpg（相机窗 %s，真实检出框 det=%.2f，top1=%s %.0f%%）"
          % (win, m["det_score"], t["zh"], t["p"] * 100))


if __name__ == "__main__":
    main()