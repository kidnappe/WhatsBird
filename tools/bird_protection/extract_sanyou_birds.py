# -*- coding: utf-8 -*-
"""从《有重要生态、科学、社会价值的陆生野生动物名录》(三有名录, 2023) 抽出**鸟类**部分。

数据来源
--------
国家林业和草原局公告 2023 年第 17 号。官方附件是 57 页**扫描件 PDF**（pypdf 抽不出文字），
改用宁乡市政府转载页挂的 `.doc` 附件（标题即该公告），本机 Word COM 转成 `.docx` 后解析。

表格结构：单张表，按「纲」分节。节头行形如 `['', '哺乳纲 MAMMALIA 7目12科91种']`，
物种行形如 `['1', '毛猬', 'Hylomys suillus', '']`，另有目/科行（序号为空）。

用法
----
    python tools/bird_protection/extract_sanyou_birds.py           # 摘要
    python tools/bird_protection/extract_sanyou_birds.py --dump    # 写 raw/sanyou_2023_birds.csv
"""
import csv
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SRC = r"tools\bird_protection\raw\sanyou_national.docx"
OUT = r"tools\bird_protection\raw\sanyou_2023_birds.csv"

CLASS_ROW = re.compile(r"^(哺乳纲|鸟纲|爬行纲|两栖纲|硬骨鱼纲|昆虫纲|蛛形纲|甲壳纲|软体纲|寡毛纲|蛭纲|多足纲)")
BIRD_CLASS = re.compile(r"鸟纲|AVES")
LATIN = re.compile(r"^[A-Z][a-z]+(?: [a-z\-]+)+$")


def cell_text(tc):
    return re.sub(r"\s+", " ", "".join(t.text or "" for t in tc.iter(W + "t"))).strip()


def main():
    with zipfile.ZipFile(SRC) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    tbl = max(root.iter(W + "tbl"), key=lambda t: len(list(t.iter(W + "tr"))))

    cls = None
    per_class = {}
    birds = []
    for tr in tbl.findall(W + "tr"):
        cells = [cell_text(tc) for tc in tr.findall(W + "tc")]
        if not any(cells):
            continue
        joined = " ".join(cells)
        if CLASS_ROW.match(joined.lstrip()):
            cls = CLASS_ROW.match(joined.lstrip()).group(1)
            per_class.setdefault(cls, 0)
            continue
        # 物种行：有编号、有中文名、有拉丁名
        if len(cells) >= 3 and cells[0].isdigit() and LATIN.match(cells[2]):
            if cls:
                per_class[cls] = per_class.get(cls, 0) + 1
            if cls and BIRD_CLASS.search(cls):
                birds.append((int(cells[0]), cells[1], cells[2], cells[3] if len(cells) > 3 else ""))

    print("=== 各纲物种数 ===")
    for k, v in per_class.items():
        print("   %-8s %4d" % (k, v))
    print("\n=== 鸟纲：%d 种 ===" % len(birds))
    print("   前 5:", [b[1] for b in birds[:5]])
    print("   后 5:", [b[1] for b in birds[-5:]])
    dup = len(birds) - len({b[2] for b in birds})
    print("   拉丁名重复: %d" % dup)

    if "--dump" in sys.argv:
        with open(OUT, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["序号", "中文名", "学名", "备注"])
            w.writerows(birds)
        print("\n已写出 %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
