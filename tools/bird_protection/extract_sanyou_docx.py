# -*- coding: utf-8 -*-
"""从《有重要生态、科学、社会价值的陆生野生动物名录》(三有名录, 2023) 的 Word 版抽物种表。

数据来源
--------
国家林业和草原局公告 2023 年第 17 号，官方附件是 57 页**扫描件 PDF**（pypdf 抽不出文字）。
改用宁乡市政府转载页挂的 `.doc` 附件（标题即该公告），本机 Word COM 转成 `.docx` 后解析。

用法
----
    python tools/bird_protection/extract_sanyou_docx.py            # 打印结构摘要
    python tools/bird_protection/extract_sanyou_docx.py --dump     # 另存 raw/sanyou_2023_rows.csv
"""
import csv
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SRC = r"tools\bird_protection\raw\sanyou_national.docx"
OUT = r"tools\bird_protection\raw\sanyou_2023_rows.csv"


def cell_text(tc):
    """一个单元格里的全部 w:t 拼起来（一个词可能被拆成多个 run）。"""
    return re.sub(r"\s+", " ", "".join(t.text or "" for t in tc.iter(W + "t"))).strip()


def rows_of(tbl):
    for tr in tbl.findall(W + "tr"):
        yield [cell_text(tc) for tc in tr.findall(W + "tc")]


def main():
    with zipfile.ZipFile(SRC) as z:
        root = ET.fromstring(z.read("word/document.xml"))

    tables = list(root.iter(W + "tbl"))
    print("表格数: %d" % len(tables))
    for i, tbl in enumerate(tables):
        rs = list(rows_of(tbl))
        widths = sorted({len(r) for r in rs})
        print("  表 %d: %d 行, 列数集合 %s" % (i, len(rs), widths))

    # 取行数最多的那张表（物种表）
    main_tbl = max(tables, key=lambda t: len(list(t.iter(W + "tr"))))
    rows = [r for r in rows_of(main_tbl) if any(c for c in r)]
    print("\n主表 %d 行（非空），前 8 行：" % len(rows))
    for r in rows[:8]:
        print("   %s" % r)

    # 只保留「像物种行」的：4 列以上、第 2 列是中文名、第 3 列是拉丁名
    species = [r for r in rows if len(r) >= 3 and re.search(r"[A-Z][a-z]+", r[2])]
    print("\n判定为物种行: %d 条" % len(species))
    print("后 5 行：")
    for r in species[-5:]:
        print("   %s" % r)

    if "--dump" in sys.argv:
        with open(OUT, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["序号", "中文名", "拉丁名", "备注"])
            for r in species:
                w.writerow((r + ["", "", "", ""])[:4])
        print("\n已写出 %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
