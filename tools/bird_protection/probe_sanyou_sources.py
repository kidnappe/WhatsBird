# -*- coding: utf-8 -*-
"""探针：找《有重要生态、科学、社会价值的陆生野生动物名录》(三有名录, 2023) 的可机读来源。

背景：gov.cn 的官方附件 P020230701333096616886.pdf 是 **57 页扫描件**，pypdf 抽不出文字。
本脚本遍历搜索到的省级转载页，看有没有挂 Excel/Word（文本型，可直接抽）附件。
"""
import gzip
import re
import sys
import urllib.request
from urllib.parse import urljoin

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/120 Safari/537.36"
}

PAGES = [
    "http://www.nxcity.gov.cn/zw229/bmxxgkml/zfgzbm85048504/7093/9140/"
    "gfxwj_138501/202307/t20230705_11156675.html",
    "https://lcj.yn.gov.cn/html/2024/zdlyxxgkqt_0930/72312.html",
    "http://lyj.xianning.gov.cn/bsfw/ggfwcx/202311/t20231121_3367242.shtml",
]

ATTR = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""", re.I)
FILE_EXT = re.compile(r"\.(pdf|xls|xlsx|doc|docx|zip|rar|txt|csv)(\?|$)", re.I)
SUSPECT = re.compile(r"(P020|W020|attach|upload|file|nrs)", re.I)


def fetch(url: str) -> str:
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", "replace")


def main() -> int:
    for page in PAGES:
        print("--- %s" % page.split("/")[-1])
        try:
            html = fetch(page)
        except Exception as exc:  # noqa: BLE001
            print("     ERR %s" % str(exc)[:80])
            continue
        links = set()
        for m in ATTR.finditer(html):
            href = m.group(1)
            if FILE_EXT.search(href) or SUSPECT.search(href):
                links.add(urljoin(page, href))
        if links:
            for link in sorted(links)[:14]:
                print("      %s" % link)
        else:
            print("      (无附件链接)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
