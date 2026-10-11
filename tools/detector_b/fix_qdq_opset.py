# -*- coding: utf-8 -*-
r"""QDQ onnx 兼容补丁：图 opset 11 → 13。

根因：export_b_onnx.py 以 opset 11 导出，quantize_static 沿用图 opset 给 Q/DQ 节点
写 per-channel 的 axis 属性——opset 11 的 Q/DQ schema 没有 axis，ORT ≥1.2x 直接拒载
（"Unrecognized attribute: axis"）。opset 13 起 axis 才是 Q/DQ 的合法属性，
故只需整体升 opset，**不动节点结构**（axis 属性原样保留）。

用官方 version_converter 处理 11→13 的连带差异（Unsqueeze/Squeeze axes 属性→输入等），
比手改节点可靠。

Usage: python fix_qdq_opset.py <in_qdq.onnx> [out_qdq.onnx]  # 缺省覆写原文件
"""
import sys

import onnx
from onnx import version_converter


def fix(path, out):
    m = onnx.load(path)
    opset = next(o.version for o in m.opset_import if o.domain == "" or o.domain == "ai.onnx")
    if opset >= 13:
        print(f"已是 opset {opset}，无需处理")
        onnx.save(m, out)
        return
    m13 = version_converter.convert_version(m, 13)
    onnx.checker.check_model(m13)
    onnx.save(m13, out)
    print(f"opset {opset} -> 13，checker 通过 -> {out}")


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    fix(src, dst)
