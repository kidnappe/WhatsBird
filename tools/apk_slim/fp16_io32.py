"""把 onnx2tf 出的「fp16 张量 IO」tflite，改造成「fp16 权重 + fp32 IO」的形态。

为什么需要这一步（2026-10-13 APK 瘦身）：
  `E:\\code\\code tools\\models\\*\\export\\tflite\\model_float16.tflite` 是 onnx2tf 产的 fp16 版，
  体积对半（10.2MB → 5.1MB），但它的**输入输出张量也是 float16**。
  端上 `TfLiteClassifier` 用的是 `ByteBuffer.putFloat` 灌输入、`Array(1){FloatArray(dim)}` 取 logits，
  换成 fp16 IO 就得改 Kotlin（自己写 float↔half 转换），等于把一个纯配置问题变成代码风险。

  TFLite 官方的 float16 量化（`converter.target_spec.supported_types=[tf.float16]`）
  本来就是「权重 fp16、IO 保持 fp32」，端上零改动、数值只差权重舍入。
  本脚本就是把这个 IO 边界补回来：在图首/图尾各插一个 `CAST`。

做法（flatbuffer 对象 API，不碰任何权重字节）：
  1. 图输入张量 idx_in 保持 fp16 不动（下游 op 引用它的**下标**，下标不变就无需改任何 op）
     ——另建一个 fp32 副本作为新的图输入；
  2. 图首插 CAST(fp32→fp16)：新输入 → 原 tensor；
  3. 图尾插 CAST(fp16→fp32)：原输出张量 → 新 fp32 图输出；
  4. 同步 signature_def 的 tensor_index（否则按签名取张量的调用方会拿到 fp16 那个）。

原样保留：opcode 表、buffers、所有权重字节、算子拓扑。改完必须跑
`eval_fp16_vs_fp32.py` 比对 fp32 原件，不许只看体积就上机。

用法：
  python tools/apk_slim/fp16_io32.py --in a.tflite --out b.tflite
"""
import argparse
import copy
import sys

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema

FLOAT32 = 0
FLOAT16 = 1
BUILTIN_CAST = 114


def _clone_as_f32(t):
    """复制一个张量定义，只把 dtype 换成 float32（形状/量化参数/稀疏性照抄）。"""
    n = copy.deepcopy(t)
    n.type = FLOAT32
    n.buffer = 0          # 0 号 buffer 是 TFLite 约定的空 buffer；新张量不携带权重
    n.name = t.name + b"_io32"
    n.isVariable = False
    return n


def _cast_opcode_index(model):
    """找到 CAST 的 opcode 下标，没有就在表尾补一个（返回下标）。"""
    for i, oc in enumerate(model.operatorCodes):
        real = oc.builtinCode if oc.builtinCode < 127 else 127 + oc.deprecatedBuiltinCode
        if real == BUILTIN_CAST:
            return i
    oc = schema.OperatorCodeT()
    oc.builtinCode = BUILTIN_CAST       # <127 走 builtin_code 字段
    oc.deprecatedBuiltinCode = BUILTIN_CAST
    oc.version = 1
    model.operatorCodes.append(oc)
    return len(model.operatorCodes) - 1


def convert(src, dst):
    buf = open(src, "rb").read()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
    if len(model.subgraphs) != 1:
        sys.exit(f"{src}: 只处理单子图模型，实得 {len(model.subgraphs)}")
    sub = model.subgraphs[0]

    if len(sub.inputs) != 1 or len(sub.outputs) != 1:
        sys.exit(f"{src}: 只处理单输入单输出，实得 {len(sub.inputs)}/{len(sub.outputs)}")
    idx_in, idx_out = sub.inputs[0], sub.outputs[0]
    if sub.tensors[idx_in].type != FLOAT16 or sub.tensors[idx_out].type != FLOAT16:
        sys.exit(f"{src}: 输入输出本来就不是 fp16，无事可做")

    op_cast = _cast_opcode_index(model)

    # 1) 新的 fp32 图输入（原 fp16 张量保留原下标，下游 op 一行都不用改）
    new_in = _clone_as_f32(sub.tensors[idx_in])
    sub.tensors.append(new_in)
    idx_in32 = len(sub.tensors) - 1

    # 2) 新的 fp32 图输出
    new_out = _clone_as_f32(sub.tensors[idx_out])
    sub.tensors.append(new_out)
    idx_out32 = len(sub.tensors) - 1

    # InitFromObj 把定长向量还原成 tuple（不可写），要改先转成 list
    sub.inputs = list(sub.inputs)
    sub.outputs = list(sub.outputs)
    sub.inputs[0] = idx_in32
    sub.outputs[0] = idx_out32

    # 3) 图首 CAST：fp32 -> fp16
    op_head = schema.OperatorT()
    op_head.opcodeIndex = op_cast
    op_head.inputs = [idx_in32]
    op_head.outputs = [idx_in]
    op_head.builtinOptionsType = schema.BuiltinOptions.NONE
    sub.operators.insert(0, op_head)

    # 4) 图尾 CAST：fp16 -> fp32
    op_tail = schema.OperatorT()
    op_tail.opcodeIndex = op_cast
    op_tail.inputs = [idx_out]
    op_tail.outputs = [idx_out32]
    op_tail.builtinOptionsType = schema.BuiltinOptions.NONE
    sub.operators.append(op_tail)

    # 5) signature_def 跟着改，否则按签名取张量的调用方拿到的还是 fp16 那个
    for sd in model.signatureDefs or []:
        for tm in sd.inputs:
            if tm.tensorIndex == idx_in:
                tm.tensorIndex = idx_in32
                tm.name = new_in.name
        for tm in sd.outputs:
            if tm.tensorIndex == idx_out:
                tm.tensorIndex = idx_out32
                tm.name = new_out.name

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    data = bytes(builder.Output())
    with open(dst, "wb") as w:
        w.write(data)
    return len(buf), len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()
    b, c = convert(a.src, a.dst)
    print(f"{a.src}\n  -> {a.dst}\n  {b/1048576:.2f}MB -> {c/1048576:.2f}MB")


if __name__ == "__main__":
    main()