# -*- coding: utf-8 -*-
r"""onnx2tf 驱动：三重补丁修本机 2.6.9 flatbuffer_direct + -oiqt 的 Windows 崩链。

根因（2026-10-07 定案，证据见 docs/handoff/detector_b/HANDOFF-DETECTOR-B-INT8-SOLVED.md）：
-oiqt 时 flatbuffer_direct 的 fast-path 会以 allow_custom_ops=[False, True] 重试两次。
attempt#1：float32.tflite 写成功 → 校准阶段 collect_calibration_ranges_from_tflite
把 NCHW 样本喂进 NHWC 输入的 tflite → resize_tensor_input 抛 concatenation.cc
(1 != 2)，错误被重试循环吞掉，但异常 traceback 持有的 interpreter 没销毁，文件仍被
映射。attempt#2 重写同一 float32 路径 → Errno 22 Invalid argument。
（最小复实测：Interpreter 正常销毁后同路径重写是 OK 的——锁只来自未销毁的实例。）

三重防护：
① copy-on-open：LiteRT Interpreter 一律打开文件副本，任何 interpreter 都不映射
   onnx2tf 稍后要重写 / os.replace 的原文件（校准副本 + 变体校验副本两处）；
② write_model_file 重写失败且目标已存在非空 → 跳过重写保留旧文件；单次失败
   延迟重试一次（杀软抖动兜底）；
③ 被重试循环吞掉的 attempt 错误强制打印——attempt#1 的真实死因不再无声。

Usage: python run_onnx2tf_safe.py <onnx2tf 原参数...>
"""
import os
import shutil
import sys
import tempfile
import time
import traceback
import weakref

_orig_write = None


def _clean(p):
    s = str(p).strip()
    while s[:1] in ('"', "'", "\u201c", "\u2018") and s[-1:] in ('"', "'", "\u201d", "\u2019"):
        s = s[1:-1].strip()
    return s


def _patch_write():
    import onnx2tf.tflite_builder as b
    import onnx2tf.tflite_builder.model_writer as mw

    global _orig_write
    _orig_write = mw.write_model_file

    def write_model_file_patched(*, schema_tflite, model_ir, output_tflite_path, timing=None):
        p = _clean(output_tflite_path)
        last = None
        for attempt in (1, 2):
            try:
                return _orig_write(schema_tflite=schema_tflite, model_ir=model_ir,
                                   output_tflite_path=p, timing=timing)
            except OSError as e:
                last = e
                print(f"[safe-driver] write 失败(第{attempt}次) {p}: {e}", flush=True)
                if os.path.isfile(p) and os.path.getsize(p) > 0:
                    print(f"[safe-driver] 目标已存在非空({os.path.getsize(p)}B) "
                          f"→ 跳过重写保留旧文件", flush=True)
                    return None
                time.sleep(1.5)
        raise last

    mw.write_model_file = write_model_file_patched
    b.write_model_file = write_model_file_patched


def _patch_interpreter_copy_on_open():
    import ai_edge_litert.interpreter as aei

    real = aei.Interpreter

    def _rm(path):
        try:
            os.remove(path)
        except OSError:
            pass

    def interpreter_factory(model_path=None, model_content=None, **kw):
        if model_path is None or model_content is not None:
            return real(model_path=model_path, model_content=model_content, **kw)
        fd, tmp = tempfile.mkstemp(suffix=".tflite", prefix="onnx2tf_safe_")
        os.close(fd)
        shutil.copyfile(model_path, tmp)
        inst = real(model_path=tmp, model_content=None, **kw)
        try:
            weakref.finalize(inst, _rm, tmp)
        except TypeError:
            pass  # 副本留在 %TEMP%，不致命
        return inst

    aei.Interpreter = interpreter_factory


def _patch_attempt_logging():
    import onnx2tf.tflite_builder as b
    orig = b.export_tflite_model_flatbuffer_direct

    def logged(**kwargs):
        try:
            return orig(**kwargs)
        except BaseException:
            print("[safe-driver] export attempt 失败，真实错误如下：", flush=True)
            traceback.print_exc()
            raise

    b.export_tflite_model_flatbuffer_direct = logged


def main():
    sys.argv = ["onnx2tf"] + sys.argv[1:]

    _patch_write()
    _patch_attempt_logging()
    _patch_interpreter_copy_on_open()

    from onnx2tf import onnx2tf as core
    core.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
