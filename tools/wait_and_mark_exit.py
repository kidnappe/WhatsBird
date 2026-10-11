"""已废弃（2026-10-05）：本脚本不再做任何事，保留空壳以免有人再发明一个标记文件。

它原来的唯一作用是「轮询一个 PID，等它退出后写 gbif_build.log.<ts>.exit 标记」，
用来补 auto_region_pipeline 被杀之后没人写 .exit 的缺口。

而那个标记本身就是这次「华东只下了 57/150 却显示 100% 已完成」事故的直接原因：

  * 它不区分「脚本跑完了」和「进程被杀掉了」—— 两种情况写出的标记一模一样；
  * 本脚本站位写死 "0"（成功码），于是"被杀"也被读成"成功结束"；
  * 加上另外两个写入者（run_detached.ps1 的 cmd 包装、auto_region_pipeline），
    盘上 30 个 .exit 内容五花八门：8 个空文件、4 个 "ECHO is off."、10 个 15、2 个 -1。

现在**完成状态只有一个来源**：`<data_dir>/progress.json`
  stage = done / interrupted / failed
  + exit_code / species_done / species_total / finished_at / pid
由 `build_dataset_gbif.py` 自己在**每一条退出路径**上用 `finalize_progress()` 写。
监控台、队列脚本、人工排查都只读它。

因此：
  * 不要再调用本脚本；
  * 不要再写或读任何 `*.exit` 标记；
  * 想判断"下载结束了没" —— 读 progress.json 的 `stage`；
    想判断"进程是不是硬死了（kill -9，没来得及写终态）" —— 看 progress.json
    的 `ts` / 日志 mtime 是否已经长时间不动（监控台按 15 分钟判定并显示
    "进程已退出（未见收尾行）"）。
"""
import sys

if __name__ == "__main__":
    print("[deprecated] wait_and_mark_exit.py 已废弃：完成状态请读 "
          "<data_dir>/progress.json 的 stage 字段（done/interrupted/failed），"
          "不要再写/读 .exit 标记。", flush=True)
    sys.exit(0)
