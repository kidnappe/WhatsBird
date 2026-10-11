# -*- coding: utf-8 -*-
"""盘点旧包（com.observebird）导出的 Room 库，确认迁移对象与行数。

背景（2026-10-09）：App 改名为 com.whatsbird 后 Room 库名 observebird.db -> whatsbird.db，
旧包数据不会自动迁移，需经 adb 手工搬运。本脚本只读地报出表结构与行数。

用法：python tools/mig_inspect.py <db路径>
"""
import sqlite3
import sys


def main(path):
    con = sqlite3.connect(path)
    tabs = [r[0] for r in con.execute(
        "select name from sqlite_master where type='table' order by name")]
    print(f"库: {path}")
    print(f"user_version = {con.execute('pragma user_version').fetchone()[0]}"
          f"   journal_mode = {con.execute('pragma journal_mode').fetchone()[0]}")
    print("表:")
    for t in tabs:
        n = con.execute(f"select count(*) from `{t}`").fetchone()[0]
        cols = [c[1] for c in con.execute(f"pragma table_info(`{t}`)")]
        print(f"  {t:<24} {n:>5} 行   列: {', '.join(cols)}")
    # 观测记录的时间跨度，帮判断是不是有价值的真实数据
    for t in tabs:
        cols = [c[1] for c in con.execute(f"pragma table_info(`{t}`)")]
        if "ts" in cols or "createdAt" in cols:
            col = "ts" if "ts" in cols else "createdAt"
            r = con.execute(
                f"select min(`{col}`), max(`{col}`) from `{t}`").fetchone()
            print(f"\n  {t} 的 `{col}` 范围: {r[0]} ~ {r[1]}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))