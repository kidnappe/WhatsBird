# -*- coding: utf-8 -*-
"""App 改名后的数据迁移：com.observebird -> com.whatsbird。

背景（2026-10-09）：App 改包名后 Room 库名与照片目录都跟着变了
（observebird.db -> whatsbird.db，/data/user/0/com.observebird/ -> .../com.whatsbird/），
旧包数据不会自动迁移。本脚本经 adb 把旧包数据搬进新包。

## 为什么必须走 base64

`adb shell` 的输出经过设备端 PTY，**高位字节会被改写**：
实测把 observebird.db 拷出来，前 3 字节从 `53 51 4c`("SQL") 变成 `83 81 76`，
文件直接打不开。设备端文件本身是好的（`od` 验证过）。
→ 所以传输一律走 **base64（纯 ASCII）**，再在本地/设备端解码。
反向推送走 `adb push`（真文件传输，不过 PTY），不需要 base64。

## 用法

    python tools/migrate_data.py plan          # 只看会搬什么，不动手
    python tools/migrate_data.py run           # 实际执行

## 会做什么

1. 停掉新 App
2. 旧库 `observebird.db` -> 本地 -> 改写 photoPath 里的包名 -> 推送为 `whatsbird.db`
3. 旧库 WAL 为空但仍按例清理；删除新包残留的 -wal/-shm（否则旧数据会被 WAL 盖掉）
4. `files/records/*.jpg` 逐个搬到新包同名目录
5. `files/datastore/theme.preferences_pb` 一并带过去（主题偏好）

## 回退

执行前新包数据为空（本工具会先检查），旧包数据全程只读、原封不动。
最坏情况是迁移后新包数据有问题——删掉 `com.whatsbird` 重装即可回到干净状态。
"""
import base64
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

ADB = r"E:\code\code tools\android-sdk\platform-tools\adb.exe"
OLD_PKG = "com.observebird"
NEW_PKG = "com.whatsbird"
OLD_DIR = f"/data/user/0/{OLD_PKG}"
NEW_DIR = f"/data/user/0/{NEW_PKG}"
TMP = "/data/local/tmp"


def adb(*args, binary=False):
    """跑 adb，统一返回 str（内部用 bytes 缓冲，避免 Windows 控制台编码问题）。"""
    r = subprocess.run([ADB, *args], capture_output=True)
    raw = r.stdout if r.returncode == 0 else r.stderr
    return raw.decode("utf-8", "replace") if not binary else raw


def pull_b64(pkg, remote, local):
    """经 base64 从应用私有目录取文件（避开 PTY 的高位字节改写）。"""
    stem = os.path.basename(remote)
    dev = f"{TMP}/mig_{stem}.b64"
    adb("shell", f"run-as {pkg} base64 {remote} | tr -d '\\n\\r' > {dev}")
    b64_local = local + ".b64"
    if os.path.exists(b64_local):
        os.remove(b64_local)
    adb("pull", dev, b64_local)
    data = base64.b64decode(open(b64_local, "rb").read())
    with open(local, "wb") as f:
        f.write(data)
    os.remove(b64_local)
    return len(data)


def push_file(pkg, local, remote):
    """把本地文件推进应用私有目录（adb push 是真文件传输，不经 PTY）。"""
    dev = f"{TMP}/mig_{os.path.basename(remote)}"
    adb("push", local, dev)
    d = os.path.dirname(remote)
    if d:
        adb("shell", f"run-as {pkg} mkdir -p {d}")
    adb("shell", f"run-as {pkg} sh -c 'cat {dev} > {remote}'")
    adb("shell", f"rm -f {dev}")


def ls(pkg, remote):
    out = adb("shell", f"run-as {pkg} ls {remote}")
    return [l.split()[-1] for l in out.strip().splitlines()
            if l.strip() and not l.startswith("ls:")]


def db_rows(path):
    try:
        con = sqlite3.connect(path)
        n = con.execute("select count(*) from observations").fetchone()[0]
        con.close()
        return n
    except sqlite3.Error:
        return None


def main(mode):
    work = tempfile.mkdtemp(prefix="ob_mig_")
    print(f"临时目录 {work}\n")

    db_src = os.path.join(work, "old.db")
    n = pull_b64(OLD_PKG, f"{OLD_DIR}/databases/observebird.db", db_src)
    head = open(db_src, "rb").read(15)
    if head != b"SQLite format 3":
        print(f"❌ 旧库头异常 {head!r}，中止（不写任何东西）")
        return 1
    rows = db_rows(db_src)
    photos = [f for f in ls(OLD_PKG, f"{OLD_DIR}/files/records/") if f.endswith(".jpg")]
    print(f"旧包：库 {n} 字节 / {rows} 条记录；照片 {len(photos)} 张")
    print(f"      主题偏好 {'有' if ls(OLD_PKG, f'{OLD_DIR}/files/datastore/') else '无'}\n")

    new_db = os.path.join(work, "new.db")
    try:
        pull_b64(NEW_PKG, f"{NEW_DIR}/databases/whatsbird.db", new_db)
        n_new = db_rows(new_db)
    except Exception:
        n_new = None
    print(f"新包现有记录：{n_new}")
    if n_new:
        print("❌ 新包已有数据，不做整体替换（避免覆盖）。请先手动处理。")
        return 1

    if mode == "plan":
        print("\n（plan 模式，未做任何改动。确认无误后跑 run）")
        return 0

    # ── 1. 改写照片路径里的包名 ──
    con = sqlite3.connect(db_src)
    con.execute("update observations set photoPath = replace(photoPath, ?, ?)",
                (OLD_PKG, NEW_PKG))
    con.commit()
    left = con.execute(
        "select count(*) from observations where photoPath like ?",
        (f"%{OLD_PKG}%",)).fetchone()[0]
    con.close()
    print(f"\n① 照片路径改写完成，残留旧包名 {left} 条")

    # ── 2. 停 App、清 WAL、推库 ──
    adb("shell", f"am force-stop {NEW_PKG}")
    adb("shell", f"run-as {NEW_PKG} rm -f databases/whatsbird.db-wal databases/whatsbird.db-shm")
    push_file(NEW_PKG, db_src, f"{NEW_DIR}/databases/whatsbird.db")
    print(f"② 库已推送为 whatsbird.db（{os.path.getsize(db_src)} 字节）")

    # ── 3. 照片 ──
    ok = 0
    for f in photos:
        src = os.path.join(work, "p_" + f)
        try:
            pull_b64(OLD_PKG, f"{OLD_DIR}/files/records/{f}", src)
            push_file(NEW_PKG, src, f"{NEW_DIR}/files/records/{f}")
            ok += 1
        except Exception as e:
            print(f"   ⚠️ {f} 失败：{e}")
    print(f"③ 照片搬运 {ok}/{len(photos)}")

    # ── 4. 主题偏好 ──
    ds = ls(OLD_PKG, f"{OLD_DIR}/files/datastore/")
    if ds:
        src = os.path.join(work, "theme.preferences_pb")
        try:
            pull_b64(OLD_PKG, f"{OLD_DIR}/files/datastore/{ds[0]}", src)
            push_file(NEW_PKG, src, f"{NEW_DIR}/files/datastore/theme.preferences_pb")
            print(f"④ 主题偏好已带过去（{ds[0]}）")
        except Exception as e:
            print(f"   ⚠️ 主题偏好失败（可忽略，用默认主题）：{e}")
    shutil.rmtree(work, ignore_errors=True)
    print("\n完成。启动新 App 检查「日志/图鉴」是否出现 "
          f"{rows} 条记录与照片。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "plan"))