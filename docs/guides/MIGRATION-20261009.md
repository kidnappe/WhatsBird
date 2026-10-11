# 改包名后的数据迁移（com.observebird → com.whatsbird）

> 2026-10-09 执行。App 改名后 Room 库名与照片目录都跟着变了，旧包数据不会自动迁移，
> 本文记录搬运方法、那个把二进制文件搞坏的坑，以及校验时踩到的两个假象。
> 工具：`tools/migrate_data.py`（主）、`tools/mig_inspect.py`（盘点）、`tools/mig_dump.py`（核对）。

---

## 1. 到底要搬什么

| 内容 | 旧位置 | 新位置 |
|---|---|---|
| Room 库 | `/data/user/0/com.observebird/databases/observebird.db` | `/data/user/0/com.whatsbird/databases/whatsbird.db` |
| 记录照片 | `/data/user/0/com.observebird/files/records/obs_*.jpg` | `/data/user/0/com.whatsbird/files/records/` |
| 主题偏好 | `.../files/datastore/theme.preferences_pb` | 同名 |

**还有一处容易漏**：观测表里 `photoPath` 存的是**绝对路径**，路径里带着包名。
不重写的话记录能列出来、**照片全是裂图**。迁移脚本用
`update observations set photoPath = replace(photoPath,'com.observebird','com.whatsbird')` 处理。

**库版本无需处理**：两包 `user_version` 都是 3，新包 Room 已声明
`MIGRATION_1_2` + `MIGRATION_2_3`，直接打开即可，不会触发重建。

---

## 2. ⚠️ 最大的坑：`adb shell` 会改写二进制

`adb shell` 的输出经过设备端 PTY，**高位字节被改写**：

```
设备端真实首字节（run-as + od 实测）：53 51 4c 69 74 65 20 66 … = "SQLite f"
直接 exec-out 拷到本地：              83 81 76 69 74 65 20 66 …
```

前 3 字节 `SQL` → `+81`，文件直接打不开（`sqlite3.DatabaseError: file is not a database`）。

**用 PowerShell 的 `>` 更糟**——那是文本重定向，53 KB 的库拷出来变成 103 KB。

### 对策

| 方向 | 做法 | 为什么 |
|---|---|---|
| **拉取** | 设备端 `run-as <pkg> base64 <file> \| tr -d '\n\r' > /data/local/tmp/x.b64`，再 `adb pull`，本地 base64 解码 | base64 是纯 ASCII，过 PTY 不变形；传输走 adb pull 是真文件传输 |
| **推送** | `adb push 本地文件 /data/local/tmp/x`，再 `run-as <pkg> sh -c 'cat /data/local/tmp/x > <目标>'` | `adb push` 本身不过 PTY，不需要 base64 |

设备端 `/data/local/tmp` 由 shell 用户持有且权限开放，`run-as` 进去的 App 能读。

---

## 3. 迁移步骤（`tools/migrate_data.py`）

```bash
python tools/migrate_data.py plan    # 只盘点，不动手
python tools/migrate_data.py run     # 实际执行
```

`run` 的动作顺序：

1. **force-stop 新 App**
2. 改写旧库 `photoPath` 里的包名（先检查残留数应为 0）
3. **删掉新包的 `-wal` / `-shm`**——残留的 WAL 会把刚推进去的库盖掉
4. 推送为 `whatsbird.db`
5. 逐张搬照片（同名落 `files/records/`）
6. 带 `theme.preferences_pb`

**安全约束**：`plan` 与 `run` 都会先读新包现有记录数，**非 0 就拒绝执行**（不做覆盖）。
旧包全程只读。回退 = 卸载 `com.whatsbird` 重装，回到干净状态。

---

## 4. 校验时的两个假象（都曾让我误报）

**① App 正在运行时回读主库文件会读到不一致快照。**
Room 开着 WAL，此时只导 `whatsbird.db`（不含 `-wal` 内容）得到的是残缺快照，
表现为"表不见了"。→ **校验前先 `am force-stop`**。

**② `run-as` 下用 `;` 串多条命令会被设备 shell 拆开。**
`run-as <pkg> "ls a/ | wc -l; ls b/ | wc -l"` 会报出并不存在的 "No such file"。
→ **校验类命令逐条单独跑**，别用分号串。

（这两个都不是数据问题，是观测手段的问题——同一类教训：
**先确认"观测手段本身对不对"，再下结论**。）

---

## 5. 本次结果（2026-10-09 23:31）

| 项 | 结果 |
|---|---|
| `observations` | **11 条**（迁移前后一致） |
| `observations_fts` | 11 条，索引完整 |
| 照片 | **9/9 张**到位，`photoPath` 残留旧包名 **0 条** |
| 主题偏好 | 已带过 |
| `user_version` | 3 → 3，Room 正常打开，**无重建、无崩溃** |
| App 运行 | `infer=92~181ms  prior=on  GPS 在华中框内 → 华中 150 类` |

记录内容：**个人观鸟记录，不随本文公开。** 迁移正确性由上面的条数（11 条）、
照片数（9/9）与 `user_version` 三项判定，与具体是哪些鸟无关。