# ⚠️ 本文件必须保存为 **UTF-8 带 BOM**。
#
# 没有 BOM 时 PowerShell 会按系统 ANSI（本机 cp936/GBK）读它 —— 中文注释和
# 字符串里的 UTF-8 字节被误解释，引号配对错位，解析器从 here-string 之后
# 全部带偏，报的却是文件末尾的 "字符串缺少终止符" / "缺少右 }"。
# 2026-10-05 踩过：一次普通编辑把 BOM 去掉，文件立刻无法解析（HEAD 版本没事）。
# 改完请验证：[System.Management.Automation.Language.Parser]::ParseFile(...) 错误数为 0。
#
# 启动一个真正脱离当前窗口的进程。
#
# 为什么不用 Start-Process：它的子进程仍在当前控制台进程组里，窗口一关
# 就跟着收到 CTRL_CLOSE_EVENT 被杀掉。2026-10-04 那次 GBIF 下载就是这样
# 变成孤儿进程然后卡死的。
#
# 走 WMI Win32_Process.Create 的话，父进程是 WmiPrvSE.exe，不属于任何
# 控制台进程组，关窗口 / 断 SSH / 关终端都不影响它。
#
# 用法:
#   .\run_detached.ps1 -Exe D:\software\Python312\python.exe `
#       -ScriptArgs @('-u','E:\code\observebird\tools\build_dataset_gbif.py','--out','x') `
#       -LogPath "E:\code\code tools\data\gbif_build.log"
param(
  [Parameter(Mandatory = $true)][string]$Exe,
  [Parameter(Mandatory = $true)][string[]]$ScriptArgs,
  [Parameter(Mandatory = $true)][string]$LogPath,
  [string]$ErrPath,
  [string]$WorkDir = "E:\code\observebird",
  [switch]$Stop
)

$ErrorActionPreference = "Stop"

# ---- 停止模式：按命令行特征找进程并杀掉 -------------------------------
if ($Stop) {
  $needle = ($ScriptArgs | Where-Object { $_ -like "*.py" } | Select-Object -First 1)
  if (-not $needle) {
    # 不要回退成 $ScriptArgs[0]：那不是 .py 时（例如 "-u"）Contains() 会命中
    # 机器上几乎所有 python 进程，包括别的项目。宁可报错让调用方补清楚。
    Write-Error "没有在 -ScriptArgs 里找到 .py 路径，拒绝按模糊特征停止进程（避免误杀无关 python）"
    exit 1
  }
  $hits = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
          Where-Object { $_.CommandLine -and $_.CommandLine.Contains($needle) }
  if (-not $hits) { Write-Host "没有匹配 '$needle' 的进程"; exit 0 }
  foreach ($h in $hits) {
    Stop-Process -Id $h.ProcessId -Force
    Write-Host "已停止 PID $($h.ProcessId)"
  }
  exit 0
}

if (-not (Test-Path -LiteralPath $LogPath)) {
  New-Item -ItemType Directory -Path (Split-Path -Parent $LogPath) -Force | Out-Null
}
if (-not $ErrPath) { $ErrPath = "$LogPath.err" }

# ---- 每次启动换一个日志，保留历史 -------------------------------------
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = "$LogPath.$stamp"
$err = "$ErrPath.$stamp"

# cmd 负责重定向：Win32_Process.Create 不过 shell，">" 必须由 cmd.exe 解释
#
# 这里**不再**写 "$log.exit"：那个标记不区分"跑完"和"被杀"，而且
# `echo %ERRORLEVEL%> file` 少个空格会被 cmd 当成流重定向（0> 重定向 stdin →
# 空文件；1> 重定向 stdout → 文件里只有 "ECHO is off."），实测 30 个 .exit 里
# 8 个空文件、4 个 "ECHO 处于关闭状态。"。完成状态一律看 <out>/progress.json。
#
# 必须显式设 PYTHONIOENCODING：本机 Python 默认 sys.stdout.encoding = gbk
# （locale.getpreferredencoding = cp936），日志里的中文会写成 GBK 字节，
# 而监控台是按 UTF-8 读的 → 全变乱码。WMI 子进程**不继承**当前 PowerShell
# 会话的环境，所以不能靠调用方先 $env:PYTHONIOENCODING='utf-8'。
# auto_region_pipeline.py 早就吃到了这个亏（它就是自己 set 的）。
# `if not defined` 保证调用方显式指定的值优先。
$quoted = ($ScriptArgs | ForEach-Object { '"' + $_ + '"' }) -join " "
$body = @"
@echo off
cd /d "$WorkDir"
if not defined PYTHONIOENCODING set PYTHONIOENCODING=utf-8
"$Exe" -u $quoted >> "$log" 2>> "$err"
"@
$wrap = Join-Path $env:TEMP "observebird-run-$stamp.cmd"
Set-Content -LiteralPath $wrap -Value $body -Encoding ASCII

$proc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
          -Arguments @{ CommandLine = "`"$wrap`"" }
if ($proc.ReturnValue -ne 0) {
  Write-Error "WMI Create 失败 ReturnValue=$($proc.ReturnValue)"
  exit 1
}

# WMI returns the pid of the cmd.exe that runs our .cmd wrapper; the real
# interpreter is its child. Report the child's pid, otherwise every monitoring
# and stop command points at cmd.exe instead of python.
$target = $proc.ProcessId
for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Milliseconds 250
  $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$target" |
           Sort-Object CreationDate | Select-Object -First 1
  if (-not $child) { break }
  $target = $child.ProcessId
  if ($child.Name -ne "cmd.exe") { break }
}

Write-Host "PID   : $target  ($((Get-CimInstance Win32_Process -Filter "ProcessId=$target").Name))"
# 解析失败时会停在 cmd.exe 上：调用方若拿这个 PID 去等待/停止，会指错对象
# （现场：队列等的是 cmd 壳，30 秒就"退出"了，真下载又跑了 1 小时 20 分）。
if ((Get-CimInstance Win32_Process -Filter "ProcessId=$target").Name -eq "cmd.exe") {
  Write-Warning "只解析到 cmd.exe 的 PID（$target），没找到 python 子进程 —— 别拿它当任务 PID 用"
}
Write-Host "日志  : $log"
Write-Host "错误  : $err"
Write-Host "完成状态: 看 <out>/progress.json 的 stage（done/interrupted/failed）—— 不再有 .exit 标记"
Write-Host "脱离链: WmiPrvSE.exe -> cmd($($proc.ProcessId)) -> $target"
Write-Host ""
Write-Host "看进度: Get-Content `"$log`" -Tail 20 -Wait"
$py = ($ScriptArgs | Where-Object { $_ -like "*.py" } | Select-Object -First 1)
Write-Host "停止  : .\run_detached.ps1 -Stop -ScriptArgs @('$py') -LogPath x -Exe x"