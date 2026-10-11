#Requires -Version 5.1
# 注意：本文件必须存为 UTF-8 **带 BOM**。Windows PowerShell 5.1 会把无 BOM 的 .ps1
# 按 GBK 解析，中文字面量在解析期即被毁（报 Unexpected token）。改动后请用字节复验首 3 字节 = ef bb bf。
<#
  publish_public.ps1 — 组装「可公开」的仓库树（给 GitHub 用）。

  口径见 AGENTS.md「公开 / 内部两层」：
    公开 = 根 README/LICENSE/LICENSE-MODELS.md/.gitattributes/.gitignore
           + android/ + tools/ + site/（GitHub Pages 下载页）
           + .github/workflows/（只有 pages.yml）
           + docs/{design,guides,legal,project}
    内部 = docs/{handoff,reports,research,session,archive} + docs/README.md
           + 根 AGENTS.md（本机维护约定；公开树里的 AGENTS.md 由
             docs/AGENTS-PUBLIC.md 换名而来，见下面的「换名」步骤）

  本脚本用**目录白名单**（不是文件名黑名单）从某个 commit 导出，
  所以以后新增的内部文档**不可能**被误发出去。

  用法：
    .\publish_public.ps1                     # 导出到 dist\public\
    .\publish_public.ps1 -InitRepo           # 导出并 git init + 首次提交（可直接 push）
    .\publish_public.ps1 -Ref v1.0.0         # 从某个 tag/commit 导出
    .\publish_public.ps1 -OutDir D:\pub      # 指定输出目录

  注意：本脚本**不 push**。推送到 GitHub 请自己执行最后打印的命令。
#>
param(
    [string]$Ref = 'HEAD',
    [string]$OutDir,
    [switch]$InitRepo,
    [switch]$KeepZip
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)   # 无 BOM：管道给原生命令时不注入 BOM

# ---- 路径 ----
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # tools\publish -> 仓库根
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot 'dist\public' }
$OutDir = [System.IO.Path]::GetFullPath($OutDir)

# ---- 公开白名单（唯一事实来源）----
$Include = @(
    'README.md'
    'LICENSE'
    'LICENSE-MODELS.md'
    '.gitattributes'
    '.gitignore'
    'android'
    'tools'
    'site'
    '.github/workflows'   # 下载页的 Pages 部署工作流（只导出这一个子目录）
    'docs/design'
    'docs/guides'
    'docs/legal'
    'docs/project'
    'docs/AGENTS-PUBLIC.md'
)

# 仓库根的 AGENTS.md 是**本机维护约定**（含开发机绝对路径、内部目录台账），不对外。
# 公开树里的 AGENTS.md 由 docs/AGENTS-PUBLIC.md 换名而来——见下面的「换名」步骤。
$PublicAgentsSource = 'docs/AGENTS-PUBLIC.md'

# 导出后必须**不存在**的路径（安全网；出现即报错退出）
$MustNotExist = @(
    'docs/handoff'
    'docs/reports'
    'docs/research'
    'docs/session'
    'docs/archive'
    'docs/README.md'
    'docs/AGENTS-PUBLIC.md'   # 换名后不应再留在 docs/ 下
)

# ---- 逐项排除（导出后删掉）----
# 白名单管**目录**，这一份只管**个别文件**——只用于「出处无法确认」的素材。
# 原则：能确定是本项目产出的就发；查不到出处的不发（2026-10-11 用户定）。
# 每条必须写明理由，否则下次没人敢动。
$ExcludePaths = @(
    @{ Path = 'docs/design/preview-paper'
       Why  = '含 13 张「华中数据集真实鸟照」（据 docs/archive/design/DESIGN-UX-v4-paper.md:11），' +
              '数据集逐条 license 记录不在仓库内，拿不出逐张署名；5 个预览页里 4 个依赖这些照片，' +
              '只删图会留死链，故整目录不发。公开文档也没有一处引用它（引用方全是 docs/archive/ 里的旧设计文档）。' }
    @{ Path = 'android/app/src/main/assets/models/selftest_bird.jpg'
       Why  = '检测器自检基准图，640x427 无 EXIF 无出处记录（docs/ 里只有基准数值没有来源）。' +
              '本机 App 仍需要它，故只从公开树剔除。' }
    @{ Path = 'docs/design/assets/device-launch-check.png'
       Why  = '本机 adb 截屏（1200x2670 / 32bpp），但内容无法确认是否嵌了图鉴里的数据集鸟照；' +
              '无任何文档引用，剔除零成本。' }
    @{ Path = 'tools/dataset_traits/2021201.zip'
       Why  = '解包后是期刊补充数据原件 Chinesebirdsdata.xlsx（王彦平等 2021《生物多样性》' +
              'DOI 10.17520/biods.2021201）。同目录的 .xlsx 原件早就按「别人的原件不入库」排除，' +
              'zip 只是换个壳，等于重新分发原件（2026-10-11 发布前审计发现它已被跟踪）。' +
              '官方下载 URL 记在 docs/guides/dataset_traits_README.md，外部读者可自行获取。' }
)

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK  $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    !!  $msg" -ForegroundColor Yellow }

# ---- 0. 环境检查 ----
Write-Step "检查仓库状态"
Push-Location $RepoRoot
try {
    $null = git rev-parse --git-dir 2>&1
    if ($LASTEXITCODE -ne 0) { throw "不是 git 仓库：$RepoRoot" }
    $sha = (git rev-parse --short $Ref).Trim()
    Write-Ok "导出 ref = $Ref ($sha)"

    $dirty = @(git status --porcelain)
    if ($dirty.Count -gt 0) {
        Write-Warn2 "工作区有 $($dirty.Count) 个未提交改动——本脚本导出的是 **commit**，不是工作区"
        Write-Warn2 "未提交的内容不会出现在公开树里。先 commit 再发布。"
    }

    # 权重是否已入库（决定公开树里有没有模型）
    $tracked = @(git ls-files 'android/app/src/main/assets/models/*.tflite')
    if ($tracked.Count -eq 0) {
        Write-Warn2 "模型权重尚未入库（android/.gitignore 仍忽略 *.tflite）"
        Write-Warn2 "→ 公开树里不会有任何 .tflite；README 的 Git LFS 说明暂时兑现不了"
        Write-Warn2 "→ 血统已结案（LICENSE-MODELS.md §1），剩下只是还没 git add"
    } else {
        Write-Ok "已入库权重 $($tracked.Count) 个（.tflite）"
    }

    # ---- 1. 导出 ----
    Write-Step "从 $Ref 导出白名单路径"
    # 先探测哪些白名单路径真的在 $Ref 里——git archive 遇到不存在的 pathspec 会直接 fatal
    $present = @(); $missing = @()
    foreach ($p in $Include) {
        if (@(git ls-tree -r --name-only $Ref -- $p).Count -gt 0) { $present += $p } else { $missing += $p }
    }
    if ($missing.Count -gt 0) {
        Write-Warn2 "白名单里有 $($missing.Count) 个路径在 $Ref 中不存在（尚未提交？），不会出现在公开树里："
        foreach ($m in $missing) { Write-Warn2 "   - $m" }
    }
    if ($present.Count -eq 0) { throw "白名单在 $Ref 里一个都没命中 —— 先把改动 commit 再发布" }
    if ($missing -contains 'README.md') {
        Write-Warn2 "!! README.md 不在 $Ref 里：公开仓库会没有首页说明文档"
    }
    $tmpZip = Join-Path $env:TEMP "whatsbird_public_$([guid]::NewGuid().ToString('N')).zip"
    $archiveArgs = @('archive', '--format=zip', "-o", $tmpZip, $Ref, '--') + $present
    & git @archiveArgs
    if ($LASTEXITCODE -ne 0) { throw "git archive 失败" }
    Write-Ok "已打包 $([math]::Round((Get-Item $tmpZip).Length/1KB,1)) KB（命中 $($present.Count) 个顶层路径）"

    # ---- 2. 解包到输出目录 ----
    Write-Step "解包到 $OutDir"
    if (Test-Path $OutDir) {
        Write-Warn2 "输出目录已存在，清空重建"
        Remove-Item -LiteralPath $OutDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Expand-Archive -LiteralPath $tmpZip -DestinationPath $OutDir -Force
    if (-not $KeepZip) { Remove-Item -LiteralPath $tmpZip -Force }
    else { Write-Ok "保留 zip: $tmpZip" }

    # ---- 2b. 换名：docs/AGENTS-PUBLIC.md → AGENTS.md ----
    # 仓库根的 AGENTS.md 是本机维护约定（含开发机绝对路径与内部目录台账），不对外；
    # 对外那份单独维护在 docs/AGENTS-PUBLIC.md，这里换名顶上去。
    Write-Step "换名对外版 AGENTS.md"
    $srcAgents = Join-Path $OutDir ($PublicAgentsSource -replace '/', '\')
    $dstAgents = Join-Path $OutDir 'AGENTS.md'
    if (Test-Path -LiteralPath $srcAgents) {
        Move-Item -LiteralPath $srcAgents -Destination $dstAgents -Force
        Write-Ok "$PublicAgentsSource → AGENTS.md（$([math]::Round((Get-Item -LiteralPath $dstAgents).Length/1KB,1)) KB）"
    } else {
        Write-Warn2 "$PublicAgentsSource 不在 $Ref 里：公开树将没有 AGENTS.md"
    }

    # ---- 2c. 逐项排除：出处无法确认的素材 ----
    Write-Step "剔除出处无法确认的素材"
    $removed = 0
    foreach ($e in $ExcludePaths) {
        $full = Join-Path $OutDir ($e.Path -replace '/', '\')
        if (Test-Path -LiteralPath $full) {
            $n = @(Get-ChildItem -LiteralPath $full -Recurse -File -ErrorAction SilentlyContinue).Count
            if ($n -eq 0) { $n = 1 }
            Remove-Item -LiteralPath $full -Recurse -Force
            $removed += $n
            Write-Ok "剔除 $($e.Path)（$n 个文件）"
        } else {
            Write-Warn2 "$($e.Path) 不在 $Ref 里，无需剔除"
        }
    }
    if ($removed -eq 0) { Write-Ok "本次无需剔除" }

    # ---- 3. 安全网：内部路径与被排除素材必须不存在 ----
    Write-Step "安全网：确认内部路径未被导出"
    $leaked = @()
    foreach ($p in ($MustNotExist + @($ExcludePaths | ForEach-Object { $_.Path }))) {
        $full = Join-Path $OutDir ($p -replace '/', '\')
        if (Test-Path $full) { $leaked += $p }
    }
    if ($leaked.Count -gt 0) {
        throw "内部路径泄漏到公开树：$($leaked -join ', ')"
    }
    Write-Ok "内部路径 0 泄漏"

    # ---- 3b. 安全网：公开文档里不得出现开发机路径或内部目录引用 ----
    # tools/ 里的脚本还有历史遗留的绝对路径（待清理），所以这里只查**文档**——
    # 文档是给外部读者看的，出现 E:\code\ 或内部目录路径就是错的。
    Write-Step "安全网：公开文档内容抽查"
    $docFiles = @()
    $docFiles += Get-Item -LiteralPath $dstAgents -ErrorAction SilentlyContinue
    $docFiles += Get-ChildItem (Join-Path $OutDir 'docs') -Recurse -File -Include *.md -ErrorAction SilentlyContinue
    $badPath = @(); $badRef = @()
    foreach ($f in $docFiles) {
        $hit1 = Select-String -LiteralPath $f.FullName -Pattern 'E:\\code\\' -ErrorAction SilentlyContinue
        if ($hit1) { $badPath += "$($f.Name):$($hit1[0].LineNumber)" }
        $hit2 = Select-String -LiteralPath $f.FullName -Pattern 'docs/(handoff|reports|research|session|archive)/' -ErrorAction SilentlyContinue
        if ($hit2) { $badRef += "$($f.Name):$($hit2[0].LineNumber)" }
    }
    if ($badPath.Count -gt 0) {
        Write-Warn2 "以下文档含开发机绝对路径（E:\code\），外部读者看不懂："
        foreach ($b in $badPath) { Write-Warn2 "   - $b" }
    } else { Write-Ok "公开文档 0 处开发机绝对路径" }
    if ($badRef.Count -gt 0) {
        throw "公开文档引用了内部目录：$($badRef -join ', ')"
    }
    Write-Ok "公开文档 0 处内部目录引用"

    # ---- 4. 统计 ----
    Write-Step "公开树统计"
    $files = @(Get-ChildItem $OutDir -Recurse -File)
    $sizeMB = [math]::Round((($files | Measure-Object Length -Sum).Sum / 1MB), 1)
    Write-Host "    文件 $($files.Count) 个 / $sizeMB MB"
    foreach ($d in @('android', 'tools', 'site', 'docs')) {
        $sub = Join-Path $OutDir $d
        if (Test-Path $sub) {
            $n = @(Get-ChildItem $sub -Recurse -File).Count
            $mb = [math]::Round(((Get-ChildItem $sub -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
            Write-Host "      $d/  $n 个 / $mb MB"
        }
    }
    $docDirs = @(Get-ChildItem (Join-Path $OutDir 'docs') -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
    Write-Host "      docs/ 子目录: $($docDirs -join ', ')"
    Write-Ok "白名单外的东西（dist/ data/ .shots/ .git/ 内部文档）全部未导出"

    # ---- 5. 可选：git init ----
    if ($InitRepo) {
        Write-Step "初始化公开仓库（单个首次提交）"
        Push-Location $OutDir
        try {
            & git init -q
            & git add -A
            & git -c user.name='WhatsBird' -c user.email='noreply@localhost' commit -q -m "拾鸟 WhatsBird：首次公开发布（源码 + 公开文档）"
            if ($LASTEXITCODE -ne 0) { throw "git commit 失败" }
            $c = (git rev-parse --short HEAD).Trim()
            Write-Ok "已创建提交 $c（历史是压平的，只有这一个 commit）"
        } finally { Pop-Location }
    }

    # ---- 6. 下一步 ----
    Write-Host ""
    Write-Host "公开树已就绪：$OutDir" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "推送到 GitHub："
    if (-not $InitRepo) {
        Write-Host "  cd `"$OutDir`""
        Write-Host "  git init && git add -A && git commit -m `"首次公开发布`""
    } else {
        Write-Host "  cd `"$OutDir`""
    }
    Write-Host "  git remote add origin git@github.com:<你>/<仓库>.git"
    Write-Host "  git branch -M main"
    Write-Host "  git push -u origin main"
    Write-Host ""
    Write-Host "发布前确认（清单见 LICENSE-MODELS.md）："
    Write-Host "  - 权重是否已入库（未入库则公开树没有 .tflite，README 的 Git LFS 说明兑现不了）"
    Write-Host "  - 两项非 Apache-2.0 资产（protection.json 的 IUCN 字段、admin_county.ndjson）"
    Write-Host "    的限制说明是否已在 README 与 LICENSE-MODELS.md 里写明"
    Write-Host ""
    Write-Host "⚠️ 公开树里有 Git LFS 资产，push 前先在这个新仓库里启用 LFS："
    Write-Host "  git lfs install          # 只需一次，会装上 pre-push 钩子"
    Write-Host "  git lfs push origin main # 兜底：显式推 LFS 对象（正常 push 会自动带上）"
    Write-Host "  漏了这步 GitHub 上就只有几百字节的指针文件。"
}
finally {
    Pop-Location
}
