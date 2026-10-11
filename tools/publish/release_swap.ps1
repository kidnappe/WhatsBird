#Requires -Version 5.1
<#
  release_swap.ps1 — 替换 GitHub Release 上的 APK 资产。

  为什么要这个脚本（本机环境的两个坑）：
    1. 这台机器的 hosts 把 github.com / api.github.com / uploads.github.com
       全部指向 127.0.0.1（本机实测 240 条有效行全是 127.0.0.1），
       所以**默认解析永远连不上**，必须显式指定可达 IP。
    2. git 自己能绕（`git -c http.curloptResolve=...`），
       但 PowerShell 的 Invoke-WebRequest / curl.exe 不读 git 的配置，
       得用 curl.exe 的 --resolve。

  ⚠️ 本脚本**不修改系统 hosts**，只用 --resolve 临时指 IP，进程结束即失效。

  用法：
    .\release_swap.ps1 -List
    .\release_swap.ps1 -Tag v1.0 -Apk <path> -Replace
#>
param(
    [string]$Tag,
    [string]$Apk,
    [switch]$List,
    [switch]$Replace,
    [string]$ApiIp = '140.82.112.6',
    [string]$UploadIp = '140.82.112.6',
    [string]$Repo = 'kidnappe/WhatsBird'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# ---- 取 token（只留在本进程内存里，绝不打印/落盘）----
function Get-GhToken {
    # ⚠️ 必须问 host=github.com：凭据管理器里 api.github.com 那条是以
    # 「GitHub - https://api.github.com/kidnappe」存的 GCM provider 条目，
    # git 的 credential lookup 匹配不上，会**挂起等浏览器授权**（实测卡死）。
    # 拿到 github.com 的 token 后直接打 api.github.com，认的是同一个账号。
    $env:GIT_TERMINAL_PROMPT = '0'
    $env:GCM_INTERACTIVE = 'never'
    $env:GCM_GUI_PROMPT = '0'
    $out = "protocol=https`nhost=github.com`n`n" | git credential fill 2>$null
    $pw = $out | Where-Object { $_ -like 'password=*' } | Select-Object -First 1
    if (-not $pw) { throw '凭据管理器里没有 github.com 的凭据' }
    return $pw.Substring(9)
}
$token = Get-GhToken

function Invoke-Gh {
    param([string]$Url, [string]$Method = 'GET', [string]$OutFile)
    $args = @('--silent', '--show-error', '--max-time', '120',
              '--resolve', "api.github.com:443:$ApiIp",
              '-H', "Authorization: Bearer $token",
              '-H', 'Accept: application/vnd.github+json',
              '-H', 'X-GitHub-Api-Version: 2022-11-28',
              '-w', "`n%{http_code}")
    if ($Method -ne 'GET') { $args += @('-X', $Method) }
    if ($OutFile) { $args += @('-o', $OutFile) }
    $args += $Url
    $resp = & curl.exe @args
    if ($LASTEXITCODE -ne 0) { throw "curl 失败 exit=$LASTEXITCODE  $Url" }
    $code = [int]$resp[-1]
    $body = ($resp[0..($resp.Count - 2)]) -join "`n"
    if ($code -ge 400) { throw "HTTP $code`n$body" }
    return @{ code = $code; body = $body }
}

if ($List) {
    $r = Invoke-Gh "https://api.github.com/repos/$Repo/releases"
    $rels = $r.body | ConvertFrom-Json
    Write-Host ''
    Write-Host ("{0,-10} {1,-12} {2,-10} {3}" -f 'TAG', 'NAME', 'PRERELEASE', 'ASSETS') -ForegroundColor Cyan
    foreach ($rel in $rels) {
        if ($rel.assets.Count -eq 0) {
            Write-Host ("{0,-10} {1,-12} {2,-10} (无资产)" -f $rel.tag_name, $rel.name, $rel.prerelease)
            continue
        }
        foreach ($a in $rel.assets) {
            Write-Host ("{0,-10} {1,-12} {2,-10} id={3}  {4}  {5:N2} MB  dl={6}" -f `
                $rel.tag_name, $rel.name, $rel.prerelease, $a.id, $a.name,
                ($a.size / 1MB), $a.download_count)
        }
    }
    Write-Host ''
    return
}

if (-not $Replace) { throw '没给 -List 也没给 -Replace，先跑 -List 看清楚再动手' }
if (-not $Apk -or -not (Test-Path -LiteralPath $Apk)) { throw "APK 不存在: $Apk" }
if (-not $Tag) { throw '要 -Tag' }

$sizeMB = [math]::Round((Get-Item -LiteralPath $Apk).Length / 1MB, 2)
Write-Host ''
Write-Host "==> 准备替换 $Tag 上的资产" -ForegroundColor Cyan
Write-Host "    新包: $Apk  ($sizeMB MB)"

$r = Invoke-Gh "https://api.github.com/repos/$Repo/releases/tags/$Tag"
$rel = $r.body | ConvertFrom-Json
Write-Host "    release id=$($rel.id)  tag=$($rel.tag_name)  当前资产 $($rel.assets.Count) 个"
foreach ($a in $rel.assets) {
    Write-Host ("      - id={0}  {1}  {2:N2} MB" -f $a.id, $a.name, ($a.size / 1MB))
}

# 顺序刻意是「先传新 → 再删旧 → 最后改名」，不是「先删旧再传新」：
# GitHub 不允许同名资产并存，中间必须用临时名过渡。
# 万一上传失败/断网，旧包还在，release 不会出现「一个资产都没有」的窗口。
$name = [IO.Path]::GetFileName($Apk)
$tmpName = "$name.incoming"
$tmp = Join-Path $env:TEMP ([guid]::NewGuid().ToString('N') + '.json')

Write-Host "    1/3 上传新资产为临时名 $tmpName ($sizeMB MB) ..." -ForegroundColor Cyan
# ⚠️ 这里**只发两个头**（Authorization + Content-Type），实测多加
# `X-GitHub-Api-Version` / `Accept: application/vnd.github+json` 会让
# uploads.github.com 直接回 404 Not Found（2026-10-13 卡在这上面半天：
# release/DELETE/PATCH 都正常，唯独 POST assets 404）。别手贱加头。
& curl.exe --silent --show-error --max-time 1800 --fail `
    --resolve "uploads.github.com:443:$UploadIp" `
    -H "Authorization: Bearer $token" `
    -H 'Content-Type: application/vnd.android.package-archive' `
    --data-binary "@$Apk" `
    -o $tmp `
    "https://uploads.github.com/repos/$Repo/releases/$($rel.id)/assets?name=$tmpName"
if ($LASTEXITCODE -ne 0) { throw "上传失败 exit=$LASTEXITCODE（旧资产未动，release 完好）" }
$new = (Get-Content -LiteralPath $tmp -Raw -Encoding UTF8 | ConvertFrom-Json)
Remove-Item -LiteralPath $tmp -Force
Write-Host ("        上传完成 id={0}  {1:N2} MB" -f $new.id, ($new.size / 1MB))

Write-Host '    2/3 删除旧资产 ...' -ForegroundColor Yellow
foreach ($a in $rel.assets) {
    Write-Host "        删除 id=$($a.id) $($a.name)" -ForegroundColor Yellow
    Invoke-Gh "https://api.github.com/repos/$Repo/releases/assets/$($a.id)" -Method DELETE | Out-Null
}

Write-Host "    3/3 改名回正式名 $name ..." -ForegroundColor Cyan
# ⚠️ JSON 必须**写进文件**再用 --data-binary @file 传：PowerShell 往原生命令传
# `-d '{"name":"x"}'` 时会把双引号吃掉，GitHub 收到 {name:x} 直接回 400（2026-10-13 踩过）。
$bodyFile = Join-Path $env:TEMP ([guid]::NewGuid().ToString('N') + '.json')
[IO.File]::WriteAllText($bodyFile, ('{"name":"' + $name + '"}'), (New-Object System.Text.UTF8Encoding($false)))
$tmp2 = Join-Path $env:TEMP ([guid]::NewGuid().ToString('N') + '.json')
& curl.exe --silent --show-error --max-time 120 --fail `
    --resolve "api.github.com:443:$ApiIp" `
    -H "Authorization: Bearer $token" `
    -X PATCH --data-binary "@$bodyFile" -o $tmp2 `
    "https://api.github.com/repos/$Repo/releases/assets/$($new.id)"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "改名失败（资产已在，只是名字带 .incoming），可稍后在页面上手动改"
} else {
    $new = (Get-Content -LiteralPath $tmp2 -Raw -Encoding UTF8 | ConvertFrom-Json)
}
Remove-Item -LiteralPath $tmp2, $bodyFile -Force -ErrorAction SilentlyContinue

Write-Host ''
Write-Host '==> 替换完成' -ForegroundColor Green
Write-Host ("    id={0}  {1}  {2:N2} MB" -f $new.id, $new.name, ($new.size / 1MB))
Write-Host ("    下载: {0}" -f $new.browser_download_url)