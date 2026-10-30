# Windows PowerShell runner for the ARMSE RTL tests (no `make` needed).
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_rtl_tests.ps1
# Needs: Python with numpy, and Icarus Verilog (iverilog + vvp) on PATH.
# NOTE: this script has not been executed by the author - please send back rtl_test_log.txt.

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$root = (Get-Location).Path

if (-not (Get-Command iverilog -ErrorAction SilentlyContinue)) {
    Write-Host "iverilog not found on PATH."
    Write-Host "Install 'Icarus Verilog' for Windows (tick 'Add to PATH' in the installer), then open a NEW terminal."
    Write-Host "Or use WSL (Ubuntu): sudo apt-get install iverilog"
    exit 2
}

python -m armse.rtlexport . --random 40 | Out-Null
python -m armse.quant . | Out-Null

$log = Join-Path $root "rtl_test_log.txt"
"ARMSE RTL test log" | Set-Content $log
(& iverilog -V 2>&1 | Select-Object -First 1) | Add-Content $log

$rtlDir = Join-Path $root "rtl"
$rtlFiles = @("armse_skew_mem.v", "armse_obuf.v", "armse_dma.v", "armse_compute.v", "armse_top.v") |
    ForEach-Object { Join-Path $rtlDir $_ }
$tbTop  = Join-Path $root "tb\tb_armse.v"
$tbSkew = Join-Path $root "tb\tb_skew_mem.v"
$fail = 0

# ---- 1. skew-memory unit test ----
$out = & iverilog -g2012 -o (Join-Path $root "skew.vvp") $tbSkew (Join-Path $rtlDir "armse_skew_mem.v") 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Host "COMPILE FAILED: tb_skew_mem"; Write-Host $out
    "COMPILE FAILED: tb_skew_mem`n$out" | Add-Content $log
    $fail = 1
} else {
    $out = & vvp (Join-Path $root "skew.vvp") 2>&1 | Out-String
    Write-Host $out; $out | Add-Content $log
    if ($out -notmatch "RESULT tb_skew_mem: PASS") { $fail = 1 }
}

# ---- 2. top-level cases ----
foreach ($d in Get-ChildItem (Join-Path $root "rtl_vectors") -Directory -Filter "t*") {
    Push-Location $d.FullName
    $out = & iverilog -g2005 -I $rtlDir -I . -o sim.vvp $tbTop @rtlFiles 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host "COMPILE FAILED: $($d.Name)"; Write-Host $out
        "COMPILE FAILED: $($d.Name)`n$out" | Add-Content $log
        $fail = 1
        Pop-Location
        continue
    }
    $out = & vvp sim.vvp 2>&1 | Out-String
    $lines = ($out -split "`r?`n") | Where-Object { $_ -match "RESULT|CYCLES|ERROR" }
    $lines | ForEach-Object { Write-Host $_ }
    $lines | Add-Content $log
    if ($out -notmatch "RESULT .*: PASS") { $fail = 1 }
    Pop-Location
}

# ---- 3. requantisation / ReLU unit (armse_post) ----
$tbPost = Join-Path $root "tb\tb_post.v"
$rtlPost = Join-Path $rtlDir "armse_post.v"
foreach ($d in Get-ChildItem (Join-Path $root "post_vectors") -Directory -Filter "p*") {
    Push-Location $d.FullName
    $out = & iverilog -g2005 -I . -o post.vvp $tbPost $rtlPost 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host "COMPILE FAILED: post $($d.Name)"; Write-Host $out
        "COMPILE FAILED: post $($d.Name)`n$out" | Add-Content $log
        $fail = 1
        Pop-Location
        continue
    }
    $out = & vvp post.vvp 2>&1 | Out-String
    $lines = ($out -split "`r?`n") | Where-Object { $_ -match "RESULT|ERROR" }
    $lines | ForEach-Object { Write-Host $_ }
    $lines | Add-Content $log
    if ($out -notmatch "RESULT .*: PASS") { $fail = 1 }
    Pop-Location
}

if ($fail -ne 0) { Write-Host "RTL TESTS: FAILURES"; "RTL TESTS: FAILURES" | Add-Content $log; exit 1 }
Write-Host "RTL TESTS: ALL PASS"; "RTL TESTS: ALL PASS" | Add-Content $log
Write-Host "Log saved to $log"
