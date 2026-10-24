# Runs the Yosys Xilinx-7-series resource estimate (see docs/SYNTHESIS.md).
# Usage (project root):  powershell -ExecutionPolicy Bypass -File scripts\run_synth.ps1
# Needs Yosys on PATH (install the YosysHQ "OSS CAD Suite" and open its environment shell, or use WSL).
# NOTE: not executed by the author - send back synth\yosys_xilinx7.log (or its last ~80 lines).
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Get-Command yosys -ErrorAction SilentlyContinue)) {
    Write-Host "yosys not found on PATH."
    Write-Host "Install the YosysHQ OSS CAD Suite (GitHub releases), start its environment shell, then rerun."
    Write-Host "Or use WSL (Ubuntu): sudo apt-get install yosys"
    exit 2
}
yosys -l synth\yosys_xilinx7.log -s synth\xilinx7.ys | Out-Null
Write-Host "---- last lines of synth\yosys_xilinx7.log ----"
Get-Content synth\yosys_xilinx7.log -Tail 80
Write-Host "Full report: synth\xilinx7_stat.txt"
