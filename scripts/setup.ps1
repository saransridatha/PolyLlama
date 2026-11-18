# ModelSwitch native setup for Windows. Requires Python 3.11+, CMake and a C++ toolchain.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Python environment"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --quiet -e ".[dev]"

Write-Host "==> llama.cpp"
$server = Join-Path $Root "vendor\llama-build\bin\Release\llama-server.exe"
if (-not (Test-Path $server) -and -not (Get-Command llama-server -ErrorAction SilentlyContinue)) {
    & (Join-Path $Root "scripts\build_llama.ps1")
}

Write-Host "==> Dashboard"
if (-not (Test-Path "app\static\dashboard\index.html")) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Push-Location frontend
        npm install --no-audit --no-fund
        npm run build
        Pop-Location
    } else {
        Write-Warning "Dashboard assets missing and npm is unavailable. The API still works."
    }
}

New-Item -ItemType Directory -Force -Path "data\logs", "data\run", "models" | Out-Null

$hasKeys = & .\.venv\Scripts\python.exe -c @"
from app.config import Settings
from app.db import Database
s=Settings(); s.ensure_dirs(); db=Database(s.db_path); n=db.count_api_keys(); db.close(); print(n)
"@
if ($hasKeys.Trim() -eq "0") {
    & .\.venv\Scripts\python.exe -m app init
}

Write-Host ""
Write-Host "Setup complete. Start with: .\scripts\start.ps1"
