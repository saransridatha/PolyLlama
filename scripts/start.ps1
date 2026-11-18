# One-command start for Windows.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & (Join-Path $Root "scripts\setup.ps1")
}

$server = Join-Path $Root "vendor\llama-build\bin\Release\llama-server.exe"
if (-not (Test-Path $server) -and -not (Get-Command llama-server -ErrorAction SilentlyContinue)) {
    & (Join-Path $Root "scripts\build_llama.ps1")
}

$hasKeys = & .\.venv\Scripts\python.exe -c @"
from app.config import Settings
from app.db import Database
s=Settings(); s.ensure_dirs(); db=Database(s.db_path); n=db.count_api_keys(); db.close(); print(n)
"@
if ($hasKeys.Trim() -eq "0") {
    & .\.venv\Scripts\python.exe -m app init
}

& .\.venv\Scripts\python.exe -m app serve @args
