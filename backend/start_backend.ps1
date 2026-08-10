$ErrorActionPreference = "Stop"

$Backend = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Backend

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Ambiente Python nao encontrado. Execute .\setup_model.ps1 primeiro."
    exit 1
}

& ".\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
