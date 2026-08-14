$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "============================================================"
Write-Host "EcoScan AI - Backend + Roboflow YOLO11"
Write-Host "============================================================"

if (!(Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[EcoScan] Criando ambiente virtual..."
    python -m venv .venv
}

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "[EcoScan] Python: $Python"
Write-Host "[EcoScan] Instalando dependências..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements-local.txt

if (-not $env:ROBOFLOW_API_KEY) {
    throw "Defina a variável de ambiente ROBOFLOW_API_KEY antes de iniciar o backend."
}

if (-not $env:ROBOFLOW_MODEL_ID) {
    $env:ROBOFLOW_MODEL_ID = "waste-sorting-smyr8/2"
}

if (-not $env:ROBOFLOW_API_URL) {
    $env:ROBOFLOW_API_URL = "https://serverless.roboflow.com"
}

Write-Host "[EcoScan] Modelo: $env:ROBOFLOW_MODEL_ID"
Write-Host "[EcoScan] Iniciando FastAPI em http://127.0.0.1:8000"

& $Python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
