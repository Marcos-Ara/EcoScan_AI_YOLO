$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "============================================================"
Write-Host "EcoScan AI - Backend"
Write-Host "============================================================"

# Usa o .venv do backend. Se ele não existir, cria.
if (!(Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[EcoScan] Criando ambiente virtual..."
    python -m venv .venv
}

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "[EcoScan] Python: $Python"

Write-Host "[EcoScan] Atualizando pip..."
& $Python -m pip install --upgrade pip

Write-Host "[EcoScan] Instalando dependências locais (conversão + API)..."
& $Python -m pip install -r requirements-local.txt

# Baixa novamente caso model.pt esteja ausente ou inválido.
Write-Host "[EcoScan] Verificando model.pt..."
& $Python download_model.py

# Converte se o ONNX não existir.
if (!(Test-Path "model.onnx")) {
    Write-Host "[EcoScan] model.onnx não encontrado. Convertendo..."
    & $Python convert_model.py
} else {
    Write-Host "[EcoScan] model.onnx encontrado."
}

# Verifica o checkpoint, o import YOLOv7, o ONNX e uma inferência real.
Write-Host "[EcoScan] Executando verificação final..."
& $Python verify_setup.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[EcoScan] model.onnx existente falhou na verificação."
    Write-Host "[EcoScan] Removendo ONNX antigo e convertendo novamente..."
    Remove-Item "model.onnx" -Force -ErrorAction SilentlyContinue
    & $Python convert_model.py
    & $Python verify_setup.py
}

if ($LASTEXITCODE -ne 0) {
    throw "A verificação do EcoScan falhou. Corrija o erro acima antes de iniciar a API."
}

Write-Host "[EcoScan] Iniciando FastAPI em http://127.0.0.1:8000"
& $Python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
