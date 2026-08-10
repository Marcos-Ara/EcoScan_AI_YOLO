$ErrorActionPreference = "Stop"

$Backend = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Backend

if (-not (Test-Path ".\GreenSorter")) {
    Write-Host "Clonando GreenSorter..."
    git clone https://github.com/1nfinityLoop/GreenSorter.git .\GreenSorter
}

if (-not (Test-Path ".\model\model.pt")) {
    Write-Host ""
    Write-Host "ATENCAO: o repositorio disponibiliza o codigo/dataset e informa um download separado do model.pt."
    Write-Host "Baixe o model.pt pelo link 'Download model' do README do GreenSorter:"
    Write-Host "https://github.com/1nfinityLoop/GreenSorter"
    Write-Host ""
    Write-Host "Depois coloque o arquivo em:"
    Write-Host "$Backend\model\model.pt"
}

if (-not (Test-Path ".\.venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup concluido."
Write-Host "Para iniciar: .\start_backend.ps1"
