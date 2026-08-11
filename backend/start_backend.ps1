$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

if (!(Test-Path "model.pt")) {
    & ".venv\Scripts\python.exe" download_model.py
}

& ".venv\Scripts\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000
