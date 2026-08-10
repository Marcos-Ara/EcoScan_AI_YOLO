$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
python -m http.server 5500 --bind 127.0.0.1
