$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m http.server 5500 --bind 0.0.0.0
} else {
    python -m http.server 5500 --bind 0.0.0.0
}
