param([string]$Conda = "conda")
$ErrorActionPreference = "Stop"
$opusRoot = $PSScriptRoot
$opusPrefix = Join-Path $opusRoot ".conda"
if (-not (Test-Path -LiteralPath (Join-Path $opusPrefix "python.exe"))) {
    & $Conda create --prefix $opusPrefix --override-channels --channel conda-forge python=3.11 pip -y
    if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed." }
}
$opusPython = Join-Path $opusPrefix "python.exe"
& $opusPython -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
Push-Location $opusRoot
try {
    & $opusPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "OPUS installation failed." }
} finally { Pop-Location }
