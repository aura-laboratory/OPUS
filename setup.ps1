param([string]$Conda = "conda")
$ErrorActionPreference = "Stop"
$opusRoot = $PSScriptRoot
if ($Conda -eq "conda" -and $env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE)) {
    $Conda = $env:CONDA_EXE
}
if (-not (Get-Command $Conda -ErrorAction SilentlyContinue)) {
    $opusCondaCandidate = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
    if ($env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE)) { $Conda = $env:CONDA_EXE }
    elseif (Test-Path -LiteralPath $opusCondaCandidate) { $Conda = $opusCondaCandidate }
    else { throw "Conda was not found. Pass its executable path with -Conda." }
}
$opusEnvironments = (& $Conda env list --json | Out-String | ConvertFrom-Json).envs
if ($LASTEXITCODE -ne 0) { throw "Could not list Conda environments." }
$opusPrefix = $opusEnvironments | Where-Object { (Split-Path $_ -Leaf) -eq "OPUS" } | Select-Object -First 1
if (-not $opusPrefix) {
    & $Conda create --name OPUS --override-channels --channel conda-forge python=3.11 pip -y
    if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed." }
}
& $Conda run --no-capture-output --name OPUS python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
Push-Location $opusRoot
try {
    & $Conda run --no-capture-output --name OPUS python -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "OPUS installation failed." }
} finally { Pop-Location }
