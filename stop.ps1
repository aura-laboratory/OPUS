param([string]$Conda = "conda")
$ErrorActionPreference = "Stop"
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
if (-not $opusPrefix) { Write-Host "The OPUS Conda environment has not been created."; return }
$opusPython = [System.IO.Path]::GetFullPath((Join-Path $opusPrefix "python.exe"))
$opusServices = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.ExecutablePath -eq $opusPython -and $_.CommandLine -match '\s-m\s+opus_rl\s+serve(?:\s|$)'
}
foreach ($opusService in $opusServices) {
    # The UI's Stop & save control should be used before stopping the service during training.
    Stop-Process -Id $opusService.ProcessId
    Write-Host "Stopped OPUS service (process $($opusService.ProcessId))."
}
if (-not $opusServices) { Write-Host "No OPUS service is running from this environment." }
