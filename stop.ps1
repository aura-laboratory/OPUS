$ErrorActionPreference = "Stop"
$opusPython = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".conda\python.exe"))
$opusServices = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.ExecutablePath -eq $opusPython -and $_.CommandLine -match '\s-m\s+opus_rl\s+serve(?:\s|$)'
}
foreach ($opusService in $opusServices) {
    # The UI's Stop & save control should be used before stopping the service during training.
    Stop-Process -Id $opusService.ProcessId
    Write-Host "Stopped OPUS service (process $($opusService.ProcessId))."
}
if (-not $opusServices) { Write-Host "No OPUS service is running from this environment." }
