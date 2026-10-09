@echo off
cd /d "%~dp0"
if not exist ".conda\python.exe" (
  echo Run setup.ps1 first to create the isolated environment.
  pause
  exit /b 1
)
set "PATH=%~dp0.conda\Library\bin;%~dp0.conda\Scripts;%PATH%"
".conda\python.exe" -m opus_rl serve %*
