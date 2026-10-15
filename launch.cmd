@echo off
cd /d "%~dp0"
set "OPUS_CONDA=%CONDA_EXE%"
if not defined OPUS_CONDA set "OPUS_CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%OPUS_CONDA%" set "OPUS_CONDA=conda"
call "%OPUS_CONDA%" run --no-capture-output --name OPUS python -m opus_rl serve %*
if errorlevel 1 (
  echo Run setup.ps1 if the OPUS Conda environment has not been created.
  pause
  exit /b 1
)
