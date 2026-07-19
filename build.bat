@echo off
rem Reproducible Windows build for nasdaq-advisor.
rem Prereq: a venv at .venv with requirements-dev.txt installed:
rem   python -m venv .venv
rem   .venv\Scripts\pip install -r requirements-dev.txt
setlocal
cd /d "%~dp0"

.venv\Scripts\pyinstaller nasdaq-advisor.spec --noconfirm
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo Running frozen selftest (fetches one ticker over the network)...
dist\nasdaq-advisor\nasdaq-advisor.exe --selftest
if errorlevel 1 (
    echo SELFTEST FAILED - see %%LOCALAPPDATA%%\nasdaq-advisor\advisor.log
    exit /b 1
)
echo Build OK: dist\nasdaq-advisor\nasdaq-advisor.exe
