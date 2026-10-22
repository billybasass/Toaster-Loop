@echo off
rem Double-click wrapper for setup.ps1 (execution-policy friendly).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
