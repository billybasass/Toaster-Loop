@echo off
rem Double-click wrapper for cleanup.ps1 (execution-policy friendly).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0cleanup.ps1" %*
