@echo off
rem Double-click wrapper for launch.ps1 (execution-policy friendly).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
