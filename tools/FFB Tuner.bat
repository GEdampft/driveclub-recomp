@echo off
rem Opens the force feedback tuner (FFB Tuner.ps1 next to this file).
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0FFB Tuner.ps1"
