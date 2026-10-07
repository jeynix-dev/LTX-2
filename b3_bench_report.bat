@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d G:\Lora\LTX-2\packages\ltx-trainer
G:\Lora\LTX-2\venv\Scripts\python.exe scripts\bench_report.py G:\Lora\LTX-2\outputs --steps 30 --warmup 10
pause
