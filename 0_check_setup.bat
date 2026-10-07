@echo off
chcp 65001 >nul
if not "%~1"=="_logged" (
    echo Checking setup... log: %~dp0check_log.txt
    call "%~f0" _logged > "%~dp0check_log.txt" 2>&1
    type "%~dp0check_log.txt"
    pause
    exit /b
)
set PYTHONUTF8=1
cd /d G:\Lora\LTX-2\packages\ltx-trainer
G:\Lora\LTX-2\venv\Scripts\python.exe scripts\check_setup.py
call G:\Lora\LTX-2\3_train_iclora.bat check
echo CHECK FINISHED
