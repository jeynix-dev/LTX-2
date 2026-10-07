@echo off
chcp 65001 >nul
REM ============================================================
REM  Replay train_log.csv into TensorBoard, live.
REM
REM  Use this when a run was started while the `tensorboard` package
REM  was not installed yet: that run has only train_log.csv, so its
REM  curves are missing in TensorBoard. This keeps following the CSV
REM  while training continues - no need to restart training.
REM
REM  Leave this window open next to 4_tensorboard.bat, then refresh
REM  http://localhost:6007 - the run appears as "<run>\tensorboard_csv".
REM ============================================================
set VPY=G:\Lora\LTX-2\venv\Scripts\python.exe
set OUTPUTS=G:\Lora\LTX-2\outputs
set PYTHONUTF8=1
cd /d G:\Lora\LTX-2\packages\ltx-trainer

"%VPY%" scripts\csv_to_tensorboard.py "%OUTPUTS%"
pause
