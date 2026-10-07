@echo off
chcp 65001 >nul
REM ============================================================
REM  TensorBoard for LTX IC-LoRA runs.
REM  Run in a SECOND window while training is going, then open
REM      http://localhost:6007
REM  Logs: G:\Lora\LTX-2\outputs\<run>\tensorboard   (+ train_log.csv)
REM  Tags: train/loss, train/loss_ema, train/learning_rate, train/step_time,
REM        train/sigma, loss_by_sigma/<sigma>  (one curve per distilled sigma)
REM ============================================================
set VPY=G:\Lora\LTX-2\venv\Scripts\python.exe
set LOGDIR=G:\Lora\LTX-2\outputs
set PORT=6007

"%VPY%" -c "import tensorboard" 2>nul
if errorlevel 1 (
    echo Installing tensorboard into the LTX venv...
    "%VPY%" -m pip install tensorboard
)

echo Starting TensorBoard on http://localhost:%PORT%  (logdir=%LOGDIR%)
echo Press Ctrl+C to stop.
start "" http://localhost:%PORT%
"%VPY%" -m tensorboard.main --logdir "%LOGDIR%" --port %PORT%
pause
