@echo off
chcp 65001 >nul
REM Self-logging: the real install runs with output in install_log.txt
if not "%~1"=="_logged" (
    echo Installing LTX-2 trainer... full log: %~dp0install_log.txt
    echo This window will show the log at the end. Please wait, it takes a while.
    call "%~f0" _logged > "%~dp0install_log.txt" 2>&1
    type "%~dp0install_log.txt"
    pause
        exit /b
)
setlocal EnableDelayedExpansion
REM ============================================================
REM  LTX-2 trainer (Lightricks) - venv install on system Python
REM  Repo: https://github.com/Lightricks/LTX-2  (packages/ltx-trainer)
REM ============================================================

set ROOT=G:\Lora\LTX-2
set VENV=%ROOT%\venv

REM System Python (no conda). Change if needed.
set PY=C:\Python313\python.exe
if not exist "%PY%" set PY=C:\Users\AI\AppData\Local\Programs\Python\Python310\python.exe

REM PyTorch CUDA wheels index (cu130 for H200 / new drivers; fallback cu128)
set TORCH_INDEX=https://download.pytorch.org/whl/cu130
set TORCH_INDEX_FALLBACK=https://download.pytorch.org/whl/cu128

REM Text encoder for LTX-2.5 (bf16 packed Gemma 4). Downloaded if missing.
set COMFY_MODELS=G:\ComfyUI_windows_portable\ComfyUI\models
set TEXT_ENCODER=%COMFY_MODELS%\text_encoders\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors

echo Python: %PY%
"%PY%" --version || (echo ERROR: python not found & exit /b 1)

cd /d "%ROOT%"

if not exist "%VENV%\Scripts\python.exe" (
    echo [1/5] Creating venv...
    "%PY%" -m venv "%VENV%" || (echo ERROR: venv failed & exit /b 1)
)
set VPY=%VENV%\Scripts\python.exe

echo [2/5] Upgrading pip...
"%VPY%" -m pip install -U pip wheel setuptools || goto :fail

echo [3/5] Installing PyTorch (CUDA)...
"%VPY%" -m pip install -U torch torchvision torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo cu130 failed, trying cu128...
    "%VPY%" -m pip install -U torch torchvision torchaudio --index-url %TORCH_INDEX_FALLBACK% || goto :fail
)

echo [4/5] Installing ltx-core + ltx-trainer (editable)...
"%VPY%" -m pip install -e packages\ltx-core || goto :fail
"%VPY%" -m pip install -e packages\ltx-trainer || goto :fail

echo.
echo ---- check ----
"%VPY%" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
"%VPY%" -c "import ltx_core, ltx_trainer; from ltx_trainer.timestep_samplers import SAMPLERS; print('ltx-trainer OK, samplers:', list(SAMPLERS))" || goto :fail

echo.
echo [5/5] Text encoder (gemma4-12b-with-proj-ltx-2.5-bf16, ~24 GB)...
if exist "%TEXT_ENCODER%" (
    echo Found: %TEXT_ENCODER%
) else (
    echo Downloading from huggingface.co/Lightricks/LTX-2.5 ...
    echo If it asks for access: run  "%VENV%\Scripts\hf.exe" auth login  and accept the license on the model page.
    "%VENV%\Scripts\hf.exe" download Lightricks/LTX-2.5 text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors --local-dir "%COMFY_MODELS%"
    if errorlevel 1 echo WARNING: download failed - download the file manually into %COMFY_MODELS%\text_encoders
)

echo.
echo ============ INSTALL DONE ============
exit /b 0

:fail
echo.
echo ERROR: install failed (see messages above)
exit /b 1
