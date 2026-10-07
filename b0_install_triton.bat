@echo off
chcp 65001 >nul
if not "%~1"=="_logged" (
    echo Installing triton-windows into the LTX venv - log: %~dp0triton_install_log.txt
    call "%~f0" _logged > "%~dp0triton_install_log.txt" 2>&1
    type "%~dp0triton_install_log.txt"
    pause
    exit /b
)
REM ============================================================
REM  torch.compile / Inductor needs Triton. On Windows that is the
REM  triton-windows package. Version 3.7.1.post27 is the one already
REM  working in G:\Lora\DiffSynth-Studio\0000\DiffSynth-Studio\deef2
REM  against the very same torch 2.10.0+cu130 this venv has.
REM
REM  --no-deps so nothing can pull a different torch. triton-windows
REM  declares no required runtime dependencies.
REM  To undo:  venv\Scripts\python.exe -m pip uninstall triton-windows
REM ============================================================
set VPY=G:\Lora\LTX-2\venv\Scripts\python.exe

echo [1/3] torch BEFORE:
"%VPY%" -c "import torch;print(' ',torch.__version__)"

echo [2/3] installing triton-windows==3.7.1.post27 ...
"%VPY%" -m pip install --no-deps triton-windows==3.7.1.post27 || goto :fail

echo [3/3] checks:
"%VPY%" -c "import torch;print('  torch AFTER:',torch.__version__)" || goto :fail
"%VPY%" -c "import triton;print('  triton:',triton.__version__)" || goto :fail
echo   compiling a tiny kernel to prove Inductor works...
"%VPY%" -c "import torch; f=torch.compile(lambda x:(x*2).sin().sum()); x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16,requires_grad=True); f(x).backward(); print('  torch.compile OK')" || goto :fail

echo.
echo ============ TRITON READY - now run b2_bench_compile.bat ============
exit /b 0
:fail
echo.
echo ERROR: triton install or torch.compile check failed - see the log above.
echo torch.compile is then simply not available here; b1 baseline still works.
exit /b 1
