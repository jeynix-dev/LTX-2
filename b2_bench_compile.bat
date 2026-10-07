@echo off
chcp 65001 >nul
if not "%~1"=="_logged" (
    echo TORCH.COMPILE - full log: %~dp0bench_compile_log.txt
    echo First steps include Inductor compilation - that is expected, the report ignores warm-up.
    call "%~f0" _logged > "%~dp0bench_compile_log.txt" 2>&1
    type "%~dp0bench_compile_log.txt"
    pause
    exit /b
)
setlocal EnableDelayedExpansion
set ROOT=G:\Lora\LTX-2
set VPY=%ROOT%\venv\Scripts\python.exe
set ACC=%ROOT%\venv\Scripts\accelerate.exe
set BASE_CONFIG=%ROOT%\packages\ltx-trainer\configs\jeynix_ltx25_distilled_v2v_ic_lora.yaml

REM ---- benchmark settings (identical in BOTH runs) ----
set STEPS=30
set DATA=G:/Lora/LTX-2/data/upscale_1920x1088x81_ref1
set RANK=128
set ALPHA=128
set LR=1.0e-4

set MODELS=G:/ComfyUI_windows_portable/ComfyUI/models
set MODEL_PATH=%MODELS%/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
set VIDEO_VAE=%MODELS%/vae/ltx-2.5-video-vae-conv-bf16.safetensors
set FIXED_CONTEXT=%MODELS%/embeddings/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors

REM ---- MSVC environment for Inductor's CPU kernels ----
REM Inductor compiles a few small CPU-side kernels with MSVC. cl.exe alone is not
REM enough: it needs INCLUDE/LIB from vcvars64.bat, otherwise it dies on omp.h.
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VCVARS%" if exist "%VSWHERE%" for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS%" echo Loading MSVC env from: %VCVARS%
if exist "%VCVARS%" call "%VCVARS%" >nul
if not exist "%VCVARS%" echo WARNING: vcvars64.bat not found - Inductor CPU codegen will fail.

set PYTHONUTF8=1
cd /d "%ROOT%\packages\ltx-trainer"

set OUTPUT_DIR=G:/Lora/LTX-2/outputs/_bench_compile
set RUN_CONFIG=%OUTPUT_DIR%/run_config.yaml

"%VPY%" scripts\make_run_config.py "%BASE_CONFIG%" "%RUN_CONFIG%" ^
    model.model_path=%MODEL_PATH% ^
    model.text_encoder_path=null ^
    model.video_vae_path=%VIDEO_VAE% ^
    model.audio_vae_path=null ^
    model.video_only=true ^
    model.fixed_context_path=%FIXED_CONTEXT% ^
    model.load_checkpoint=null ^
    lora.rank=%RANK% ^
    lora.alpha=%ALPHA% ^
    optimization.learning_rate=%LR% ^
    optimization.steps=%STEPS% ^
    optimization.batch_size=1 ^
    optimization.gradient_accumulation_steps=1 ^
    acceleration.quantization=null ^
    data.preprocessed_data_root=%DATA% ^
    checkpoints.interval=null ^
    validation.interval=null ^
    tensorboard=false ^
    output_dir=%OUTPUT_DIR%
if errorlevel 1 (echo ERROR: bad config & exit /b 1)

echo.
echo Pre-flight: compiling a tiny GPU graph...
"%VPY%" -c "import torch;f=torch.compile(lambda x:(x*2).sin().sum());x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16,requires_grad=True);f(x).backward();print('  GPU codegen OK')" || goto :nocompile
echo Pre-flight: compiling a tiny CPU graph ^(this is what failed on omp.h^)...
"%VPY%" -c "import torch;f=torch.compile(lambda x:(x*2).sin().sum());f(torch.randn(256,256));print('  CPU codegen OK')" || goto :nocompile

echo.
echo ===== TORCH.COMPILE : %STEPS% steps =====
echo Per-step times go to %OUTPUT_DIR%\train_log.csv
echo.
"%ACC%" launch --config_file "configs\accelerate\single_gpu_compile.yaml" scripts\train.py "%RUN_CONFIG%" --disable-progress-bars
if errorlevel 1 (echo ERROR: benchmark run failed & exit /b 1)
echo.
echo ===== DONE: TORCH.COMPILE =====
exit /b 0

:nocompile
echo.
echo ===== torch.compile is NOT usable on this machine =====
echo Inductor could not build its kernels. Nothing was changed in your training setup -
echo keep using the normal pipeline. The baseline number from b1 still stands.
exit /b 1
