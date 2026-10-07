@echo off
chcp 65001 >nul
if not "%~1"=="_logged" (
    echo BASELINE - no torch.compile - full log: %~dp0bench_baseline_log.txt
    call "%~f0" _logged > "%~dp0bench_baseline_log.txt" 2>&1
    type "%~dp0bench_baseline_log.txt"
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

set PYTHONUTF8=1
cd /d "%ROOT%\packages\ltx-trainer"

set OUTPUT_DIR=G:/Lora/LTX-2/outputs/_bench_nocompile
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
echo ===== BASELINE - no torch.compile : %STEPS% steps =====
echo Per-step times go to %OUTPUT_DIR%\train_log.csv
echo.
"%ACC%" launch --config_file "configs\accelerate\single_gpu.yaml" scripts\train.py "%RUN_CONFIG%" --disable-progress-bars
if errorlevel 1 (echo ERROR: benchmark run failed & exit /b 1)
echo.
echo ===== DONE: BASELINE - no torch.compile =====
