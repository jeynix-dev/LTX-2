@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM ============================================================
REM  Preprocess paired dataset for LTX-2.5 IC-LoRA
REM    control_video\*.mp4  (distorted)  -> reference_latents
REM    video\*.mp4          (clean)      -> latents (target)
REM  No audio is extracted/encoded. With FIXED_CONTEXT no captions at all.
REM ============================================================

set ROOT=G:\Lora\LTX-2
set VPY=%ROOT%\venv\Scripts\python.exe

REM ---- dataset ----
set DATASET_CSV=E:\PROJECT_UPSKALE\Dataset\metadata.csv
set TARGET_COLUMN=video
set REFERENCE_COLUMN=control_video
set CAPTION_COLUMN=prompt
REM Leave empty to keep captions from CSV ("up")
set CAPTION=

REM ---- fixed text context instead of the text encoder (as Lightricks HDR IC-LoRA) ----
REM Set -> NO prompt / NO Gemma: only target + reference video latents are encoded.
REM Leave empty to encode the captions from the CSV with Gemma.
REM Must match FIXED_CONTEXT in 3_train_iclora.bat
set FIXED_CONTEXT=G:\ComfyUI_windows_portable\ComfyUI\models\embeddings\ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors

REM ---- resolution buckets WxHxF (W,H multiple of 32; F = 8k+1) ----
REM Several buckets are separated by ";" - each clip goes to the bucket with the
REM largest frame count that still fits in it, e.g.
REM     set BUCKET=1920x1088x81;1920x1088x41
REM   81-frame clips -> the 81 bucket, 41-frame clips -> the 41 bucket.
REM A clip SHORTER than the smallest bucket is skipped with a warning in the log.
REM Target and reference of one pair must have the SAME frame count.
REM Training must then use optimization.batch_size: 1 (it already does).
set BUCKET=1920x1088x81
REM 1 = reference at full res, 2 = reference at half res (4x fewer ref tokens, faster)
set REF_DOWNSCALE=1
REM 0 = all pairs, otherwise random subset
set MAX_SAMPLES=0
REM Validation clips, excluded from training (must match 3_train_iclora.bat)
set HOLDOUT=0_00003.mp4,9_00062.mp4

REM ---- output (~6 MB per pair at 1920x1088x41, ~12 MB at x81, REF_DOWNSCALE=1) ----
REM The folder name is built from the FIRST bucket, so adding a second bucket does not
REM silently start a new folder. To write into a folder that already exists under a
REM different name, uncomment the override below - finished files there are then reused.
for /f "tokens=1 delims=;" %%B in ("%BUCKET%") do set BUCKET_TAG=%%B
if not "%BUCKET%"=="%BUCKET:;=%" set BUCKET_TAG=%BUCKET_TAG%_multi
set PRECOMPUTED=%ROOT%\data\upscale_%BUCKET_TAG%_ref%REF_DOWNSCALE%
REM Manual override (use the exact same path in DATA of 3_train_iclora.bat):
REM set PRECOMPUTED=G:\Lora\LTX-2\data\upscale_1920x1088x81_multi_ref1

REM ---- models ----
set MODELS=G:\ComfyUI_windows_portable\ComfyUI\models
set MODEL_PATH=%MODELS%\diffusion_models\ltx-2.5-22b-distilled-transformer-bf16.safetensors
set TEXT_ENCODER=%MODELS%\text_encoders\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
set VIDEO_VAE=%MODELS%\vae\ltx-2.5-video-vae-conv-bf16.safetensors

set PYTHONUTF8=1
cd /d "%ROOT%\packages\ltx-trainer"

for %%F in ("%MODEL_PATH%" "%VIDEO_VAE%" "%DATASET_CSV%") do (
    if not exist %%F (echo ERROR: not found %%F & pause & exit /b 1)
)

set "MULTI_BUCKET="
if not "%BUCKET%"=="%BUCKET:;=%" set MULTI_BUCKET=1
if defined MULTI_BUCKET if not "%REF_DOWNSCALE%"=="1" (
    echo ERROR: several buckets require REF_DOWNSCALE=1 ^(the preprocessor rejects scaled
    echo        references together with multiple buckets^).
    pause
    exit /b 1
)

set CAPTION_ARG=
if not "%CAPTION%"=="" set CAPTION_ARG=--caption "%CAPTION%"
set FIXED_ARG=--text-encoder-path "%TEXT_ENCODER%"
if not "%FIXED_CONTEXT%"=="" set FIXED_ARG=--fixed-context "%FIXED_CONTEXT%"

echo Buckets: %BUCKET%
echo Output:  %PRECOMPUTED%
"%VPY%" scripts\prepare_iclora_pairs.py "%DATASET_CSV%" ^
    --output-dir "%PRECOMPUTED%" ^
    --model-path "%MODEL_PATH%" ^
    --video-vae-path "%VIDEO_VAE%" ^
    --resolution-buckets "%BUCKET%" ^
    --target-column %TARGET_COLUMN% ^
    --reference-column %REFERENCE_COLUMN% ^
    --caption-column %CAPTION_COLUMN% %CAPTION_ARG% %FIXED_ARG% ^
    --holdout "%HOLDOUT%" ^
    --max-samples %MAX_SAMPLES% ^
    --reference-downscale-factor %REF_DOWNSCALE%
if errorlevel 1 (echo ERROR: preprocessing failed & pause & exit /b 1)

echo.
echo ===== DONE: %PRECOMPUTED% =====
echo (Re-running is safe: finished files are skipped.)
pause
