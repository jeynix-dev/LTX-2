@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set PAUSE=pause
if /i "%~1"=="check" set PAUSE=type nul
REM ============================================================
REM  LTX-2.5 DISTILLED  IC-LoRA training (distorted -> clean)
REM  Training sigmas = 8-step distilled schedule:
REM    1.0 0.99375 0.9875 0.98125 0.975 0.909375 0.725 0.421875 (0.0)
REM  Validation = the same 8 sigmas, CFG 1, STG 0.
REM ============================================================

set ROOT=G:\Lora\LTX-2
set VPY=%ROOT%\venv\Scripts\python.exe
set BASE_CONFIG=%ROOT%\packages\ltx-trainer\configs\jeynix_ltx25_distilled_v2v_ic_lora.yaml

REM ---- run name / output ----
set RUN_NAME=ltx25_distilled_upscale_iclora_mixed
set OUTPUT_DIR=G:/Lora/LTX-2/outputs/%RUN_NAME%

REM ---- data (output of 2_prepare_dataset.bat) ----
REM Must match PRECOMPUTED in 2_prepare_dataset.bat.
REM BUCKET=1920x1088x81;1920x1088x41 there -> this folder name.
set DATA=G:/Lora/LTX-2/data/upscale_1920x1088x81_multi_ref1

REM ---- models ----
set MODELS=G:/ComfyUI_windows_portable/ComfyUI/models
set MODEL_PATH=%MODELS%/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
set TEXT_ENCODER=%MODELS%/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
set VIDEO_VAE=%MODELS%/vae/ltx-2.5-video-vae-conv-bf16.safetensors

REM ---- fixed text context (no text encoder / no connectors), as Lightricks HDR IC-LoRA ----
REM null = normal mode (Gemma captions from 2_prepare_dataset.bat)
set FIXED_CONTEXT=%MODELS%/embeddings/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors
set TE_ARG=%TEXT_ENCODER%
if not "%FIXED_CONTEXT%"=="null" set TE_ARG=null

REM ---- video only: audio blocks/weights of the transformer are not built (less VRAM), no audio VAE ----
set VIDEO_ONLY=true

REM ---- LoRA / optimisation ----
set RANK=256
set ALPHA=256
set LR=1.0e-4
set STEPS=58000
set GRAD_ACCUM=1
set SAVE_EVERY=2000
REM null = no quantization (H200). Options: int8-quanto, fp8-quanto
set QUANT=null

REM ---- sigmas (training) : 8 distilled steps, 0.0 dropped ----
set SIGMAS=[1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875]
REM null = uniform. Example favouring detail steps: [1,1,1,1,1,2,2,2]
set SIGMA_WEIGHTS=null

REM ---- validation (encoded on the fly; these clips MUST be in HOLDOUT of 2_prepare_dataset.bat) ----
set VAL_EVERY=500
REM Validation renders ONE fixed size. Both hold-out clips below are 41 frames,
REM so keep 41 here; change it only together with VAL_REF1/VAL_REF2.
set VAL_DIMS=[1920,1088,41]
set VAL_REF_DOWNSCALE=1
set VAL_REF1=G:/Lora/DiffSynth-Studio/0000/DiffSynth-Studio/data/refiner_control/Old/control_video/0_00003.mp4
set VAL_REF2=G:/Lora/DiffSynth-Studio/0000/DiffSynth-Studio/data/refiner_control/Old/control_video/9_00062.mp4

REM ---- resume: path to a .safetensors checkpoint or checkpoints folder, or null ----
REM Run "3b_resume_training.bat" (or this file with the argument: 3_train_iclora.bat resume)
REM to continue from the latest checkpoint in %OUTPUT_DIR%\checkpoints.
set RESUME=null
if /i "%~1"=="resume" set RESUME=%OUTPUT_DIR%/checkpoints
REM To carry the weights of the OLD 41-frame-only run into this one, uncomment:
REM set RESUME=G:/Lora/LTX-2/outputs/ltx25_distilled_upscale_iclora/checkpoints

REM ---- what to save next to each checkpoint for resuming ----
REM   minimal = scheduler + RNG + step  (a few KB; AdamW moments restart on resume)
REM   full    = the above + optimizer state (~1.5 GB per checkpoint, seamless resume)
REM   off     = nothing, resuming impossible
set SAVE_STATE=minimal

set PYTHONUTF8=1
cd /d "%ROOT%\packages\ltx-trainer"

set RUN_CONFIG=%OUTPUT_DIR%/run_config.yaml
"%VPY%" scripts\make_run_config.py "%BASE_CONFIG%" "%RUN_CONFIG%" ^
    model.model_path=%MODEL_PATH% ^
    model.text_encoder_path=%TE_ARG% ^
    model.video_vae_path=%VIDEO_VAE% ^
    model.audio_vae_path=null ^
    model.video_only=%VIDEO_ONLY% ^
    model.fixed_context_path=%FIXED_CONTEXT% ^
    model.load_checkpoint=%RESUME% ^
    lora.rank=%RANK% ^
    lora.alpha=%ALPHA% ^
    optimization.learning_rate=%LR% ^
    optimization.steps=%STEPS% ^
    optimization.gradient_accumulation_steps=%GRAD_ACCUM% ^
    acceleration.quantization=%QUANT% ^
    data.preprocessed_data_root=%DATA% ^
    flow_matching.timestep_sampling_mode=distilled ^
    flow_matching.timestep_sampling_params.sigmas=%SIGMAS% ^
    flow_matching.timestep_sampling_params.weights=%SIGMA_WEIGHTS% ^
    checkpoints.interval=%SAVE_EVERY% ^
    checkpoints.save_training_state=%SAVE_STATE% ^
    validation.interval=%VAL_EVERY% ^
    validation.video_dims=%VAL_DIMS% ^
    validation.samples.0.conditions.0.video=%VAL_REF1% ^
    validation.samples.1.conditions.0.video=%VAL_REF2% ^
    validation.samples.0.conditions.0.downscale_factor=%VAL_REF_DOWNSCALE% ^
    validation.samples.1.conditions.0.downscale_factor=%VAL_REF_DOWNSCALE% ^
    output_dir=%OUTPUT_DIR%
if errorlevel 1 (echo ERROR: bad config & %PAUSE% & exit /b 1)
if /i "%~1"=="check" (echo CONFIG OK: %RUN_CONFIG% & exit /b 0)

echo.
if /i "%~1"=="resume" (
    echo ===== RESUMING: %RUN_NAME%  from %RESUME% =====
) else (
    echo ===== Training: %RUN_NAME% =====
)
"%VPY%" scripts\train.py "%RUN_CONFIG%"
if errorlevel 1 (echo ERROR: training failed & %PAUSE% & exit /b 1)

echo.
echo ===== DONE. LoRA checkpoints: %OUTPUT_DIR%\checkpoints =====
%PAUSE%
