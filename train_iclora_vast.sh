#!/bin/bash
set -euo pipefail
# ============================================================================
#  LTX-2.5 DISTILLED  IC-LoRA training on vast.ai  (RTX PRO 6000 Blackwell)
#
#  Mirrors 3_train_iclora.bat. Same sigmas, same fixed context, video-only,
#  same rank / steps / schedule - only the paths are Linux ones.
#
#  Run it inside tmux, the session survives a dropped ssh:
#      tmux new -s t 'bash /workspace/LTX-2/train_iclora_vast.sh'
#      tmux new -s t 'bash /workspace/LTX-2/train_iclora_vast.sh resume'
#
#  "resume" continues from the newest checkpoint in OUTPUT_DIR/checkpoints.
# ============================================================================

ROOT=/workspace/LTX-2
VENV=/workspace/venv
MODELS=/workspace/models
VPY="$VENV/bin/python"
BASE_CONFIG="$ROOT/packages/ltx-trainer/configs/jeynix_ltx25_distilled_v2v_ic_lora.yaml"

# ---- run name / output ----
RUN_NAME=ltx25_distilled_upscale_iclora_mixed
OUTPUT_DIR=/workspace/outputs/$RUN_NAME

# ---- data: the preprocessed latents you uploaded ----
DATA=/workspace/data/upscale_1920x1088x81_multi_ref1

# ---- models (downloaded by the onstart script) ----
MODEL_PATH=$MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
VIDEO_VAE=$MODELS/vae/ltx-2.5-video-vae-conv-bf16.safetensors
FIXED_CONTEXT=$MODELS/embeddings/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors

# ---- no text encoder: the fixed scene embedding replaces Gemma entirely ----
TE_ARG=null

# ---- video only: audio blocks of the transformer are never built ----
VIDEO_ONLY=true

# ---- LoRA / optimisation (same as 3_train_iclora.bat) ----
RANK=256
ALPHA=256
LR=1.0e-4
STEPS=58000
GRAD_ACCUM=1
SAVE_EVERY=2000
QUANT=null

# ---- training sigmas: the 8 distilled steps, 0.0 dropped ----
SIGMAS='[1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875]'
SIGMA_WEIGHTS=null

# ---- validation: hold-out clips, must NOT be in the training latents ----
VAL_EVERY=500
VAL_DIMS='[1920,1088,41]'
VAL_REF_DOWNSCALE=1
VAL_REF1=/workspace/data/val/0_00003.mp4
VAL_REF2=/workspace/data/val/9_00062.mp4

# ---- what to store next to each checkpoint for resuming ----
#   minimal = scheduler + RNG + step (a few KB)
#   full    = the above + optimizer state (~3 GB per checkpoint at rank 256)
SAVE_STATE=minimal

# ---- resume ----
RESUME=null
if [ "${1:-}" = "resume" ]; then
    RESUME=$OUTPUT_DIR/checkpoints
fi

# ============================================================================

source "$VENV/bin/activate"
cd "$ROOT/packages/ltx-trainer"
mkdir -p "$OUTPUT_DIR"

for f in "$MODEL_PATH" "$VIDEO_VAE" "$FIXED_CONTEXT"; do
    [ -s "$f" ] || { echo "ERROR: missing model file: $f"; exit 1; }
done
for d in "$DATA/latents" "$DATA/reference_latents"; do
    [ -d "$d" ] || { echo "ERROR: missing data directory: $d"; exit 1; }
done
echo "latents: $(find "$DATA/latents" -name '*.pt' | wc -l)   references: $(find "$DATA/reference_latents" -name '*.pt' | wc -l)"

# Validation needs the two hold-out clips. Without them the ValidationRunner
# would die at start-up, so switch validation off instead of losing the run.
VAL_ARGS=( "validation.interval=$VAL_EVERY"
           "validation.video_dims=$VAL_DIMS"
           "validation.samples.0.conditions.0.video=$VAL_REF1"
           "validation.samples.1.conditions.0.video=$VAL_REF2"
           "validation.samples.0.conditions.0.downscale_factor=$VAL_REF_DOWNSCALE"
           "validation.samples.1.conditions.0.downscale_factor=$VAL_REF_DOWNSCALE" )
if [ ! -s "$VAL_REF1" ] || [ ! -s "$VAL_REF2" ]; then
    echo "WARNING: validation clips not found - validation disabled for this run."
    echo "         upload them to /workspace/data/val/ and restart to get samples."
    VAL_ARGS=( "validation.interval=null" )
fi

RUN_CONFIG=$OUTPUT_DIR/run_config.yaml
"$VPY" scripts/make_run_config.py "$BASE_CONFIG" "$RUN_CONFIG" \
    "model.model_path=$MODEL_PATH" \
    "model.text_encoder_path=$TE_ARG" \
    "model.video_vae_path=$VIDEO_VAE" \
    "model.audio_vae_path=null" \
    "model.video_only=$VIDEO_ONLY" \
    "model.fixed_context_path=$FIXED_CONTEXT" \
    "model.load_checkpoint=$RESUME" \
    "lora.rank=$RANK" \
    "lora.alpha=$ALPHA" \
    "optimization.learning_rate=$LR" \
    "optimization.steps=$STEPS" \
    "optimization.gradient_accumulation_steps=$GRAD_ACCUM" \
    "acceleration.quantization=$QUANT" \
    "data.preprocessed_data_root=$DATA" \
    "flow_matching.timestep_sampling_mode=distilled" \
    "flow_matching.timestep_sampling_params.sigmas=$SIGMAS" \
    "flow_matching.timestep_sampling_params.weights=$SIGMA_WEIGHTS" \
    "checkpoints.interval=$SAVE_EVERY" \
    "checkpoints.save_training_state=$SAVE_STATE" \
    "${VAL_ARGS[@]}" \
    "output_dir=$OUTPUT_DIR"

echo
if [ "$RESUME" != "null" ]; then
    echo "===== RESUMING: $RUN_NAME  from $RESUME ====="
else
    echo "===== Training: $RUN_NAME ====="
fi
echo "TensorBoard: $OUTPUT_DIR/tensorboard    CSV: $OUTPUT_DIR/train_log.csv"
echo

# Less allocator fragmentation on long runs. Costs nothing, changes no numerics.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$VPY" scripts/train.py "$RUN_CONFIG" --disable-progress-bars

echo
echo "===== DONE. LoRA checkpoints: $OUTPUT_DIR/checkpoints ====="
