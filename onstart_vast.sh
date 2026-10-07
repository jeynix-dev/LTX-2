#!/bin/bash
# =============================================================================
#  vast.ai ONSTART  -  LTX-2.5 distilled IC-LoRA trainer
#  Target GPU: RTX PRO 6000 Blackwell 96 GB  (sm_120)
#
#  What it does
#    1. clean venv from the image's system python (no conda)
#    2. torch from the cu128 index, then HARD-PINS the torch family so no
#       later dependency can silently swap it
#    3. clones our fork of ltx-trainer and installs it with --no-deps
#       + an explicit dependency list (skips the packages the training path
#       never imports: torchcodec, bitsandbytes, google-genai, openai, scenedetect)
#    4. attention sanity check (see the ATTENTION note at the bottom)
#    5. downloads DiT + conv VAE + the scene embedding from HuggingFace
#
#  Set these as ENV VARS in the vast.ai template (NOT in this file):
#    HF_TOKEN   - HuggingFace read token (required, LTX-2.5 is a gated repo)
#    GH_TOKEN   - GitHub token, ONLY needed if the fork is private
#
#  Progress / errors:  tail -f /workspace/onstart.log
#  When it finishes:   tmux new -s t 'bash /workspace/LTX-2/train_iclora_vast.sh'
# =============================================================================

set -uo pipefail
mkdir -p /workspace
exec > >(tee -a /workspace/onstart.log) 2>&1
echo "=============== onstart $(date -u +%FT%TZ) ==============="

# ---------------------------------------------------------------- settings ---
FORK_URL="${FORK_URL:-https://github.com/jeynix-dev/LTX-2.git}"
FORK_BRANCH="${FORK_BRANCH:-jeynix-iclora}"

ROOT=/workspace/LTX-2
VENV=/workspace/venv
MODELS=/workspace/models

# Leave empty to take the newest cu128 build (always has sm_120 kernels).
# After the first successful run, copy the versions the log prints in here to
# make the box reproducible.
TORCH_VERSION="${TORCH_VERSION:-}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

# ---------------------------------------------------------------- packages ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs tmux htop ffmpeg libgl1 libglib2.0-0 \
                       python3-venv python3-dev build-essential curl >/dev/null
git lfs install --skip-repo >/dev/null 2>&1 || true

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ------------------------------------------------------------------- venv ----
# System python, deliberately NOT conda.
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo ">>> base python: $PY  ($($PY -V 2>&1))"
rm -rf "$VENV"
"$PY" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip setuptools wheel

# -------------------------------------------------------------------- torch --
echo ">>> installing torch from $TORCH_INDEX"
if [ -n "$TORCH_VERSION" ]; then
    pip install -q "torch==$TORCH_VERSION" torchvision torchaudio --index-url "$TORCH_INDEX"
else
    pip install -q torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

python - <<'PY'
import sys, torch
archs = torch.cuda.get_arch_list()
print(f">>> torch {torch.__version__}  cuda {torch.version.cuda}")
print(f">>> arch list: {' '.join(archs)}")
if not any(a.endswith("120") for a in archs):
    sys.exit(
        "FATAL: this torch build has no sm_120 kernels, the RTX PRO 6000 Blackwell "
        "cannot run it.\n       Set TORCH_INDEX to a newer CUDA index (e.g. .../whl/cu130) "
        "and re-run."
    )
PY
[ $? -eq 0 ] || { echo "ABORT: torch/GPU mismatch"; exit 1; }

# Freeze the torch family. Every pip install after this line reads this file,
# so a dependency asking for a different torch fails loudly instead of pulling
# a CPU-only or wrong-CUDA wheel over a working one.
mkdir -p /workspace/pip
python - > /workspace/pip/constraints.txt <<'PY'
import importlib.metadata as md
for p in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        pass
PY
cat /workspace/pip/constraints.txt
export PIP_CONSTRAINT=/workspace/pip/constraints.txt

# ------------------------------------------------------------------- clone ----
echo ">>> cloning $FORK_URL ($FORK_BRANCH)"
rm -rf "$ROOT"
CLONE_URL="$FORK_URL"
if [ -n "${GH_TOKEN:-}" ]; then
    # private fork: inject the token only into the clone URL, never store it
    CLONE_URL=$(printf '%s' "$FORK_URL" | sed "s#^https://#https://x-access-token:${GH_TOKEN}@#")
fi
git clone -q --branch "$FORK_BRANCH" --single-branch "$CLONE_URL" "$ROOT" || {
    echo "FATAL: clone failed."
    echo "  - public repo?  check the URL and branch name"
    echo "  - private repo? set GH_TOKEN in the vast template"
    exit 1
}
git -C "$ROOT" remote set-url origin "$FORK_URL"   # drop the token from .git/config
git -C "$ROOT" log -1 --format='>>> HEAD %h %s'

# ---------------------------------------------------------------- install ----
# ltx-core / ltx-trainer are installed WITHOUT their dependency closure, then
# the deps the training path actually imports are installed by hand.
# Deliberately NOT installed:
#   torchcodec    - only used by torchaudio.load(); we train video-only
#   bitsandbytes  - only for the 8-bit Gemma text encoder; we use no text encoder
#   google-genai, openai - captioning / prompt enhancement only
#   scenedetect   - dataset splitting, already done offline
echo ">>> installing ltx-core + ltx-trainer (--no-deps)"
pip install -q --no-deps -e "$ROOT/packages/ltx-core"
pip install -q --no-deps -e "$ROOT/packages/ltx-trainer"

echo ">>> installing dependencies"
pip install -q \
    "einops" "numpy" "scipy>=1.14" "colour-science" "av>=14.2.1" \
    "transformers>=5.8.0,<5.15" "safetensors>=0.5.0" "accelerate>=1.2.1" \
    "huggingface-hub[hf-xet,cli]>=0.31.4" \
    "imageio>=2.37.0" "imageio-ffmpeg>=0.6.0" \
    "opencv-python-headless>=4.11.0.86" "pillow-heif>=0.21.0" \
    "optimum-quanto>=0.2.6" "peft>=0.14.0" \
    "pandas>=2.2.3" "pydantic>=2.10.4" "rich>=13.9.4" \
    "sentencepiece>=0.2.0" "soundfile>=0.12.1" \
    "typer>=0.15.1" "setuptools>=79.0.0" \
    "tensorboard" "wandb>=0.27.0"

python - <<'PY'
import ltx_core  # noqa: F401
from ltx_trainer.config import ModelConfig
from ltx_trainer.timestep_samplers import SAMPLERS
print(">>> ltx_trainer imports OK")
assert "distilled" in SAMPLERS, "the discrete sigma sampler is missing - wrong branch?"
for f in ("fixed_context_path", "video_only"):
    assert f in ModelConfig.model_fields, f"{f} missing - wrong branch?"
print(">>> fork patches present: distilled sampler, fixed_context_path, video_only")
PY
[ $? -eq 0 ] || { echo "ABORT: trainer import / patch check failed"; exit 1; }

# -------------------------------------------------------------- attention ----
# ltx-core can only call a real FlashAttention kernel in two cases:
#   flash_attn_interface (FA3)  -> sm_90  (H100/H200)
#   flash_attn.cute      (FA4)  -> sm_100 (datacenter Blackwell B100/B200)
# The RTX PRO 6000 Blackwell is sm_120, so neither import path is ever used -
# installing flash-attn 2 would change nothing, ltx-core has no code path to it.
# What it does use is torch SDPA with CUDNN > FLASH > EFFICIENT > MATH priority,
# and the cuDNN/flash SDPA backends are fully supported on sm_120.
python - <<'PY'
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
q = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
for name, backend in (("CUDNN", SDPBackend.CUDNN_ATTENTION), ("FLASH", SDPBackend.FLASH_ATTENTION)):
    try:
        with sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        print(f">>> SDPA {name}: OK")
    except Exception as e:                                  # noqa: BLE001
        print(f">>> SDPA {name}: unavailable ({type(e).__name__})")
PY

# ------------------------------------------------------------------ models ----
if [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: HF_TOKEN is not set. Add it as an env var in the vast.ai template."
    exit 1
fi
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$MODELS/diffusion_models" "$MODELS/vae" "$MODELS/embeddings"

get () {   # repo  file-in-repo  destination-dir
    local repo="$1" file="$2" dest="$3"
    echo ">>> $repo :: $file"
    hf download "$repo" "$file" --local-dir "$dest" --token "$HF_TOKEN" >/dev/null || {
        echo "FATAL: download failed. Files actually in $repo:"
        REPO="$repo" python - <<'PY'
import os
from huggingface_hub import list_repo_files
try:
    for f in list_repo_files(os.environ["REPO"], token=os.environ["HF_TOKEN"]):
        if f.endswith(".safetensors"):
            print("   ", f)
except Exception as e:                                      # noqa: BLE001
    print("    could not list the repo:", e)
    print("    -> the token may be missing access; accept the licence at")
    print("       https://huggingface.co/" + os.environ["REPO"])
PY
        exit 1
    }
}

get Lightricks/LTX-2.5 diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors "$MODELS"
get Lightricks/LTX-2.5 vae/ltx-2.5-video-vae-conv-bf16.safetensors                          "$MODELS"
get Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors "$MODELS/embeddings"

mkdir -p /workspace/data /workspace/outputs /workspace/data/val

echo
echo ">>> model files:"
find "$MODELS" -name '*.safetensors' -printf '    %10s  %p\n' | sort -k2

MISSING=0
for f in "$MODELS/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
         "$MODELS/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
         "$MODELS/embeddings/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"; do
    [ -s "$f" ] || { echo "MISSING: $f"; MISSING=1; }
done

echo
echo "=============== onstart finished $(date -u +%FT%TZ) ==============="
if [ "$MISSING" = 1 ]; then
    echo "!! some model files are missing - fix the paths above before training."
    exit 1
fi
cat <<'EOF'

NEXT STEPS
  1. upload the preprocessed latents to  /workspace/data/upscale_1920x1088x81_multi_ref1
         (it must contain  latents/  and  reference_latents/)
  2. optional, for validation samples:
         /workspace/data/val/0_00003.mp4
         /workspace/data/val/9_00062.mp4
         without them the script just runs with validation off
  3. start training inside tmux so an ssh drop does not kill it:
         tmux new -s t 'bash /workspace/LTX-2/train_iclora_vast.sh'
     resume from the newest checkpoint:
         tmux new -s t 'bash /workspace/LTX-2/train_iclora_vast.sh resume'
  4. loss curves:
         source /workspace/venv/bin/activate
         tensorboard --logdir /workspace/outputs --host 0.0.0.0 --port 6006
EOF
