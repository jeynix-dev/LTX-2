# LTX-2.5 distilled IC-LoRA — video upscaler / refiner

Fork of `Lightricks/LTX-2` with the patches needed to train an in-context LoRA
that maps **distorted video → clean video**, on the LTX-2.5 **22B distilled**
checkpoint, using the exact sigma schedule the distilled model is sampled with:

```
1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
```

The LoRA is trained on those 8 noise levels (0.0 dropped — nothing to denoise),
so at inference it is used with the same 8 steps instead of a generic schedule.

## What is different from upstream

| Area | Change |
|---|---|
| `timestep_samplers.py` | `DiscreteSigmaTimestepSampler`, selected by `timestep_sampling_mode: distilled` (alias `discrete`). Samples sigma from a fixed list, with optional per-sigma `weights` and `jitter`. |
| `trainer.py` | A fixed scene embedding replaces the text encoder completely — no Gemma, no embedding connectors, no prompt. This is how the Lightricks SDR-to-HDR IC-LoRA is conditioned. Context is passed unmasked so attention stays on the fast path. Adds TensorBoard + `train_log.csv` logging with per-sigma loss buckets. |
| `model_loader.py` / `config.py` | `model.video_only: true` builds the transformer without any audio blocks (≈33 GB instead of ≈120 GB at rank 256). Plus `model.fixed_context_path`, `validation.sigmas`. |
| `validation_runner.py` | Validation denoises with the explicit sigma list rather than a computed schedule. |
| `scripts/prepare_iclora_pairs.py` | Builds target + reference latents from a paired CSV. With `--fixed-context` no captions are encoded at all. Mixed frame buckets (e.g. `1920x1088x81;1920x1088x41`) are supported. |

Nothing in `ltx-core` is touched.

## Running it

**Windows** — the numbered `.bat` files, in order: `1_install`, `2_prepare_dataset`,
`3_train_iclora` (`3b_continue_training` to resume), `4_tensorboard`.

**Rented Linux GPU box** — `onstart_vast.sh` provisions the machine
(venv, torch, this fork, model downloads), then:

```bash
tmux new -s t 'bash /workspace/LTX-2/train_iclora_vast.sh'
tmux new -s t 'bash /workspace/LTX-2/train_iclora_vast.sh resume'
```

## Attention backends — read before hunting for wheels

`ltx-core` imports a real FlashAttention kernel from exactly two places:
`flash_attn_interface` (FA3, **sm_90** — H100/H200) and `flash_attn.cute`
(FA4, **sm_100** — datacenter Blackwell). Everything else falls back to torch
SDPA with priority `CUDNN > FLASH > EFFICIENT > MATH`.

There is **no code path to FlashAttention 2**, on any GPU. On an RTX PRO 6000
Blackwell (sm_120) installing `flash-attn` changes nothing — SDPA's cuDNN and
flash backends are what actually run, and they are fully supported there.

## Notes measured on this setup

- LoRA size: **5,111,808** trainable parameters per rank unit (rank 256 →
  1.31 B params, ≈2.6 GB at bf16).
- `torch.compile` was benchmarked over 30 steps: 34.85 s vs 35.14 s per step,
  i.e. **0.8 %** — inside the noise, and not worth the Triton/MSVC setup.
- The far bigger effect is thermal: step 1 runs at 23.1 s and the plateau is
  35.1 s, a **52 %** slowdown as the card heats up. Worth checking
  `nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv` on a
  long run.
- Upstream bug, still present: `_infer_reference_scale_factors_from_config` in
  `flexible.py` builds a relative `Path(self.config.video.latents_dir)` without
  `preprocessed_data_root`, so the rglob finds nothing and the scale factors
  fall back to 1. Keep `--reference-downscale-factor 1` until that is fixed,
  otherwise reference positions are silently wrong.
