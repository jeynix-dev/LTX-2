#!/usr/bin/env python3
"""
Prepare a paired (distorted -> clean) video dataset for LTX IC-LoRA training.

Input: a CSV with columns  video, control_video, prompt   (paths relative to the CSV folder)
       video          = clean / original clip   -> training TARGET
       control_video  = distorted clip          -> IC-LoRA REFERENCE (conditioning)
       prompt         = caption

Steps:
  1. Writes  metadata_ltx_iclora.csv  (caption, video, reference_video) next to the source CSV,
     skipping missing files and the hold-out clips used for validation.
  2. If every caption is identical (e.g. "up"), encodes it ONCE with Gemma and hard-links the
     embedding for every sample (NTFS hard links -> no extra disk space; ~10 MB per sample otherwise).
  3. Runs the official LTX preprocessing (scripts/process_dataset.py) for target + reference latents.

Usage (see 2_prepare_dataset.bat):
  python scripts/prepare_iclora_pairs.py E:/.../metadata.csv --output-dir G:/.../precomputed ^
      --model-path ... --text-encoder-path ... --video-vae-path ... --resolution-buckets 1920x1088x81
"""

import csv
import os
import random
import shutil
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_captions import compute_captions_embeddings  # noqa: E402
from process_dataset import preprocess_dataset  # noqa: E402

from ltx_trainer import logger  # noqa: E402
from ltx_trainer.gpu_utils import free_gpu_memory_context  # noqa: E402
from ltx_trainer.model_loader import read_video_scale_factors, resolve_video_vae_path  # noqa: E402
from ltx_trainer.process_videos import parse_resolution_buckets  # noqa: E402

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

OUT_NAME = "metadata_ltx_iclora.csv"
ONE_CAPTION_NAME = "metadata_ltx_iclora__one_caption.csv"


def _norm(p: str) -> str:
    return p.strip().replace("\\", "/")


@app.command()
def main(  # noqa: PLR0913, PLR0915
    source_csv: str = typer.Argument(..., help="CSV with columns video, control_video, prompt"),
    output_dir: str = typer.Option(..., help="Where to write precomputed latents (latents/, reference_latents/, conditions/)"),
    model_path: str = typer.Option(..., help="LTX transformer .safetensors"),
    text_encoder_path: str | None = typer.Option(None, help="Packed Gemma text encoder .safetensors (not needed with --fixed-context)"),
    video_vae_path: str = typer.Option(..., help="Video VAE .safetensors"),
    audio_vae_path: str | None = typer.Option(None, help="Audio VAE .safetensors (not needed for video-only)"),
    resolution_buckets: str = typer.Option("1920x1088x81", help='"WxHxF" (W,H multiple of 32, F % 8 == 1)'),
    target_column: str = typer.Option("video", help="Column with the clean target clip"),
    reference_column: str = typer.Option("control_video", help="Column with the distorted reference clip"),
    caption_column: str = typer.Option("prompt", help="Column with the caption"),
    caption: str | None = typer.Option(None, help="Override caption for ALL samples"),
    fixed_context: str | None = typer.Option(
        None,
        help="Train with a fixed context (e.g. the HDR scene-emb): no captions/Gemma, only video latents "
        "are encoded. Training must use model.fixed_context_path.",
    ),
    holdout: str = typer.Option("", help="Comma-separated file names to exclude (validation clips)"),
    max_samples: int = typer.Option(0, help="Use at most N random pairs (0 = all)"),
    seed: int = typer.Option(42),
    reference_downscale_factor: int = typer.Option(1),
    reference_temporal_scale_factor: int = typer.Option(1),
    vae_tiling: bool = typer.Option(False),
    device: str = typer.Option("cuda"),
    overwrite: bool = typer.Option(False),
) -> None:
    src = Path(source_csv).resolve()
    root = src.parent
    out_csv = root / OUT_NAME
    pre = Path(output_dir)
    pre.mkdir(parents=True, exist_ok=True)

    holdout_names = {h.strip() for h in holdout.split(",") if h.strip()}

    # ---------------------------------------------------------------- 1. metadata
    rows: list[dict[str, str]] = []
    missing = 0
    with open(src, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tgt, ref = _norm(r[target_column]), _norm(r[reference_column])
            cap = caption if caption is not None else (r.get(caption_column) or "").strip()
            if Path(tgt).name in holdout_names or Path(ref).name in holdout_names:
                continue
            if not (root / tgt).is_file() or not (root / ref).is_file():
                missing += 1
                continue
            rows.append({"caption": cap, "video": tgt, "reference_video": ref})

    if max_samples and len(rows) > max_samples:
        random.Random(seed).shuffle(rows)
        rows = sorted(rows[:max_samples], key=lambda x: x["video"])

    if not rows:
        raise typer.BadParameter("No valid pairs found")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["caption", "video", "reference_video"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Pairs: {len(rows):,} (missing skipped: {missing}, hold-out: {len(holdout_names)}) -> {out_csv}")

    # ---------------------------------------------------------------- 2. shared caption
    captions = {r["caption"] for r in rows}
    cond_dir = pre / "conditions"
    master = cond_dir / Path(rows[0]["video"]).with_suffix(".pt")
    if fixed_context:
        _latents_only(
            out_csv, pre, model_path, video_vae_path, resolution_buckets, reference_downscale_factor,
            reference_temporal_scale_factor, vae_tiling, device, overwrite,
        )
        logger.info(f"Done (fixed context, no captions). Set data.preprocessed_data_root to: {pre}")
        return
    if not text_encoder_path:
        raise typer.BadParameter("--text-encoder-path is required unless --fixed-context is used")
    if len(captions) == 1:
        cap = next(iter(captions))
        logger.info(f'All captions identical ("{cap}") - encoding once and hard-linking')
        one_csv = root / ONE_CAPTION_NAME
        with open(one_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["caption", "video"])
            w.writeheader()
            w.writerow({"caption": cap, "video": rows[0]["video"]})
        with free_gpu_memory_context():
            compute_captions_embeddings(
                dataset_file=str(one_csv),
                output_dir=str(cond_dir),
                model_path=model_path,
                text_encoder_path=text_encoder_path,
                caption_column="caption",
                media_column="video",
                batch_size=1,
                device=device,
                overwrite=overwrite,
            )
        if not master.is_file():
            raise RuntimeError(f"Caption embedding was not created: {master}")
        linked = copied = 0
        for r in rows[1:]:
            dst = cond_dir / Path(r["video"]).with_suffix(".pt")
            if dst.exists():
                if not overwrite:
                    continue
                dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(master, dst)
                linked += 1
            except OSError:
                shutil.copy2(master, dst)
                copied += 1
        logger.info(f"Caption embeddings: hard-linked {linked:,}, copied {copied:,}")

    # ---------------------------------------------------------------- 3. latents
    vae = resolve_video_vae_path(model_path, video_vae_path)
    buckets = parse_resolution_buckets(resolution_buckets, read_video_scale_factors(vae))
    preprocess_dataset(
        dataset_file=str(out_csv),
        resolution_buckets=buckets,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        device=device,
        video_vae_path=video_vae_path,
        audio_vae_path=audio_vae_path,
        output_dir=str(pre),
        batch_size=1,
        vae_tiling=vae_tiling,
        reference_downscale_factor=reference_downscale_factor,
        reference_temporal_scale_factor=reference_temporal_scale_factor,
        skip_audio=True,
        overwrite=overwrite,
    )
    logger.info(f"Done. Set data.preprocessed_data_root to: {pre}")


def _latents_only(  # noqa: PLR0913
    out_csv: Path,
    pre: Path,
    model_path: str,
    video_vae_path: str,
    resolution_buckets: str,
    reference_downscale_factor: int,
    reference_temporal_scale_factor: int,
    vae_tiling: bool,
    device: str,
    overwrite: bool,
) -> None:
    """Encode target + reference latents only (no text encoder, no audio)."""
    from ltx_trainer.process_videos import compute_latents, compute_scaled_resolution_buckets  # noqa: PLC0415

    vae = resolve_video_vae_path(model_path, video_vae_path)
    scale_factors = read_video_scale_factors(vae)
    buckets = parse_resolution_buckets(resolution_buckets, scale_factors)
    if (reference_downscale_factor > 1 or reference_temporal_scale_factor > 1) and len(buckets) > 1:
        raise ValueError("Reference scale factors > 1 need a single resolution bucket")
    with free_gpu_memory_context():
        compute_latents(
            dataset_file=str(out_csv),
            video_column="video",
            resolution_buckets=buckets,
            output_dir=str(pre / "latents"),
            model_path=model_path,
            video_vae_path=vae,
            audio_vae_path=None,
            batch_size=1,
            device=device,
            vae_tiling=vae_tiling,
            with_audio=False,
            audio_output_dir=None,
            overwrite=overwrite,
        )
    with free_gpu_memory_context():
        compute_latents(
            dataset_file=str(out_csv),
            main_media_column="video",
            video_column="reference_video",
            resolution_buckets=compute_scaled_resolution_buckets(buckets, reference_downscale_factor, scale_factors),
            output_dir=str(pre / "reference_latents"),
            model_path=model_path,
            video_vae_path=vae,
            audio_vae_path=None,
            batch_size=1,
            device=device,
            vae_tiling=vae_tiling,
            overwrite=overwrite,
            temporal_subsample_factor=reference_temporal_scale_factor,
        )


if __name__ == "__main__":
    app()
