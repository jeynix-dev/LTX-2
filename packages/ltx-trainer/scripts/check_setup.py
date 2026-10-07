"""Quick environment / data sanity check for the Jeynix LTX-2.5 IC-LoRA setup."""
import sys
from pathlib import Path

print("python", sys.version)
import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free {free/2**30:.1f} / {total/2**30:.1f} GiB")
import torchaudio  # noqa: E402,F401
import torchvision  # noqa: E402

print("torchaudio", torchaudio.__version__, "torchvision", torchvision.__version__)
import ltx_trainer.process_videos  # noqa: E402,F401
from ltx_trainer.timestep_samplers import DiscreteSigmaTimestepSampler  # noqa: E402

s = DiscreteSigmaTimestepSampler()
x = s.sample(80000)
vals, counts = torch.unique(x, return_counts=True)
print("distilled sampler:", {round(v.item(), 6): c.item() for v, c in zip(vals, counts, strict=True)})

from ltx_trainer.model_loader import is_split_transformer  # noqa: E402

m = Path("G:/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors")
print("split transformer:", is_split_transformer(m))

from ltx_trainer.video_utils import read_video  # noqa: E402

for p in ["E:/PROJECT_UPSKALE/Dataset/video/Tenet_scene2191_000.mp4",
          "E:/PROJECT_UPSKALE/Dataset/control_video/Tenet_scene2191_000.mp4"]:
    v, fps = read_video(p, max_frames=81)
    print("read", p, tuple(v.shape), round(fps, 3))
print("CHECK_SETUP OK")
