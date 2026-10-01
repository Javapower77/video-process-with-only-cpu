from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cpu_count() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    upscale_backend: str = "pytorch"
    workers: int = 4
    host: str = "127.0.0.1"
    port: int = 7860

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        return cls(ffmpeg=os.getenv("FFMPEG", "ffmpeg"),
                   ffprobe=os.getenv("FFPROBE", "ffprobe"),
                   upscale_backend=os.getenv("UPSCALE_BACKEND", "pytorch"),
                   workers=max(1, min(cpu_count(), int(os.getenv("CPU_WORKERS", "4")))),
                   host=os.getenv("HOST", "127.0.0.1"),
                   port=int(os.getenv("PORT", "7860")))

    @property
    def weights(self):
        return self.root / "weights"


@dataclass(frozen=True)
class Options:
    upscale_model: str = "realesr-animevideov3"
    rife_model: str = "rife-v4"
    scale: int = 2
    fps: int = 60
    workers: int = 4
    restore_faces: bool = True
    keep_temp: bool = False

    def validate(self):
        if self.upscale_model not in {"realesr-animevideov3", "realesr-general-x4v3"}:
            raise ValueError("Unsupported upscale model")
        if not self.rife_model.startswith("rife-v4") or Path(self.rife_model).name != self.rife_model:
            raise ValueError("Use an installed RIFE v4 model directory name")
        if self.scale not in {2, 3, 4} or not 1 <= self.fps <= 120:
            raise ValueError("Scale must be 2–4 and FPS must be 1–120")
        if not 1 <= self.workers <= cpu_count():
            raise ValueError("Workers must fit the available CPU count")