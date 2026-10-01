from dataclasses import dataclass
from fractions import Fraction
import json
import math
import subprocess

from .runtime import frames, run


@dataclass(frozen=True)
class VideoInfo:
    fps: Fraction
    width: int
    height: int


def probe(path, settings):
    result = subprocess.run([settings.ffprobe, "-v", "error", "-select_streams", "v:0",
                             "-show_streams", "-of", "json", str(path)],
                            capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("Input has no video stream")
    stream = streams[0]
    try:
        fps = Fraction(stream.get("avg_frame_rate", "0/1"))
        if fps <= 0:
            fps = Fraction(stream["r_frame_rate"])
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        raise ValueError("Cannot determine source frame rate") from exc
    if fps <= 0 or not math.isfinite(float(fps)):
        raise ValueError("Invalid source frame rate")
    return VideoInfo(fps, int(stream["width"]), int(stream["height"]))


def extract(path, output, info, settings, ctx):
    output.mkdir()
    # Normalize VFR sources to a uniform time base before interpolation.
    run([settings.ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", path,
         "-map", "0:v:0", "-vf", f"fps={info.fps}", "-start_number", "1",
         output / "%08d.png"], ctx, 0.05)
    return len(frames(output))