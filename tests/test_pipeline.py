from fractions import Fraction
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import pytest

from pipeline.config import Options, Settings, cpu_count
from pipeline import encoder, extractor, interpolator, engine
from pipeline.runtime import Cancelled, Context, run, verify_frames
from scripts.download_models import extract_safe


def sequence(directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        (directory / f"{i:08d}.png").write_bytes(b"frame")


@pytest.mark.parametrize("count,fps,target,expected", [
    (240, Fraction(30), 60, 480),
    (240, Fraction(30000, 1001), 60, 480),
    (1, Fraction(24), 60, 2),
    (60, Fraction(60), 24, 24),
])
def test_target_count(count, fps, target, expected):
    assert interpolator.target_count(count, fps, target) == expected


def test_options_validation():
    Options(workers=1).validate()
    for option in (Options(scale=1), Options(fps=0), Options(workers=cpu_count() + 1),
                   Options(rife_model="../rife-v4"), Options(upscale_model="unknown")):
        with pytest.raises(ValueError):
            option.validate()


def test_sequence_integrity(tmp_path):
    sequence(tmp_path, 2)
    assert len(verify_frames(tmp_path, 2)) == 2
    (tmp_path / "00000002.png").unlink()
    with pytest.raises(RuntimeError, match="Incomplete"):
        verify_frames(tmp_path, 2)


def test_rife_command_is_cpu_and_total_count(tmp_path, monkeypatch):
    source = tmp_path / "source"
    sequence(source, 3)
    calls = []

    def fake_run(command, ctx, fraction, **kwargs):
        calls.append([str(x) for x in command])
        sequence(tmp_path / "output", 6)

    monkeypatch.setattr(interpolator, "run", fake_run)
    output, count = interpolator.interpolate(source, tmp_path / "output",
                                            extractor.VideoInfo(Fraction(30), 16, 16),
                                            Settings(root=tmp_path), Options(workers=1), Context())
    command = calls[0]
    assert command[command.index("-g") + 1] == "-1"
    assert command[command.index("-n") + 1] == "6"
    assert count == 6 and len(list(output.glob("*.png"))) == 6


def test_single_frame_interpolation(tmp_path):
    source = tmp_path / "source"
    sequence(source, 1)
    output, count = interpolator.interpolate(source, tmp_path / "output",
                                            extractor.VideoInfo(Fraction(24), 16, 16),
                                            Settings(root=tmp_path), Options(workers=1), Context())
    assert count == 2 and len(list(output.glob("*.png"))) == 2


def test_equal_fps_skips_rife(tmp_path):
    sequence(tmp_path / "source", 3)
    source, count = interpolator.interpolate(tmp_path / "source", tmp_path / "output",
                                            extractor.VideoInfo(Fraction(60), 16, 16),
                                            Settings(root=tmp_path), Options(workers=1), Context())
    assert source == tmp_path / "source" and count == 3
    assert not (tmp_path / "output").exists()


def test_subprocess_failure():
    with pytest.raises(RuntimeError, match="diagnostic"):
        run([sys.executable, "-c", "import sys; print('diagnostic'); sys.exit(7)"], Context())


def test_subprocess_cancellation():
    ctx = Context()
    timer = threading.Timer(0.2, ctx.cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            run([sys.executable, "-c", "while True: pass"], ctx)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 5


def test_cpu_visibility():
    messages = []
    run([sys.executable, "-c", "import os; print('CPU=' + os.environ['CUDA_VISIBLE_DEVICES'])"],
        Context(lambda fraction, message: messages.append(message)))
    assert "CPU=" in messages


def test_archive_traversal_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_safe(archive, tmp_path / "target")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("keep", [False, True])
def test_failed_job_cleanup(tmp_path, monkeypatch, keep):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    monkeypatch.setattr(engine, "preflight", lambda *args: None)
    monkeypatch.setattr(extractor, "probe", lambda *args: extractor.VideoInfo(Fraction(60), 16, 16))

    def fail(*args):
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(extractor, "extract", fail)
    with pytest.raises(RuntimeError, match="extraction failed"):
        engine.process_video(source, Settings(root=tmp_path),
                             Options(workers=1, keep_temp=keep), Context())
    assert bool(list((tmp_path / "temp").iterdir())) == keep
    assert not list((tmp_path / "outputs").iterdir())
    logs = list((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 1
    assert "extraction failed" in logs[0].read_text()


def test_successful_pipeline_order_and_cleanup(tmp_path, monkeypatch):
    from pipeline import upscaler, face_restorer
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    stages = []
    monkeypatch.setattr(engine, "preflight", lambda *args: None)
    monkeypatch.setattr(extractor, "probe", lambda *args: extractor.VideoInfo(Fraction(60), 16, 16))

    def extract(path, output, *args):
        stages.append("extract")
        sequence(output, 3)
        return 3

    def upscale(source, output, *args):
        stages.append("upscale")
        sequence(output, 3)

    def restore(source, output, *args):
        stages.append("restore")
        sequence(output, 3)

    def encode(source, original, output, *args):
        stages.append("encode")
        verify_frames(source, 3)
        output.write_bytes(b"encoded")

    monkeypatch.setattr(extractor, "extract", extract)
    monkeypatch.setattr(upscaler, "upscale", upscale)
    monkeypatch.setattr(face_restorer, "restore", restore)
    monkeypatch.setattr(encoder, "encode", encode)
    output = engine.process_video(source, Settings(root=tmp_path), Options(workers=1), Context())
    assert stages == ["extract", "upscale", "restore", "encode"]
    assert output.read_bytes() == b"encoded"
    assert not list((tmp_path / "temp").iterdir())


def test_face_free_frame_skips_gfpgan(tmp_path, monkeypatch):
    import numpy as np
    import cv2
    from pipeline import face_restorer

    class Detector:
        def setInput(self, blob):
            pass

        def forward(self):
            return np.zeros((1, 1, 1, 7))

    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    cv2.imwrite(str(source), np.zeros((16, 16, 3), dtype=np.uint8))
    monkeypatch.setattr(face_restorer, "_detector", Detector())
    monkeypatch.setattr(face_restorer, "_restorer", None)
    assert "skipped" in face_restorer.restore_frame((str(source), str(target)))
    assert face_restorer._restorer is None
    assert source.read_bytes() == target.read_bytes()


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_real_ffmpeg_extract_encode(tmp_path, audio):
    source = tmp_path / "source.mp4"
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=32x24:r=24:d=0.5"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)]
    subprocess.run(command, check=True)
    settings = Settings(root=tmp_path)
    info = extractor.probe(source, settings)
    count = extractor.extract(source, tmp_path / "frames", info, settings, Context())
    output = tmp_path / "encoded.mp4"
    encoder.encode(tmp_path / "frames", source, output, count, settings,
                   Options(fps=24, workers=1), Context())
    encoded = extractor.probe(output, settings)
    assert encoded.fps == 24 and (encoded.width, encoded.height) == (32, 24)
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                             "stream=codec_name", "-of", "csv=p=0", str(output)],
                            capture_output=True, text=True, check=True)
    assert ("aac" in result.stdout) == audio