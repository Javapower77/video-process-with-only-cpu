import os
from pathlib import Path
import shutil
import traceback
import uuid

from . import encoder, extractor, face_restorer, interpolator, upscaler


def preflight(settings, options):
    options.validate()
    for tool in (settings.ffmpeg, settings.ffprobe):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Missing executable: {tool}")
    if settings.upscale_backend not in {"pytorch", "ncnn"}:
        raise ValueError("UPSCALE_BACKEND must be pytorch or ncnn")
    required = []
    if settings.upscale_backend == "pytorch":
        required.append(settings.weights / f"{options.upscale_model}.pth")
    else:
        if options.upscale_model != "realesr-animevideov3":
            raise ValueError("NCNN mode supports the installed anime model only")
        required.append(settings.root / "bin" / "realesrgan-ncnn-vulkan")
        for suffix in ("param", "bin"):
            required.append(settings.weights / "models" / f"realesr-animevideov3-x{options.scale}.{suffix}")
    if options.restore_faces:
        required.extend(settings.weights / name for name in (
            "GFPGANv1.4.pth", "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"))
        required.extend(settings.root / "gfpgan" / "weights" / name for name in (
            "detection_Resnet50_Final.pth", "parsing_parsenet.pth"))
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Run bash download_models.sh; missing assets:\n" + "\n".join(missing))
    if settings.upscale_backend == "ncnn" and not os.access(required[0], os.X_OK):
        raise RuntimeError("Real-ESRGAN NCNN binary is not executable")


def process_video(path, settings, options, ctx):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Input video does not exist")
    preflight(settings, options)
    ctx.report(0, f"Inspecting {path.name}")
    info = extractor.probe(path, settings)
    if options.fps != info.fps:
        binary = settings.root / "bin" / "rife-ncnn-vulkan"
        model = settings.weights / options.rife_model
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError("Install executable RIFE using bash download_models.sh")
        if not list(model.glob("*.param")) or not list(model.glob("*.bin")):
            raise RuntimeError(f"Missing NCNN .param/.bin files: {model}")
    job = uuid.uuid4().hex
    work = settings.root / "temp" / job
    work.mkdir(parents=True)
    outputs = settings.root / "outputs"
    outputs.mkdir(exist_ok=True)
    logs = settings.root / "logs"
    logs.mkdir(exist_ok=True)
    previous_log = ctx.log_path
    ctx.log_path = logs / f"{job}.log"
    output = outputs / f"enhanced-{job}.mp4"
    try:
        ctx.report(0, f"Job {job}; diagnostic log: {ctx.log_path}")
        ctx.report(0.05, "Extracting normalized, lossless frames")
        count = extractor.extract(path, work / "extracted", info, settings, ctx)
        ctx.report(0.15, f"Upscaling {count} frames on CPU")
        upscaler.upscale(work / "extracted", work / "upscaled", settings, options, ctx)
        current = work / "upscaled"
        if options.restore_faces:
            ctx.report(0.5, "Detecting faces; restoring only matching frames")
            face_restorer.restore(current, work / "restored", settings, options, ctx)
            current = work / "restored"
        ctx.report(0.7, f"Resampling to {options.fps} FPS using CPU RIFE")
        current, count = interpolator.interpolate(current, work / "interpolated", info, settings, options, ctx)
        ctx.report(0.9, "Encoding video and preserving original audio")
        encoder.encode(current, path, output, count, settings, options, ctx)
        ctx.report(1, f"Complete: {output.name}")
        return output
    except BaseException as exc:
        ctx.log(traceback.format_exc())
        message = f"FAILED: {exc}\nDiagnostic log: {ctx.log_path}"
        ctx.log(message)
        ctx.emit(0, message)
        print(message, flush=True)
        output.unlink(missing_ok=True)
        raise
    finally:
        if not options.keep_temp:
            shutil.rmtree(work, ignore_errors=True)
        ctx.log_path = previous_log