import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from .runtime import frames, run, verify_frames
from .config import cpu_count

_upsampler = None


def compatibility():
    # BasicSR 1.4.2 imports a torchvision module removed in newer versions.
    import sys
    from torchvision.transforms import functional
    sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)


def init_upscaler(model, weight, threads):
    global _upsampler
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    compatibility()
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer
    net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                         num_conv=16 if model == "realesr-animevideov3" else 32,
                         upscale=4, act_type="prelu")
    _upsampler = RealESRGANer(scale=4, model_path=weight, model=net, tile=256,
                             tile_pad=10, pre_pad=0, half=False, device=torch.device("cpu"))


def upscale_frame(task):
    import cv2
    source, dest, scale = task
    image = cv2.imread(source)
    if image is None:
        raise RuntimeError(f"Cannot read {source}")
    result, _ = _upsampler.enhance(image, outscale=scale)
    if not cv2.imwrite(dest, result):
        raise RuntimeError(f"Cannot write {dest}")


def parallel_frames(function, initializer, initargs, tasks, workers, ctx, start, span):
    """Bound pending work to limit memory. Cancellation waits for running frames."""
    tasks = list(tasks)
    total = max(1, len(tasks))
    tasks = iter(tasks)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=initializer, initargs=initargs)
    pending = set()
    done_count = 0
    try:
        exhausted = False
        while pending or not exhausted:
            ctx.check()
            while not exhausted and len(pending) < workers * 2:
                try:
                    pending.add(pool.submit(function, next(tasks)))
                except StopIteration:
                    exhausted = True
            ready, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in ready:
                value = future.result()
                done_count += 1
                ctx.report(start + span * done_count / total,
                           f"Processed {done_count}/{total} frames" if value is None else str(value))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def upscale(source, output, settings, options, ctx):
    output.mkdir()
    inputs = frames(source)
    if settings.upscale_backend == "ncnn":
        # Only a separately supplied CPU-capable fork supports this contract.
        model = options.upscale_model
        if model != "realesr-animevideov3":
            raise ValueError("NCNN mode supports the installed anime model only")
        binary = settings.root / "bin" / "realesrgan-ncnn-vulkan"
        run([binary, "-i", source, "-o", output, "-m", settings.weights / "models",
             "-n", model, "-s", options.scale, "-g", "-1", "-j",
             f"2:{options.workers}:2", "-f", "png"], ctx, 0.15)
    else:
        weight = settings.weights / f"{options.upscale_model}.pth"
        threads = max(1, min(8, cpu_count() // options.workers))
        parallel_frames(upscale_frame, init_upscaler,
                        (options.upscale_model, str(weight), threads),
                        ((str(p), str(output / p.name), options.scale) for p in inputs),
                        options.workers, ctx, 0.15, 0.3)
    verify_frames(output, len(inputs))