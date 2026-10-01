import shutil
from .runtime import frames, run, verify_frames


def target_count(count, source_fps, target_fps):
    return max(1, round(count * target_fps / source_fps))


def interpolate(source, output, info, settings, options, ctx):
    inputs = frames(source)
    count = target_count(len(inputs), info.fps, options.fps)
    if options.fps == info.fps:
        return source, count
    output.mkdir()
    if len(inputs) == 1:
        for index in range(1, count + 1):
            ctx.check()
            shutil.copy2(inputs[0], output / f"{index:08d}.png")
    else:
        run([settings.root / "bin" / "rife-ncnn-vulkan", "-i", source, "-o", output,
             "-m", settings.weights / options.rife_model, "-g", "-1",
             "-n", count, "-j", f"2:{options.workers}:2", "-f", "%08d.png", "-v"], ctx, 0.7,
            activity=lambda: f"{sum(1 for _ in output.glob('*.png'))}/{count} interpolated frames")
    verify_frames(output, count)
    return output, count