from .runtime import run


def encode(source, original, output, count, settings, options, ctx):
    # Optional audio mapping handles silent inputs. AAC is safe across MP4 sources.
    run([settings.ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
         "-framerate", options.fps, "-start_number", "1", "-i", source / "%08d.png",
         "-i", original, "-map", "0:v:0", "-map", "1:a:0?",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-threads", options.workers, "-c:a", "aac", "-b:a", "192k",
         "-t", f"{count / options.fps:.9f}", "-movflags", "+faststart", output], ctx, 0.9)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Encoder produced no video")