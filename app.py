"""Gradio entry point. Run using Python 3.11 via run.sh."""
import os
import sys
import threading
from collections import deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pipeline.config import Options, Settings, cpu_count
from pipeline.engine import process_video
from pipeline.runtime import Context

_active = {}
_lock = threading.Lock()
_jobs = {}
_processing = threading.Lock()


def build_ui(settings):
    import gradio as gr

    def completed_outputs():
        return [str(path.resolve()) for path in sorted(
            (settings.root / "outputs").glob("enhanced-*.mp4"),
            key=lambda path: path.stat().st_mtime, reverse=True) if path.stat().st_size > 0]

    def enhance(files, model, rife, scale, fps, workers, faces, keep, request: gr.Request):
        if not files:
            raise gr.Error("Upload at least one video")
        options = Options(model, rife, int(scale), int(fps), int(workers), faces, keep)
        options.validate()
        if not _processing.acquire(blocking=False):
            raise gr.Error("An enhancement job is already running. Wait for it to finish or stop it.")
        job = {"logs": deque(maxlen=150), "results": [], "status": "Starting…",
               "running": True, "delivered": 0}

        def emit(fraction, text):
            with _lock:
                job["logs"].append(text)
                job["status"] = f"{fraction:.0%} — {text[:160]}"

        ctx = Context(emit)
        with _lock:
            _active[request.session_hash] = ctx
            _jobs[request.session_hash] = job

        def worker():
            try:
                for index, file in enumerate(files, 1):
                    ctx.report(0, f"Video {index}/{len(files)}")
                    output = Path(process_video(file, settings, options, ctx)).resolve()
                    if not output.is_file() or output.stat().st_size == 0:
                        raise RuntimeError(f"Completed video is missing or empty: {output}")
                    with _lock:
                        job["results"].append(str(output))
                        job["logs"].append(f"Ready to download: {output.name}")
                with _lock:
                    job["status"] = "Complete — videos are ready to download."
                    job["logs"].append("Batch complete")
            except Exception as exc:
                with _lock:
                    job["status"] = "Stopped — see the log for details."
                    job["logs"].append(str(exc))
            finally:
                with _lock:
                    job["running"] = False
                    _active.pop(request.session_hash, None)
                _processing.release()

        thread = threading.Thread(target=worker, daemon=True)
        try:
            thread.start()
        except BaseException:
            with _lock:
                _active.pop(request.session_hash, None)
                _jobs.pop(request.session_hash, None)
            _processing.release()
            raise
        return gr.skip(), "Job started. Status refreshes every two seconds.", "Starting…"

    def poll(request: gr.Request):
        with _lock:
            job = _jobs.get(request.session_hash)
            if job is None:
                return gr.skip(), gr.skip(), gr.skip()
            # File postprocessing is needed only when new outputs are available.
            downloads = gr.skip()
            if len(job["results"]) != job["delivered"]:
                downloads = job["results"][:]
                job["delivered"] = len(downloads)
            return downloads, "\n".join(job["logs"]), job["status"]

    def cancel(request: gr.Request):
        with _lock:
            ctx = _active.get(request.session_hash)
        if ctx:
            ctx.cancel.set()
            return "Cancellation requested; waiting for running CPU frames to finish."
        return "No active job for this session."

    installed = sorted(p.name for p in settings.weights.glob("rife-v4*") if p.is_dir())
    rife_models = installed or ["rife-v4"]
    with gr.Blocks(title="CPU Video Studio") as ui:
        gr.Markdown("# CPU Video Studio\nUpscale • Restore faces • Smooth motion\n\n"
                    "Private, CPU-only processing. Outputs remain available after temporary frames are removed.")
        with gr.Row():
            with gr.Column(scale=2):
                upload = gr.File(label="Videos — single or batch", file_count="multiple",
                                 file_types=["video"], type="filepath")
                model = gr.Dropdown(["realesr-animevideov3", "realesr-general-x4v3"],
                                    value="realesr-animevideov3", label="Upscaling model")
                rife = gr.Dropdown(rife_models, value=rife_models[0], label="Installed RIFE model")
                with gr.Row():
                    scale = gr.Radio([2, 3, 4], value=2, label="Scale factor")
                    fps = gr.Slider(1, 120, value=60, step=1, label="Target FPS")
                workers = gr.Slider(1, cpu_count(), value=settings.workers, step=1,
                                    label="CPU workers / NCNN inference threads")
                faces = gr.Checkbox(value=True, label="Selective GFPGAN face restoration")
                keep = gr.Checkbox(value=False, label="Keep temporary frames (uses substantial disk space)")
                with gr.Row():
                    start = gr.Button("Enhance videos", variant="primary")
                    stop = gr.Button("Stop", variant="stop")
                cancellation = gr.Textbox(label="Cancellation status", interactive=False)
            with gr.Column(scale=3):
                status = gr.Textbox(label="Processing status", value="Idle", interactive=False)
                downloads = gr.File(label="Enhanced MP4 downloads", file_count="multiple", interactive=False)
                refresh = gr.Button("Refresh completed videos")
                logs = gr.Textbox(label="Live processing log", lines=22, interactive=False)
        gr.Markdown(f"Backend: **{settings.upscale_backend}** · CPU capacity: **{cpu_count()}** · "
                    "Batch jobs run sequentially to avoid oversubscription. Audio is preserved as AAC.")
        timer = gr.Timer(2)
        timer.tick(poll, outputs=[downloads, logs, status], queue=False, show_progress="hidden")
        start.click(enhance, [upload, model, rife, scale, fps, workers, faces, keep],
                [downloads, logs, status], concurrency_limit=1, concurrency_id="enhancement",
                show_progress="hidden")
        refresh.click(completed_outputs, outputs=downloads, queue=False, show_progress="hidden")
        ui.load(completed_outputs, outputs=downloads, queue=False, show_progress="hidden")
        stop.click(cancel, outputs=cancellation, queue=False)
    return ui.queue(max_size=8)


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Python 3.11 is required. Run bash setup_venv.sh first.")
    settings = Settings.from_env()
    os.chdir(settings.root)
    build_ui(settings).launch(server_name=settings.host, server_port=settings.port,
                              share=False, allowed_paths=[str(settings.root / "outputs")])