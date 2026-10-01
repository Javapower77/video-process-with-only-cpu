# CPU Video Studio

Python **3.11**, Ubuntu, and a private Gradio web interface for ×2/×3/×4
Real-ESRGAN upscaling, selective GFPGAN restoration, and RIFE frame interpolation.
Supports single uploads and sequential batches. No CUDA or physical GPU required.

## Important corrections to the original design

The upstream [Real-ESRGAN NCNN source](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/blob/master/src/main.cpp)
rejects negative GPU IDs. Its official binary **does not implement `-g -1` CPU mode**.
Therefore the default is CPU PyTorch Real-ESRGAN. A user-supplied CPU-capable
NCNN fork may be selected explicitly with `UPSCALE_BACKEND=ncnn`; it is never
downloaded or silently substituted with GPU processing.

[RIFE NCNN](https://github.com/nihui/rife-ncnn-vulkan) does support `-g -1`.
Its `-n` argument is the **total target frame count**, not the multiplication
factor. The application computes `round(input_frames × target_fps / source_fps)`.
Only v4 models support this arbitrary time resampling. The official 20221029
Ubuntu bundle supplies the baseline `rife-v4`; newer `rife-v4.*-lite` models
are not included or claimed compatible. Install them manually only after
confirming compatibility with your binary.

NCNN models require `.param` **and** `.bin` files; `.pth` checkpoints cannot
be passed to an NCNN executable. The original 15–35 minute estimates are not
benchmarks and should not be treated as performance guarantees.

## Installation

1. Install Python **3.11**, its `venv` support, FFmpeg/FFprobe, and the Ubuntu
   `libglib2.0-0` and `libvulkan1` packages using your system administrator.
   Python 3.11 package availability depends on the Ubuntu release; use a
   trusted Python distribution if the release does not provide it. RIFE's
   binary links the Vulkan loader even when inference runs on CPU.
2. Run `bash setup_venv.sh`. This creates `venv`, installs CPU-only
   Torch 2.5.1/Torchvision 0.20.1, and copies `.env.example` to `.env` if absent.
3. Run `bash download_models.sh`. Downloads official Real-ESRGAN/GFPGAN,
   OpenCV detection weights, facexlib detection/parsing weights, and the
   approximately **431 MB** RIFE archive. Downloads are reusable and atomic.
   `bash download_models.sh --skip-rife` skips that bundle for equal-FPS jobs.
4. Run `bash run.sh` and open **http://127.0.0.1:7860**.

The scripts deliberately do not invoke `sudo` or modify system packages.
Python 3.12 is not an accepted substitute. Use `venv/bin/python -m pytest`
for tests and `venv/bin/python -m ruff check .` for lint validation.

## Usage

Upload one or more videos, choose the content model, scale, FPS, and worker
budget, and select **Enhance videos**. Each completed MP4 appears in the
download panel. **Stop** cancels only the requesting session's active job.
Completed videos in a partially failed batch remain downloadable; failures
are displayed in the log.

* **Anime video v3**: compact 16-convolution upscaler; best suited to animation.
* **General x4v3**: compact 32-convolution upscaler; better for general imagery.
* **Face restoration**: OpenCV detects faces first. Frames without confident
  faces are copied unchanged; matching frames use GFPGAN v1.4 on CPU.
* **Workers**: number of spawned PyTorch processes or NCNN inference threads.
  Start with 4 on a 32-core host; each process maintains its own model copy.
  PyTorch inference threads per process are bounded by CPU capacity and 8.
  For a supplied NCNN upscaler, ensure loader/saver plus inference thread
  budgets fit the host. CPU `-j` inference values are threads, not GPU queues.
* **Keep temporary frames**: disabled by default. Keeping frames can require
  many gigabytes, particularly with upscaled 60 FPS footage.

## Configuration

`.env` contains `HOST`, `PORT`, `CPU_WORKERS`, `UPSCALE_BACKEND`, `FFMPEG`, and
`FFPROBE`. Environment variables take precedence. Output and asset locations
are under the project directory. Never commit local secrets or model files.

Default `HOST=127.0.0.1`, no public sharing, and a single active enhancement
job protect CPU resources. Binding to `0.0.0.0` exposes an unauthenticated UI:
use an authenticated HTTPS reverse proxy and upload limits before any remote
deployment. Gradio's queue accepts up to eight requests. This is a local
application, not a hardened multi-tenant hosted service.

### Optional CPU-capable Real-ESRGAN NCNN fork

Place its executable at `bin/realesrgan-ncnn-vulkan` and anime model files at
`weights/models/realesr-animevideov3-x{2,3,4}.{param,bin}`. Set
`UPSCALE_BACKEND=ncnn`. It must explicitly support `-g -1`, `-j`, directory
input/output, and PNG format. Official Vulkan-only binaries will fail clearly;
there is no automatic GPU fallback. Only the anime model is offered through
this backend. Do not install untrusted executable binaries.

## Pipeline and operational behavior

`Extract → Upscale → Selective face restoration → RIFE → Encode`

* FFmpeg normalizes variable-frame-rate inputs to their average frame rate.
  Lossless PNG is used instead of repeated lossy JPEG intermediates.
* Upscaling and restoration use bounded `ProcessPoolExecutor` queues with
  the `spawn` context. CUDA is hidden and all model devices are explicitly CPU.
* Every output frame sequence must be contiguous, correctly named, and nonempty.
* RIFE can upsample or downsample FPS. Single-frame clips are repeated safely.
  Output duration differs from the normalized source by at most half an
  output frame due to count rounding; the final RIFE interval holds the last frame.
* FFmpeg encodes H.264, `veryfast`, CRF 18, YUV420P, and AAC audio. It preserves
  the **first original audio track**, transcoding rather than copying its codec.
  Silent videos work. Odd image dimensions receive at most one padding pixel.
  Subtitles, chapters, additional audio tracks, HDR, and original metadata are
  not preserved. Audio is clipped to the calculated video duration.
* Temporary job directories are UUID-isolated and removed on success, failure,
  and cancellation unless explicitly retained. Partial MP4s are deleted.
  Completed `outputs/` files are never auto-deleted: remove them manually when
  no longer needed. A process crash may leave temporary directories behind.
* Cancellation kills FFmpeg/NCNN process groups. Running PyTorch frames finish
  before their workers exit; pending work is cancelled. Cancellation may take
  minutes on a slow frame, and must finish before temporary data can be removed.

## Troubleshooting

Each processing job writes a persistent `logs/<job-id>.log` with subprocess
commands and errors. The UI polls backend status every two seconds rather than
holding a long-lived streaming request; interrupted UI updates do not cancel
processing. One job runs at a time, and completed videos can be recovered using
**Refresh completed videos**.
The logs include subprocess
commands, output, exit codes, and failure tracebacks. These survive temporary
frame cleanup. RIFE reports completed frames and an activity update every ten
seconds; a slow CPU job is not necessarily a failed job. If no MP4 appears,
check the last stage and error in this log. Enable **Keep temporary frames**
before a diagnostic retry if frame inspection is needed.

* **Missing Python 3.11 / FFmpeg**: install the prerequisites and retry setup.
* **Missing assets**: rerun the downloader; confirm nonzero file sizes.
* **RIFE loader errors**: inspect the binary with `ldd`; install the Vulkan
  loader and required runtime libraries. Official Ubuntu assets target x86-64;
  other architectures must build a compatible RIFE binary from source.
* **Unknown/new lite model**: use the bundled baseline model, or match a
  verified binary/model pair; newer versions are not automatically interchangeable.
* **BasicSR import failure**: the application provides a narrow compatibility
  alias for torchvision's removed `functional_tensor`; use the pinned CPU
  Torch/Torchvision pair and NumPy <2.
* **`libGL.so.1` missing**: upstream dependencies install GUI OpenCV by name.
  The setup script replaces overlapping OpenCV wheels with the headless build;
  rerun setup instead of installing both wheels together.
* **Slow inference / memory pressure**: lower workers, disable restoration,
  and test a short low-resolution clip. Worker-count speedup is not linear.
* **Disk full**: disable temporary retention and free `temp/` and old outputs.

## Project layout

`app.py` is the UI; `pipeline/` contains independent processing stages and
runtime/configuration helpers; `scripts/download_models.py` handles assets;
`tests/` contains deterministic unit tests and optional real FFmpeg smoke tests.
`setup_venv.sh`, `download_models.sh`, and `run.sh` cover setup and execution.
See `Changelog.md` and `docs/VALIDATION.md` for implementation/validation status.

## Trust and licenses

Download only trusted checkpoints: PyTorch `.pth` loading can execute code.
SHA256 values printed by the downloader are informational, not verified against
publisher checksums. Archives are checked for path traversal and symlinks.
Review upstream model/software licenses (Real-ESRGAN, GFPGAN, facexlib,
RIFE NCNN, OpenCV, Torch, FFmpeg, Gradio) before redistribution or commercial use.
No model weights or third-party source are committed here.