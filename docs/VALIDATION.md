# Validation — 2026-10-01

## Verified

- Python 3.11.13 installed user-locally with uv; isolated `.venv` created.
- Application dependencies installed; Torch 2.5.1+cpu reports CUDA unavailable.
- Real-ESRGAN/GFPGAN imports pass after the BasicSR compatibility alias.
- Headless OpenCV loads without `libGL`; setup removes overlapping GUI wheels.
- **19 automated tests passed**, including real FFmpeg extraction/encoding
  with and without audio, rational frame-count calculation, CPU subprocess
  options, face-free GFPGAN skipping, cancellation, unsafe archive rejection,
  successful pipeline orchestration, and failure/success cleanup.
- Ruff passes; Python compilation and all Bash syntax checks pass.
- Gradio 5.50.0 UI instantiated with 25 components, served on localhost,
  and inspected in the integrated browser. Stop correctly returns
  “No active job for this session” when idle. Validation server stopped.

## Not yet verified

- End-to-end pretrained model inference, including real face restoration.
- RIFE executable runtime libraries, downloaded baseline model inference,
  and compatibility of optional newer lite models.
- Any user-supplied CPU Real-ESRGAN NCNN fork.
- High-resolution/long-video resource consumption and performance benchmarks.
- Active-job browser cancellation while real PyTorch inference runs.

Model assets have not been downloaded; they are large and are installed
explicitly by `download_models.sh`. FFmpeg tests used a temporary x86-64 static
distribution under `/tmp/cpu-video-tools`, not a system installation.
The application test environment is `.venv`; supported setup/run scripts
create and use `venv`. Install system FFmpeg and run setup/download scripts
before processing real videos. User-local Python 3.11 is now available under
`$HOME/.local/bin`; the setup script includes this location in PATH.