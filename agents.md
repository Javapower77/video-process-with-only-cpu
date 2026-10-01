# AGENTS.md
# Video Enhancement Pipeline – Agent Context & Instructions

**Project Goal**  
Build a complete, production-ready Python application that performs:
1. Video Upscaling (×2) using Real-ESRGAN
2. Face Restoration using GFPGAN
3. Frame Interpolation to 60 FPS using RIFE

Target platform: **Linux Ubuntu**  
Python version: **3.11** (mandatory)  
Interface: **Gradio** web UI  
Hardware target: High-core CPU machines (especially 32 cores + 128 GB RAM) **without GPU**

---

## 1. Core Requirements & Constraints

- Must run efficiently on **CPU-only** systems (no CUDA / no GPU required)
- Prefer **NCNN** implementations for maximum speed on CPU
- Use light models for best performance/quality trade-off
- Support batch processing and progress feedback
- Clean temporary files after processing
- Preserve original audio
- Python 3.11 + virtual environment (`venv`)
- Modern UI controls
- Create documentation of everything and Changelog.md
- Use .env to store environmet variables
- Create bash shell to download models, configure app and run the app.

---

## 2. Recommended Technology Stack

| Component              | Preferred Implementation                  | Fallback                  |
|------------------------|-------------------------------------------|---------------------------|
| Upscaling              | `realesrgan-ncnn-vulkan` (CPU mode)       | Real-ESRGAN (PyTorch)     |
| Face Restoration       | GFPGAN v1.4 (PyTorch)                     | -                         |
| Frame Interpolation    | `rife-ncnn-vulkan` (CPU mode)             | Practical-RIFE (PyTorch)  |
| Web UI                 | Gradio ≥ 4.x                              | -                         |
| Video I/O              | FFmpeg                                    | -                         |
| Face Detection (skip)  | OpenCV DNN (res10_300x300)                | -                         |

---

## 3. Recommended Light Models (Best for CPU)

### Real-ESRGAN
- `realesr-animevideov3` (best for anime / fast)
- `realesr-general-x4v3` (general content)

Download:
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
- https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

### GFPGAN
- `GFPGANv1.4.pth` (recommended)

Download:
- https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

### RIFE (NCNN)
- `rife-v4.25-lite` (best balance)
- `rife-v4.22-lite`
- `rife-v4.15-lite`

Download:
- https://drive.google.com/file/d/1zlKblGuKNatulJNFf5jdB-emp9AqGK05/view

---

## 4. Performance Expectations (8-second 480p video – 64 cores / 128 GB RAM)

| Pipeline                              | Expected Time     |
|---------------------------------------|-------------------|
| Optimized NCNN + face skipping        | 15–35 minutes     |
| Pure PyTorch (well parallelized)      | 25–50 minutes     |
| Heavy models + no optimizations       | 60+ minutes       |

---

## 5. Critical Speed Optimizations (Must Implement)

1. **Always force CPU mode** in NCNN binaries: `-g -1`
2. Use high thread counts for NCNN: `-j 8:32:8` (tune according to cores)
3. Run face detection **before** GFPGAN and skip frames without faces
4. Use `ProcessPoolExecutor` for GFPGAN (PyTorch part)
5. Prefer NCNN binaries over pure PyTorch whenever possible
6. Process in this order:  
   Extract → Upscale (NCNN) → Face Restore (selective) → Interpolate (NCNN) → Encode
7. Use multi-threaded FFmpeg encoding (`-preset veryfast`)

---

## 6. Project Structure (Recommended)
video-process-with-cpu-only/
├── app.py                     # Gradio interface
├── pipeline/
│   ├── init.py
│   ├── extractor.py
│   ├── upscaler.py            # realesrgan-ncnn wrapper
│   ├── face_restorer.py       # GFPGAN + face detection
│   ├── interpolator.py        # rife-ncnn wrapper
│   └── encoder.py
├── bin/                       # NCNN binaries
│   ├── realesrgan-ncnn-vulkan
│   └── rife-ncnn-vulkan
├── weights/                   # Models
├── temp/                      # Temporary frames (auto-cleaned)
├── requirements.txt
├── setup_venv.sh
└── README.md

---

## 7. Environment Setup (Ubuntu + Python 3.11)

```bash
# Create and activate venv
python3.11 -m venv venv
source venv/bin/activate

# Install system dependencies
sudo apt update
sudo apt install -y ffmpeg libgl1 libglib2.0-0

# Install Python packages
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install gfpgan opencv-python-headless gradio tqdm basicsr facexlib
```

---

## 8. Gradio UI Requirements

The Gradio interface should include:

- Video upload
- Model selection (Upscale model, RIFE model, GFPGAN on/off)
- Scale factor (default ×2)
- Target FPS (default 60)
- Number of CPU workers
- Progress bar + live logs
- Download button for the final video
- Option to keep / delete temporary files

---

## 9. Key Implementation Notes for the Agent

- Always use subprocess to call the NCNN binaries (do not try to reimplement them).
- GFPGAN must run in separate processes (ProcessPoolExecutor) because it is still PyTorch-based.
- Clean up the temp/ directory after successful processing (or provide a toggle).
- Preserve original audio track using FFmpeg.
- Make the pipeline fully interruptible if possible.
- Log every major step clearly so the user can see progress in the Gradio UI.
- Support both single video and future batch mode.

---

## 10. Preferred Processing Pipeline (Final)

Input Video
    ↓
FFmpeg Extract frames (high quality JPEG)
    ↓
realesrgan-ncnn-vulkan  (-g -1 -s 2 -j 8:32:8)
    ↓
Face Detection → GFPGAN only on frames with faces
    ↓
rife-ncnn-vulkan  (-g -1 -n 2 -j 8:32:8)
    ↓
FFmpeg Encode (libx264 + original audio)
    ↓
Output Video

---

# 11. Current Status of Conversation

- Performance estimates for 8s 480p videos on high-core CPU machines have been provided.
- Two complete skeletons were delivered:
    - Pure PyTorch version
    - High-speed NCNN + selective GFPGAN version (recommended)
- All official model download links were provided.
- User requested this AGENTS.md to continue development with an agentic coding system.

---

Agent Instructions

When helping the user:

- Always target Python 3.11 + venv.
- Prioritize the NCNN-based pipeline for speed.
- Keep the code clean, modular, and well-commented.
- Make the Gradio interface user-friendly and informative.
- Include proper error handling and progress reporting.
- Prefer official model sources and NCNN binaries.
- Ask. A 30-second clarifying question is cheaper than a 30-minute revert.
