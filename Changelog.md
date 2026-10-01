# Changelog

## Unreleased

- Replaced long-lived UI streaming with two-second status polling and independent job execution.
- Added persistent processing logs, RIFE frame-count activity updates, and visible failure notifications.
- Fixed the ParseNet download URL to use the official facexlib v0.2.2 release.

## 0.1.0 — 2026-10-01

- Added modular CPU Real-ESRGAN, selective GFPGAN, CPU NCNN RIFE, and FFmpeg pipeline.
- Added batch Gradio UI, live logs, progress, downloads, and per-session cancellation.
- Added bounded spawned worker pools, frame verification, and isolated job cleanup.
- Added Python 3.11 venv setup, CPU dependency pins, `.env` configuration, and official asset downloads.
- Added automated unit tests and optional FFmpeg smoke validation.
- Corrected upstream Real-ESRGAN NCNN CPU-mode assumption and RIFE frame-count semantics.
- Documented baseline RIFE model compatibility, audio behavior, and performance limitations.