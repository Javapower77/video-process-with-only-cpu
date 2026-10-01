#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="$HOME/.local/bin:$PATH"
# Resolve user-local symlinks so relocated Python builds record the correct
# standard-library location in pyvenv.cfg.
PYTHON311="$(python3.11 -c 'import os, sys; print(os.path.realpath(sys.executable))')"
if ! command -v "$PYTHON311" >/dev/null; then
    echo 'Python 3.11 is required. Install Python 3.11 and its venv package, then retry.' >&2
    exit 1
fi
for tool in ffmpeg ffprobe; do
    if ! command -v "$tool" >/dev/null; then
        echo "Missing $tool. Install the Ubuntu ffmpeg package, then retry." >&2
        # exit 1
    fi
done
"$PYTHON311" -m venv venv
venv/bin/python -m pip install --upgrade pip wheel setuptools
venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
# Upstream packages require opencv-python by name; replace its overlapping cv2
# files with headless wheels after dependency resolution on server installations.
venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless
venv/bin/python -m pip install --no-deps opencv-python-headless==4.11.0.86
if [[ ! -f .env ]]; then cp .env.example .env; fi
mkdir -p bin weights temp outputs
echo 'Environment ready. Next: bash download_models.sh, then bash run.sh.'