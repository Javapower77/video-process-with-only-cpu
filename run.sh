#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x venv/bin/python ]]; then
    echo 'Run bash setup_venv.sh first.' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec venv/bin/python app.py