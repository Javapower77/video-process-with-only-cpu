#!/usr/bin/env bash
# Run in a subshell so sourcing this file cannot replace the terminal shell,
# change its directory, or enable shell-wide exit-on-error settings.
download_models_main() (
    cd "$(dirname "${BASH_SOURCE[0]}")" || return 1
    if [[ ! -x venv/bin/python ]]; then
        echo 'Run bash setup_venv.sh first: venv/bin/python is missing.' >&2
        return 1
    fi

    venv/bin/python -u scripts/download_models.py "$@"
    status=$?
    if [[ "$status" -ne 0 ]]; then
        printf '\nModel download failed (exit code %s). See the error above.\n' "$status" >&2
        echo 'Retry with: bash download_models.sh' >&2
    fi
    return "$status"
)

# A conditional also prevents a caller's `set -e` from closing a sourced shell.
if download_models_main "$@"; then
    :
else
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        exit 1
    fi
    echo 'Returned to the current shell; the terminal has not been replaced.' >&2
fi