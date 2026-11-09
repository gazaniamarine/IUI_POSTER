#!/usr/bin/env bash

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate habitat

cd "$REPO"
echo "[nav] starting"
python -m nav.gui "$@"
