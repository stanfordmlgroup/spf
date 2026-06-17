#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_DIR="${ENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
INSTALL_DEV="${INSTALL_DEV:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"

export UV_CACHE_DIR

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is not installed on PATH.

Install uv without modifying the sai-v4 conda environment, then rerun this script.
Examples:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or, if your site provides a module:
  module load uv

You can also place a uv binary on PATH manually.
EOF
  exit 127
fi

sync_args=()
if [[ "$INSTALL_DEV" == "1" ]]; then
  sync_args+=(--extra dev)
fi

echo "Creating uv environment at ${ENV_DIR} with Python ${PYTHON_VERSION}"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --python "$PYTHON_VERSION" --link-mode=copy "${sync_args[@]}"

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  echo "Installing flash-attn without build isolation after torch is available"
  UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv pip install --no-build-isolation "flash-attn==2.7.2.post1"
fi

echo "Running import smoke test"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv run python - <<'PY'
import torch
import hydra
import pytorch_lightning
import xarray
import zarr

from emulator.data.loading.datamodule import ClimateDataModule
from emulator.data.datasets import ClimateModelDataset, TimescaleAlignedClimateModelDataset, MultiClimateModelDataset
from emulator.models.spatiotemporal_pyramid_flow_model import SpatiotemporalPyramidFlow
from emulator.models.single_timescale_spatiotemporal_pyramid_flow import SingleTimescaleSpatiotemporalPyramidFlow
from emulator.models.pyramidal_flow_model import PyramidalFlowModel

print("python ok")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

echo "uv environment is ready: ${ENV_DIR}"
