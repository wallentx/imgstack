#!/usr/bin/env sh
# Build a one-file binary using uv-managed deps.
# Env knobs (can also be set via CLI flags below):
#   USE_SYSTEM_OPENCV=1   reuse system cv2 via system site-packages
#                         (default on Termux; disabled elsewhere)
#   SKIP_SYNC=1           skip uv sync (faster when deps already synced)
#   VENV_DIR=path         virtualenv location (default: .venv)
set -eu

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options (override env vars):
  -s, --use-system-opencv    reuse system cv2 via system site-packages (default on Termux)
  -n, --no-system-opencv     install opencv-python into the venv (sets USE_SYSTEM_OPENCV=0)
  -k, --skip-sync            skip uv sync (sets SKIP_SYNC=1)
  -r, --sync                 force uv sync even if SKIP_SYNC=1
  -v, --venv-dir PATH        virtualenv location (default: .venv)
  -h, --help                 show this help
EOF
}

ROOT="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
USE_SYSTEM_OPENCV="${USE_SYSTEM_OPENCV:-auto}"
SKIP_SYNC="${SKIP_SYNC:-0}"

IS_TERMUX=0
if python -c 'import sys; raise SystemExit(not hasattr(sys, "getandroidapilevel"))' 2>/dev/null; then
  IS_TERMUX=1
fi

if [ "$USE_SYSTEM_OPENCV" = "auto" ]; then
  USE_SYSTEM_OPENCV="$IS_TERMUX"
fi

while [ $# -gt 0 ]; do
  case "$1" in
    -s|--use-system-opencv|--system-cv2|--system-opencv)
      USE_SYSTEM_OPENCV=1
      shift
      ;;
    -n|--no-system-opencv)
      USE_SYSTEM_OPENCV=0
      shift
      ;;
    -k|--skip-sync)
      SKIP_SYNC=1
      shift
      ;;
    -r|--sync)
      SKIP_SYNC=0
      shift
      ;;
    -v|--venv-dir)
      if [ $# -lt 2 ]; then
        echo "Missing value for --venv-dir" >&2
        usage
        exit 1
      fi
      VENV_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

maybe_recreate_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    return 0
  fi
  if [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
    return 0
  fi

  HAS_SYSTEM_PACKAGES=0
  if grep -q "include-system-site-packages = true" "$VENV_DIR/pyvenv.cfg"; then
    HAS_SYSTEM_PACKAGES=1
  fi
  if [ "$HAS_SYSTEM_PACKAGES" != "$USE_SYSTEM_OPENCV" ]; then
    return 0
  fi
  return 1
}

if maybe_recreate_venv; then
  rm -rf "$VENV_DIR"
  VENV_FLAGS=""
  [ "$USE_SYSTEM_OPENCV" = "1" ] && VENV_FLAGS="$VENV_FLAGS --system-site-packages"
  uv venv $VENV_FLAGS "$VENV_DIR"
fi

# Make uv sync/run use the environment selected by --venv-dir.
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

SYNC_ARGS="--group build"
if [ "$USE_SYSTEM_OPENCV" != "1" ]; then
  SYNC_ARGS="$SYNC_ARGS --extra opencv"
fi
if [ "$IS_TERMUX" = "1" ] && [ "$USE_SYSTEM_OPENCV" = "1" ]; then
  # PyPI does not publish Android wheels for these native packages. They are
  # supplied by Termux and visible through --system-site-packages.
  SYNC_ARGS="$SYNC_ARGS --no-install-package numpy --no-install-package pillow"
fi

if [ "${SKIP_SYNC:-0}" != "1" ]; then
  uv sync $SYNC_ARGS
fi

PYINSTALLER_ARGS="--clean --onefile --name imgstack --additional-hooks-dir pyinstaller_hooks --add-data models:models"
if [ "$USE_SYSTEM_OPENCV" = "1" ]; then
  PYINSTALLER_ARGS="$PYINSTALLER_ARGS --runtime-hook pyinstaller_hooks/system_site.py"
fi

uv run --no-sync --group build pyinstaller $PYINSTALLER_ARGS imgstack.py

echo "Binary written to dist/imgstack"
