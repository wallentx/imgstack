#!/usr/bin/env sh
# Build a one-file binary using uv-managed deps. Set USE_SYSTEM_OPENCV=1 to re-use system cv2 via system site-packages.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
USE_SYSTEM_OPENCV="${USE_SYSTEM_OPENCV:-0}"

maybe_recreate_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    return 0
  fi
  if [ "$USE_SYSTEM_OPENCV" != "1" ]; then
    return 1
  fi
  if [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
    return 0
  fi
  if ! grep -q "include-system-site-packages = true" "$VENV_DIR/pyvenv.cfg"; then
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

SYNC_ARGS="--group build"
if [ "$USE_SYSTEM_OPENCV" != "1" ]; then
  SYNC_ARGS="$SYNC_ARGS --extra opencv"
fi

uv sync $SYNC_ARGS

PYINSTALLER_ARGS="--onefile --name imgstack --collect-all cv2"
if [ "$USE_SYSTEM_OPENCV" = "1" ]; then
  PYINSTALLER_ARGS="$PYINSTALLER_ARGS --runtime-hook pyinstaller_hooks/system_site.py"
fi

uv run --group build pyinstaller $PYINSTALLER_ARGS imgstack.py

echo "Binary written to dist/imgstack"
