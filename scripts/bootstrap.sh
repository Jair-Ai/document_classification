#!/usr/bin/env bash
#
# bootstrap.sh — make a clean clone fully reproducible in one command.
#
# Trains the full-accuracy model and regenerates the evaluation reports. The
# dataset and trained model are intentionally NOT committed (large /
# proprietary), so this script obtains the data, then trains and evaluates.
#
# It gets the dataset from the first source that applies:
#   1. DATA_SRC  — a local folder or .zip you already have (no download)
#   2. an existing data/trellis_assessment_ds (unless FORCE=1)
#   3. DATA_URL  — download from Dropbox (the case-study link, by default)
#
# For an out-of-the-box smoke model with no dataset at all, use the committed
# sample corpus instead: `make sample-model`.
#
# Usage:
#   scripts/bootstrap.sh                              # auto: existing data, else download
#   DATA_SRC=~/Downloads/trellis_assessment_ds  scripts/bootstrap.sh   # use a local folder
#   DATA_SRC=~/Downloads/trellis.zip            scripts/bootstrap.sh   # use a local zip
#   DATA_URL="https://..."                      scripts/bootstrap.sh   # custom download URL
#   FORCE=1                                      scripts/bootstrap.sh   # re-acquire even if present
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Dropbox folder link from the case study. Appending dl=1 yields a zip.
DATA_URL="${DATA_URL:-https://www.dropbox.com/scl/fo/bsx6t0y86eicr15xm2haa/AJvvER3VtuXJ090Bcvnh1mI?rlkey=mf7s184ymqlw7pdz64n1eymc0&st=z99aunov&dl=1}"
DEST="data/trellis_assessment_ds"
# Labels we expect under the dataset root, used to locate the correct
# directory inside a downloaded/extracted archive or a supplied folder.
EXPECT_DIRS=("business" "sport" "other")

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[bootstrap] %s\033[0m\n' "$*" >&2; exit 1; }

has_dataset() {
  [[ -d "$DEST" ]] && [[ -n "$(find "$DEST" -mindepth 2 -name '*.txt' -print -quit 2>/dev/null)" ]]
}

# Find the directory under $1 that contains the expected category folders,
# then move it into place at $DEST.
install_from_tree() {
  local search="$1" root=""
  while IFS= read -r candidate; do
    local ok=1
    for d in "${EXPECT_DIRS[@]}"; do
      [[ -d "$candidate/$d" ]] || { ok=0; break; }
    done
    if [[ "$ok" == 1 ]]; then root="$candidate"; break; fi
  done < <(find "$search" -type d)

  [[ -n "$root" ]] || die "Could not find category folders (${EXPECT_DIRS[*]}) under $search."

  mkdir -p "$(dirname "$DEST")"
  rm -rf "$DEST"
  cp -R "$root" "$DEST"
  log "Dataset ready at $DEST"
}

import_local() {
  local src="$1"
  [[ -e "$src" ]] || die "DATA_SRC not found: $src"
  if [[ -d "$src" ]]; then
    log "Importing dataset from local folder: $src"
    install_from_tree "$src"
  elif [[ "$src" == *.zip ]]; then
    command -v unzip >/dev/null || die "unzip is required to use a .zip DATA_SRC"
    local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
    log "Extracting local zip: $src"
    unzip -q "$src" -d "$tmp/extracted"
    install_from_tree "$tmp/extracted"
  else
    die "DATA_SRC must be a directory or a .zip file: $src"
  fi
}

download_dataset() {
  command -v curl >/dev/null || die "curl is required to download the dataset"
  command -v unzip >/dev/null || die "unzip is required to extract the dataset"
  local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  log "Downloading dataset archive..."
  curl -L --fail -o "$tmp/dataset.zip" "$DATA_URL"
  log "Extracting..."
  unzip -q "$tmp/dataset.zip" -d "$tmp/extracted"
  install_from_tree "$tmp/extracted"
}

# --- acquire the dataset -------------------------------------------------
if [[ -n "${DATA_SRC:-}" ]]; then
  import_local "$DATA_SRC"
elif has_dataset && [[ "${FORCE:-0}" != "1" ]]; then
  log "Dataset already present at $DEST (set FORCE=1 to re-acquire); skipping."
else
  download_dataset
fi

# --- train + evaluate ----------------------------------------------------
log "Installing dependencies..."
uv sync --dev

log "Training model on the full dataset..."
uv run python -m src.train \
  --data-dir "$DEST" \
  --output models/document_classifier.joblib \
  --report-dir reports

log "Evaluating (with leave-one-class-out OOD probe)..."
uv run python -m src.evaluate \
  --data-dir "$DEST" \
  --model-path models/document_classifier.joblib \
  --report-dir reports \
  --loco

log "Done. Start the API with: make serve"
