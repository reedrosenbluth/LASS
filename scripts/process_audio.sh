#!/usr/bin/env bash
set -euo pipefail
set -m

# Trap SIGINT (Ctrl-C) and kill all children
trap 'echo "Interrupted – killing all subprocesses…"; kill 0; exit 1' SIGINT

# Determine script & project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LASS_ROOT="$( dirname "$SCRIPT_DIR" )"

# Input dirs
CLOTHO_IN="/scratch/$USER/clotho_dataset"
FSD_IN_ROOT="/scratch/$USER/fsd50k_dataset"

# Output under LASS/processed_data_files
BASE_OUT="$LASS_ROOT/processed_data_files"
CLOTHO_OUT="$BASE_OUT/clotho"
FSD_OUT_ROOT="$BASE_OUT/fsd50k"

# Create top-level processed_data_files and subdirs
mkdir -p "$BASE_OUT"
for split in development validation evaluation; do
  mkdir -p "$CLOTHO_OUT/$split"
done
mkdir -p "$FSD_OUT_ROOT/dev_audio" "$FSD_OUT_ROOT/eval_audio"

echo "Resampling → $BASE_OUT …"

# Convert Clotho splits
for split in development validation evaluation; do
  in_dir="$CLOTHO_IN/$split"
  out_dir="$CLOTHO_OUT/$split"
  echo "→ Clotho: $split"
  find "$in_dir" -type f -iname '*.wav' | \
    while read -r src; do
      fname="$(basename "$src")"
      sox -q -G "$src" -r 16000 -c 1 "$out_dir/$fname" gain -n -3 &
    done
done

# Convert FSD50K dev/eval
for tag in dev eval; do
  in_dir="$FSD_IN_ROOT/FSD50K.${tag}_audio"
  out_dir="$FSD_OUT_ROOT/${tag}_audio"
  echo "→ FSD50K: $tag"
  find "$in_dir" -type f -iname '*.wav' | \
    while read -r src; do
      fname="$(basename "$src")"
      sox -q -G "$src" -r 16000 -c 1 "$out_dir/$fname" gain -n -3 &
    done
done

# Wait for all background jobs
wait

echo "Done! Processed audio in:"
echo "  $CLOTHO_OUT/{development,validation,evaluation}"
echo "  $FSD_OUT_ROOT/{dev_audio,eval_audio}"

