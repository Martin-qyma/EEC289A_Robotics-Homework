#!/usr/bin/env bash
# Bundle the artifacts the assignment requires into ./submission/.
#
# Usage (from the repo root, after a Colab training + eval run):
#   scripts/prepare_submission.sh artifacts/run_baseline
#
# Argument: directory containing best_checkpoint/ from train.py.
# Assumes generate_public_rollout.py + public_eval.py have already produced
# artifacts/public_eval_bundle/ and that test_policy.py has produced
# artifacts/demo_bundle/. custom_eval.py output (if any) is bundled too.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run-dir>" >&2
  echo "  e.g. $0 artifacts/run_baseline" >&2
  exit 2
fi

RUN_DIR="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="submission"
mkdir -p "$OUT"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "[error] missing required input: $1" >&2
    return 1
  fi
}

# Required files per configs/course_config.json:submission.required_files
require_file "$RUN_DIR/best_checkpoint"
require_file "configs/colab_runtime_config.json"
require_file "artifacts/public_eval_bundle/public_eval.json"
require_file "artifacts/demo_bundle/demo.mp4"
require_file "$OUT/short_report.pdf"

# best_checkpoint/
rm -rf "$OUT/best_checkpoint"
cp -r "$RUN_DIR/best_checkpoint" "$OUT/best_checkpoint"

# configs/colab_runtime_config.json
mkdir -p "$OUT/configs"
cp configs/colab_runtime_config.json "$OUT/configs/colab_runtime_config.json"

# public_eval_bundle/public_eval.json (and the rollout npz + first-episode video for context)
mkdir -p "$OUT/public_eval_bundle"
cp artifacts/public_eval_bundle/public_eval.json "$OUT/public_eval_bundle/"
[[ -e artifacts/public_eval_bundle/rollout_public_eval.npz ]] && \
  cp artifacts/public_eval_bundle/rollout_public_eval.npz "$OUT/public_eval_bundle/"
[[ -e artifacts/public_eval_bundle/public_eval_episode0.mp4 ]] && \
  cp artifacts/public_eval_bundle/public_eval_episode0.mp4 "$OUT/public_eval_bundle/"
[[ -e artifacts/public_eval_bundle/rollout_summary.json ]] && \
  cp artifacts/public_eval_bundle/rollout_summary.json "$OUT/public_eval_bundle/"

# demo_bundle/demo.mp4 (and the per-step npz that drove it)
mkdir -p "$OUT/demo_bundle"
cp artifacts/demo_bundle/demo.mp4 "$OUT/demo_bundle/"
[[ -e artifacts/demo_bundle/rollout_public_eval.npz ]] && \
  cp artifacts/demo_bundle/rollout_public_eval.npz "$OUT/demo_bundle/"

# Optional: custom_eval outputs (bonus)
if [[ -d artifacts/custom_eval ]]; then
  mkdir -p "$OUT/custom_eval"
  cp -r artifacts/custom_eval/. "$OUT/custom_eval/"
fi

# Create a tar.gz for upload
TAR="$OUT.tar.gz"
tar -czf "$TAR" "$OUT"

echo
echo "Submission bundle ready:"
echo "  directory: $OUT/"
echo "  archive:   $TAR"
echo
echo "Contents:"
find "$OUT" -maxdepth 3 -mindepth 1 | sort
