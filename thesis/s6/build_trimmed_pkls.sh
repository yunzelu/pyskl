#!/bin/bash

set -euo pipefail

JSONL_ROOT="${JSONL_ROOT:-data/radar_v4/raw_jsonl/yolo26xpose}"
ORIGIN_ROOT="${ORIGIN_ROOT:-data/radar_v4/origin}"
OUT_DIR="${OUT_DIR:-data/radar_v4/pyskl/8111teacher4_s6}"
CLIP_LEN="${CLIP_LEN:-60}"
LABEL_SOURCE="${LABEL_SOURCE:-origin}"

mkdir -p "${OUT_DIR}"

build_pkl() {
  local fold="$1"
  local teacher="$2"
  local val_subject="$3"
  local calib_subject="$4"
  local pseudo_slug="$5"
  local train_subjects="$6"
  local pseudo_subjects="$7"
  local pkl="radarv4_yolo26xpose_clip${CLIP_LEN}_8111teacher4_s6_${fold}_${teacher}_val_${val_subject}_calib_${calib_subject}_pseudo_${pseudo_slug}"

  echo "[INFO] Building ${pkl}"
  # shellcheck disable=SC2086
  python tools/data/radar_v4/build_pyskl_pkl.py \
    --jsonl-root "${JSONL_ROOT}" \
    --origin-root "${ORIGIN_ROOT}" \
    --output "${OUT_DIR}/${pkl}.pkl" \
    --clip-len "${CLIP_LEN}" \
    --label-source "${LABEL_SOURCE}" \
    --val-subject "${val_subject}" \
    --calibration-subject "${calib_subject}" \
    --train-subjects ${train_subjects} \
    --test-subjects ${pseudo_subjects}
}

# Fold A: validation han, calibration dengdeng, original test chenzhe.
build_pkl fold_a t1 han dengdeng hui_jiadi "li mia rose saad xilai yunze" "hui jiadi"
build_pkl fold_a t2 han dengdeng li_mia "hui jiadi rose saad xilai yunze" "li mia"
build_pkl fold_a t3 han dengdeng rose_saad "hui jiadi li mia xilai yunze" "rose saad"
build_pkl fold_a t4 han dengdeng xilai_yunze "hui jiadi li mia rose saad" "xilai yunze"

# Fold B: validation mia, calibration li, original test jiadi.
build_pkl fold_b t1 mia li chenzhe_dengdeng "han hui rose saad xilai yunze" "chenzhe dengdeng"
build_pkl fold_b t2 mia li han_hui "chenzhe dengdeng rose saad xilai yunze" "han hui"
build_pkl fold_b t3 mia li rose_saad "chenzhe dengdeng han hui xilai yunze" "rose saad"
build_pkl fold_b t4 mia li xilai_yunze "chenzhe dengdeng han hui rose saad" "xilai yunze"

# Fold C: validation yunze, calibration xilai, original test saad.
build_pkl fold_c t1 yunze xilai chenzhe_dengdeng "han hui jiadi li mia rose" "chenzhe dengdeng"
build_pkl fold_c t2 yunze xilai han_hui "chenzhe dengdeng jiadi li mia rose" "han hui"
build_pkl fold_c t3 yunze xilai jiadi_li "chenzhe dengdeng han hui mia rose" "jiadi li"
build_pkl fold_c t4 yunze xilai mia_rose "chenzhe dengdeng han hui jiadi li" "mia rose"

echo "[DONE] Wrote zero-frame-filtered S6 trimmed teacher pkls under ${OUT_DIR}"
