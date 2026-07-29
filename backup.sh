STAMP=$(date +%Y%m%d_%H%M%S)
for d in work_dirs/posec3d/8111 work_dirs/thesis/e2 work_dirs/thesis/s2; do
  if [[ -e "$d" ]]; then
    mv "$d" "${d}_before_zero_filter_${STAMP}"
  fi
done