#!/bin/sh
# Train + evaluate all four C-MAPSS sub-datasets (cross-validated model selection for each)
cd "$(dirname "$0")"
for s in FD001 FD002 FD003 FD004; do
  python -u -m cmapss_health.train --subset $s --compare 2>&1 | tee logs/$s.log
done
