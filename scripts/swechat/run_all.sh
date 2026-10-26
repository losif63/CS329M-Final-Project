#!/usr/bin/env bash
# Rebuild the SWE-chat slop-comment dataset end to end (see README.md).
#   scripts/swechat/run_all.sh                 full dataset (~15 min extraction, ~11 GB download once)
#   LIMIT=50 SEED=0 scripts/swechat/run_all.sh  quick sample
#   PUBLISH=1 scripts/swechat/run_all.sh        also upload the viewer to the private HF Space
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}

$PY scripts/swechat/download.py
$PY -m pytest -q scripts/swechat/tests
$PY scripts/swechat/extract.py ${LIMIT:+--limit-sessions "$LIMIT"} --seed "${SEED:-0}"
$PY scripts/swechat/viewer/export.py
if [[ "${PUBLISH:-0}" == "1" ]]; then
  $PY scripts/swechat/viewer/publish.py
fi
