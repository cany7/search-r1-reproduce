#!/bin/bash
# Eval concurrency test: run offline_eval on the same dev subset with each concurrency in $@
source "$(dirname -- "${BASH_SOURCE[0]}")/../env_common.sh" || exit 1
cd "$PROJ"
for C in "$@"; do
  S=$(date +%s)
  echo "START c=$C $S" >> "$POD_ROOT/logs/conc_test.times"
  rc=0
  uv run --frozen python -m search_r1.offline_eval backend=tinker skyrl_base_eval=true tinker_base_url="$TINKER_BASE_URL" \
    eval_file=eval_sets/dev.jsonl limit_per_source=40 concurrency=$C run_name=cloud-conc-c$C > "$POD_ROOT/logs/conc_c$C.log" 2>&1 || rc=$?
  echo "END c=$C $(date +%s) rc=$rc" >> "$POD_ROOT/logs/conc_test.times"
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo CONC_DONE >> "$POD_ROOT/logs/conc_test.times"
