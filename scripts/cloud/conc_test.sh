#!/bin/bash
# Eval concurrency test: run offline_eval on the same dev subset with each concurrency in $@
source /workspace/bin/env_common.sh
cd $PROJ
for C in "$@"; do
  S=$(date +%s)
  echo "START c=$C $S" >> /workspace/logs/conc_test.times
  uv run --frozen python -m search_r1.offline_eval backend=tinker tinker_base_url=http://127.0.0.1:8000 \
    eval_file=eval_sets/dev.jsonl limit_per_source=40 concurrency=$C run_name=cloud-conc-c$C > /workspace/logs/conc_c$C.log 2>&1
  echo "END c=$C $(date +%s) rc=$?" >> /workspace/logs/conc_test.times
done
echo CONC_DONE >> /workspace/logs/conc_test.times
