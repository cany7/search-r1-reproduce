#!/bin/bash
# Log GPU util/memory and container memory every 5 s to /workspace/logs/sysmon.csv
# cgroup_mem includes page cache; anon_mem is non-reclaimable process memory.
OUT=/workspace/logs/sysmon.csv
echo "time,gpu_util,gpu_mem_mib,cgroup_mem_gib,anon_mem_gib,file_cache_gib" > $OUT
while true; do
  g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
  m=$(awk '{printf "%.1f", $1/1073741824}' /sys/fs/cgroup/memory.current)
  a=$(awk '$1=="anon"{printf "%.1f", $2/1073741824}' /sys/fs/cgroup/memory.stat)
  f=$(awk '$1=="file"{printf "%.1f", $2/1073741824}' /sys/fs/cgroup/memory.stat)
  echo "$(date +%s),$g,$m,$a,$f" >> $OUT
  sleep 5
done
