#!/usr/bin/env bash
# Match the training module as a complete argv word for both Python entries.
pattern='^([^[:space:]]*/)?python([0-9]+([.][0-9]+)*)?[[:space:]]+-m[[:space:]]+search_r1[.]train(_dynamic_sampling)?([[:space:]]|$)'
tmux kill-session -t =train 2>/dev/null || true
pkill -f "$pattern" 2>/dev/null || [ "$?" -eq 1 ] || exit 1
sleep 2
pgrep -af "$pattern" && exit 1
rc=$?
[ "$rc" -eq 1 ] || exit "$rc"
echo "train stopped"
