#!/bin/bash
# Stop a running train.py test (kept in a script so pkill does not match the ssh command line).
tmux kill-session -t train 2>/dev/null
pkill -f "python -m search_r1[.]train" 2>/dev/null
pkill -f "python3 -m search_r1[.]train" 2>/dev/null
sleep 2; ps -eo pid,args | grep "search_r1[.]train" || echo "train stopped"
