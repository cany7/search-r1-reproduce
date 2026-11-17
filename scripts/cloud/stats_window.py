"""Summarize sysmon.csv and vLLM engine stats between two unix timestamps.
usage: python stats_window.py START END infra_log_glob"""
import csv, glob, re, sys, datetime
s, e = int(sys.argv[1]), int(sys.argv[2])
rows = [r for r in csv.DictReader(open("/workspace/logs/sysmon.csv")) if s <= int(r["time"]) <= e]
if rows:
    gu = [float(r["gpu_util"]) for r in rows]; gm = [float(r["gpu_mem_mib"]) for r in rows]
    an = [float(r["anon_mem_gib"]) for r in rows]
    print(f"gpu_util mean {sum(gu)/len(gu):.0f}% max {max(gu):.0f}%; gpu_mem max {max(gm)/1024:.1f} GiB; anon_mem max {max(an):.1f} GiB")
pat = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers.py.*Avg prompt throughput: ([\d.]+).*Avg generation throughput: ([\d.]+).*Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")
year = datetime.datetime.now().year
st = []
for f in glob.glob(sys.argv[3]):
    for line in open(f, errors="ignore"):
        m = pat.search(line)
        if not m: continue
        t = datetime.datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S").timestamp()
        if s <= t <= e: st.append([float(x) for x in m.groups()[1:]])
if st:
    col = lambda i: [x[i] for x in st]
    print(f"vLLM: gen tok/s mean {sum(col(1))/len(st):.0f} max {max(col(1)):.0f}; running max {max(col(2)):.0f}; waiting max {max(col(3)):.0f}; KV max {max(col(4)):.1f}%; prefix hit last {col(5)[-1]:.1f}%")
for f in glob.glob(sys.argv[3]):
    pre = sum(1 for line in open(f, errors="ignore") if "preempt" in line.lower())
    print(f"preemption mentions in {f.split('/')[-1]}: {pre}")
