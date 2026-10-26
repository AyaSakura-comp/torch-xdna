"""Re-run a traced build with other core trace events: patch the event-select words in the cached
ELF (Triton cache) and in input_with_addresses.mlir, run again, parse, summarize per tile.

usage: python tools/npu_trace/retrace.py OUTDIR "37,28,26,23" "24,38,39,22" -- <trace_gemm.py args...>
The default events aircc programs are 37,34,33,1 / 79,75,45,44.
"""
import glob, hashlib, json, os, re, subprocess, sys, collections

out = os.path.abspath(sys.argv[1]); ev0 = [int(x) for x in sys.argv[2].split(",")]; ev1 = [int(x) for x in sys.argv[3].split(",")]
gemm_args = sys.argv[sys.argv.index("--") + 1:]
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
word = lambda evs: sum((e & 0xFF) << (8 * i) for i, e in enumerate(evs))
OLD0, OLD1 = word([37, 34, 33, 1]), word([79, 75, 45, 44])

def run():
    r = subprocess.run([py, os.path.join(here, gemm_args[0]), out] + gemm_args[1:], capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if "exact" in l or "mlir for" in l))
    if r.returncode:
        print(r.stderr[-3000:]); sys.exit(1)

run()                                        # build (default events) -> cache entry
elf = os.path.join(out, "air_project", "aie.elf")
h = hashlib.md5(open(elf, "rb").read()).hexdigest()
cache_dir = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
cached = [p for p in glob.glob(os.path.join(cache_dir, "*", "aie.elf")) if hashlib.md5(open(p, "rb").read()).hexdigest() == h]
assert cached, "cached ELF not found"
for p in cached + [elf]:
    data = open(p, "rb").read()
    n0, n1 = data.count(OLD0.to_bytes(4, "little")), data.count(OLD1.to_bytes(4, "little"))
    data = data.replace(OLD0.to_bytes(4, "little"), word(ev0).to_bytes(4, "little")).replace(OLD1.to_bytes(4, "little"), word(ev1).to_bytes(4, "little"))
    open(p, "wb").write(data)
    print(f"patched {p}: {n0}+{n1} words")
m = os.path.join(out, "input_with_addresses.mlir")
txt = open(m).read()
txt = txt.replace(f"%c{OLD0}_i32", f"%c{word(ev0)}_i32").replace(f"arith.constant {OLD0} : i32", f"arith.constant {word(ev0)} : i32")
txt = txt.replace(f"%c{OLD1}_i32", f"%c{word(ev1)}_i32").replace(f"arith.constant {OLD1} : i32", f"arith.constant {word(ev1)} : i32")
open(m, "w").write(txt)
run()                                        # cache hit -> patched ELF
subprocess.run([py, "-m", "aie.utils.trace.parse", "--input", "trace.txt", "--mlir", "input_with_addresses.mlir",
                "--output", "trace.json"], cwd=out, capture_output=True)
ev = json.load(open(os.path.join(out, "trace.json")))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
pname, tname = {}, {}
for e in ev:
    if e.get("ph") == "M":
        if e["name"] == "process_name": pname[e["pid"]] = e["args"]["name"]
        if e["name"] == "thread_name": tname[(e["pid"], e["tid"])] = e["args"]["name"]
dur = collections.defaultdict(float); cnt = collections.Counter(); span = {}; open_ = {}
for e in ev:
    if e.get("ph") in ("B", "E"):
        key = (e["pid"], e["tid"])
        if e["ph"] == "B": open_[key] = e["ts"]
        elif key in open_:
            dur[key] += e["ts"] - open_.pop(key); cnt[key] += 1
    if "ts" in e:
        p = e["pid"]; lo, hi = span.get(p, (1e30, -1e30)); span[p] = (min(lo, e["ts"]), max(hi, e["ts"]))
for p in sorted(pname):
    lo, hi = span.get(p, (0, 0))
    if hi - lo < 1000:
        continue
    print(f"== {pname[p]}  span {hi - lo:.0f} cycles")
    for (pp, t), d in sorted(dur.items()):
        if pp == p:
            print(f"   {tname.get((pp, t), t):26s} n={cnt[(pp, t)]:7d} cycles={d:10.0f} ({100 * d / max(hi - lo, 1):5.1f}%)")
