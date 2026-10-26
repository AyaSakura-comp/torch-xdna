"""torch_xdna backend tests. Run: python tests/test_backend.py (each case in its own process).

Cases that need the NPU skip themselves when it is unavailable.
"""

import os
import subprocess
import sys

import torch

import torch_xdna
from torch_xdna import runtime
from torch_xdna.conv import conv3x3_int8

torch.manual_seed(0)


class Net(torch.nn.Module):
    """Odd shapes on purpose: K=200 and N=300 are not block multiples."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(200, 300)
        self.b = torch.nn.Linear(300, 1024, bias=False)
        self.w = torch.nn.Parameter(torch.randn(1024, 520) * 0.03)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.a(x))
        h = torch.nn.functional.silu(self.b(h))
        return h @ self.w


def run(net, x, **opts):
    torch._dynamo.reset()
    runtime.stats.reset()
    m = torch.compile(net, backend="xdna", options=opts)
    with torch.no_grad():
        return m(x)


def rel_err(y, ref):
    return ((y - ref).norm() / ref.norm()).item()


def case_numerics():
    """policy npu == policy cpu bit for bit (same int8 math), both close to fp32 eager."""
    if not torch_xdna.npu_available():
        print("SKIP numerics: no NPU:", torch_xdna.kernels.unavailable_reason())
        return
    net = Net().eval()
    for rows in (37, 300, 700):
        x = torch.randn(rows, 200)
        with torch.no_grad():
            ref = net(x)
        y_npu = run(net, x, policy="npu")
        s = runtime.stats.as_dict()
        assert s["calls"].get("npu", 0) == 3, s
        y_cpu = run(net, x, policy="cpu")
        assert torch.equal(y_npu, y_cpu), f"rows={rows}: NPU and CPU int8 differ"
        e = rel_err(y_npu, ref)
        assert e < 0.03, f"rows={rows}: rel err vs fp32 {e}"
        print(f"PASS numerics rows={rows}: npu == cpu-int8 bitwise, rel err vs fp32 {e:.4f}, "
              f"{s['npu_launches']} NPU launches")


def case_split_and_eager():
    """More rows than the largest bucket -> several launches; the eager fallback path."""
    if not torch_xdna.npu_available():
        print("SKIP split: no NPU")
        return
    net = Net().eval()
    x = torch.randn(3, 233, 200)  # 699 rows, 3-D input
    y_cpu = run(net, x, policy="cpu", fallback="eager")
    y = run(net, x, policy="npu", fallback="eager", row_buckets=(128, 256))
    s = runtime.stats.as_dict()
    assert torch.equal(y, y_cpu)
    assert s["npu_launches"] == 9, s  # 3 GEMMs x (256 + 256 + 187->256)
    print(f"PASS split+eager: 699 rows in {s['npu_launches']} launches, bitwise == cpu-int8")


def case_auto():
    if not torch_xdna.npu_available():
        print("SKIP auto: no NPU")
        return
    net = Net().eval()
    x = torch.randn(512, 200)
    y = run(net, x, policy="auto")
    y_cpu = run(net, x, policy="cpu")
    assert torch.equal(y, y_cpu)
    run(net, x, policy="auto")
    d = runtime.stats.as_dict()["decisions"]
    assert len(d) == 3, d
    print("PASS auto:", d)


class Attn(torch.nn.Module):
    """q/k/v read the same input -> one fused GEMM; o reads another."""

    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(256, 256)
        self.k = torch.nn.Linear(256, 128, bias=False)
        self.v = torch.nn.Linear(256, 128)
        self.o = torch.nn.Linear(256, 256)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        return self.o(q * torch.cat([k, v], -1).sigmoid())


def case_fusion():
    net = Attn().eval()
    x = torch.randn(200, 256)
    with torch.no_grad():
        ref = net(x)
    y_cpu = run(net, x, policy="cpu")
    calls_fused = runtime.stats.as_dict()["calls"]
    y_unfused = run(net, x, policy="cpu", fuse=False)
    calls_unfused = runtime.stats.as_dict()["calls"]
    assert calls_fused == {"cpu": 2} and calls_unfused == {"cpu": 4}, (calls_fused, calls_unfused)
    assert rel_err(y_cpu, ref) < 0.03 and rel_err(y_unfused, ref) < 0.03
    msg = f"PASS fusion: q/k/v -> 1 GEMM (2 calls vs 4), rel err {rel_err(y_cpu, ref):.4f}"
    if torch_xdna.npu_available():
        y_npu = run(net, x, policy="npu")
        assert torch.equal(y_npu, y_cpu)
        msg += "; npu == cpu-int8 bitwise"
    print(msg)


def case_silent_wrong():
    """An NPU program that runs but returns a wrong value (a silent miscompile) is caught on its
    first launch; that shape moves to the CPU and the model's output is still right."""
    if not torch_xdna.npu_available():
        print("SKIP silent wrong: no NPU")
        return
    from torch_xdna import kernels

    real = kernels.mm_i8

    def corrupt(a, w, c, t, *args, **kw):
        real(a, w, c, t, *args, **kw)
        if w.shape[1] == 512:          # only self.a (N=300, padded to 512)
            c[0, 0] += 1

    kernels.mm_i8 = corrupt
    net = Net().eval()
    x = torch.randn(300, 200)
    y = run(net, x, policy="npu")
    s = runtime.stats.as_dict()
    y_cpu = run(net, x, policy="cpu")
    assert torch.equal(y, y_cpu), "wrong NPU value reached the output"
    assert len(s["fallbacks"]) == 1 and "disagrees with the CPU" in s["fallbacks"][0][1], s["fallbacks"]
    assert s["calls"] == {"npu": 2, "cpu": 1}, s["calls"]
    print("PASS silent wrong ->", s["fallbacks"][0][1][:90], "...; that GEMM on the CPU, output bitwise == cpu-int8")


def case_launch_failure():
    """An NPU launch that raises: the call still returns the right answer, on the CPU, and the NPU is retired."""
    if not torch_xdna.npu_available():
        print("SKIP launch failure: no NPU")
        return
    from torch_xdna import kernels

    def boom(*a, **k):
        raise RuntimeError("simulated XRT error")

    kernels.mm_i8 = boom
    net = Net().eval()
    x = torch.randn(300, 200)
    y = run(net, x, policy="npu")
    s = runtime.stats.as_dict()
    y_cpu = run(net, x, policy="cpu")
    assert torch.equal(y, y_cpu)
    assert s["fallbacks"] and not s["npu_available"] and s["calls"].get("cpu") == 3, s
    print("PASS launch failure ->", s["fallbacks"][0][1], "; all 3 GEMMs on the CPU")


def case_no_npu():
    """TORCH_XDNA_DISABLE_NPU=1 (as on a machine without one): everything on the CPU, fp kernel too."""
    assert not torch_xdna.npu_available()
    net = Net().eval()
    x = torch.randn(300, 200)
    with torch.no_grad():
        ref = net(x)
    y = run(net, x, policy="npu")
    assert runtime.stats.as_dict()["calls"] == {"cpu": 3}
    y_fp = run(net, x, policy="npu", cpu_kernel="fp")
    e_int8, e_fp = rel_err(y, ref), rel_err(y_fp, ref)
    assert e_int8 < 0.03 and e_fp < 1e-5, (e_int8, e_fp)
    cnn = CNN().eval()
    xc = torch.randn(1, 64, 32, 64)
    with torch.no_grad():
        ref_c = cnn(xc)
    y_c = run(cnn, xc, policy="npu")
    assert runtime.stats.as_dict()["calls"] == {}
    e_c = rel_err(y_c, ref_c)
    assert e_c < 1e-5, e_c
    print(f"PASS no NPU: cpu int8 rel err {e_int8:.4f}, cpu fp rel err {e_fp:.2e}; convs left as fp convs, "
          f"rel err {e_c:.2e}")


class CNN(torch.nn.Module):
    """3x3 convs: with a bias (its ReLU stays in Inductor), without (ReLU fused on the NPU), padding
    "same", and a stride-2 one the NPU op does not take."""

    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.c2 = torch.nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.c3 = torch.nn.Conv2d(128, 32, 3, padding="same")
        self.down = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)

    def forward(self, x):
        h = torch.relu(self.c1(x))
        h = torch.relu(self.c2(h))
        return self.down(self.c3(h))


def case_conv_pass():
    """Which convs the pass rewrites, and which ReLU it fuses."""
    if not torch_xdna.npu_available():
        print("SKIP conv pass: no NPU (convs are left alone)")
        return
    graphs = []

    def capture(gm, example_inputs):
        torch_xdna.offload_gemms(gm, example_inputs, torch_xdna.Config(policy="npu"))
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(CNN().eval(), backend=capture)(torch.randn(1, 64, 16, 64))
    nodes = [n for n in graphs[0].graph.nodes if n.op == "call_function"]
    convs = [n for n in nodes if n.target is conv3x3_int8]
    assert [n.args[3] for n in convs] == [False, True, False], [n.args for n in convs]
    assert sum(n.target in (torch.relu, torch.nn.functional.relu) for n in nodes) == 1
    assert sum(n.target is torch.conv2d for n in nodes) == 1          # the stride-2 one
    print("PASS conv pass: 3 of 4 convs -> xdna::conv3x3, the bias-free one's ReLU fused, stride 2 left alone")


def case_conv():
    """3x3 convs on the NPU: close to fp32 eager, every layer's first launch bitwise == the CPU integers."""
    if not torch_xdna.npu_available():
        print("SKIP conv: no NPU")
        return
    net = CNN().eval()
    x = torch.randn(2, 64, 32, 64)
    with torch.no_grad():
        ref = net(x)
    y = run(net, x, policy="npu")
    s = runtime.stats.as_dict()
    assert s["calls"] == {"npu": 3} and not s["fallbacks"], s
    e = rel_err(y, ref)
    assert e < 0.05, f"rel err vs fp32 {e}"
    from torch_xdna import conv as xconv

    for _, layer in xconv._layers.values():      # the same layers (and shifts), routed to the CPU
        layer.device = "cpu"
    y_cpu = run(net, x, policy="npu")
    assert runtime.stats.as_dict()["calls"] == {"cpu": 3}
    assert torch.equal(y, y_cpu), "NPU and CPU integers differ"
    print(f"PASS conv: 3 convs (batch 2) on the NPU in {s['npu_launches']} launches, == the CPU's integers, "
          f"rel err vs fp32 {e:.4f}")


def case_conv_silent_wrong():
    """A conv launch returning one wrong value is caught on the layer's first launch; that layer
    moves to the CPU (fp conv) and the model's output is still right."""
    if not torch_xdna.npu_available():
        print("SKIP conv silent wrong: no NPU")
        return
    from triton.language.extra.npu import conv3x3

    real = conv3x3.Conv3x3.__call__

    def corrupt(self, x, out=None):
        y = real(self, x, out)
        if self.ci == 128 and self.co == 32:     # only c3
            y[0, 0, 0] += 1
        return y

    conv3x3.Conv3x3.__call__ = corrupt
    net = CNN().eval()
    x = torch.randn(1, 64, 32, 64)
    with torch.no_grad():
        ref = net(x)
    y = run(net, x, policy="npu")
    s = runtime.stats.as_dict()
    assert len(s["fallbacks"]) == 1 and "disagrees with the CPU" in s["fallbacks"][0][1], s["fallbacks"]
    assert s["calls"] == {"npu": 2, "cpu": 1}, s["calls"]
    e = rel_err(y, ref)
    assert e < 0.05, e
    print(f"PASS conv silent wrong -> {s['fallbacks'][0][1][:70]}...; c3 on the CPU, rel err vs fp32 {e:.4f}")


CASES = {f.__name__[5:]: f for f in (case_numerics, case_split_and_eager, case_auto, case_fusion, case_silent_wrong,
                                      case_launch_failure, case_no_npu, case_conv_pass, case_conv,
                                      case_conv_silent_wrong)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        CASES[sys.argv[1]]()
        sys.exit(0)
    failed = 0
    for name in CASES:
        env = dict(os.environ)
        if name == "no_npu":
            env["TORCH_XDNA_DISABLE_NPU"] = "1"
        r = subprocess.run([sys.executable, __file__, name], env=env, capture_output=True, text=True)
        out = [l for l in r.stdout.splitlines() if l.startswith(("PASS", "SKIP", "FAIL"))]
        print("\n".join(out) if r.returncode == 0 else f"FAIL {name}:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
        failed += r.returncode != 0
    sys.exit(1 if failed else 0)
