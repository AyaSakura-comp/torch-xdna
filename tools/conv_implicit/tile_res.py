"""Prototype: ResNet-50 convs as implicit GEMM on the NPU (no im2col), vs CPU conv."""
import math, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from torch_xdna import kernels
from torch_xdna.kernels import Tiling
import conv_kernel

torch.manual_seed(0); torch.set_num_threads(16)
assert kernels.available(), kernels.unavailable_reason()
res = kernels.resident()
M = kernels._mod

def ceil(a, b): return -(-a // b) * b

class ConvPlan:
    """Offline: weights quantized per out-channel, reordered (ky,kx,ci)xCo, padded, resident."""
    def __init__(self, conv, H, tiling, B=1):
        self.t = t = tiling
        co, ci, kh, kw = conv.weight.shape
        self.S, self.P, self.KH, self.KW, self.co = conv.stride[0], conv.padding[0], kh, kw, co
        self.CI = ceil(ci, t.l2_k); self.Np = ceil(co, t.block_n)
        w = conv.weight.detach().float()
        self.sw = w.abs().amax(dim=(1, 2, 3)).clamp(min=1e-8) / 127
        wq = torch.round(w / self.sw[:, None, None, None]).clamp(-127, 127).to(torch.int8)
        wk = torch.zeros(kh, kw, self.CI, self.Np, dtype=torch.int8)
        wk[:, :, :ci, :co] = wq.permute(2, 3, 1, 0)
        self.w_kn = wk.reshape(-1, self.Np).contiguous()
        self.w_npu = res.to_npu(self.w_kn, static=True)
        S, P = self.S, self.P
        self.Hp = H + 2 * P
        self.Ho = (self.Hp - kh) // S + 1
        self.Hph = self.WP = -(-self.Hp // S)          # phase grid height / width (square input)
        self.B = B
        self.M = B * self.Hph * self.WP if B > 1 else self.Ho * self.WP
        self.Mp = ceil(self.M, t.block_m)
        self.PH = max(B * self.Hph * self.WP, self.Mp + ((kh - 1) // S) * self.WP + (kw - 1) // S)
        self.R = S * S * self.PH
        self.a = res.empty((self.R, self.CI), dtype=torch.int8)
        self.c = res.empty((self.Mp, self.Np), dtype=torch.int32)
        self.host = torch.zeros(S * S, self.PH, self.CI, dtype=torch.int8)

    def layout(self, x):
        """Per-tensor quantize + pad + polyphase split into the resident buffer (1x data, no im2col)."""
        S, P = self.S, self.P
        sx = x.abs().amax().clamp(min=1e-8) / 127
        xq = torch.round(x / sx).clamp(-127, 127).to(torch.int8)          # B,C,H,W
        ci = xq.shape[1]
        xp = F.pad(xq, (P, P + S, P, P + S))                               # extra zeros so every phase is full
        h = self.host
        n = self.B * self.Hph * self.WP
        for p in range(S):
            for q in range(S):
                ph = xp[:, :, p::S, q::S][:, :, : self.Hph, : self.WP]    # B, C, Hph, WP
                h[p * S + q, :n, :ci] = ph.permute(0, 2, 3, 1).reshape(-1, ci)
        self.a.copy_(h.view(self.R, self.CI))
        return sx

    def launch(self):
        t = self.t
        cfg = M["npu_config"]; M["triton"].runtime.driver.set_active(M["driver"])
        saved = (cfg._transform_tiling_script, cfg._air_project_path)
        cfg._transform_tiling_script = kernels.schedule(t); cfg._air_project_path = kernels.project_dir()
        try:
            conv_kernel.conv_i8_implicit[(self.Mp // t.block_m, self.Np // t.block_n)](
                self.a, self.w_npu, self.c, self.Np, self.CI, self.KH, self.KW, self.S, self.WP, self.PH,
                BLOCK_M=t.block_m, BLOCK_N=t.block_n, BLOCK_K=t.l2_k)
        finally:
            cfg._transform_tiling_script, cfg._air_project_path = saved

    def reference(self):
        """Same integers on the CPU: gather the implicit A explicitly, torch._int_mm."""
        a = self.host.view(self.R, self.CI)
        cols = []
        for ky in range(self.KH):
            for kx in range(self.KW):
                row = ((ky % self.S) * self.S + kx % self.S) * self.PH + (ky // self.S) * self.WP + kx // self.S
                cols.append(a[row: row + self.Mp])
        return torch._int_mm(torch.cat(cols, 1).contiguous(), self.w_kn)

    def finish(self, sx):
        y = self.c[: self.M, : self.co].cpu().float() * (sx * self.sw)
        if self.B == 1:
            return y.view(self.Ho, self.WP, self.co)[:, : self.Ho].permute(2, 0, 1).unsqueeze(0)
        return y.view(self.B, self.Hph, self.WP, self.co)[:, : self.Ho, : self.Ho].permute(0, 3, 1, 2)

    def __call__(self, x):
        sx = self.layout(x); self.launch(); return self.finish(sx)


def med(fn, reps=30):
    fn(); fn(); ts = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
    return sorted(ts)[reps // 2] * 1e3

T = Tiling
# ResNet-50 3x3 layers at batch 1, input resolution 224 / 512 / 1024 (square): x*y is the GEMM's M
LAYERS = [("layer1 3x3", 64, 64, 4, [T(1024,64,128,32,64), T(512,64,64,32,64)]),
          ("layer2 3x3", 128, 128, 8, [T(512,128,128,64,128), T(1024,128,128,64,128), T(1024,128,128,32,128)]),
          ("layer3 3x3", 256, 256, 16, [T(256,256,64,128,128), T(512,256,64,128,128), T(512,256,128,64,128)]),
          ("layer4 3x3", 512, 512, 32, [T(64,512,32,128,128), T(256,512,64,128,128), T(512,512,64,128,128)])]
for res_in in (224, 512, 1024):
    for name, ci, co, down, cands in LAYERS:
        H = res_in // down
        conv = torch.nn.Conv2d(ci, co, 3, 1, 1, bias=False).eval()
        x = torch.randn(1, ci, H, H).relu()
        with torch.no_grad():
            ref = conv(x); t_cpu = med(lambda: conv(x), 15)
        best = None
        for t in cands:
            try:
                p = ConvPlan(conv, H, t, 1)
                sx = p.layout(x); p.launch()
                if not torch.equal(p.c.cpu(), p.reference()):
                    print("  MISMATCH", name, H, t); continue
                tk = med(p.launch, 15)
                if best is None or tk < best[0]:
                    best = (tk, t, p)
            except Exception as e:
                print(f"  {name} {H} {t} FAIL", str(e).splitlines()[-1][:100])
        tk, t, p = best
        te = med(lambda: p(x), 15)
        err = ((p(x) - ref).norm() / ref.norm()).item()
        print(f"input {res_in:4d} {name} {H:3d}x{H:<3d} M={H*H:6d} | cpu fp32 {t_cpu:7.3f} ms | npu kernel {tk:7.3f} ms "
              f"e2e {te:7.3f} ms ({t}) | kernel/cpu {tk/t_cpu:.2f}  err {err:.4f}", flush=True)
