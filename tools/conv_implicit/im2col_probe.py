"""ResNet-50 conv layers (batch 1, 224x224) as im2col GEMMs: CPU conv vs xdna (NPU / CPU int8)."""
import time, torch, torch.nn.functional as F
import torch_xdna
from torch_xdna import runtime
torch.manual_seed(0)
torch.set_num_threads(16)

class ConvGemm(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.k, self.s, self.p, self.o = conv.kernel_size, conv.stride, conv.padding, conv.out_channels
        self.w = torch.nn.Parameter(conv.weight.detach().reshape(self.o, -1).clone())
    def forward(self, x):
        B, C, H, W = x.shape
        Ho = (H + 2*self.p[0] - self.k[0]) // self.s[0] + 1
        Wo = (W + 2*self.p[1] - self.k[1]) // self.s[1] + 1
        cols = F.unfold(x, self.k, padding=self.p, stride=self.s).transpose(1, 2)   # B, L, C*k*k
        y = F.linear(cols, self.w)                                                  # B, L, O
        return y.transpose(1, 2).reshape(B, self.o, Ho, Wo)

def t(fn, reps=20):
    fn(); fn()
    ts = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
    return sorted(ts)[reps // 2] * 1e3

LAYERS = [  # name, Cin, Cout, k, stride, H_in, count in ResNet-50
    ("conv1 7x7/2", 3, 64, 7, 2, 224, 1),
    ("layer1 3x3", 64, 64, 3, 1, 56, 3),
    ("layer1 1x1 64->256", 64, 256, 1, 1, 56, 3),
    ("layer2 3x3", 128, 128, 3, 1, 28, 4),
    ("layer3 3x3", 256, 256, 3, 1, 14, 6),
    ("layer3 1x1 256->1024", 256, 1024, 1, 1, 14, 6),
    ("layer4 3x3", 512, 512, 3, 1, 7, 3),
    ("layer4 1x1 512->2048", 512, 2048, 1, 1, 7, 3),
]
print(f"{'layer':22s} {'M':>6s} {'K':>5s} {'N':>5s} | {'cpu conv fp32':>13s} {'im2col only':>11s} {'npu int8':>9s} {'cpu int8':>9s} | launches  rel_err")
for name, ci, co, k, s, H, cnt in LAYERS:
    conv = torch.nn.Conv2d(ci, co, k, s, k // 2, bias=False).eval()
    x = torch.randn(1, ci, H, H)
    g = ConvGemm(conv).eval()
    with torch.no_grad():
        ref = conv(x)
        t_conv = t(lambda: conv(x))
        t_unf = t(lambda: F.unfold(x, conv.kernel_size, padding=conv.padding, stride=conv.stride).transpose(1, 2).contiguous())
        res = {}
        for pol in ("npu", "cpu"):
            torch._dynamo.reset(); runtime.stats.reset()
            m = torch.compile(g, backend="xdna", options=dict(policy=pol, min_rows=1, min_macs=0))
            y = m(x)
            runtime.stats.reset()
            res[pol] = (t(lambda: m(x)), y, runtime.stats.as_dict())
    Ho = ref.shape[-1]; M = Ho * Ho; K = ci * k * k
    tn, yn, sn = res["npu"]; tc, yc, _ = res["cpu"]
    assert torch.equal(yn, yc)
    launches = sn["npu_launches"] // 22
    err = ((yn - ref).norm() / ref.norm()).item()
    print(f"{name:22s} {M:6d} {K:5d} {co:5d} | {t_conv:10.3f} ms {t_unf:8.3f} ms {tn:6.3f} ms {tc:6.3f} ms | {launches:3d} x{cnt}  {err:.4f}", flush=True)
