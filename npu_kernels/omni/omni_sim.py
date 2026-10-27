"""OmniVoice's backbone step in the numerics an all-NPU version would use, on the CPU.

The NPU's matrix units multiply bfp16 (blocks of 8 values sharing an exponent, 8-bit
mantissas) and accumulate in fp32. This runs the 28 Qwen3 layers, the final norm and the
audio heads with every GEMM operand rounded to bfp16 the way the cores round it, the
RMSNorm weights folded into the following GEMM's weights and the norm itself applied to the
GEMM's output rows, and the residual stream stored in a chosen format between ops. With
everything set to fp32 it is the model's own forward (checked against HF), so the effect of
each format on the logits, and on the audio, can be measured before building anything.

    sim = OmniSim(model, act="bfp16", resid="bfp16")
    sim.install()            # model.generate now runs the simulated backbone
"""

import math

import torch
import torch.nn.functional as F


def bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def bfp16(x):
    """Round the last dim, in groups of 8, to bfp16ebs8 as the AIE2P cores do: bf16 values,
    the group's largest exponent, 8-bit mantissas (round half to even), a mantissa that
    rounds to 128 moving the group up an exponent."""
    n = x.shape[-1]
    if n % 8:  # the cores see zero padding past the end
        return bfp16(F.pad(x, (0, 8 - n % 8)))[..., :n]
    shape = x.shape
    g = bf16(x).reshape(*shape[:-1], shape[-1] // 8, 8)
    top = g.abs().amax(-1, keepdim=True)
    _, e = torch.frexp(top)                      # top = f * 2^e, f in [0.5, 1)
    scale = torch.ldexp(torch.ones_like(top), 7 - e)
    m = torch.round(g * scale)
    over = m.abs().amax(-1, keepdim=True) > 127
    scale = torch.where(over, scale * 0.5, scale)
    m = torch.round(g * scale)
    out = m / scale
    out = torch.where(top < 2.0 ** -119, torch.zeros_like(out), out)   # biased exponent < 8
    return out.reshape(shape)


FMT = {"fp32": lambda x: x, "bf16": bf16, "bfp16": bfp16}


class OmniSim:
    """The backbone (+ final norm + audio heads) with bfp16 GEMMs.

    act    format of every GEMM's activation operand (and attention's q, k, v, p)
    w      format of the weights
    resid  format of the residual stream as stored between ops
    """

    def __init__(self, model, act="bfp16", w="bfp16", resid="bfp16", p="bf16"):
        self.model = model
        self.fa, self.fw, self.fr, self.fp = FMT[act], FMT[w], FMT[resid], FMT[p]
        llm = model.llm
        cfg = llm.config
        self.eps = cfg.rms_norm_eps
        self.nh, self.nkv, self.dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.inv_freq = llm.rotary_emb.inv_freq.float()
        self.layers = []
        with torch.no_grad():
            for lay in llm.layers:
                a, m = lay.self_attn, lay.mlp
                g_in = lay.input_layernorm.weight.float()
                g_post = lay.post_attention_layernorm.weight.float()
                wqkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).float() * g_in
                wgu = torch.cat([m.gate_proj.weight, m.up_proj.weight]).float() * g_post
                self.layers.append(dict(
                    wqkv=self.fw(wqkv), wo=self.fw(a.o_proj.weight.float()), wgu=self.fw(wgu),
                    wd=self.fw(m.down_proj.weight.float()),
                    gq=a.q_norm.weight.float(), gk=a.k_norm.weight.float()))
            self.wh = self.fw(model.audio_heads.weight.float() * llm.norm.weight.float())
        self.calls = 0

    def gemm(self, x, w):
        return self.fa(x) @ w.t()

    def rstd(self, x):
        xq = self.fa(x)
        return torch.rsqrt(xq.pow(2).mean(-1, keepdim=True) + self.eps)

    def rope(self, x, cos, sin):
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def head_norm(self, x, g):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * g

    def attention(self, q, k, v, mask):
        """q [B, H, T, d], k / v [B, Hkv, T, d], mask [B, 1, T, T] bool."""
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        s = (self.fa(q) @ self.fa(k).transpose(-1, -2)) / math.sqrt(self.dh)
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, -1)
        # the kernel keeps p unnormalized in bf16 and divides at the end; normalized here
        # V is consumed transposed (blocks of 8 keys share an exponent)
        return self.fa(self.fp(p)) @ self.fa(v.transpose(-1, -2)).transpose(-1, -2)

    def backbone(self, x, mask):
        """x [B, T, 1024] input embeddings, mask [B, 1, T, T] -> audio logits [B, T, 8200]."""
        B, T, _ = x.shape
        pos = torch.arange(T, dtype=torch.float32)
        ang = pos[:, None] * self.inv_freq[None, :]
        cos, sin = ang.cos(), ang.sin()
        nq, nk = self.nh * self.dh, self.nkv * self.dh
        x = self.fr(x.float())
        for L in self.layers:
            r = self.rstd(x)
            qkv = self.gemm(x, L["wqkv"]) * r
            q = qkv[..., :nq].view(B, T, self.nh, self.dh).transpose(1, 2)
            k = qkv[..., nq:nq + nk].view(B, T, self.nkv, self.dh).transpose(1, 2)
            v = qkv[..., nq + nk:].view(B, T, self.nkv, self.dh).transpose(1, 2)
            q = self.rope(self.head_norm(q, L["gq"]), cos, sin)
            k = self.rope(self.head_norm(k, L["gk"]), cos, sin)
            o = self.attention(q, k, v, mask).transpose(1, 2).reshape(B, T, nq)
            x = self.fr(x + self.gemm(o, L["wo"]))
            r = self.rstd(x)
            gu = self.gemm(x, L["wgu"]) * r
            g, u = gu.chunk(2, -1)
            h = F.silu(g) * u
            x = self.fr(x + self.gemm(h, L["wd"]))
        return self.gemm(x, self.wh) * self.rstd(x)

    def forward(self, input_ids, audio_mask, attention_mask=None, **kw):
        """Stands in for OmniVoice.forward (inference: no labels)."""
        m = self.model
        with torch.no_grad():
            emb = m._prepare_embed_inputs(input_ids, audio_mask)
            logits = self.backbone(emb, attention_mask)
        self.calls += 1
        B, T, _ = logits.shape
        out = logits.view(B, T, m.config.num_audio_codebook, m.config.audio_vocab_size).permute(0, 2, 1, 3)

        class Out:
            pass

        o = Out()
        o.logits = out
        return o

    def install(self):
        self._orig = self.model.forward
        self.model.forward = self.forward

    def uninstall(self):
        self.model.forward = self._orig
