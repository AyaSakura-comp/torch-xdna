"""OmniVoice's backbone step on the NPU without torch.compile: one launch of Triton-XDNA's
tl.extra.npu.qwen3_stack per unmasking step.

    from omni_npu import NPUBackbone
    NPUBackbone(model).install()        # model.generate(...) now runs each step on the NPU

torch.compile(model.forward, backend="xdna") finds the same stack in the graph and runs the same
op (torch_xdna/stack.py), with bit-identical logits; that is the way to use it. This route skips
Dynamo's per-call work (~2.5-3 ms per step for OmniVoice's 317 graph inputs) and checks nothing:
for A/B comparisons and profiling. The text/audio embeddings and the sampling stay on the CPU; a
step longer than the op takes (384 positions per sequence) runs the original forward.
"""

import time

import torch


def stack_op(model, layers=None):
    """tl.extra.npu.qwen3_stack over OmniVoice's backbone (its first ``layers`` layers) and audio heads."""
    from triton.language.extra.npu import qwen3_stack

    def weights(lay):
        a, m = lay.self_attn, lay.mlp
        return dict(q=a.q_proj.weight, k=a.k_proj.weight, v=a.v_proj.weight, o=a.o_proj.weight,
                    gate=m.gate_proj.weight, up=m.up_proj.weight, down=m.down_proj.weight,
                    input_norm=lay.input_layernorm.weight, post_norm=lay.post_attention_layernorm.weight,
                    q_norm=a.q_norm.weight, k_norm=a.k_norm.weight)

    llm = model.llm
    with torch.no_grad():
        return qwen3_stack.Qwen3Stack([weights(lay) for lay in llm.layers[:layers]], norm=llm.norm.weight,
                                      head=model.audio_heads.weight, inv_freq=llm.rotary_emb.inv_freq)


class NPUBackbone:
    def __init__(self, model):
        self.model = model
        self.op = stack_op(model)
        self.V = model.config.audio_vocab_size
        self.C = model.config.num_audio_codebook
        self.calls = {"npu": 0, "cpu": 0}
        self.timing = []   # per NPU step: (host seconds, NPU seconds)

    def forward(self, input_ids, audio_mask, attention_mask=None, **kw):
        m = self.model
        B2, _, L = input_ids.shape
        ok = B2 == 2 and attention_mask is not None
        if ok:
            c_len = int(attention_mask[0, 0, 0].sum())
            u_len = int(attention_mask[1, 0, 0].sum())
            ok = max(c_len, u_len) <= self.op.max_tokens
        if not ok:
            self.calls["cpu"] += 1
            return self._orig(input_ids=input_ids, audio_mask=audio_mask, attention_mask=attention_mask, **kw)
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = m._prepare_embed_inputs(input_ids, audio_mask).float()
            lc, lu = self.op(emb[0, :c_len], emb[1, :u_len])
        n = self.C * self.V
        out = torch.zeros((2, L, n))
        out[0, :c_len] = lc
        out[1, :u_len] = lu
        logits = out.view(2, L, self.C, self.V).permute(0, 2, 1, 3)
        self.calls["npu"] += 1
        npu_s = self.op.last_npu_s
        self.timing.append((time.perf_counter() - t0 - npu_s, npu_s))

        class Out:
            pass

        o = Out()
        o.logits = logits
        return o

    def install(self):
        self._orig = self.model.forward
        self.model.forward = self.forward
        return self

    def uninstall(self):
        self.model.forward = self._orig
