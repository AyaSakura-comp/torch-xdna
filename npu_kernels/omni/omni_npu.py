"""OmniVoice with its whole backbone step on the NPU: one dispatch per unmasking step.

    from omni_npu import NPUBackbone
    NPUBackbone(model).install()        # model.generate(...) now runs each step on the NPU

The text/audio embeddings stay on the CPU (a gather and a sum), so does the sampling between
steps; the 28 Qwen3 layers, the final norm and the audio heads are one NPU dispatch
(omni_step.py). The build takes the sequences in row groups of 64 positions each: a step
uses the build with the fewest groups that holds its longer sequence (built and loaded on
first use, then kept); a step longer than MAX_NG groups runs on the CPU as before.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omni_host as H  # noqa: E402


class NPUBackbone:
    MAX_NG = 6            # row groups: sequences of up to 384 positions

    def __init__(self, model, NL=28):
        self.model = model
        self.NL = NL
        self.weights = H.pack_weights(model, NL)
        self.engines = {}
        self.V = model.config.audio_vocab_size
        self.C = model.config.num_audio_codebook
        self.calls = {"npu": 0, "cpu": 0}
        self.timing = []

    def engine(self, NG):
        if NG not in self.engines:
            self.engines[NG] = H.Engine(self.model, NL=self.NL, NG=NG, weights=self.weights)
        return self.engines[NG]

    def forward(self, input_ids, audio_mask, attention_mask=None, **kw):
        m = self.model
        B2, _, L = input_ids.shape
        ok = B2 == 2 and attention_mask is not None
        if ok:
            c_len = int(attention_mask[0, 0, 0].sum())
            u_len = int(attention_mask[1, 0, 0].sum())
            NG = -(-max(c_len, u_len) // 64)
            ok = NG <= self.MAX_NG
        if not ok:
            self.calls["cpu"] += 1
            return self._orig(input_ids=input_ids, audio_mask=audio_mask, attention_mask=attention_mask, **kw)
        eng = self.engine(NG)
        t0 = time.perf_counter()
        with torch.no_grad():
            emb = m._prepare_embed_inputs(input_ids, audio_mask).float()
        t1 = time.perf_counter()
        tm = []
        lc, lu = eng.run(emb[0, :c_len], emb[1, :u_len], tm)
        t2 = time.perf_counter()
        n = self.C * self.V
        out = torch.zeros((2, L, n))
        out[0, :c_len] = torch.from_numpy(lc[:, :n])
        out[1, :u_len] = torch.from_numpy(lu[:, :n])
        logits = out.view(2, L, self.C, self.V).permute(0, 2, 1, 3)
        self.calls["npu"] += 1
        # host (embeddings, tiles in, logits out), NPU dispatch
        self.timing.append((t1 - t0 + tm[0][0] + tm[0][2] + time.perf_counter() - t2, tm[0][1]))

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
