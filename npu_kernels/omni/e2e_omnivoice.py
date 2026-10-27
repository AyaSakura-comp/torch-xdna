"""OmniVoice end to end with the backbone on the NPU (one dispatch per step) vs the CPU.

Per mode: generate() wall time (median of reps after a warm-up), the time inside the NPU
dispatches, the first step's logits vs the fp32 model, and Whisper's transcript / CER of the
audio over a few seeds.

usage (IRON env + OmniVoice on the path):
    python e2e_omnivoice.py [--texts short] [--seeds 0 1 2] [--reps 5] [--modes npu cpu-int8 fp32]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/chihmin/src/torch-xdna/examples")

T2S = dict(zip("來氣錯們園剛幫臺會溫記帶傘這對說話裡點時", "来气错们园刚帮台会温记带伞这对说话里点时"))
PUNCT = set("，。、！？,.!? ；;：:「」\"'")


def norm(s):
    return "".join(T2S.get(c, c) for c in s if c not in PUNCT)


def cer(hyp, ref):
    h, r = norm(hyp), norm(ref)
    d = np.arange(len(h) + 1)
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", nargs="+", default=["short"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["npu", "cpu-int8", "fp32"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from omnivoice import OmniVoice
    from omnivoice_xdna import TEXTS

    torch.set_num_threads(16)
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
    if hasattr(model.llm, "_orig_mod"):
        model.llm = model.llm._orig_mod
    orig = model.forward
    asr = None
    if a.asr:
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cpu")
    results = {}
    npu = None
    for mode in a.modes:
        model.forward = orig
        if mode == "npu":
            from omni_npu import NPUBackbone

            npu = NPUBackbone(model).install()
        elif mode == "cpu-int8":
            import torch_xdna  # noqa: F401

            model.forward = torch.compile(orig, backend="xdna", options={"policy": "cpu"})
        rows = []
        for name in a.texts:
            text = TEXTS[name]
            torch.manual_seed(0)
            model.generate(text=text, num_step=a.steps)          # warm-up (compiles / loads)
            if npu is not None:
                npu.calls = {"npu": 0, "cpu": 0}
            ts = []
            for _ in range(a.reps):
                if npu is not None:
                    npu.timing.clear()
                t = time.perf_counter()
                model.generate(text=text, num_step=a.steps)
                ts.append(time.perf_counter() - t)
            row = dict(mode=mode, text=name, median_s=round(float(np.median(ts)), 4),
                       all_s=[round(x, 4) for x in ts])
            if mode == "npu":
                tm = np.array(npu.timing)
                row.update(npu_ms_per_step=round(1e3 * float(np.median(tm[:, 1])), 2),
                           host_ms_per_step=round(1e3 * float(np.median(tm[:, 0])), 2),
                           calls=dict(npu.calls))
            cers = []
            for seed in a.seeds if asr is not None else []:
                torch.manual_seed(seed)
                audio = np.asarray(model.generate(text=text, num_step=a.steps)[0], np.float32).squeeze()
                if a.wav_dir:
                    import soundfile as sf

                    os.makedirs(a.wav_dir, exist_ok=True)
                    sf.write(os.path.join(a.wav_dir, f"{mode}_{name}_s{seed}.wav"), audio, model.sampling_rate)
                hyp = asr({"raw": audio, "sampling_rate": model.sampling_rate},
                          generate_kwargs={"language": "zh"})["text"]
                cers.append((round(cer(hyp, text), 3), hyp))
            if cers:
                row["cer"] = round(float(np.mean([c for c, _ in cers])), 3)
                row["hyps"] = [h for _, h in cers]
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        results[mode] = rows
        if npu is not None and mode == "npu":
            npu.uninstall()
            npu = None
    if a.out:
        json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
