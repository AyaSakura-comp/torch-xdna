"""OmniVoice TTS with torch.compile(backend="xdna"): its GEMMs on the NPU, the rest on the CPU.

No OmniVoice code changes: the model's forward (embeddings, the 28-layer
Qwen3 backbone, the audio heads) is compiled with the xdna backend, and
generate() calls it once per unmasking step as usual.

usage (bash):
  source ~/src/Triton-XDNA/npu-env.sh
  python examples/omnivoice_xdna.py --policy npu|auto|cpu [--fallback inductor|eager] [--steps 8] [--wav-dir out]
  python examples/omnivoice_xdna.py --eager          # the uncompiled fp32 baseline
"""

import argparse
import json
import os
import statistics
import time

import torch

import torch_xdna

TEXTS = {
    "short": "好啊，我來看看。",
    "mid": "今天天氣不錯，我們下午可以去公園走走。",
    "long": "我剛剛幫你查了一下，明天台北會下雨，氣溫大概在二十度左右，出門記得帶傘，也可以多穿一件外套。",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="npu", choices=["npu", "auto", "cpu"])
    ap.add_argument("--fallback", default="inductor", choices=["inductor", "eager"])
    ap.add_argument("--eager", action="store_true", help="no torch.compile at all (fp32 baseline)")
    ap.add_argument("--no-tuned", action="store_true", help="default tiling per row bucket, not the swept table")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
    if hasattr(model.llm, "_orig_mod"):  # the repo compiles the backbone for the GPU when triton imports
        model.llm = model.llm._orig_mod
    tag = "eager-fp32" if args.eager else f"xdna-{args.policy}-{args.fallback}" + ("-untuned" if args.no_tuned else "")
    if not args.eager:
        model.forward = torch.compile(model.forward, backend="xdna",
                                      options={"policy": args.policy, "fallback": args.fallback,
                                               "tuned_tilings": not args.no_tuned})

    t = time.perf_counter()
    for text in TEXTS.values():  # compile: static shape first, then the dynamic-shape graph
        model.generate(text=text, num_step=args.steps)
    warm_s = time.perf_counter() - t
    print(f"{tag}: warmup (compile + NPU builds) {warm_s:.1f}s", flush=True)

    rows = []
    for name, text in TEXTS.items():
        times = []
        torch_xdna.stats.reset()
        for r in range(args.reps):
            t = time.perf_counter()
            audio = model.generate(text=text, num_step=args.steps)[0]
            times.append(time.perf_counter() - t)
            if args.wav_dir and r == 0:
                import soundfile as sf
                os.makedirs(args.wav_dir, exist_ok=True)
                sf.write(os.path.join(args.wav_dir, f"{tag}_s{args.steps}_{name}.wav"), audio.squeeze(),
                         model.sampling_rate)
        dur = audio.shape[-1] / model.sampling_rate
        s = torch_xdna.stats.as_dict()
        row = dict(mode=tag, text=name, steps=args.steps, median_s=round(statistics.median(times), 3),
                   audio_s=round(dur, 2), rtf=round(statistics.median(times) / dur, 3),
                   all_s=[round(x, 3) for x in times],
                   gemm_calls_per_rep={k: v // args.reps for k, v in s["calls"].items()},
                   npu_launches_per_rep=s["npu_launches"] // args.reps)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(torch_xdna.report())
    if args.out:
        json.dump(dict(mode=tag, warmup_s=warm_s, rows=rows, stats=torch_xdna.stats.as_dict()),
                  open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
