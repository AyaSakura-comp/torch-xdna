"""Off-the-shelf PyTorch models through torch.compile(backend="xdna"): do they still give the right answer?

Each model runs three ways, in its own process:
  eager   fp32, uncompiled: the reference
  npu     xdna, policy "npu", min_rows=1, min_macs=0: every GEMM the pass can take goes to the NPU
  cpu8    xdna, policy "cpu": the same W8A8 math on the CPU

"npu == cpu8 bitwise" checks the NPU computed what it should; "vs fp32" measures what W8A8
quantization does to the model's actual output (text, top-1, embeddings). "NPU FLOPs" is the
share of the run's FLOPs (counted on the eager run) that the NPU executed.

usage: HF_HUB_OFFLINE=1 python tests/model_zoo.py [model ...]   (all when none given)
"""

import difflib
import glob
import json
import os
import subprocess
import sys
import time

import torch

_SK = os.path.expanduser("~/.local/lib/python3.13/site-packages/skimage/data")
_MPL = os.path.expanduser("~/.local/lib/python3.13/site-packages/matplotlib/mpl-data/sample_data")
IMAGES = [os.path.join(_SK, f) for f in ("astronaut.png", "chelsea.png", "coffee.png", "rocket.jpg",
                                         "motorcycle_left.png", "brick.png", "coins.png")] + \
    [os.path.join(_MPL, "grace_hopper.jpg")]
WAVS = sorted(glob.glob(os.path.expanduser("~/src/torch-xdna/results/wav_q/eager-fp32_s8_*.wav")))
PRECISION = os.environ.get("ZOO_PRECISION", "int8")
NPU_OPTS = dict(policy="npu", min_rows=1, min_macs=0, precision=PRECISION)


# ------------------------------------------------------------------ comparisons
def cmp_tensor(ref, out):
    ref, out = ref.float(), out.float()
    d = dict(cos=round(torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item(), 5),
             rel_err=round(((out - ref).norm() / ref.norm()).item(), 4))
    if ref.dim() >= 2:
        d["top1_agree"] = round((ref.argmax(-1) == out.argmax(-1)).float().mean().item(), 4)
    return d


def cmp_text(ref, out):
    ref, out = list(ref), list(out)
    return dict(identical=sum(a == b for a, b in zip(ref, out)), of=len(ref),
                char_sim=round(sum(difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(ref, out)) / len(ref), 4))


def same(a, b):
    if isinstance(a, torch.Tensor):
        return bool(torch.equal(a, b))
    if isinstance(a, (list, tuple)):
        return all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return all(same(a[k], b[k]) for k in a)
    return a == b


# ------------------------------------------------------------------ models
def m_resnet50():
    import torchvision
    from PIL import Image

    model = torchvision.models.resnet50()
    model.load_state_dict(torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")))
    tf = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in IMAGES])
    return model, (lambda m: m(x)), cmp_tensor, "ImageNet logits, 8 images"


def m_dinov3():
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    name = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
    proc, model = AutoImageProcessor.from_pretrained(name), AutoModel.from_pretrained(name)
    px = proc(images=[Image.open(p).convert("RGB") for p in IMAGES], return_tensors="pt")["pixel_values"]
    return model, (lambda m: m(pixel_values=px).pooler_output), cmp_tensor, "ViT-L/16 image embeddings, 8 images"


def _lm_case(name, prompt, ppl_text, n_new):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    if tok.chat_template:
        prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    ppl_ids = tok(ppl_text, return_tensors="pt")["input_ids"]

    def run(m):
        logits = m(input_ids=ppl_ids).logits
        nll = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ppl_ids[0, 1:])
        gen = m.generate(**ids, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return dict(logits=logits, ppl=torch.exp(nll), text=tok.decode(gen[0, ids["input_ids"].shape[1]:],
                                                                         skip_special_tokens=True))

    def cmp(ref, out):
        d = cmp_tensor(ref["logits"], out["logits"])
        d.update(ppl_fp32=round(ref["ppl"].item(), 3), ppl_xdna=round(out["ppl"].item(), 3),
                 greedy_text_identical=ref["text"] == out["text"],
                 text_fp32=ref["text"][:120], text_xdna=out["text"][:120])
        return d
    return model, run, cmp


PPL_EN = ("The Industrial Revolution began in Great Britain in the late eighteenth century and spread to "
          "continental Europe and North America. It replaced hand production with machines, new chemical "
          "manufacturing and iron production processes, and the increasing use of steam power.")
PPL_ZH = ("台灣位於東亞，北臨東海，東臨太平洋，南隔巴士海峽與菲律賓相望。台灣的氣候屬於亞熱帶與熱帶季風氣候，"
          "夏季炎熱潮濕，冬季溫和，山區則較為涼爽。")


def m_gpt2():
    model, run, cmp = _lm_case("gpt2", "The secret of making good coffee is", PPL_EN, 32)
    return model, run, cmp, "perplexity + 32-token greedy generation (Conv1D -> addmm)"


def m_qwen25():
    model, run, cmp = _lm_case("Qwen/Qwen2.5-0.5B-Instruct", "用兩句話介紹台北。", PPL_ZH, 48)
    return model, run, cmp, "perplexity + 48-token greedy chat reply"


def _mlm_case(name, sents):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name)
    sents = [s.replace("[MASK]", tok.mask_token) for s in sents]
    enc = tok(sents, return_tensors="pt", padding=True)
    mask = enc["input_ids"] == tok.mask_token_id

    def run(m):
        logits = m(**enc).logits
        return dict(logits=logits[mask], words=tok.convert_ids_to_tokens(logits[mask].argmax(-1)))

    def cmp(ref, out):
        d = cmp_tensor(ref["logits"], out["logits"])
        d.update(words_fp32=ref["words"], words_xdna=out["words"])
        return d
    return model, run, cmp


def m_bert_zh():
    model, run, cmp = _mlm_case("bert-base-chinese", ["台灣的首都是台[MASK]。", "我今天早上喝了一杯[MASK]啡。",
                                                       "下雨了，記得帶[MASK]。", "他在醫[MASK]當護理師。",
                                                       "這本[MASK]很好看。", "我們去公[MASK]散步吧。"])
    return model, run, cmp, "fill-mask top-1, 6 sentences"


def m_roberta():
    model, run, cmp = _mlm_case("roberta-base", ["The capital of France is[MASK].", "She drank a cup of hot[MASK].",
                                                  "Take an[MASK] because it is raining.", "The cat sat on the[MASK]."])
    return model, run, cmp, "fill-mask top-1, 4 sentences"


def m_e5():
    from transformers import AutoModel, AutoTokenizer

    name = "intfloat/multilingual-e5-small"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)
    qs = ["query: 台北的天氣", "query: how to make coffee", "query: NPU 是什麼", "query: 櫻花什麼時候開"]
    ps = ["passage: 台北明天會下雨，氣溫約二十度。", "passage: Grind beans, use 93°C water, brew four minutes.",
          "passage: NPU 是專門做神經網路運算的處理器。", "passage: 日本的櫻花通常在三月底到四月初盛開。",
          "passage: The stock market fell two percent today.", "passage: 台灣高鐵從台北到高雄約一個半小時。"]
    enc = tok(qs + ps, return_tensors="pt", padding=True)

    def run(m):
        h = m(**enc).last_hidden_state
        e = (h * enc["attention_mask"][..., None]).sum(1) / enc["attention_mask"].sum(1, keepdim=True)
        e = torch.nn.functional.normalize(e, dim=-1)
        sim = e[: len(qs)] @ e[len(qs):].T
        return dict(emb=e, rank=sim.argsort(-1, descending=True))

    def cmp(ref, out):
        d = cmp_tensor(ref["emb"], out["emb"])
        d.pop("top1_agree", None)
        d.update(min_row_cos=round(torch.nn.functional.cosine_similarity(ref["emb"], out["emb"]).min().item(), 5),
                 ranking_identical=bool(torch.equal(ref["rank"], out["rank"])))
        return d
    return model, run, cmp, "retrieval embeddings, 4 queries x 6 passages"


def m_whisper():
    import soundfile as sf
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    name = "openai/whisper-base"
    proc, model = WhisperProcessor.from_pretrained(name), WhisperForConditionalGeneration.from_pretrained(name)
    audio = []
    for p in WAVS:
        y, sr = sf.read(p)
        audio.append(librosa.resample(y.astype("float32"), orig_sr=sr, target_sr=16000))
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features

    def run(m):
        ids = m.generate(input_features=feats, language="zh", task="transcribe", max_new_tokens=64)
        return proc.batch_decode(ids, skip_special_tokens=True)
    return model, run, cmp_text, f"ASR of {len(WAVS)} OmniVoice clips (zh), greedy"


def m_omnivoice():
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
    if hasattr(model.llm, "_orig_mod"):
        model.llm = model.llm._orig_mod

    def run(m):
        torch.manual_seed(0)  # its unmasking samples positions; fixed seed -> comparable runs
        return torch.as_tensor(m.generate(text="今天天氣不錯，我們下午可以去公園走走。", num_step=16)[0]).flatten()

    def cmp(ref, out):
        n = min(ref.numel(), out.numel())
        return dict(samples_fp32=ref.numel(), samples_xdna=out.numel(), identical=bool(torch.equal(ref, out)),
                    wave_cos=round(torch.nn.functional.cosine_similarity(ref[:n], out[:n], dim=0).item(), 4))
    return model, run, cmp, "TTS 19 chars, 16 steps, seed 0 (waveform)"


MODELS = {f.__name__[2:]: f for f in (m_resnet50, m_dinov3, m_gpt2, m_qwen25, m_e5, m_whisper, m_omnivoice)}


# ------------------------------------------------------------------ harness
def evaluate(key):
    import torch_xdna
    from torch.utils.flop_counter import FlopCounterMode

    torch.manual_seed(0)
    torch.set_num_threads(16)
    model, run, cmp, task = MODELS[key]()
    model.eval()
    res = dict(model=key, task=task)
    with torch.no_grad():
        run(model)
        t = time.perf_counter()
        ref = run(model)
        res["eager_s"] = round(time.perf_counter() - t, 3)
        with FlopCounterMode(display=False) as fc:
            run(model)
        total_flops = fc.get_total_flops()
        orig_forward = model.forward
        outs = {}
        for tag, opts in (("npu", NPU_OPTS), ("cpu8", dict(policy="cpu", precision=PRECISION))):
            torch._dynamo.reset()
            model.forward = torch.compile(orig_forward, backend="xdna", options=opts)
            t = time.perf_counter()
            run(model)                       # compiles
            res[f"{tag}_first_s"] = round(time.perf_counter() - t, 1)
            torch_xdna.stats.reset()
            t = time.perf_counter()
            outs[tag] = run(model)
            res[f"{tag}_s"] = round(time.perf_counter() - t, 3)
            s = torch_xdna.stats.as_dict()
            res[f"{tag}_gemm_calls"] = s["calls"]
            if tag == "npu":
                res["npu_flops_share"] = round(2e9 * s["gmacs"].get("npu", 0) / max(total_flops, 1), 4)
                res["npu_fallbacks"] = s["fallbacks"]
                res["npu_programs_verified"] = len(torch_xdna.runtime._verified)
            model.forward = orig_forward
    res["precision"] = PRECISION
    if PRECISION == "int8":
        res["npu_eq_cpu8_bitwise"] = same(outs["npu"], outs["cpu8"])
    else:  # bf16: the NPU's bfp16 vs the CPU route's fp32 -- close, not equal
        res["npu_vs_cpu"] = cmp(outs["cpu8"], outs["npu"])
    res["vs_fp32"] = cmp(ref, outs["npu"])
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        print("RESULT " + json.dumps(evaluate(sys.argv[2]), ensure_ascii=False), flush=True)
        sys.exit(0)
    keys = sys.argv[1:] or list(MODELS)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"model_zoo_{PRECISION}.jsonl")
    for k in keys:
        t = time.perf_counter()
        r = subprocess.run([sys.executable, __file__, "--one", k], capture_output=True, text=True,
                           env=dict(os.environ, HF_HUB_OFFLINE="1"))
        line = next((l[7:] for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            err = (r.stderr.strip().splitlines() or ["?"])[-1]
            line = json.dumps(dict(model=k, error=err[:400]), ensure_ascii=False)
        d = json.loads(line)
        d["wall_s"] = round(time.perf_counter() - t, 1)
        with open(out_path, "a") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(json.dumps(d, ensure_ascii=False), flush=True)
