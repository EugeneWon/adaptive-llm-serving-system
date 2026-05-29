"""
Ablation Study for Profiling-Guided Adaptive Optimization

Three ablations are measured, each targeting a specific claim of the paper:

Ablation A — Wrong optimization per regime  ("특정 optimization 제거")
    Run all four configs at one representative (batch, seq) per regime.
    Shows that the adaptive policy's mapping is *necessary*: applying the
    wrong optimization in a regime either gives no benefit or regresses.
    Output: results/ablation_a_wrong_opt.csv

Ablation B — Rule removal  ("rule 제거")
    Compare three static baselines (always compile, always fp16, always
    baseline) against the adaptive policy across ALL (batch, seq) points.
    The adaptive policy should match the *best static* config in each regime
    and outperform any single static policy when aggregated.
    Output: results/ablation_b_no_rule.csv

Ablation C — Diverse workloads  ("다양한 workload")
    Repeat the adaptive-policy measurements on five prompt categories
    (code, QA, story, chat, math) to verify that regime boundaries and
    speedup magnitudes are prompt-invariant.
    Output: results/ablation_c_diverse.csv
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "transformers==4.44.2", "nvidia-ml-py", "accelerate", "pandas", "tqdm"])

import time, csv, math, os, warnings, statistics
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = [
    {"name": "gpt2",         "model_id": "gpt2"},
    {"name": "gpt2-large",   "model_id": "gpt2-large"},
    {"name": "gpt-neo-125m", "model_id": "EleutherAI/gpt-neo-125m"},
]

# ── Configs ───────────────────────────────────────────────────────────────────
ALL_CONFIGS = [
    {"name": "baseline",     "fp16": False, "compile": False},
    {"name": "compile",      "fp16": False, "compile": True},
    {"name": "fp16",         "fp16": True,  "compile": False},
    {"name": "fp16+compile", "fp16": True,  "compile": True},
]

STATIC_POLICIES = [          # for Ablation B
    {"name": "always-baseline", "fp16": False, "compile": False},
    {"name": "always-compile",  "fp16": False, "compile": True},
    {"name": "always-fp16",     "fp16": True,  "compile": False},
]

# Adaptive policy: regime → config name
ADAPTIVE_CONFIG = {
    "low-utilization":       "compile",
    "kernel-overhead-bound": "compile",
    "memory-bound":          "fp16",
}

# ── Regime classifier (matches adaptive_policy.py) ────────────────────────────
_GPU_UTIL_LOW          = 35
_GPU_UTIL_MEMORY_BOUND = 90

def classify_regime(batch_size, seq_length, gpu_util_pct=None):
    if gpu_util_pct is not None:
        if gpu_util_pct < _GPU_UTIL_LOW:
            return "low-utilization"
        elif gpu_util_pct >= _GPU_UTIL_MEMORY_BOUND and seq_length >= 256:
            return "memory-bound"
        return "kernel-overhead-bound"
    if batch_size <= 4:
        return "low-utilization"
    elif batch_size >= 16 and seq_length >= 256:
        return "memory-bound"
    return "kernel-overhead-bound"

# ── Representative ablation-A points (one per regime) ─────────────────────────
ABLATION_A_POINTS = [
    (1,  128, "low-utilization"),       # gpu_util ≈ 27%
    (8,  128, "kernel-overhead-bound"), # gpu_util ≈ 39%
    (16, 512, "memory-bound"),          # gpu_util ≈ 96%
]

# ── Full sweep for Ablation B ─────────────────────────────────────────────────
ABLATION_B_BATCH_SIZES   = [1, 4, 8, 16, 32]
ABLATION_B_INPUT_LENGTHS = [32, 128, 512]

# ── Diverse prompts for Ablation C ───────────────────────────────────────────
# Five prompt categories that cover typical LLM usage patterns.
# Each is truncated / repeated to the target input_length at runtime.
DIVERSE_PROMPTS = {
    "code":  "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
    "qa":    "Question: What is the capital of France? Answer: Paris. Question: Who wrote Hamlet? Answer: Shakespeare. Question: ",
    "story": "Once upon a time in a land far away, a young wizard discovered a hidden library filled with ancient spell books and forgotten knowledge. ",
    "chat":  "User: Hello, how are you today? Assistant: I'm doing well, thank you for asking! User: Can you help me with something? Assistant: ",
    "math":  "Let x = 2 and y = 3. Then x + y = 5, x * y = 6, x^2 = 4, y^2 = 9. The sum of squares is x^2 + y^2 = ",
}
ABLATION_C_BATCH    = 8     # representative middle-ground batch
ABLATION_C_SEQ      = 128   # representative sequence length
ABLATION_C_CONFIGS  = [
    {"name": "baseline", "fp16": False, "compile": False},
    {"name": "compile",  "fp16": False, "compile": True},
    {"name": "fp16",     "fp16": True,  "compile": False},
]

# ── Timing constants ──────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 50
WARMUP_RUNS    = 10
COMPILE_WARMUP = 30
MEASURE_RUNS   = 20
_T95           = 2.093   # t(0.975, df=19) for 95% CI

pynvml.nvmlInit()
_visible  = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
_gpu_idx  = int(_visible.split(",")[0]) if _visible.strip() else 0
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(_gpu_idx)


# ── Utilities ─────────────────────────────────────────────────────────────────
def get_gpu_stats():
    mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return {"gpu_util_pct": util.gpu, "gpu_mem_used_mb": mem.used // (1024 ** 2)}


def make_input_from_prompt(tokenizer, prompt, batch_size, input_length, device):
    ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    ids = ids.repeat((input_length // len(ids)) + 1)[:input_length]
    input_ids      = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def make_input(tokenizer, batch_size, input_length, device):
    prompt = "The quick brown fox jumps over the lazy dog"
    return make_input_from_prompt(tokenizer, prompt, batch_size, input_length, device)


def run_timed(model, tokenizer, batch_size, input_length, device, compiled,
              prompt=None):
    if prompt:
        input_ids, attention_mask = make_input_from_prompt(
            tokenizer, prompt, batch_size, input_length, device)
    else:
        input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id,
    )
    n_warmup = COMPILE_WARMUP if compiled else WARMUP_RUNS
    try:
        for _ in range(n_warmup):
            with torch.no_grad():
                model.generate(input_ids, **gen_kwargs)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(MEASURE_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(input_ids, **gen_kwargs)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    avg = sum(latencies) / len(latencies)
    std = statistics.stdev(latencies)
    thr = (MAX_NEW_TOKENS * batch_size) / avg
    return {
        "avg_latency_s":    round(avg, 4),
        "std_latency_s":    round(std, 5),
        "ci95_half_ms":     round(_T95 * std * 1000 / math.sqrt(MEASURE_RUNS), 3),
        "ms_per_token":     round(avg * 1000 / MAX_NEW_TOKENS, 3),
        "throughput_tok_s": round(thr, 2),
        **get_gpu_stats(),
    }


def load_model(model_cfg, opt_cfg, device):
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"])
    if opt_cfg["fp16"]:
        model = model.half()
    model = model.to(device).eval()
    if opt_cfg["compile"]:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model


def speedup_and_ci(opt_r, base_r, _t95=_T95, n=MEASURE_RUNS):
    sp = round(opt_r["throughput_tok_s"] / base_r["throughput_tok_s"], 4)
    cv_opt = opt_r["std_latency_s"] / opt_r["avg_latency_s"]
    cv_ref = base_r["std_latency_s"] / base_r["avg_latency_s"]
    ci = round(sp * _t95 * math.sqrt(cv_opt**2 + cv_ref**2) / math.sqrt(n), 4)
    return sp, ci


# ═════════════════════════════════════════════════════════════════════════════
# Ablation A — Wrong optimization per regime
# ═════════════════════════════════════════════════════════════════════════════
def run_ablation_a(device):
    print("\n" + "="*65)
    print("ABLATION A: Wrong optimization per regime")
    print("="*65)

    out_path   = os.path.join(RESULTS_DIR, "ablation_a_wrong_opt.csv")
    fieldnames = [
        "model_name", "batch_size", "input_length", "regime",
        "config", "fp16", "compiled",
        "throughput_tok_s", "avg_latency_s", "std_latency_s", "ci95_half_ms",
        "ms_per_token", "gpu_util_pct", "gpu_mem_used_mb",
        "baseline_throughput_tok_s", "speedup", "speedup_ci95",
        "is_adaptive_choice",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            print(f"\nModel: {model_cfg['name']}")
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            for bs, seq, regime in ABLATION_A_POINTS:
                print(f"\n  Regime: [{regime}]  batch={bs}  seq={seq}")

                # In-session baseline
                base_model = load_model(model_cfg, {"fp16": False, "compile": False}, device)
                max_pos    = base_model.config.max_position_embeddings
                if seq + MAX_NEW_TOKENS > max_pos:
                    print("    exceeds max context — skipped")
                    del base_model; torch.cuda.empty_cache(); continue

                print(f"    [baseline      ] ...", end=" ", flush=True)
                base_r = run_timed(base_model, tokenizer, bs, seq, device, compiled=False)
                del base_model; torch.cuda.empty_cache()
                if base_r is None:
                    print("OOM"); continue
                print(f"thr={base_r['throughput_tok_s']:.1f}  GPU={base_r['gpu_util_pct']}%")

                writer.writerow({
                    "model_name": model_cfg["name"], "batch_size": bs,
                    "input_length": seq, "regime": regime,
                    "config": "baseline", "fp16": False, "compiled": False,
                    **base_r,
                    "baseline_throughput_tok_s": base_r["throughput_tok_s"],
                    "speedup": 1.0, "speedup_ci95": 0.0,
                    "is_adaptive_choice": 0,
                })
                f.flush()

                for opt_cfg in ALL_CONFIGS:
                    if opt_cfg["name"] == "baseline":
                        continue
                    print(f"    [{opt_cfg['name']:13s}] ...", end=" ", flush=True)
                    model = load_model(model_cfg, opt_cfg, device)
                    r     = run_timed(model, tokenizer, bs, seq, device, compiled=opt_cfg["compile"])
                    del model; torch.cuda.empty_cache()
                    if r is None:
                        print("OOM"); continue

                    sp, ci  = speedup_and_ci(r, base_r)
                    is_adap = int(ADAPTIVE_CONFIG.get(regime) == opt_cfg["name"])
                    writer.writerow({
                        "model_name": model_cfg["name"], "batch_size": bs,
                        "input_length": seq, "regime": regime,
                        "config": opt_cfg["name"],
                        "fp16": opt_cfg["fp16"], "compiled": opt_cfg["compile"],
                        **r,
                        "baseline_throughput_tok_s": base_r["throughput_tok_s"],
                        "speedup": sp, "speedup_ci95": ci,
                        "is_adaptive_choice": is_adap,
                    })
                    f.flush()
                    tag = " ← adaptive" if is_adap else ""
                    print(f"thr={r['throughput_tok_s']:.1f}  speedup={sp:.3f}±{ci:.4f}{tag}")

    print(f"\n  → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Ablation B — Rule removal (static policy vs adaptive)
# ═════════════════════════════════════════════════════════════════════════════
def run_ablation_b(device):
    print("\n" + "="*65)
    print("ABLATION B: Rule removal — static vs adaptive policy")
    print("  Static policies: always-baseline, always-compile, always-fp16")
    print("  Adaptive policy: compile for low-util/kernel-overhead, fp16 for memory-bound")
    print("="*65)

    out_path   = os.path.join(RESULTS_DIR, "ablation_b_no_rule.csv")
    fieldnames = [
        "model_name", "batch_size", "input_length", "regime",
        "policy", "config_used",
        "throughput_tok_s", "avg_latency_s", "std_latency_s", "ci95_half_ms",
        "ms_per_token", "gpu_util_pct", "gpu_mem_used_mb",
        "baseline_throughput_tok_s", "speedup", "speedup_ci95",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            print(f"\nModel: {model_cfg['name']}")
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Phase 0: measure in-session baseline for all (batch, seq)
            print("  Phase 0: in-session baseline reference...")
            baseline_ref = {}
            ref_model = load_model(model_cfg, {"fp16": False, "compile": False}, device)
            max_pos   = ref_model.config.max_position_embeddings
            for bs in ABLATION_B_BATCH_SIZES:
                for seq in ABLATION_B_INPUT_LENGTHS:
                    if seq + MAX_NEW_TOKENS > max_pos:
                        continue
                    r = run_timed(ref_model, tokenizer, bs, seq, device, compiled=False)
                    if r:
                        baseline_ref[(bs, seq)] = r
                        gpu_u = r["gpu_util_pct"]
                        regime = classify_regime(bs, seq, gpu_util_pct=gpu_u)
                        print(f"    batch={bs:2d} seq={seq:3d}  [{regime:22s}]  "
                              f"thr={r['throughput_tok_s']:.1f}  GPU={gpu_u}%")
            del ref_model; torch.cuda.empty_cache()

            # Static policies
            for static_cfg in STATIC_POLICIES:
                print(f"\n  Policy: {static_cfg['name']}")
                model = load_model(model_cfg, static_cfg, device)
                for bs in ABLATION_B_BATCH_SIZES:
                    for seq in ABLATION_B_INPUT_LENGTHS:
                        base_r = baseline_ref.get((bs, seq))
                        if base_r is None:
                            continue
                        print(f"    batch={bs:2d} seq={seq:3d} ...", end=" ", flush=True)
                        r = run_timed(model, tokenizer, bs, seq, device,
                                      compiled=static_cfg["compile"])
                        if r is None:
                            print("OOM"); continue
                        sp, ci = speedup_and_ci(r, base_r)
                        regime = classify_regime(bs, seq,
                                                 gpu_util_pct=base_r["gpu_util_pct"])
                        writer.writerow({
                            "model_name": model_cfg["name"],
                            "batch_size": bs, "input_length": seq,
                            "regime": regime,
                            "policy": static_cfg["name"],
                            "config_used": static_cfg["name"].replace("always-", ""),
                            **r,
                            "baseline_throughput_tok_s": base_r["throughput_tok_s"],
                            "speedup": sp, "speedup_ci95": ci,
                        })
                        f.flush()
                        print(f"thr={r['throughput_tok_s']:.1f}  speedup={sp:.3f}")
                del model; torch.cuda.empty_cache()

            # Adaptive policy — pick config per (batch, seq) using baseline regime
            # Pre-load all configs used by the adaptive policy so that measurement
            # is not contaminated by model-reload and compile-warmup overhead.
            # This matches always-compile / always-fp16 which stay loaded throughout.
            print(f"\n  Policy: adaptive")
            adaptive_cfgs = {name: next(c for c in ALL_CONFIGS if c["name"] == name)
                             for name in set(ADAPTIVE_CONFIG.values())}
            preloaded = {}
            for cfg_name, opt_cfg in adaptive_cfgs.items():
                print(f"  [pre-load] {cfg_name} ...", end=" ", flush=True)
                preloaded[cfg_name] = load_model(model_cfg, opt_cfg, device)
                print("ready")

            for bs in ABLATION_B_BATCH_SIZES:
                for seq in ABLATION_B_INPUT_LENGTHS:
                    base_r = baseline_ref.get((bs, seq))
                    if base_r is None:
                        continue
                    regime   = classify_regime(bs, seq,
                                               gpu_util_pct=base_r["gpu_util_pct"])
                    cfg_name = ADAPTIVE_CONFIG[regime]
                    opt_cfg  = adaptive_cfgs[cfg_name]
                    model    = preloaded[cfg_name]

                    print(f"    batch={bs:2d} seq={seq:3d}  [{regime:22s}]→{cfg_name} ...",
                          end=" ", flush=True)
                    r = run_timed(model, tokenizer, bs, seq, device,
                                  compiled=opt_cfg["compile"])
                    if r is None:
                        print("OOM"); continue
                    sp, ci = speedup_and_ci(r, base_r)
                    writer.writerow({
                        "model_name": model_cfg["name"],
                        "batch_size": bs, "input_length": seq,
                        "regime": regime,
                        "policy": "adaptive",
                        "config_used": cfg_name,
                        **r,
                        "baseline_throughput_tok_s": base_r["throughput_tok_s"],
                        "speedup": sp, "speedup_ci95": ci,
                    })
                    f.flush()
                    print(f"thr={r['throughput_tok_s']:.1f}  speedup={sp:.3f}")

            for m in preloaded.values():
                del m
            torch.cuda.empty_cache()

    print(f"\n  → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Ablation C — Diverse workloads
# ═════════════════════════════════════════════════════════════════════════════
def run_ablation_c(device):
    print("\n" + "="*65)
    print("ABLATION C: Diverse workloads — prompt-invariance check")
    print(f"  Prompts: {list(DIVERSE_PROMPTS.keys())}")
    print(f"  batch={ABLATION_C_BATCH}  seq={ABLATION_C_SEQ}")
    print("="*65)

    out_path   = os.path.join(RESULTS_DIR, "ablation_c_diverse.csv")
    fieldnames = [
        "model_name", "prompt_category", "batch_size", "input_length",
        "config", "fp16", "compiled",
        "throughput_tok_s", "avg_latency_s", "std_latency_s", "ci95_half_ms",
        "ms_per_token", "gpu_util_pct", "gpu_mem_used_mb",
        "baseline_throughput_tok_s", "speedup", "speedup_ci95",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            print(f"\nModel: {model_cfg['name']}")
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            for prompt_cat, prompt_text in DIVERSE_PROMPTS.items():
                print(f"\n  Prompt: [{prompt_cat}]")

                # In-session baseline for this prompt
                base_model = load_model(model_cfg, {"fp16": False, "compile": False}, device)
                max_pos    = base_model.config.max_position_embeddings
                if ABLATION_C_SEQ + MAX_NEW_TOKENS > max_pos:
                    del base_model; torch.cuda.empty_cache(); continue

                print(f"    [baseline   ] ...", end=" ", flush=True)
                base_r = run_timed(base_model, tokenizer,
                                   ABLATION_C_BATCH, ABLATION_C_SEQ, device,
                                   compiled=False, prompt=prompt_text)
                del base_model; torch.cuda.empty_cache()
                if base_r is None:
                    print("OOM"); continue
                print(f"thr={base_r['throughput_tok_s']:.1f}  GPU={base_r['gpu_util_pct']}%")

                writer.writerow({
                    "model_name": model_cfg["name"], "prompt_category": prompt_cat,
                    "batch_size": ABLATION_C_BATCH,
                    "input_length": ABLATION_C_SEQ,
                    "config": "baseline", "fp16": False, "compiled": False,
                    **base_r,
                    "baseline_throughput_tok_s": base_r["throughput_tok_s"],
                    "speedup": 1.0, "speedup_ci95": 0.0,
                })
                f.flush()

                for opt_cfg in ABLATION_C_CONFIGS:
                    if opt_cfg["name"] == "baseline":
                        continue
                    print(f"    [{opt_cfg['name']:10s}] ...", end=" ", flush=True)
                    model = load_model(model_cfg, opt_cfg, device)
                    r     = run_timed(model, tokenizer,
                                      ABLATION_C_BATCH, ABLATION_C_SEQ, device,
                                      compiled=opt_cfg["compile"], prompt=prompt_text)
                    del model; torch.cuda.empty_cache()
                    if r is None:
                        print("OOM"); continue
                    sp, ci = speedup_and_ci(r, base_r)
                    writer.writerow({
                        "model_name": model_cfg["name"], "prompt_category": prompt_cat,
                        "batch_size": ABLATION_C_BATCH,
                        "input_length": ABLATION_C_SEQ,
                        "config": opt_cfg["name"],
                        "fp16": opt_cfg["fp16"], "compiled": opt_cfg["compile"],
                        **r,
                        "baseline_throughput_tok_s": base_r["throughput_tok_s"],
                        "speedup": sp, "speedup_ci95": ci,
                    })
                    f.flush()
                    print(f"thr={r['throughput_tok_s']:.1f}  speedup={sp:.3f}±{ci:.4f}")

    print(f"\n  → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    print(f"Device: {device} | GPU: {gpu_name}\n")

    run_ablation_a(device)   # wrong optimization per regime
    run_ablation_b(device)   # rule removal (static vs adaptive)
    run_ablation_c(device)   # diverse workloads

    pynvml.nvmlShutdown()
    print("\nAll ablation experiments complete.")


if __name__ == "__main__":
    main()
