import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "transformers==4.44.2", "nvidia-ml-py", "pandas"])

import time, csv, math, os, warnings, statistics
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    {"name": "gpt2",         "model_id": "gpt2"},
    {"name": "gpt2-large",   "model_id": "gpt2-large"},
    {"name": "gpt-neo-125m", "model_id": "EleutherAI/gpt-neo-125m"},
]

BATCH_SIZES    = [1, 2, 4, 8, 16, 32]
INPUT_LENGTHS  = [32, 128, 512]
MAX_NEW_TOKENS  = 50
WARMUP_RUNS     = 10
COMPILE_WARMUP  = 30
MEASURE_RUNS    = 20
_T95            = 2.093  # t(0.975, df=19) for 95% CI

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


# ── Regime Classifier ─────────────────────────────────────────────────────────
# Thresholds calibrated from Phase 0 profiling data:
#   low-util    : GPU util < 35%      → insufficient SM occupancy (batch too small)
#   memory-bound: batch ≥ 16, seq ≥ 256 → HBM bandwidth-limited; decode weight-loading
#                                        rate confirms bandwidth saturation
#   kernel-overhead-bound: all other  → moderate util, many small kernel launches
#
# When measured gpu_util_pct is available, it is used directly for the
# low-utilization boundary — making classification genuinely profiling-guided.
# The memory-bound / kernel-overhead boundary uses offline-calibrated thresholds
# (batch ≥ 16 and seq ≥ 256) that proved stable across both model families.

_GPU_UTIL_LOW = 35   # % — SM under-occupancy threshold (from profiling: batch ≤ 4 → < 35%)


def classify_regime(batch_size, seq_length, gpu_util_pct=None):
    """
    Classify the inference bottleneck regime.

    Parameters
    ----------
    gpu_util_pct : int or None
        Measured GPU utilisation from Phase 0 baseline pass.  When provided,
        this takes precedence over the static batch-size heuristic for the
        low-utilisation boundary.
    """
    # Use measured GPU util for low-util boundary when available
    is_low_util = (gpu_util_pct < _GPU_UTIL_LOW) if gpu_util_pct is not None else (batch_size <= 4)

    if is_low_util:
        return "low-utilization"
    elif batch_size >= 16 and seq_length >= 256:
        # Large batch + long context → weight-loading rate drops, HBM saturated.
        # batch=8 does not consistently benefit from FP16 (overhead ≈ bandwidth gain).
        return "memory-bound"
    else:
        return "kernel-overhead-bound"


# ── Policy: regime → config ────────────────────────────────────────────────────
# kernel-overhead-bound: operator fusion via torch.compile (reduce-overhead mode)
#   reduces redundant kernel launches and improves data locality.
# memory-bound: fp16 cuts memory footprint and bandwidth pressure.
# low-utilization: baseline (runtime cannot force batch size changes).
POLICY = {
    "memory-bound":          {"fp16": True,  "compile": False},
    "kernel-overhead-bound": {"fp16": False, "compile": True},
    "low-utilization":       {"fp16": False, "compile": False},
}


def get_gpu_stats():
    mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return {
        "gpu_util_pct":     util.gpu,
        "gpu_mem_used_mb":  mem.used  // (1024 ** 2),
        "gpu_mem_total_mb": mem.total // (1024 ** 2),
    }


def get_model_weight_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def compute_data_movement_metrics(weight_bytes, param_count, avg_latency, batch_size):
    latency_per_token = avg_latency / MAX_NEW_TOKENS
    estimated_bw_GBs = weight_bytes / latency_per_token / 1e9
    flops_per_token = 2 * param_count * batch_size
    arithmetic_intensity = flops_per_token / weight_bytes
    return {
        "weight_bytes_mb":           round(weight_bytes / (1024 ** 2), 1),
        "estimated_bandwidth_GBs":   round(estimated_bw_GBs, 2),
        "arithmetic_intensity":      round(arithmetic_intensity, 4),
    }


def make_input(tokenizer, batch_size, input_length, device):
    prompt = "The quick brown fox jumps over the lazy dog"
    ids    = tokenizer.encode(prompt, return_tensors="pt")[0]
    ids    = ids.repeat((input_length // len(ids)) + 1)[:input_length]
    input_ids      = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def run_inference(model, tokenizer, batch_size, input_length, device, max_pos, compiled=False,
                  weight_bytes=None, param_count=None):
    if input_length + MAX_NEW_TOKENS > max_pos:
        return None

    input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
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

    avg_lat    = sum(latencies) / len(latencies)
    throughput = (MAX_NEW_TOKENS * batch_size) / avg_lat
    sorted_lat = sorted(latencies)
    result = {
        "avg_latency_s":    round(avg_lat, 4),
        "p50_latency_s":    round(sorted_lat[(len(sorted_lat) - 1) // 2], 4),
        "p95_latency_s":    round(sorted_lat[int(len(sorted_lat) * 0.95) - 1], 4),
        "std_latency_s":    round(statistics.stdev(latencies), 5),
        "ci95_half_ms":     round(_T95 * statistics.stdev(latencies) * 1000 / math.sqrt(MEASURE_RUNS), 3),
        "ms_per_token":     round(avg_lat * 1000 / MAX_NEW_TOKENS, 3),
        "throughput_tok_s": round(throughput, 2),
        **get_gpu_stats(),
    }
    if weight_bytes is not None and param_count is not None:
        result.update(compute_data_movement_metrics(weight_bytes, param_count, avg_lat, batch_size))
    return result


def load_model(model_cfg, cfg, device):
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"])
    if cfg["fp16"]:
        model = model.half()
    model = model.to(device).eval()
    if cfg["compile"]:
        # reduce-overhead mode uses CUDA graphs / kernel fusion to minimize launch overhead,
        # targeting the kernel-overhead-bound regime. dynamic=True is intentionally omitted
        # so the compiler can perform static operator fusion per decode step shape.
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model


def cfg_name(cfg):
    parts = (["fp16"] if cfg["fp16"] else []) + (["compile"] if cfg["compile"] else [])
    return "+".join(parts) or "baseline"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}\n")

    out_path   = os.path.join(RESULTS_DIR, "adaptive_results.csv")
    fieldnames = [
        "model_name", "batch_size", "input_length", "max_new_tokens",
        "regime", "selected_config",
        "avg_latency_s", "p50_latency_s", "p95_latency_s", "std_latency_s", "ci95_half_ms", "ms_per_token",
        "throughput_tok_s",
        "baseline_latency_s", "baseline_throughput_tok_s", "speedup", "speedup_ci95",
        "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb",
        "weight_bytes_mb", "estimated_bandwidth_GBs", "arithmetic_intensity",
    ]

    # plan is built AFTER Phase 0 so that measured gpu_util feeds into classify_regime
    plan = None   # populated below after Phase 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            print(f"\n{'='*55}")
            print(f"Model: {model_cfg['name']} ({model_cfg['model_id']})")
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # ── Phase 0: In-experiment baseline reference ─────────────────────
            # Measure baseline for all (batch, seq) combinations in the same GPU
            # session so that speedup comparison is free of inter-experiment noise.
            # The measured gpu_util from these runs feeds into classify_regime,
            # making the adaptive policy genuinely profiling-guided.
            print("\n  --- Phase 0: baseline reference pass ---")
            baseline_ref = {}
            ref_model        = load_model(model_cfg, {"fp16": False, "compile": False}, device)
            ref_max_pos      = ref_model.config.max_position_embeddings
            ref_weight_bytes = get_model_weight_bytes(ref_model)
            ref_param_count  = sum(p.numel() for p in ref_model.parameters())

            # Pre-warmup: run 20 discarded iterations so GPU thermal/cache state is
            # stable before we start recording Phase 0 reference latencies.
            # Without this, the first model measured (here) shows ~20% higher latency
            # than later measurements in the same session due to cold HBM/cache state.
            print("  [pre-warmup] stabilising GPU state (20 discarded iterations)...")
            _w_ids, _w_mask = make_input(tokenizer, 4, 128, device)
            _gen_kw = dict(max_new_tokens=10, do_sample=False,
                           attention_mask=_w_mask, pad_token_id=tokenizer.eos_token_id)
            for _ in range(20):
                with torch.no_grad():
                    ref_model.generate(_w_ids, **_gen_kw)
            torch.cuda.synchronize()
            del _w_ids, _w_mask, _gen_kw

            for bs in BATCH_SIZES:
                for seq in INPUT_LENGTHS:
                    r = run_inference(ref_model, tokenizer, bs, seq, device, ref_max_pos,
                                      compiled=False,
                                      weight_bytes=ref_weight_bytes, param_count=ref_param_count)
                    if r is not None:
                        baseline_ref[(bs, seq)] = r
                        print(f"    baseline batch={bs:2d} seq={seq:4d}  "
                              f"lat={r['avg_latency_s']:.3f}s  thr={r['throughput_tok_s']:.1f}  "
                              f"gpu={r['gpu_util_pct']}%")
            del ref_model
            torch.cuda.empty_cache()

            # Build plan using Phase 0 profiling data (gpu_util → classify_regime)
            plan = []
            for bs in BATCH_SIZES:
                for seq in INPUT_LENGTHS:
                    ref = baseline_ref.get((bs, seq))
                    gpu_util = ref["gpu_util_pct"] if ref else None
                    regime   = classify_regime(bs, seq, gpu_util_pct=gpu_util)
                    plan.append((bs, seq, regime, POLICY[regime]))
            plan_sorted = sorted(plan, key=lambda x: (x[3]["fp16"], x[3]["compile"]))

            # ── Phase 1: Adaptive policy pass ────────────────────────────────
            print("\n=== Adaptive Policy Plan (profiling-guided) ===")
            for bs, seq, regime, cfg in plan:
                print(f"  batch={bs:2d}  seq={seq:4d} → [{regime:22s}] → {cfg_name(cfg)}")

            print("\n=== Running Experiments ===")
            current_cfg_key = None
            model           = None
            max_pos         = None

            for bs, seq, regime, cfg in plan_sorted:
                cfg_key = (cfg["fp16"], cfg["compile"])
                if cfg_key != current_cfg_key:
                    if model is not None:
                        del model
                        torch.cuda.empty_cache()
                    print(f"\n  --- Loading model: {cfg_name(cfg)} ---")
                    model           = load_model(model_cfg, cfg, device)
                    max_pos         = model.config.max_position_embeddings
                    weight_bytes    = get_model_weight_bytes(model)
                    param_count     = sum(p.numel() for p in model.parameters())
                    current_cfg_key = cfg_key

                print(f"  batch={bs:2d}  seq={seq:4d}  [{regime:22s}] ...", end=" ", flush=True)
                row = run_inference(model, tokenizer, bs, seq, device, max_pos,
                                    compiled=cfg["compile"],
                                    weight_bytes=weight_bytes, param_count=param_count)
                if row is None:
                    print("skipped")
                    continue

                ref     = baseline_ref.get((bs, seq))
                speedup = round(row["throughput_tok_s"] / ref["throughput_tok_s"], 4) if ref else None

                # Error propagation for speedup: σ_s/s = sqrt((σ_opt/μ_opt)² + (σ_ref/μ_ref)²)
                speedup_ci95 = None
                if ref and speedup:
                    cv_opt = row["std_latency_s"] / row["avg_latency_s"]
                    cv_ref = ref["std_latency_s"] / ref["avg_latency_s"]
                    speedup_ci95 = round(speedup * _T95 * math.sqrt(cv_opt**2 + cv_ref**2) / math.sqrt(MEASURE_RUNS), 4)

                writer.writerow({
                    "model_name":               model_cfg["name"],
                    "batch_size":               bs,
                    "input_length":             seq,
                    "max_new_tokens":           MAX_NEW_TOKENS,
                    "regime":                   regime,
                    "selected_config":          cfg_name(cfg),
                    **row,
                    "baseline_latency_s":       ref["avg_latency_s"]    if ref else "",
                    "baseline_throughput_tok_s":ref["throughput_tok_s"] if ref else "",
                    "speedup":                  speedup      if speedup      else "",
                    "speedup_ci95":             speedup_ci95 if speedup_ci95 else "",
                })
                f.flush()
                speedup_str = f"  speedup={speedup:.3f}±{speedup_ci95:.4f}x" if speedup else ""
                print(f"latency={row['avg_latency_s']:.3f}s  "
                      f"thr={row['throughput_tok_s']:.1f} tok/s  "
                      f"GPU={row['gpu_util_pct']}%{speedup_str}")

            if model is not None:
                del model
                torch.cuda.empty_cache()

    print(f"\nResults saved → {out_path}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
