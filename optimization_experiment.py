import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "transformers==4.44.2", "nvidia-ml-py", "pandas"])

import time, csv, os, warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    {"name": "gpt2",         "model_id": "gpt2"},
    {"name": "gpt-neo-125m", "model_id": "EleutherAI/gpt-neo-125m"},
]

BATCH_SIZES    = [1, 2, 4, 8, 16, 32]
INPUT_LENGTHS  = [32, 128, 512]
MAX_NEW_TOKENS  = 50
WARMUP_RUNS     = 10
COMPILE_WARMUP  = 30  # compiled graphs need extra runs to JIT-stabilize all decode shapes
MEASURE_RUNS    = 20

CONFIGS = [
    {"name": "baseline",     "fp16": False, "compile": False},
    {"name": "fp16",         "fp16": True,  "compile": False},
    {"name": "compile",      "fp16": False, "compile": True},
    {"name": "fp16+compile", "fp16": True,  "compile": True},
]

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def get_gpu_stats():
    mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return {"gpu_util_pct": util.gpu, "gpu_mem_used_mb": mem.used // (1024 ** 2)}


def make_input(tokenizer, batch_size, input_length, device):
    prompt = "The quick brown fox jumps over the lazy dog"
    ids    = tokenizer.encode(prompt, return_tensors="pt")[0]
    ids    = ids.repeat((input_length // len(ids)) + 1)[:input_length]
    input_ids      = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def run_experiment(model, tokenizer, batch_size, input_length, device, max_pos, model_cfg, opt_cfg):
    if input_length + MAX_NEW_TOKENS > max_pos:
        return None

    input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )

    n_warmup = COMPILE_WARMUP if opt_cfg["compile"] else WARMUP_RUNS
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

    avg_latency = sum(latencies) / len(latencies)
    throughput  = (MAX_NEW_TOKENS * batch_size) / avg_latency

    return {
        "model_name":       model_cfg["name"],
        "config":           opt_cfg["name"],
        "fp16":             opt_cfg["fp16"],
        "compiled":         opt_cfg["compile"],
        "batch_size":       batch_size,
        "input_length":     input_length,
        "avg_latency_s":    round(avg_latency, 4),
        "p50_latency_s":    round(sorted(latencies)[(len(latencies) - 1) // 2], 4),
        "throughput_tok_s": round(throughput, 2),
        **get_gpu_stats(),
    }


def load_model(model_cfg, opt_cfg, device):
    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"])
    if opt_cfg["fp16"]:
        model = model.half()
    model = model.to(device).eval()
    if opt_cfg["compile"]:
        # reduce-overhead mode uses CUDA graphs / kernel fusion to minimize launch overhead,
        # targeting the kernel-overhead-bound regime. dynamic=True is intentionally omitted
        # so the compiler can perform static operator fusion per decode step shape.
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}\n")

    out_path   = os.path.join(RESULTS_DIR, "optimization_results.csv")
    fieldnames = [
        "model_name", "config", "fp16", "compiled",
        "batch_size", "input_length",
        "avg_latency_s", "p50_latency_s", "throughput_tok_s",
        "gpu_util_pct", "gpu_mem_used_mb",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_cfg in MODELS:
            print(f"\n{'='*55}")
            print(f"Model: {model_cfg['name']} ({model_cfg['model_id']})")
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            for opt_cfg in CONFIGS:
                print(f"\n  --- Config: {opt_cfg['name']} ---")
                model   = load_model(model_cfg, opt_cfg, device)
                max_pos = model.config.max_position_embeddings

                for bs in BATCH_SIZES:
                    for seq_len in INPUT_LENGTHS:
                        print(f"    batch={bs:2d}  seq={seq_len:4d} ...", end=" ", flush=True)
                        row = run_experiment(
                            model, tokenizer, bs, seq_len, device,
                            max_pos, model_cfg, opt_cfg
                        )
                        if row is None:
                            print("skipped")
                            continue
                        writer.writerow(row)
                        f.flush()
                        print(f"latency={row['avg_latency_s']:.3f}s  "
                              f"thr={row['throughput_tok_s']:.1f} tok/s  "
                              f"GPU={row['gpu_util_pct']}%")

                del model
                torch.cuda.empty_cache()

    print(f"\nResults saved → {out_path}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
