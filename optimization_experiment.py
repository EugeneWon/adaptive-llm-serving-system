import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "transformers==4.44.2", "nvidia-ml-py", "pandas"])

import time, csv, os, warnings
import torch
import pynvml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

warnings.filterwarnings("ignore")

RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZES    = [1, 2, 4, 8, 16, 32]
INPUT_LENGTHS  = [32, 128, 512]   # 대표 seq만 (전체 sweep은 baseline에서 완료)
MAX_NEW_TOKENS = 50
WARMUP_RUNS    = 10
MEASURE_RUNS   = 20

CONFIGS = [
    {"name": "baseline",        "fp16": False, "compile": False},
    {"name": "fp16",            "fp16": True,  "compile": False},
    {"name": "compile",         "fp16": False, "compile": True},
    {"name": "fp16+compile",    "fp16": True,  "compile": True},
]

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def get_gpu_stats():
    mem  = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return {
        "gpu_util_pct":    util.gpu,
        "gpu_mem_used_mb": mem.used // (1024 ** 2),
    }


def make_input(tokenizer, batch_size, input_length, device):
    prompt = "The quick brown fox jumps over the lazy dog"
    ids    = tokenizer.encode(prompt, return_tensors="pt")[0]
    ids    = ids.repeat((input_length // len(ids)) + 1)[:input_length]
    input_ids      = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def run_experiment(model, tokenizer, batch_size, input_length, device, model_config):
    max_pos = 1024  # GPT-2 hard limit
    if input_length + MAX_NEW_TOKENS > max_pos:
        return None

    input_ids, attention_mask = make_input(tokenizer, batch_size, input_length, device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )

    try:
        for _ in range(WARMUP_RUNS):
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
        "config":           model_config["name"],
        "fp16":             model_config["fp16"],
        "compiled":         model_config["compile"],
        "batch_size":       batch_size,
        "input_length":     input_length,
        "avg_latency_s":    round(avg_latency, 4),
        "p50_latency_s":    round(sorted(latencies)[len(latencies) // 2], 4),
        "throughput_tok_s": round(throughput, 2),
        **get_gpu_stats(),
    }


def load_model(config, device):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if config["fp16"]:
        model = model.half()
    model = model.to(device).eval()
    if config["compile"]:
        model = torch.compile(model, dynamic=True)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}\n")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    out_path   = os.path.join(RESULTS_DIR, "optimization_results.csv")
    fieldnames = [
        "config", "fp16", "compiled", "batch_size", "input_length",
        "avg_latency_s", "p50_latency_s", "throughput_tok_s",
        "gpu_util_pct", "gpu_mem_used_mb",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cfg in CONFIGS:
            print(f"=== Config: {cfg['name']} ===")
            model = load_model(cfg, device)

            for bs in BATCH_SIZES:
                for seq_len in INPUT_LENGTHS:
                    print(f"  batch={bs:2d}  seq={seq_len:4d} ...", end=" ", flush=True)
                    row = run_experiment(model, tokenizer, bs, seq_len, device, cfg)
                    if row is None:
                        print("skipped")
                        continue
                    writer.writerow(row)
                    f.flush()
                    print(f"latency={row['avg_latency_s']:.3f}s  "
                          f"throughput={row['throughput_tok_s']:.1f} tok/s  "
                          f"GPU={row['gpu_util_pct']}%  "
                          f"mem={row['gpu_mem_used_mb']}MB")

            del model
            torch.cuda.empty_cache()
            print()

    print(f"Results saved → {out_path}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
