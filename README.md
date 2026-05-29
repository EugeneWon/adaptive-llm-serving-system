# Profiling-Guided Adaptive Optimization for Memory-Bound LLM Inference via Data Movement Analysis

> Graduate-level AI systems project focused on analyzing and optimizing **memory-bound LLM inference** through runtime profiling and workload-aware adaptive optimization.
>
> This project investigates how modern LLM inference workloads become bottlenecked by **memory bandwidth** and **data movement**, and proposes adaptive optimization strategies based on runtime behavior analysis.

---

# Overview

Large Language Model (LLM) inference is increasingly limited not by raw computation, but by **memory access overhead** and **data movement cost**.

In autoregressive decoding:

- model weights are repeatedly loaded from memory
- KV cache continuously grows
- arithmetic intensity remains low
- GPU compute units often stay underutilized

As a result, practical LLM inference becomes fundamentally **memory-bound**.

This project focuses on analyzing those bottlenecks from a systems perspective and developing a lightweight adaptive optimization framework that dynamically selects optimization strategies according to workload characteristics.

---

# Project Goals

This project aims to:

- analyze memory-bound behavior in LLM inference
- profile runtime bottlenecks under varying workloads
- understand how data movement affects inference efficiency
- evaluate practical optimization techniques
- design adaptive optimization policies based on runtime conditions

---

# Key Features

## Runtime Profiling System

The framework collects runtime metrics including:

- latency (avg, p50, p95, std)
- throughput (tokens/sec)
- ms per token
- GPU utilization
- memory usage (MB)

using PyTorch with CUDA synchronization and `pynvml` for GPU monitoring.

---

## Bottleneck Characterization

The system classifies inference behavior into three regimes based on batch size and sequence length:

| Regime | Condition (profiling-guided) | Condition (static fallback) | Description |
|---|---|---|---|
| Low-utilization | GPU util < 35% | batch ≤ 4 | Insufficient parallelism, low GPU utilization |
| Kernel-overhead-bound | GPU util < 90% or seq < 256 | otherwise | Many small kernel launches, moderate utilization |
| Memory-bound | GPU util ≥ 90% and seq ≥ 256 | batch ≥ 16 and seq ≥ 256 | Bandwidth-limited, high data movement cost |

---

## Adaptive Optimization Framework

Based on the classified regime, the system selects an optimization strategy:

| Regime | Selected Optimization | Rationale |
|---|---|---|
| Kernel-overhead-bound | `torch.compile` (reduce-overhead mode) | Kernel fusion reduces redundant launches |
| Memory-bound | FP16 mixed precision | Cuts memory footprint and bandwidth pressure |
| Low-utilization | `torch.compile` (reduce-overhead mode) | Kernel launch overhead dominates at small batch sizes; operator fusion reduces redundant launches and delivers the largest per-regime speedup (+25–28% throughput) |

Four optimization configurations are benchmarked:

| Config | FP16 | torch.compile |
|---|---|---|
| baseline | ✗ | ✗ |
| fp16 | ✓ | ✗ |
| compile | ✗ | ✓ |
| fp16+compile | ✓ | ✓ |

Unlike static optimization pipelines, this framework adapts policies according to workload-dependent bottlenecks.

---

# Motivation

Modern AI workloads are rapidly shifting toward a **memory-centric computing paradigm**.

Although GPUs provide massive computational throughput, practical LLM inference often suffers from:

- low arithmetic intensity
- excessive memory traffic
- inefficient GPU utilization
- latency-sensitive small-batch execution

This project explores how:

- workload characteristics influence bottlenecks
- optimization methods alter effective data movement
- adaptive strategies improve inference efficiency

from a system-level perspective.

---

# System Workflow

```text
LLM Inference Request
          ↓
Runtime Profiling
          ↓
Bottleneck Analysis
          ↓
Workload Classification (regime)
          ↓
Adaptive Optimization Selection
          ↓
Optimized Inference Execution
```

---

# Tech Stack

## Frameworks

- Python
- PyTorch
- HuggingFace Transformers

## Models

- GPT-2 (124M parameters, max context: 1024)
- GPT-2 Large (762M parameters, max context: 1024)
- GPT-Neo-125M (EleutherAI, max context: 2048)

## Optimization

- `torch.compile` (reduce-overhead mode, CUDA graph-based kernel fusion)
- FP16 mixed precision (`.half()`)
- workload-aware adaptive policy

## Profiling & Analysis

- `pynvml` — GPU utilization and memory monitoring
- CUDA synchronization-based latency measurement
- Jupyter notebook for result visualization

## Environment

- Linux
- CUDA
- Python 3.x

---

# Experiments

Experiments are run across all combinations of the following:

| Parameter | Values |
|---|---|
| Models | GPT-2, GPT-Neo-125M |
| Batch sizes | 1, 2, 4, 8, 16, 32 |
| Input lengths | 32, 64, 128, 256, 512 (baseline) / 32, 128, 512 (optimization, adaptive) |
| Max new tokens | 50 |
| Warmup runs | 10 (30 for compiled configs) |
| Measure runs | 20 |

Speedup in the adaptive experiment is computed relative to a same-session baseline reference to eliminate inter-experiment GPU noise.

Results are saved to `results/` as CSV files and visualized as figures.

---

# How to Run

```bash
# 1. Install dependencies
bash setup.sh

# 2. Run baseline profiling
python baseline_inference.py

# 3. Run optimization comparison (baseline / fp16 / compile / fp16+compile)
python optimization_experiment.py

# 4. Run adaptive policy experiment
python adaptive_policy.py

# 5. Plot results
python plot_baseline.py
python plot_optimization.py
python plot_adaptive.py
```

> **Note:** Scripts write results to `/workspace/results/` by default (container path). Adjust `RESULTS_DIR` in each script if running locally.

---

# Repository Structure

```text
.
├── baseline_inference.py       # Baseline profiling across batch sizes and input lengths
├── optimization_experiment.py  # Comparison of 4 optimization configs
├── adaptive_policy.py          # Workload-aware adaptive optimization with regime classifier
├── plot_baseline.py            # Visualization for baseline results
├── plot_optimization.py        # Visualization for optimization results
├── plot_adaptive.py            # Visualization for adaptive policy results
├── results_analysis.ipynb      # Notebook for result analysis
├── setup.sh                    # Dependency installation script
├── results/
│   ├── baseline_results.csv
│   ├── optimization_results.csv
│   ├── adaptive_results.csv
│   └── figures/                # Generated plots (PNG)
└── README.md
```

---

# Future Extensions

Potential future research directions include:

- KV cache optimization
- memory-aware scheduling
- dynamic batching
- adaptive compilation
- memory-efficient serving
- hardware-aware AI systems
- SRAM/HBM-aware optimization
- multi-GPU inference optimization

---

# Author

**Eugene Won**  
Information Coding and Processing Lab  
Department of Electronic & Electrical Engineering, AI Software Engineering  
Ewha Womans University  
Seoul, Republic of Korea  
eugenewon12@ewhain.net

---

# References

[1] PyTorch Team, "TorchDynamo and TorchInductor," 2023.  
[2] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS, 2022.  
[3] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP, 2023.
