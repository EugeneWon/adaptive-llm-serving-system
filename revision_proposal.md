# Proposal 수정 사항 (pro_1.pdf → 지금 당장 반영)

Proposal은 "할 것이다"를 설득하는 문서이므로 결과 기술은 하지 않음.
수정 범위: 설계 오류 수정 + 계획 보강 + 참고문헌 보강.

---

## 1. Section III.C — Adaptive Optimization Policy ★ 가장 중요

### 문제
현재 proposal 텍스트:
> "Low-utilization → increase batch size to improve parallelism"

이 처방은 두 가지 이유로 틀렸습니다:
1. Serving runtime은 외부 요청 batch를 마음대로 늘릴 수 없음 (SLA 위반)
2. 실제 bottleneck이 batch size 부족이 아니라 kernel launch overhead임

### 수정 후 텍스트 (미래형 유지, proposal 어조)
```
Based on our profiling methodology, we derive the following adaptive policy:

  - Kernel-overhead-bound: Apply operator fusion via torch.compile
    (reduce-overhead mode). At batch=1–4, GPU SM occupancy remains low
    (<35%) and kernel launch latency dominates per-token cost. Operator
    fusion collapses multiple small kernel launches into CUDA graph
    executions, directly targeting this bottleneck.

  - Memory-bound (batch ≥ 16, seq ≥ 256): Apply FP16 mixed precision.
    When HBM bandwidth is saturated (GPU util ≥ 90%), halving model
    weight bytes via FP16 reduces memory traffic proportionally.

  - Kernel-overhead-bound (intermediate regime): Apply torch.compile.
    Moderate GPU utilization (35–90%) still exhibits significant kernel
    launch overhead relative to weight-streaming cost.

We expect this policy to outperform any single static optimization because
each decision is conditioned on the measured runtime bottleneck rather than
applied uniformly.
```

### Policy 표 (proposal용 — "Expected" 수치)
| Regime | GPU util | Optimization | Expected benefit |
|--------|----------|--------------|------------------|
| Low-utilization | < 35% | torch.compile | Reduce kernel launch overhead |
| Kernel-overhead-bound | 35–90% | torch.compile | Fuse redundant kernel launches |
| Memory-bound | ≥ 90%, seq ≥ 256 | FP16 | Halve HBM traffic |

---

## 2. Section II — Related Work (참고문헌 보강)

현재 3개 → 최소 8개 이상으로 보강. 아래 추가:

```
[4] T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism
    and Work Partitioning," ICLR 2024.

[5] G. Yu et al., "Orca: A Distributed Serving System for Transformer-Based
    Generative Models," OSDI 2022.

[6] S. Williams et al., "Roofline: An Insightful Visual Performance Model
    for Multicore Architectures," CACM 2009.

[7] R. Y. Aminabadi et al., "DeepSpeed-Inference: Enabling Efficient
    Inference of Transformer Models at Unprecedented Scale," SC 2022.

[8] Y. Leviathan et al., "Fast Inference from Transformers via Speculative
    Decoding," ICML 2023.

[9] Y. Sheng et al., "FlexGen: High-Throughput Generative Inference of
    Large Language Models with a Single GPU," ICML 2023.
```

관련 문장도 추가:
```
The roofline model [6] provides a principled framework for characterizing
memory-bound vs. compute-bound behavior, which we adopt to classify LLM
inference regimes. Unlike prior work that applies optimizations statically
[4, 7], our approach selects optimizations based on measured runtime regime.
Speculative decoding [8] and memory-efficient serving [5] are complementary
techniques that could be integrated into our adaptive framework.
```

---

## 3. Section III.A — 레짐 분류 기준 명시

현재 proposal에 세 레짐이 정의되어 있지만 분류 기준(threshold)이 없음.
아래 추가:

```
We classify execution regimes based on measured GPU utilization:
  - Low-utilization:        gpu_util < 35%
  - Kernel-overhead-bound:  35% ≤ gpu_util < 90%
  - Memory-bound:           gpu_util ≥ 90% with seq_len ≥ 256

These thresholds will be validated empirically by sweeping batch sizes
(1–32) and sequence lengths (32–512) and correlating GPU utilization
with the dominant bottleneck metric (estimated HBM bandwidth vs.
kernel launch count).
```

---

## 4. Section IV — Experimental Plan 확장

현재 plan에 ablation과 generalization이 없음. 아래 추가:

```
IV.D Ablation Study
  To validate the necessity of regime-based policy selection, we conduct:
  (1) Wrong-optimization ablation: Apply each optimization in the
      incorrect regime and measure throughput regression.
  (2) Rule-removal ablation: Compare static policies (always-compile,
      always-fp16) against the adaptive policy across all workload points.
      A correct adaptive policy should match or exceed the best static
      policy in every regime.
  (3) Prompt-diversity check: Repeat measurements with five prompt
      categories (code, QA, story, chat, math) to verify that regime
      boundaries and speedup magnitudes are prompt-invariant.

IV.E Generalization
  We evaluate across three model families to confirm architecture-
  independence: GPT-2, OPT (facebook/opt-125m, opt-350m), and
  BLOOM (bigscience/bloom-560m). Regime boundaries and speedup
  magnitudes should be consistent across architectures sharing
  similar decode-phase memory access patterns.
```

---

## 수정하지 않아도 되는 것

| 항목 | 이유 |
|------|------|
| Abstract의 수치 ("up to X%") | 아직 실험 전 — final paper에서 추가 |
| Section V "Expected Contributions" | Proposal이므로 "Expected" 유지 |
| Conclusion의 발견 사항 | 아직 결과 없음 |
| IV를 "Experiments"로 rename | Proposal이므로 "Experimental Plan" 유지 |
| Figure (실험 결과) | 실험 후 final paper에서 추가 |
