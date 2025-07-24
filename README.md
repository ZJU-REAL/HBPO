# Hierarchical Budget Policy Optimization

**A Reinforcement Learning Framework for Adaptive Reasoning Efficiency**

[![Paper](https://img.shields.io/badge/arXiv-2507.15844-b31b1b.svg)](http://arxiv.org/abs/2507.15844)
[![Discussion](https://img.shields.io/badge/alphaXiv-Discussion-blue)](https://www.alphaxiv.org/abs/2507.15844)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/zju-real/HBPO)

## Abstract

Modern reasoning models suffer from computational inefficiency, generating unnecessarily verbose explanations regardless of problem complexity. While these models can solve complex mathematical proofs requiring thousands of tokens, they often apply the same extensive reasoning to simple arithmetic problems.

**Hierarchical Budget Policy Optimization (HBPO)** solves this fundamental challenge by teaching models to adapt their reasoning depth to problem complexity. Through structured exploration across multiple token budgets, HBPO enables models to automatically allocate computational resources—using concise reasoning for simple problems and extended chains for complex tasks.

**Key Achievement**: 60.6% token reduction with 3.14% accuracy improvement across mathematical reasoning benchmarks.

## Problem Statement

Current efficient reasoning approaches face two critical limitations:

1. **Exploration Space Collapse**: Length penalties systematically bias models away from necessary long reasoning paths during training
2. **Uniform Resource Allocation**: Static constraints fail to capture the heterogeneous nature of reasoning requirements across different problem types

## Method Overview

![Method](figures/method.pdf)

HBPO introduces a hierarchical training framework with three core components:

### 1. Hierarchical Budget Exploration
- Partitions rollout samples into subgroups with distinct token budgets (512, 1024, 2048, 2560)
- Maintains exploration diversity throughout training
- Prevents systematic degradation of reasoning capabilities

### 2. Differentiated Reward Mechanisms  
- Budget-specific piecewise reward functions
- Monotonically non-decreasing rewards within allocated budgets
- Deviation penalties for responses exceeding budget constraints

### 3. Emergent Adaptive Behavior
- Models learn to recognize problem complexity indicators
- Automatic computational effort adjustment without external control
- Natural correspondence between task requirements and resource allocation

## Experimental Results

### Main Performance Metrics

**Reasoning Performance**:

| Method | GSM8K | MATH500 | Olympiad | AIME25 | Average |
|--------|-------|---------|----------|---------|---------|
| **Accuracy (%)** |
| Baseline | 82.3 | 81.6 | 42.3 | 18.9 | 56.3 |
| HBPO | **85.3** | **81.0** | **45.2** | **31.1** | **59.4** |
| **Token Usage** |
| Baseline | 1,111 | 4,696 | 10,225 | 15,651 | 7,921 |
| HBPO | **408** | **1,948** | **4,403** | **4,717** | **3,120** |

### Comparative Analysis

| Method | Strategy | Avg Accuracy | Avg Tokens | Trade-off |
|--------|----------|--------------|------------|-----------|
| TLMRE | Length penalty | 49.5% | 4,162 | -6.8% accuracy |
| AdaptThink | Binary selection | 56.6% | 2,838 | Same accuracy, fewer tokens |
| L1-Max | Explicit control | 55.6% | 3,142 | -0.7% accuracy |
| **HBPO** | **Hierarchical exploration** | **59.4%** | **3,120** | **+3.14% accuracy, -60.6% tokens** |

### Adaptive Reasoning Behavior

HBPO demonstrates genuine adaptability in token allocation:

- **GSM8K (Basic Math)**: 408 tokens on average
- **MATH500 (Intermediate)**: 1,948 tokens on average  
- **Olympiad (Advanced)**: 4,403 tokens on average
- **AIME25 (Competition)**: 4,717 tokens on average

Unlike existing methods that maintain uniform token usage, HBPO naturally scales computational effort with problem complexity.

## Installation & Usage

### Requirements

```bash
conda create -n hbpo python=3.10
conda activate hbpo
pip install -e .
```

### Training

Execute HBPO training with hierarchical budget exploration:

```bash
bash examples/grpo_trainer/run_qwen2-7b_deepscale.sh
```

### Evaluation

Setup evaluation environment:

```bash
conda create -n eval_env python=3.10
conda activate eval_env

git clone https://github.com/NovaSky-AI/SkyThought.git
cd SkyThought && pip install -e .
```

Run benchmark evaluation:

```bash
skythought evaluate \
  --model <model_path> \
  --task math500 \
  --backend vllm \
  --sampling-params temperature=0.6,top_p=0.95,max_tokens=32768 \
  --n 1 \
  --batch-size 128 \
  --result-dir results/
```

## Technical Implementation

### Hierarchical Sampling Strategy

For each training query, HBPO generates responses across multiple budget constraints:

```python
budgets = [512, 1024, 2048, 2560]  # Token limits per subgroup
prompts = [f"I will answer within {b} tokens" for b in budgets]
```

### Reward Function Design

Piecewise reward structure balancing exploration and efficiency:

```python
def hierarchical_reward(correctness, length, budget):
    if correctness and length <= budget:
        return min(exploration_reward(length), budget_reward(budget))
    elif correctness:
        return apply_deviation_penalty(length, budget)
    else:
        return 0
```

### Policy Optimization

Advantage computation incorporates both intra-subgroup and inter-subgroup comparisons:

- **Intra-subgroup**: Compare responses within same budget constraint
- **Inter-subgroup**: Enable cross-budget learning through global baselines

## Repository Structure

```
HBPO/
├── verl/                    # Core framework
│   ├── trainer/            # Training algorithms
│   ├── workers/            # Distributed components  
│   └── utils/              # Utilities
├── examples/               # Training scripts
│   ├── grpo_trainer/      # HBPO implementation
│   └── data_preprocess/   # Data preparation
├── tests/                 # Test suites
├── docs/                  # Documentation
└── figures/               # Paper figures
```


## Citation

```bibtex
@misc{lyu2025hierarchicalbudgetpolicyoptimization,
      title={Hierarchical Budget Policy Optimization for Adaptive Reasoning}, 
      author={Shangke Lyu and Linjuan Wu and Yuchen Yan and Xingyu Wu and Hao Li and Yongliang Shen and Peisheng Jiang and Weiming Lu and Jun Xiao and Yueting Zhuang},
      year={2025},
      eprint={2507.15844},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.15844}, 
}
```

## Contact

For questions about the research or implementation, please open an issue or contact the author: `lyusk@zju.edu.cn`.
