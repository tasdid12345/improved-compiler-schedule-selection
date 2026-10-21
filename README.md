# Class-balanced, CV-tuned LightGBM vs random-forest schedule-cost surrogate

This experiment is motivated by "Schedgehammer: Auto-Tuning Compiler Optimizations Beyond Numerical Parameters" (2026), which argues that compiler auto-tuning must handle non-numerical choices such as loop order, vectorization, parallelization and tile-size categories. A learned cost model that ranks candidate schedules is a common way to guide such search, so the quality of its ranking on unseen programs matters.

**Question:** Can a class-balanced, CV-tuned gradient-boosting cost model with categorical schedule features choose faster compiler schedules on held-out programs than a standard random-forest cost surrogate?

**Dataset:** A seeded synthetic loop-nest simulator generated with numpy (no download). It is not the paper's own dataset. Programs are split into train and held-out test sets, and each program's sampled schedules are ranked by predicted runtime.

**Baseline:** A random-forest regressor on one-hot encoded schedule features plus numeric program features, predicting raw runtime. It stands in for the paper's search-guiding model, whose details are not public.

**Improved:** A LightGBM regressor with native categorical handling, hyper-parameters chosen by grouped cross-validation on training programs only, a per-program standardised log-runtime target, and program-balanced sample weights.

**How to run**

```
pip install -r requirements.txt
python analysis.py
```

**Results:** Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On a synthetic loop-nest schedule dataset, a standard random-forest cost surrogate reached 46.9% mean regret and 17% top-1 accuracy, while my class-balanced, CV-tuned LightGBM model reached 3.7% regret and 53% top-1, though the baseline was untuned.

| Metric | Value |
|---|---|
| baseline_score | 0.469 |
| baseline_score_std | 0.00549 |
| baseline_top1_accuracy | 0.1704 |
| baseline_spearman | 0.7632 |
| improved_score | 0.03749 |
| improved_score_std | 0 |
| improved_top1_accuracy | 0.5278 |
| improved_spearman | 0.9251 |
| improvement | 0.4316 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic data, not the paper's dataset; the paper's own method is not reproduced (only a generic random-forest baseline is used). The baseline uses fixed hyper-parameters while the improved model is CV-tuned, so gains may partly reflect tuning. Regret and top-1 use noisy observed runtimes on a modest number of test programs. Seeds vary model randomness only; the data split and generator are fixed at seed 42.

## Reference

This project is motivated by the following paper:

> J Lenfers, S Spehr, J Dieckmann, J Jansen (2026). *Schedgehammer: Auto-Tuning Compiler Optimizations Beyond Numerical Parameters*. Proceedings of the 35th. https://dl.acm.org/doi/abs/10.1145/3771775.3786282

Paper author: Prof. Dr. Sergei Gorlatch, University of Münster (gorlatch@uni-muenster.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
