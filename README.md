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

## Reference

This project is motivated by the following paper:

> J Lenfers, S Spehr, J Dieckmann, J Jansen (2026). *Schedgehammer: Auto-Tuning Compiler Optimizations Beyond Numerical Parameters*. Proceedings of the 35th. https://dl.acm.org/doi/abs/10.1145/3771775.3786282

Paper author: Prof. Dr. Sergei Gorlatch, University of Münster (gorlatch@uni-muenster.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
