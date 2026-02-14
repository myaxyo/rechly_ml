from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def bootstrap_confidence_interval(
    values: Iterable[float],
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"lower": None, "upper": None, "mean": None}

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats.append(metric_fn(sample))

    stats_arr = np.asarray(stats)
    lower = float(np.quantile(stats_arr, alpha / 2))
    upper = float(np.quantile(stats_arr, 1 - alpha / 2))
    mean = float(stats_arr.mean())

    return {"lower": lower, "upper": upper, "mean": mean}


def paired_t_test(values_a: Iterable[float], values_b: Iterable[float]):
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return {"t_stat": None, "p_value": None}

    diff = a - b
    n = diff.size
    std_diff = diff.std(ddof=1) if n > 1 else 0.0
    if std_diff == 0.0:
        return {"t_stat": 0.0, "p_value": 1.0}

    stat = ttest_rel(a, b, alternative="two-sided")
    return {"t_stat": float(stat.statistic), "p_value": float(stat.pvalue)}


def wilcoxon_signed_rank_test(values_a: Iterable[float], values_b: Iterable[float]):
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return {"w_stat": None, "p_value": None}

    if np.allclose(a, b):
        return {"w_stat": 0.0, "p_value": 1.0}

    stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
    return {"w_stat": float(stat.statistic), "p_value": float(stat.pvalue)}


def _cohens_d_paired(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    diff = values_a - values_b
    if diff.size < 2:
        return None
    std_diff = float(diff.std(ddof=1))
    if std_diff == 0.0:
        return 0.0
    return float(diff.mean() / std_diff)


def compare_models_statistically(
    baseline_errors: Iterable[float],
    candidate_errors: Iterable[float],
):
    baseline = np.asarray(list(baseline_errors), dtype=float)
    candidate = np.asarray(list(candidate_errors), dtype=float)

    if baseline.size == 0 or candidate.size == 0 or baseline.size != candidate.size:
        return {
            "winner": "undetermined",
            "mean_baseline_error": None,
            "mean_candidate_error": None,
            "paired_test": {"t_stat": None, "p_value": None},
        }

    mean_baseline = float(baseline.mean())
    mean_candidate = float(candidate.mean())
    mean_diff = float((baseline - candidate).mean())
    relative_improvement_pct = (
        float((mean_diff / mean_baseline) * 100.0) if mean_baseline != 0 else None
    )

    paired = paired_t_test(baseline, candidate)
    wilcoxon_result = wilcoxon_signed_rank_test(baseline, candidate)
    diff_ci = bootstrap_confidence_interval(
        baseline - candidate,
        metric_fn=lambda x: float(np.mean(x)),
        n_bootstrap=2000,
        alpha=0.05,
    )
    effect_size = _cohens_d_paired(baseline, candidate)

    if mean_candidate < mean_baseline:
        winner = "candidate"
    elif mean_candidate > mean_baseline:
        winner = "baseline"
    else:
        winner = "tie"

    return {
        "winner": winner,
        "n_samples": int(baseline.size),
        "mean_baseline_error": mean_baseline,
        "mean_candidate_error": mean_candidate,
        "mean_error_reduction": mean_diff,
        "relative_improvement_pct": relative_improvement_pct,
        "mean_error_reduction_ci_95": diff_ci,
        "effect_size_cohens_d": effect_size,
        "paired_test": paired,
        "wilcoxon_test": wilcoxon_result,
        "significant_at_0_05": bool(
            paired.get("p_value") is not None and paired["p_value"] < 0.05
        ),
    }
