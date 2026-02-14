from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


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
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1) if n > 1 else 0.0

    if std_diff == 0.0:
        return {"t_stat": 0.0, "p_value": 1.0}

    t_stat = mean_diff / (std_diff / np.sqrt(n))

    # Normal approximation (sufficient for service-level comparison).
    p_value = float(2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2)))))
    return {"t_stat": float(t_stat), "p_value": p_value}


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

    paired = paired_t_test(baseline, candidate)

    if mean_candidate < mean_baseline:
        winner = "candidate"
    elif mean_candidate > mean_baseline:
        winner = "baseline"
    else:
        winner = "tie"

    return {
        "winner": winner,
        "mean_baseline_error": mean_baseline,
        "mean_candidate_error": mean_candidate,
        "paired_test": paired,
    }
