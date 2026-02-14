from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from analytics.utils.config import get_model_artifact_dir, get_seed


MODEL_PATH = "revenue_forecast_model.joblib"
METRICS_PATH = "revenue_forecast_metrics.json"


@dataclass
class RevenueArtifacts:
    model: Any
    residual_std: float
    trained_at: str
    model_version: str
    metrics: dict[str, Any]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def _build_daily_paid_series(invoices_df: pd.DataFrame) -> pd.DataFrame:
    paid = invoices_df[invoices_df["status"] == "paid"].copy()
    if paid.empty:
        return pd.DataFrame(columns=["ds", "y"])

    paid["paid_day"] = pd.to_datetime(paid["paid_date"], utc=True, errors="coerce").dt.floor("D")
    paid = paid.dropna(subset=["paid_day"])

    grouped = (
        paid.groupby("paid_day", as_index=False)["total_gross"]
        .sum()
        .rename(columns={"paid_day": "ds", "total_gross": "y"})
    )

    full_range = pd.date_range(grouped["ds"].min(), grouped["ds"].max(), freq="D", tz="UTC")
    grouped = (
        grouped.set_index("ds")
        .reindex(full_range, fill_value=0.0)
        .rename_axis("ds")
        .reset_index()
    )

    return grouped


def _create_regression_frame(series_df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    frame = series_df.copy()
    for lag in lags:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)

    frame["ma_7"] = frame["y"].rolling(7).mean().shift(1)
    frame["ma_30"] = frame["y"].rolling(30).mean().shift(1)

    frame["day_of_week"] = frame["ds"].dt.dayofweek
    frame["day_of_month"] = frame["ds"].dt.day
    frame["month"] = frame["ds"].dt.month

    frame = frame.dropna().reset_index(drop=True)
    return frame


class RevenueForecastModelService:
    def __init__(self) -> None:
        self.seed = get_seed()
        self.artifact_dir = Path(get_model_artifact_dir())
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def train_model(self, invoices_df: pd.DataFrame) -> dict[str, Any]:
        series = _build_daily_paid_series(invoices_df)
        if len(series) < 120:
            raise ValueError("Need at least 120 daily points for robust revenue training.")

        lags = [1, 7, 14, 30]
        frame = _create_regression_frame(series, lags)
        feature_columns = [
            *[f"lag_{lag}" for lag in lags],
            "ma_7",
            "ma_30",
            "day_of_week",
            "day_of_month",
            "month",
        ]

        model = GradientBoostingRegressor(random_state=self.seed)

        # Rolling-window validation
        window = 30
        step = 7
        preds, actuals = [], []
        naive_preds, ma_preds = [], []
        naive_errors, ma_errors, model_errors = [], [], []

        start = max(60, int(len(frame) * 0.5))
        for end_idx in range(start, len(frame) - window, step):
            train_slice = frame.iloc[:end_idx]
            valid_slice = frame.iloc[end_idx : end_idx + window]

            model.fit(train_slice[feature_columns], train_slice["y"])
            pred = model.predict(valid_slice[feature_columns])

            preds.extend(pred.tolist())
            actuals.extend(valid_slice["y"].tolist())

            naive_pred = valid_slice["lag_1"].to_numpy()
            ma_pred = valid_slice["ma_7"].to_numpy()

            naive_preds.extend(naive_pred.tolist())
            ma_preds.extend(ma_pred.tolist())

            naive_errors.extend(np.abs(valid_slice["y"].to_numpy() - naive_pred).tolist())
            ma_errors.extend(np.abs(valid_slice["y"].to_numpy() - ma_pred).tolist())
            model_errors.extend(np.abs(valid_slice["y"].to_numpy() - pred).tolist())

        y_true = np.array(actuals)
        y_pred = np.array(preds)
        y_naive = np.array(naive_preds)
        y_ma = np.array(ma_preds)

        metrics = {
            "naive": {
                "mae": float(np.mean(naive_errors)) if naive_errors else None,
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_naive))) if len(y_true) else None,
                "mape": _mape(y_true, y_naive) if len(y_true) else None,
            },
            "moving_average": {
                "mae": float(np.mean(ma_errors)) if ma_errors else None,
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_ma))) if len(y_true) else None,
                "mape": _mape(y_true, y_ma) if len(y_true) else None,
            },
            "gradient_boosting": {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mape": _mape(y_true, y_pred),
            },
        }

        model.fit(frame[feature_columns], frame["y"])
        residuals = frame["y"].to_numpy() - model.predict(frame[feature_columns])
        residual_std = float(np.std(residuals))

        trained_at = datetime.now(timezone.utc).isoformat()
        version = f"revenue-{trained_at}"

        output = {
            "trained_at": trained_at,
            "model_version": version,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "seed": self.seed,
        }

        artifact = RevenueArtifacts(
            model=model,
            residual_std=residual_std,
            trained_at=trained_at,
            model_version=version,
            metrics=output,
        )

        self._save(artifact, feature_columns, lags)
        return output

    def _save(self, artifact: RevenueArtifacts, feature_columns: list[str], lags: list[int]) -> None:
        model_file = self.artifact_dir / MODEL_PATH
        metrics_file = self.artifact_dir / METRICS_PATH

        joblib.dump(
            {
                "model": artifact.model,
                "residual_std": artifact.residual_std,
                "trained_at": artifact.trained_at,
                "model_version": artifact.model_version,
                "feature_columns": feature_columns,
                "lags": lags,
            },
            model_file,
        )

        metrics_file.write_text(json.dumps(artifact.metrics, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        model_file = self.artifact_dir / MODEL_PATH
        if not model_file.exists():
            raise FileNotFoundError("Revenue model not found. Train first.")
        return joblib.load(model_file)

    def forecast(self, invoices_df: pd.DataFrame) -> dict[str, Any]:
        bundle = self._load()
        model: GradientBoostingRegressor = bundle["model"]
        residual_std: float = float(bundle["residual_std"])
        lags: list[int] = bundle["lags"]

        daily = _build_daily_paid_series(invoices_df)
        if daily.empty:
            return {
                "next_30_days": 0.0,
                "next_90_days": 0.0,
                "interval": {"lower": 0.0, "upper": 0.0},
            }

        current = daily.copy()
        horizon = 90
        forecasts = []

        for _ in range(horizon):
            next_day = current["ds"].max() + pd.Timedelta(days=1)
            row = {
                "ds": next_day,
                "y": np.nan,
            }
            temp = pd.concat([current, pd.DataFrame([row])], ignore_index=True)

            for lag in lags:
                temp[f"lag_{lag}"] = temp["y"].shift(lag)
            temp["ma_7"] = temp["y"].rolling(7).mean().shift(1)
            temp["ma_30"] = temp["y"].rolling(30).mean().shift(1)
            temp["day_of_week"] = temp["ds"].dt.dayofweek
            temp["day_of_month"] = temp["ds"].dt.day
            temp["month"] = temp["ds"].dt.month

            latest = temp.iloc[-1:].copy().fillna(0)
            pred = float(model.predict(latest[bundle["feature_columns"]])[0])
            pred = max(0.0, pred)

            current = pd.concat(
                [current, pd.DataFrame([{"ds": next_day, "y": pred}])],
                ignore_index=True,
            )
            forecasts.append(pred)

        next_30 = float(np.sum(forecasts[:30]))
        next_90 = float(np.sum(forecasts[:90]))

        # Approximate prediction interval via residual std
        interval_multiplier = 1.96
        total_std_30 = residual_std * np.sqrt(30)
        lower_30 = max(0.0, next_30 - interval_multiplier * total_std_30)
        upper_30 = next_30 + interval_multiplier * total_std_30

        return {
            "next_30_days": next_30,
            "next_90_days": next_90,
            "interval_30_days": {
                "lower": lower_30,
                "upper": upper_30,
            },
        }

    def get_latest_metrics(self) -> dict[str, Any]:
        metrics_file = self.artifact_dir / METRICS_PATH
        if not metrics_file.exists():
            raise FileNotFoundError("No revenue metrics found. Train first.")
        return json.loads(metrics_file.read_text(encoding="utf-8"))
