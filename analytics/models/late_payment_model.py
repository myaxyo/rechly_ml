from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analytics.services.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_row_from_payload,
    temporal_split,
)
from analytics.utils.config import get_model_artifact_dir, get_seed


MODEL_PATH = "late_payment_model.joblib"
METRICS_PATH = "late_payment_metrics.json"


@dataclass
class LatePaymentArtifacts:
    model: Any
    model_name: str
    feature_columns: list[str]
    version: str
    trained_at: str
    metrics: dict[str, Any]


class LatePaymentModelService:
    def __init__(self) -> None:
        self.seed = get_seed()
        self.artifact_dir = Path(get_model_artifact_dir())
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _build_candidates(self) -> dict[str, Pipeline]:
        return {
            "logistic_regression": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            random_state=self.seed,
                            max_iter=2000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "gradient_boosting": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GradientBoostingClassifier(random_state=self.seed),
                    ),
                ]
            ),
        }

    def _compute_metrics(self, y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else 0.0,
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
        }
        return metrics

    def train_model(self, dataset: pd.DataFrame) -> dict[str, Any]:
        if dataset.empty:
            raise ValueError("Dataset is empty. Cannot train late-payment model.")

        if "prediction_date" not in dataset.columns:
            raise ValueError("Dataset must contain prediction_date for temporal split.")

        train_df, val_df, test_df = temporal_split(dataset, date_column="prediction_date")

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("Insufficient rows for temporal 70/15/15 split.")

        x_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["late"].astype(int)
        x_val = val_df[FEATURE_COLUMNS]
        y_val = val_df["late"].astype(int)
        x_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["late"].astype(int)

        if len(np.unique(y_train)) < 2:
            raise ValueError("Training set needs both target classes for classification.")

        candidates = self._build_candidates()

        best_name = None
        best_model = None
        best_val_pr_auc = -np.inf

        candidate_validation_metrics: dict[str, dict[str, float]] = {}

        for name, model in candidates.items():
            model.fit(x_train, y_train)
            val_prob = model.predict_proba(x_val)[:, 1]
            val_metrics = self._compute_metrics(y_val, val_prob)
            candidate_validation_metrics[name] = val_metrics
            if val_metrics["pr_auc"] > best_val_pr_auc:
                best_val_pr_auc = val_metrics["pr_auc"]
                best_name = name
                best_model = model

        assert best_model is not None and best_name is not None

        test_prob = best_model.predict_proba(x_test)[:, 1]
        test_metrics = self._compute_metrics(y_test, test_prob)

        trained_at = datetime.now(timezone.utc).isoformat()
        version = f"late-payment-{trained_at}"

        result = {
            "model_name": best_name,
            "trained_at": trained_at,
            "model_version": version,
            "split_sizes": {
                "train": int(len(train_df)),
                "validation": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "validation_metrics": candidate_validation_metrics,
            "test_metrics": test_metrics,
            "feature_columns": FEATURE_COLUMNS,
            "seed": self.seed,
        }

        artifacts = LatePaymentArtifacts(
            model=best_model,
            model_name=best_name,
            feature_columns=FEATURE_COLUMNS,
            version=version,
            trained_at=trained_at,
            metrics=result,
        )

        self._save(artifacts)
        return result

    def _save(self, artifacts: LatePaymentArtifacts) -> None:
        model_file = self.artifact_dir / MODEL_PATH
        metrics_file = self.artifact_dir / METRICS_PATH

        joblib.dump(
            {
                "model": artifacts.model,
                "model_name": artifacts.model_name,
                "feature_columns": artifacts.feature_columns,
                "version": artifacts.version,
                "trained_at": artifacts.trained_at,
            },
            model_file,
        )

        metrics_file.write_text(json.dumps(artifacts.metrics, indent=2), encoding="utf-8")

    def _load_model_bundle(self) -> dict[str, Any]:
        model_file = self.artifact_dir / MODEL_PATH
        if not model_file.exists():
            raise FileNotFoundError("Late-payment model artifact not found. Train first.")
        return joblib.load(model_file)

    def predict_proba(self, payload: dict[str, Any]) -> float:
        bundle = self._load_model_bundle()
        model = bundle["model"]
        feature_df = build_feature_row_from_payload(payload)[bundle["feature_columns"]]
        probability = float(model.predict_proba(feature_df)[:, 1][0])
        return probability

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        probability = self.predict_proba(payload)
        label = "high" if probability >= 0.67 else "medium" if probability >= 0.34 else "low"
        return {
            "late_payment_probability": probability,
            "risk_label": label,
        }

    def get_latest_metrics(self) -> dict[str, Any]:
        metrics_file = self.artifact_dir / METRICS_PATH
        if not metrics_file.exists():
            raise FileNotFoundError("No metrics snapshot found. Train first.")
        return json.loads(metrics_file.read_text(encoding="utf-8"))
