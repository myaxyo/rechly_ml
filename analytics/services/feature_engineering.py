from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "invoice_amount",
    "client_late_rate",
    "client_avg_days_late",
    "client_invoice_frequency",
    "client_overdue_rate",
    "invoice_age",
    "days_until_due",
    "client_total_revenue",
    "recency_days",
]


@dataclass
class FeatureDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str = "late"


def _temporal_base(invoices_df: pd.DataFrame) -> pd.DataFrame:
    df = invoices_df.copy()
    df = df.sort_values(["issue_date", "invoice_id"]).reset_index(drop=True)

    df["prediction_date"] = df["issue_date"].dt.floor("D")
    df["invoice_amount"] = df["total_gross"].fillna(0).clip(lower=0)

    return df


def build_late_payment_training_dataset(invoices_df: pd.DataFrame) -> FeatureDataset:
    if invoices_df.empty:
        return FeatureDataset(frame=pd.DataFrame(columns=FEATURE_COLUMNS + ["late", "prediction_date"]), feature_columns=FEATURE_COLUMNS)

    df = _temporal_base(invoices_df)

    grp = df.groupby("client_id", dropna=False)

    previous_count = grp.cumcount()
    df["client_invoice_frequency"] = previous_count.astype(float)

    prev_late_sum = grp["late"].cumsum().shift(1).fillna(0)
    df["client_late_rate"] = np.where(
        previous_count > 0,
        prev_late_sum / previous_count.replace(0, np.nan),
        0.0,
    )
    df["client_late_rate"] = df["client_late_rate"].fillna(0)

    late_days_series = pd.Series(
        np.where(df["late"] == 1, df["days_late"], np.nan),
        index=df.index,
        name="late_days_only",
    )
    prev_late_days_sum = (
        late_days_series.groupby(df["client_id"], dropna=False).apply(
            lambda s: s.fillna(0).cumsum().shift(1).fillna(0)
        ).reset_index(level=0, drop=True)
    )
    prev_late_count = grp["late"].cumsum().shift(1).fillna(0)
    df["client_avg_days_late"] = np.where(
        prev_late_count > 0,
        prev_late_days_sum / prev_late_count.replace(0, np.nan),
        0.0,
    )
    df["client_avg_days_late"] = df["client_avg_days_late"].fillna(0)

    overdue_indicator = (
        (df["status"].isin(["draft", "sent"]))
        & (df["due_date"].notna())
        & (df["due_date"].dt.floor("D") < df["prediction_date"])
    ).astype(int)
    prev_overdue_sum = overdue_indicator.groupby(df["client_id"]).cumsum().shift(1).fillna(0)
    df["client_overdue_rate"] = np.where(
        previous_count > 0,
        prev_overdue_sum / previous_count.replace(0, np.nan),
        0.0,
    )
    df["client_overdue_rate"] = df["client_overdue_rate"].fillna(0)

    prev_revenue_sum = grp["invoice_amount"].cumsum().shift(1).fillna(0)
    df["client_total_revenue"] = prev_revenue_sum

    prev_issue_date = grp["issue_date"].shift(1)
    df["recency_days"] = (
        (df["prediction_date"] - prev_issue_date.dt.floor("D")).dt.days
    )
    df["recency_days"] = df["recency_days"].fillna(365).clip(lower=0)

    df["invoice_age"] = (
        (df["prediction_date"] - df["created_at"].dt.floor("D")).dt.days
    )
    df["invoice_age"] = df["invoice_age"].fillna(0).clip(lower=0)

    df["days_until_due"] = (
        (df["due_date"].dt.floor("D") - df["prediction_date"]).dt.days
    )
    df["days_until_due"] = df["days_until_due"].fillna(0)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["late"] = df["late"].astype(int)

    return FeatureDataset(frame=df, feature_columns=FEATURE_COLUMNS)


def build_feature_row_from_payload(payload: dict) -> pd.DataFrame:
    feature_data = {col: [float(payload.get(col, 0.0))] for col in FEATURE_COLUMNS}
    return pd.DataFrame(feature_data)


def temporal_split(
    frame: pd.DataFrame,
    date_column: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    ordered = frame.sort_values(date_column).reset_index(drop=True)
    n = len(ordered)

    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()

    if validation.empty and not test.empty:
        validation = test.iloc[:1].copy()
        test = test.iloc[1:].copy()

    return train, validation, test
