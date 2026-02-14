from __future__ import annotations

from datetime import datetime, timezone

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from analytics.api.serializers import (
    EmptyTrainSerializer,
    LatePaymentPredictionSerializer,
)
from analytics.models.late_payment_model import LatePaymentModelService
from analytics.models.revenue_forecast_model import RevenueForecastModelService
from analytics.services.data_loader import load_appwrite_data
from analytics.services.experiment_store import ExperimentStore
from analytics.services.feature_engineering import build_late_payment_training_dataset
from analytics.services.evaluation import compare_models_statistically


@api_view(["POST"])
def train_late_payment(request):
    serializer = EmptyTrainSerializer(data=request.data or {})
    serializer.is_valid(raise_exception=True)

    try:
        data = load_appwrite_data()
        feature_dataset = build_late_payment_training_dataset(data.invoices)

        service = LatePaymentModelService()
        metrics = service.train_model(feature_dataset.frame)
        experiment_store = ExperimentStore()

        data_summary = {
            "rows_used": int(len(feature_dataset.frame)),
            "label_positive_count": int((feature_dataset.frame["late"] == 1).sum()),
            "label_negative_count": int((feature_dataset.frame["late"] == 0).sum()),
            "label_positive_rate": float(feature_dataset.frame["late"].mean())
            if len(feature_dataset.frame)
            else None,
        }

        run_record = experiment_store.record_run(
            task="train_late_payment",
            metrics=metrics,
            data_summary=data_summary,
            model_version=metrics.get("model_version"),
        )

        return Response(
            {
                "status": "ok",
                "task": "train_late_payment",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "experiment_run": run_record,
            }
        )
    except Exception as exc:
        return Response(
            {
                "status": "error",
                "task": "train_late_payment",
                "message": str(exc),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["POST"])
def predict_late_payment(request):
    serializer = LatePaymentPredictionSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    try:
        service = LatePaymentModelService()
        prediction = service.predict(serializer.validated_data)

        return Response(
            {
                "status": "ok",
                "task": "predict_late_payment",
                "result": prediction,
            }
        )
    except Exception as exc:
        return Response(
            {
                "status": "error",
                "task": "predict_late_payment",
                "message": str(exc),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["POST"])
def train_revenue(request):
    serializer = EmptyTrainSerializer(data=request.data or {})
    serializer.is_valid(raise_exception=True)

    try:
        data = load_appwrite_data()

        service = RevenueForecastModelService()
        metrics = service.train_model(data.invoices)

        errors = metrics.get("validation_errors", {})
        gb_errors = errors.get("gradient_boosting", [])
        statistical = {
            "gradient_boosting_vs_naive": compare_models_statistically(
                errors.get("naive", []), gb_errors
            ),
            "gradient_boosting_vs_moving_average": compare_models_statistically(
                errors.get("moving_average", []), gb_errors
            ),
        }

        experiment_store = ExperimentStore()
        data_summary = {
            "raw_invoice_rows": int(len(data.invoices)),
            "paid_invoice_rows": int((data.invoices["status"] == "paid").sum()),
            "validation_error_points": {
                "naive": int(len(errors.get("naive", []))),
                "moving_average": int(len(errors.get("moving_average", []))),
                "gradient_boosting": int(len(gb_errors)),
            },
        }
        run_record = experiment_store.record_run(
            task="train_revenue",
            metrics=metrics,
            statistical_summary=statistical,
            data_summary=data_summary,
            model_version=metrics.get("model_version"),
        )

        return Response(
            {
                "status": "ok",
                "task": "train_revenue",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "statistical_comparison": statistical,
                "experiment_run": run_record,
            }
        )
    except Exception as exc:
        return Response(
            {
                "status": "error",
                "task": "train_revenue",
                "message": str(exc),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["GET"])
def forecast_revenue(_request):
    try:
        data = load_appwrite_data()
        service = RevenueForecastModelService()
        result = service.forecast(data.invoices)

        return Response(
            {
                "status": "ok",
                "task": "forecast_revenue",
                "forecast": result,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        return Response(
            {
                "status": "error",
                "task": "forecast_revenue",
                "message": str(exc),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["GET"])
def late_payment_metrics(_request):
    try:
        service = LatePaymentModelService()
        metrics = service.get_latest_metrics()
        return Response(
            {
                "status": "ok",
                "task": "late_payment_metrics",
                "metrics": metrics,
            }
        )
    except Exception as exc:
        return Response(
            {
                "status": "error",
                "task": "late_payment_metrics",
                "message": str(exc),
            },
            status=status.HTTP_404_NOT_FOUND,
        )


@api_view(["GET"])
def shap_importance(_request):
    return Response(
        {
            "status": "ok",
            "task": "shap_importance",
            "message": "SHAP endpoint placeholder. Enable after adding shap dependency and explainer pipeline.",
        }
    )


@api_view(["POST"])
def retrain_models(_request):
    return Response(
        {
            "status": "ok",
            "task": "retrain_models",
            "message": "Retraining endpoint placeholder. Wire scheduler or async worker for production use.",
        }
    )


@api_view(["GET"])
def drift_status(_request):
    return Response(
        {
            "status": "ok",
            "task": "drift_status",
            "drift_detected": False,
            "message": "Drift detection placeholder. Add PSI/KS checks for production monitoring.",
        }
    )
