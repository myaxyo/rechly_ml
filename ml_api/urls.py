from django.urls import path
from analytics.api.views import (
    train_late_payment,
    predict_late_payment,
    train_revenue,
    forecast_revenue,
    late_payment_metrics,
    shap_importance,
    retrain_models,
    drift_status,
)

urlpatterns = [
    path("train/late-payment", train_late_payment),
    path("predict/late-payment", predict_late_payment),
    path("train/revenue", train_revenue),
    path("forecast/revenue", forecast_revenue),
    path("metrics/late-payment", late_payment_metrics),
    path("api/train/late-payment", train_late_payment),
    path("api/predict/late-payment", predict_late_payment),
    path("api/train/revenue", train_revenue),
    path("api/forecast/revenue", forecast_revenue),
    path("api/metrics/late-payment", late_payment_metrics),
    path("api/shap/late-payment", shap_importance),
    path("api/retrain", retrain_models),
    path("api/drift/status", drift_status),
]
