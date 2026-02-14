from rest_framework import serializers

from analytics.services.feature_engineering import FEATURE_COLUMNS


class LatePaymentPredictionSerializer(serializers.Serializer):
    invoice_amount = serializers.FloatField(required=True)
    client_late_rate = serializers.FloatField(required=True)
    client_avg_days_late = serializers.FloatField(required=True)
    client_invoice_frequency = serializers.FloatField(required=True)
    client_overdue_rate = serializers.FloatField(required=True)
    invoice_age = serializers.FloatField(required=True)
    days_until_due = serializers.FloatField(required=True)
    client_total_revenue = serializers.FloatField(required=True)
    recency_days = serializers.FloatField(required=True)

    def validate(self, attrs):
        missing = [key for key in FEATURE_COLUMNS if key not in attrs]
        if missing:
            raise serializers.ValidationError({"missing_features": missing})
        return attrs


class EmptyTrainSerializer(serializers.Serializer):
    force = serializers.BooleanField(required=False, default=False)
