from django.conf import settings
from django.http import JsonResponse


class MLSecretMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        protected_prefixes = (
            "/api/",
            "/train/",
            "/predict/",
            "/forecast/",
            "/metrics/",
        )

        if request.path.startswith(protected_prefixes):
            configured_secret = settings.ML_SECRET
            if not configured_secret:
                return JsonResponse(
                    {"error": "ML_SECRET is not configured"}, status=500
                )

            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JsonResponse(
                    {"error": "Unauthorized: missing bearer token"},
                    status=401,
                )

            provided_secret = auth_header.replace("Bearer ", "", 1).strip()
            if provided_secret != configured_secret:
                return JsonResponse(
                    {"error": "Unauthorized: invalid token"},
                    status=401,
                )

        return self.get_response(request)
