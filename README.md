# Rechly ML API (Django)

Independent ML and analytics microservice for Rechly.

## Features

- Late-payment model training and scoring
- Revenue forecasting (30/90 days)
- Temporal-safe feature engineering
- Metrics snapshots and model versioning
- Bearer-token protected API

## Project Layout

- `ml_api/` Django project
- `analytics/models/` ML model modules
- `analytics/services/` data loading + features + evaluation
- `analytics/api/` DRF serializers/views
- `analytics/utils/` config + security middleware

## Environment

Copy `.env.example` to `.env` and fill values.

Required:

- `ML_SECRET`
- `APPWRITE_ENDPOINT`
- `APPWRITE_PROJECT_ID`
- `APPWRITE_API_KEY`
- `APPWRITE_DATABASE_ID`
- `APPWRITE_INVOICES_COLLECTION_ID`
- `APPWRITE_CLIENTS_COLLECTION_ID`

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8000
```

## API Endpoints

Protected by `Authorization: Bearer <ML_SECRET>`.

- `POST /train/late-payment`
- `POST /predict/late-payment`
- `POST /train/revenue`
- `GET /forecast/revenue`
- `GET /metrics/late-payment`

Compatibility aliases with `/api/*` prefix are also exposed.

## Render Deploy

- `Procfile` includes Gunicorn entrypoint
- `runtime.txt` pins Python runtime
- Set all env variables in Render dashboard
