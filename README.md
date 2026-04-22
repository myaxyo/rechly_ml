# Rechly ML API

Optional Django service for Rechly analytics, forecasting, and late-payment risk scoring.

## Features

- Late-payment model training and prediction
- Revenue forecasting for the next 30 and 90 days
- Temporal-safe feature engineering and evaluation
- Metrics snapshots and experiment metadata
- Bearer-token protected API endpoints

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and fill in all required values.
4. Run Django migrations.
5. Start the development server.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

## Required Environment Variables

- `DJANGO_SECRET_KEY`
- `ML_SECRET`
- `APPWRITE_ENDPOINT`
- `APPWRITE_PROJECT_ID`
- `APPWRITE_API_KEY`
- `APPWRITE_DATABASE_ID`
- `APPWRITE_INVOICES_COLLECTION_ID`
- `APPWRITE_CLIENTS_COLLECTION_ID`

The service expects an Appwrite project that already contains the invoice and client data required for training and prediction. It does not yet include a sample data loader or seeded model artifacts.

## Training and Artifacts

- Trained artifacts are written to `MODEL_ARTIFACT_DIR`.
- `services/ml_api/artifacts/` is ignored by git and should be backed by persistent storage in production.
- Predict endpoints will fail until the corresponding models have been trained.

## API Endpoints

Requests must include `Authorization: Bearer <ML_SECRET>`.

- `POST /train/late-payment`
- `POST /predict/late-payment`
- `POST /train/revenue`
- `GET /forecast/revenue`
- `GET /metrics/late-payment`

Compatibility aliases with `/api/*` are also exposed.

## Deployment Notes

- `Procfile` contains the Gunicorn entrypoint.
- `runtime.txt` pins the Python runtime.
- Set every environment variable explicitly in your hosting provider.
- Persist `MODEL_ARTIFACT_DIR` if you expect trained models to survive redeploys.
