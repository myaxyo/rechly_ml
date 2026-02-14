from dataclasses import dataclass
from django.conf import settings


@dataclass(frozen=True)
class AppwriteConfig:
    endpoint: str
    project_id: str
    api_key: str
    database_id: str
    invoices_collection_id: str
    clients_collection_id: str


def get_appwrite_config() -> AppwriteConfig:
    return AppwriteConfig(
        endpoint=settings.APPWRITE_ENDPOINT,
        project_id=settings.APPWRITE_PROJECT_ID,
        api_key=settings.APPWRITE_API_KEY,
        database_id=settings.APPWRITE_DATABASE_ID,
        invoices_collection_id=settings.APPWRITE_INVOICES_COLLECTION_ID,
        clients_collection_id=settings.APPWRITE_CLIENTS_COLLECTION_ID,
    )


def get_model_artifact_dir() -> str:
    return settings.MODEL_ARTIFACT_DIR


def get_seed() -> int:
    return settings.MODEL_RANDOM_SEED
