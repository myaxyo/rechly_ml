from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query

from analytics.utils.config import get_appwrite_config


@dataclass
class LoadedData:
    invoices: pd.DataFrame
    clients: pd.DataFrame


def _safe_get(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _to_records(response: Any) -> list[dict[str, Any]]:
    documents = _safe_get(response, "documents", [])
    normalized: list[dict[str, Any]] = []
    for doc in documents:
        if isinstance(doc, dict):
            normalized.append(doc)
        else:
            normalized.append(doc.__dict__)
    return normalized


def _paginate_documents(
    databases: Databases,
    database_id: str,
    collection_id: str,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    offset = 0
    all_records: list[dict[str, Any]] = []

    while True:
        response = databases.list_documents(
            database_id=database_id,
            collection_id=collection_id,
            queries=[Query.limit(page_size), Query.offset(offset)],
        )
        records = _to_records(response)
        if not records:
            break

        all_records.extend(records)
        if len(records) < page_size:
            break
        offset += page_size

    return all_records


def _build_client_df(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    clients_df = pd.DataFrame(list(records))
    if clients_df.empty:
        return pd.DataFrame(
            columns=["client_id", "client_name", "client_created_at"]
        )

    clients_df = clients_df.rename(
        columns={
            "$id": "client_id",
            "name": "client_name",
            "$createdAt": "client_created_at",
        }
    )

    clients_df["client_created_at"] = pd.to_datetime(
        clients_df.get("client_created_at"), errors="coerce", utc=True
    )

    return clients_df[["client_id", "client_name", "client_created_at"]].copy()


def _build_invoice_df(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    invoices_df = pd.DataFrame(list(records))
    if invoices_df.empty:
        return pd.DataFrame(
            columns=[
                "invoice_id",
                "client_id",
                "invoice_number",
                "status",
                "issue_date",
                "due_date",
                "total_gross",
                "created_at",
                "updated_at",
                "paid_date",
                "days_late",
                "late",
                "label_available",
            ]
        )

    invoices_df = invoices_df.rename(
        columns={
            "$id": "invoice_id",
            "clientId": "client_id",
            "invoiceNumber": "invoice_number",
            "issueDate": "issue_date",
            "dueDate": "due_date",
            "totalGross": "total_gross",
            "$createdAt": "created_at",
            "$updatedAt": "updated_at",
        }
    )

    invoices_df["issue_date"] = pd.to_datetime(
        invoices_df.get("issue_date"), errors="coerce", utc=True
    )
    invoices_df["due_date"] = pd.to_datetime(
        invoices_df.get("due_date"), errors="coerce", utc=True
    )
    invoices_df["created_at"] = pd.to_datetime(
        invoices_df.get("created_at"), errors="coerce", utc=True
    )
    invoices_df["updated_at"] = pd.to_datetime(
        invoices_df.get("updated_at"), errors="coerce", utc=True
    )

    invoices_df["total_gross"] = (
        pd.to_numeric(invoices_df.get("total_gross"), errors="coerce")
        .fillna(0.0)
        .clip(lower=0)
    )
    invoices_df["status"] = invoices_df.get("status", "draft").fillna("draft")

    invoices_df["paid_date"] = pd.NaT
    paid_mask = invoices_df["status"].eq("paid")
    invoices_df.loc[paid_mask, "paid_date"] = invoices_df.loc[paid_mask, "updated_at"]

    invoices_df["days_late"] = np.where(
        paid_mask & invoices_df["due_date"].notna() & invoices_df["paid_date"].notna(),
        (
            invoices_df["paid_date"].dt.floor("D")
            - invoices_df["due_date"].dt.floor("D")
        ).dt.days,
        np.nan,
    )

    invoices_df["days_late"] = invoices_df["days_late"].clip(lower=0)

    label_mask = (
        paid_mask
        & invoices_df["due_date"].notna()
        & invoices_df["paid_date"].notna()
        & invoices_df["issue_date"].notna()
        & (invoices_df["paid_date"] >= invoices_df["issue_date"])
    )
    invoices_df["late"] = np.where(
        label_mask,
        (invoices_df["days_late"] > 7).astype(int),
        np.nan,
    )
    invoices_df["label_available"] = label_mask.astype(int)

    for col in ["issue_date", "due_date", "created_at"]:
        invoices_df[col] = invoices_df[col].fillna(invoices_df["created_at"])

    return invoices_df[
        [
            "invoice_id",
            "client_id",
            "invoice_number",
            "status",
            "issue_date",
            "due_date",
            "total_gross",
            "created_at",
            "updated_at",
            "paid_date",
            "days_late",
            "late",
            "label_available",
        ]
    ].copy()


def load_appwrite_data() -> LoadedData:
    config = get_appwrite_config()

    client = Client().set_endpoint(config.endpoint).set_project(config.project_id)
    client.set_key(config.api_key)

    databases = Databases(client)

    invoice_records = _paginate_documents(
        databases=databases,
        database_id=config.database_id,
        collection_id=config.invoices_collection_id,
    )
    client_records = _paginate_documents(
        databases=databases,
        database_id=config.database_id,
        collection_id=config.clients_collection_id,
    )

    invoices_df = _build_invoice_df(invoice_records)
    clients_df = _build_client_df(client_records)

    return LoadedData(invoices=invoices_df, clients=clients_df)
