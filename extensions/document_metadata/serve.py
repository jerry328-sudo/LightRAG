"""Expose bibliographic metadata through the existing document list."""

import copy
import json
import logging
import os
from functools import wraps

import asyncpg


logger = logging.getLogger(__name__)

_SUMMARY_FIELDS = (
    "title",
    "document_type",
    "published_date",
    "publisher",
    "overall_confidence",
    "evidence_strength",
    "review_status",
    "summary",
)
_SOURCE_FIELDS = (*_SUMMARY_FIELDS, "contributors", "identifiers")


def _preview_text(value, limit):
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _summary(metadata):
    """Fit a bibliographic preview in the upstream plain-text dialog."""
    summary = {
        key: _preview_text(metadata[key], 240)
        for key in _SUMMARY_FIELDS
        if isinstance(metadata.get(key), str) and metadata[key].strip()
    }
    for field, label, limit in (
        ("contributors", "name", 6),
        ("identifiers", "value", 3),
    ):
        values = metadata.get(field)
        if not isinstance(values, list):
            continue
        labels = [
            _preview_text(item[label], 64)
            for item in values
            if isinstance(item, dict)
            and isinstance(item.get(label), str)
            and item[label].strip()
        ]
        if labels:
            summary[field] = labels[:limit]
            if len(labels) > limit:
                summary[field].append(f"+{len(labels) - limit}")
    return summary


def attach_document_metadata(storage):
    original = storage.get_docs_paginated

    @wraps(original)
    async def paginated(*args, **kwargs):
        documents, total = await original(*args, **kwargs)
        if not documents:
            return documents, total

        try:
            rows = await storage.db.query(
                """SELECT id, (
                           SELECT jsonb_object_agg(key, value)
                           FROM jsonb_each(echem_document_metadata)
                           WHERE key = ANY($3::text[])
                       ) AS document_metadata
                     FROM lightrag_doc_status
                    WHERE workspace = $1 AND id = ANY($2)
                      AND echem_document_metadata IS NOT NULL""",
                [
                    storage.workspace,
                    [document_id for document_id, _ in documents],
                    _SOURCE_FIELDS,
                ],
                multirows=True,
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError) as exc:
            logger.warning("Could not read document metadata: %s", type(exc).__name__)
            return documents, total

        metadata_by_id = {}
        for row in rows or ():
            metadata = row["document_metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if isinstance(metadata, dict) and metadata:
                summary = _summary(metadata)
                if summary:
                    metadata_by_id[row["id"]] = summary

        result = []
        for document_id, document in documents:
            if document_id in metadata_by_id:
                document = copy.copy(document)
                document.metadata = {
                    **(document.metadata or {}),
                    "document_metadata": metadata_by_id[document_id],
                }
            result.append((document_id, document))
        return result, total

    storage.get_docs_paginated = paginated


def main():
    from fastapi import APIRouter, Depends

    from lightrag.api.config import initialize_config
    from lightrag.api.run_with_gunicorn import main as run_gunicorn

    initialize_config()
    os.environ["LIGHTRAG_GUNICORN_MODE"] = "1"

    from lightrag.api import lightrag_server
    from lightrag.api.utils_api import get_combined_auth_dependency
    from lightrag.kg.postgres_impl import PGDocStatusStorage

    from .content import register_content_route
    from .documents import register_document_routes
    from .images import register_image_route
    from .query import attach_query_document_ids

    create_routes = lightrag_server.create_document_routes
    create_queries = lightrag_server.create_query_routes
    curation_file = os.getenv("LIGHTRAG_GRAPH_CURATION_FILE")

    @wraps(create_routes)
    def create_document_routes(rag, doc_manager, api_key=None):
        routes = create_routes(rag, doc_manager, api_key)
        if curation_file:
            from extensions.graph_curation.rules import attach_graph_curation

            attach_graph_curation(rag, curation_file)
        if isinstance(rag.doc_status, PGDocStatusStorage):
            attach_document_metadata(rag.doc_status)
            agent_routes = APIRouter(
                dependencies=[
                    Depends(
                        get_combined_auth_dependency(api_key, respect_whitelist=False)
                    )
                ]
            )
            register_document_routes(agent_routes, rag)
            register_content_route(agent_routes, rag, doc_manager.input_dir)
            register_image_route(agent_routes, rag, doc_manager.input_dir)
            routes.include_router(agent_routes)
        return routes

    @wraps(create_queries)
    def create_query_routes(rag, *args, **kwargs):
        routes = create_queries(rag, *args, **kwargs)
        if isinstance(rag.doc_status, PGDocStatusStorage):
            attach_query_document_ids(rag)
            for route in routes.routes:
                if route.path == "/query/data":
                    route.description += (
                        "\n\nRetrieved chunks and references include `document_id` "
                        "when their stored source can be verified. Use it to open "
                        "the document's metadata or read a specific passage. "
                        "Image description hits also include `image_id` for image retrieval."
                    )
        return routes

    # Let the regular application still own authentication and serialization.
    lightrag_server.create_document_routes = create_document_routes
    lightrag_server.create_query_routes = create_query_routes
    run_gunicorn()


if __name__ == "__main__":
    main()
