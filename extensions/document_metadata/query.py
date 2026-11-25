"""Attach document IDs to retrieved chunks for follow-up source reads."""

import logging
from collections import defaultdict
from functools import wraps

import asyncpg

from lightrag.chunk_schema import normalize_chunk_sidecar


logger = logging.getLogger(__name__)


def attach_query_document_ids(rag):
    original = rag.aquery_data

    @wraps(original)
    async def query_data(*args, **kwargs):
        result = await original(*args, **kwargs)
        data = result.get("data", {})
        chunks = data.get("chunks", [])
        ids = list(
            dict.fromkeys(
                chunk["chunk_id"] for chunk in chunks if chunk.get("chunk_id")
            )
        )
        if not ids:
            return result
        try:
            rows = await rag.text_chunks.get_by_ids(ids)
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError) as exc:
            logger.warning(
                "Could not resolve query document IDs: %s", type(exc).__name__
            )
            return result

        rows_by_id = {row["id"]: row for row in rows if row and row.get("id")}
        documents_by_reference = defaultdict(set)
        for chunk in chunks:
            row = rows_by_id.get(chunk.get("chunk_id"))
            # A reused chunk ID can point to another file after an index update.
            # Only use the stored ownership when it matches the returned source.
            if (
                row
                and row.get("full_doc_id")
                and row.get("file_path") == chunk.get("file_path")
            ):
                document_id = row["full_doc_id"]
                chunk["document_id"] = document_id
                sidecar = normalize_chunk_sidecar(row)
                if sidecar and sidecar["type"] == "drawing":
                    chunk["image_id"] = sidecar["id"]
                if chunk.get("reference_id"):
                    documents_by_reference[chunk["reference_id"]].add(document_id)

        for reference in data.get("references", []):
            documents = documents_by_reference.get(reference.get("reference_id"), ())
            if len(documents) == 1:
                reference["document_id"] = next(iter(documents))
        return result

    rag.aquery_data = query_data
