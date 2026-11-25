"""Document metadata stored on the existing PostgreSQL document row."""

import json
import time


_MIGRATION_LOCK = 430895494991544824
WORKER_LOCK = 430895494991544825


async def migrate(db):
    async with db.transaction():
        await db.execute("SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK)
        # Existing documents retain NULL, including any still in the pipeline.
        # Only documents inserted after the migration opt into model extraction.
        await db.execute(
            """ALTER TABLE lightrag_doc_status
                 ADD COLUMN IF NOT EXISTS echem_document_metadata jsonb,
                 ADD COLUMN IF NOT EXISTS echem_document_metadata_job jsonb;
               ALTER TABLE lightrag_doc_status ALTER COLUMN
                 echem_document_metadata_job SET DEFAULT '{"status":"pending"}'::jsonb"""
        )


async def next_document(db, workspace):
    return await db.fetchrow(
        """SELECT s.id, s.file_path, s.created_at, d.content_hash, d.content,
                  d.parse_format, d.sidecar_location,
                  COALESCE((s.echem_document_metadata_job->>'attempts')::int, 0) AS attempts
             FROM lightrag_doc_status s
             JOIN lightrag_doc_full d USING (workspace, id)
            WHERE s.workspace = $1 AND s.status IN ('processing', 'processed')
              AND lower(s.file_path) LIKE '%.pdf'
              AND s.metadata->>'is_duplicate' IS DISTINCT FROM 'true'
              AND s.echem_document_metadata IS NULL
              AND s.echem_document_metadata_job->>'status' = 'pending'
              AND COALESCE((s.echem_document_metadata_job->>'after')::bigint, 0) <= $2
              AND d.parse_format IS DISTINCT FROM 'pending_parse'
              AND length(trim(d.content)) > 0
            ORDER BY s.created_at, s.id LIMIT 1""",
        workspace,
        int(time.time()),
    )


async def finish_document(db, workspace, document, metadata, job):
    # Matching the parsed revision and the original row prevents a late call
    # from publishing onto a replaced or deleted document. A retry can pick up
    # the new revision without touching any of the original pipeline columns.
    return await db.fetchval(
        """UPDATE lightrag_doc_status s
              SET echem_document_metadata = $5::jsonb,
                  echem_document_metadata_job = $6::jsonb
             FROM lightrag_doc_full d
            WHERE s.workspace = $1 AND s.id = $2
              AND d.workspace = s.workspace AND d.id = s.id
              AND s.created_at IS NOT DISTINCT FROM $3
              AND d.content_hash IS NOT DISTINCT FROM $4
              AND s.metadata->>'is_duplicate' IS DISTINCT FROM 'true'
              AND s.status IN ('processing', 'processed')
              AND s.echem_document_metadata_job->>'status' = 'pending'
              AND s.echem_document_metadata IS NULL
        RETURNING s.id""",
        workspace,
        document["id"],
        document["created_at"],
        document["content_hash"],
        json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
        json.dumps(job),
    )


async def legacy_documents(db, workspace):
    rows = await db.fetch(
        """SELECT id, file_path FROM lightrag_doc_status
            WHERE workspace = $1 AND status = 'processed'
              AND metadata->>'is_duplicate' IS DISTINCT FROM 'true'
              AND (echem_document_metadata_job IS NULL
                   OR echem_document_metadata_job->>'status' = 'imported')""",
        workspace,
    )
    result = {}
    for row in rows:
        if row["file_path"] in result:
            raise ValueError(
                f"Ambiguous file name in workspace {workspace!r}: {row['file_path']!r}"
            )
        result[row["file_path"]] = row["id"]
    return result


async def import_metadata(db, workspace, document_id, metadata):
    """Fill a legacy row once. Re-running an import keeps the recorded value."""
    return await db.fetchval(
        """UPDATE lightrag_doc_status
              SET echem_document_metadata = $3::jsonb,
                  echem_document_metadata_job = '{"status":"imported"}'::jsonb
            WHERE workspace = $1 AND id = $2 AND status = 'processed'
              AND metadata->>'is_duplicate' IS DISTINCT FROM 'true'
              AND echem_document_metadata_job IS NULL
              AND echem_document_metadata IS NULL
        RETURNING id""",
        workspace,
        document_id,
        json.dumps(metadata, ensure_ascii=False),
    )


async def show_metadata(db, workspace, document_id):
    row = await db.fetchrow(
        """SELECT id, file_path, echem_document_metadata_job AS job,
                  echem_document_metadata AS metadata
             FROM lightrag_doc_status WHERE workspace = $1 AND id = $2""",
        workspace,
        document_id,
    )
    if row is None:
        raise ValueError("Document ID not found in this workspace")
    result = dict(row)
    for key in ("job", "metadata"):
        if result[key] is not None:
            result[key] = json.loads(result[key])
    return result


async def retry_document(db, workspace, document_id):
    return await db.fetchval(
        """UPDATE lightrag_doc_status
              SET echem_document_metadata_job = '{"status":"pending"}'::jsonb
            WHERE workspace = $1 AND id = $2
              AND echem_document_metadata_job->>'status' = 'failed'
              AND echem_document_metadata IS NULL
        RETURNING id""",
        workspace,
        document_id,
    )
