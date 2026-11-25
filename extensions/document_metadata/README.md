# LightRAG document metadata

This deployment extension uses the existing PostgreSQL document rows. Its two
columns hold bibliographic metadata and extraction state. LightRAG owns the
original columns, including its processing metadata. The worker uses the
existing OpenAI extraction binding with one in-flight metadata request across
the two workspaces. It reads completed parsing output once a PDF reaches
`processing` or `processed`. Existing rows are excluded from model extraction
when the migration first adds the columns. A job interrupted before writing a
result remains pending. A failed extraction can be requeued independently.

Use the deployment Python and the workspace's existing `.env`. From the project
root, `python -m extensions.document_metadata --env PATH show DOC_ID` displays
the stored record. Replace `show` with `retry` to requeue a failed metadata task.
Run `migrate` once against the deployment database before starting the workers.
For the one-time OKF import, use `backfill RECORDS --revision COMMIT --dry-run`
to verify the file matches, then remove `--dry-run`. Use
`import-one FILE_NAME METADATA_JSON` for a separately sourced legacy record.

Start the API with `python -m extensions.document_metadata.serve` in place of
`lightrag-gunicorn`, with the same launcher arguments. The document list includes
a bounded preview under `metadata.document_metadata`, which the existing status
details dialog renders alongside the processing metadata. Use `show DOC_ID` for
the full record, including the original contributors, identifiers and evidence.
