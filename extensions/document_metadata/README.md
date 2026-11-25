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

## Agent retrieval API

The same server documents the following authenticated endpoints at `/docs`
under each knowledge base's URL prefix. The existing workspace permissions
also apply to these read-only operations.

| Method and endpoint | Purpose |
| --- | --- |
| `POST /documents/search` | Filter by bibliography or research metadata, and search text with literal or regular-expression matching. An optional `document_ids` list limits the search scope. Results include document IDs, short excerpts and a continuation cursor. |
| `GET /documents/{document_id}/metadata` | Read the full record. Use `view=fields` to list populated field paths and their types. Repeat `fields` to request selected paths, for example `fields=domain_details.datasets&fields=evidence`. |
| `GET /documents/{document_id}/content` | Discover the available sections and reading selectors, then use them to verify source passages. Local reads have a response limit. Explicit `scope=full` streams the entire parsed text. |

Use `/query/data` for semantic retrieval. Its chunks and references now include
`document_id` when the indexed source can be verified, so an agent can follow
a result into the same document. Page information is returned when supported
by the available source data.

The existing MCP gateway exposes `search_documents`, `get_document_metadata`
and `read_document`. Use a local read for passages that should enter the
model context. A full read provides an authenticated download for the complete
parsed text. The gateway leaves Ollama compatibility and WebUI assets to
the HTTP APIs. Knowledge-base management tools remain available in MCP.

Install `mcp_documents.py` as `document_metadata_mcp.py` alongside the gateway
modules and apply `mcp_gateway.conf.in` with the local gateway root. Restart
the gateway after all of its LightRAG backends advertise the new operations.
