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
Use `enrich BATCH.json --dry-run` to preview additions to existing records from
reviewed bibliographic sources, then remove `--dry-run` to apply them. Each
entry in the batch's `records` array identifies a workspace, document ID,
file path, field values, and a source locator for each value. The command
preserves populated fields and adds provenance under `field_sources`, for
example `field_sources.published_date`.

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
| `POST /documents/search` | Browse the catalog, select metadata with dotted `fields`, or search the source text with a literal or regular expression. Pass `document_ids` and `fields` for a batch metadata read. Results include document IDs, applicable excerpts and a continuation cursor. |
| `GET /documents/{document_id}/metadata` | Read the full record. Use `view=fields` to list populated field paths and their types. Repeat `fields` to request selected paths, for example `fields=domain_details.datasets&fields=evidence`. |
| `GET /documents/{document_id}/content` | Discover reading selectors and retrieve bounded passages. Set `include_citations=true` to get a versioned `passage` locator, or supply a locator to replay its exact excerpt. Explicit `scope=full` streams the entire parsed text. |
| `GET /documents/{document_id}/images/{image_id}` | Read one extracted JPEG, PNG, GIF, or WebP asset from this document. Images up to 4 MiB are returned in their stored format. |

Use `/query/data` for semantic retrieval. Its chunks and references now include
`document_id` when the indexed source can be verified, so an agent can follow
a result into the same document. Page information is returned when supported
by the available source data.

Image-description chunks include `image_id` when their stored document is
verified. Pass `kb` and the `document_id` and `image_id` from `retrieve_kb_data`
to `get_document_image`. The MCP result includes one `ImageContent` block and
an `image_url` that opens the same image after signing in to the selected
knowledge base. Assets are read from the existing parse artifacts.
`search_kb` also includes both IDs on image hits. Following an image hit with
`fetch` preserves the IDs and the knowledge base in `metadata`.

A compact catalog can be requested without searching the full text.

```json
{
  "filters": {"document_type": "paper", "year_from": 2025},
  "fields": ["title", "published_date", "type_details.journal", "evidence_strength"],
  "limit": 200
}
```

For the next page, repeat the request with the returned `next_cursor`. Use
`year_from` and `year_to` to include records with year-only dates.
`year_min` and `year_max` are aliases for these inclusive bounds. The
`published_from` and `published_to` filters use complete dates. Methodology
filters use recorded metadata. A missing value differs from an explicit
negative, such as `cross_domain_validation: false`.

When a filter-only request selects fields explicitly, it can ask for up to
200 records. A projected page may return fewer records when it reaches its
response size budget. Follow `next_cursor` to continue. Selecting `id` in
`fields` refers to the always-returned `document_id`.

The existing MCP gateway exposes `search_documents`, `get_document_metadata`
and `read_document`. Use a local read for passages that should enter the
model context. A full read provides an authenticated download for the complete
parsed text. Bounded MCP reads include a `citation_url` that opens the excerpt
with its verified page, section, or line information. The link checks access
to the same knowledge base. A parsed revision change invalidates the locator.
For a chunk locator in the REST API, replay it with its original `chunk_id`.
The gateway leaves Ollama compatibility and WebUI assets to the HTTP APIs.
Knowledge-base management tools remain available in MCP.

Use `search_documents` for a research catalogue and `list_documents` for
processing status. MCP clients that read `structuredContent` can pass
`response_format: "structured"` alongside `kb` and `body` to get a short
text summary. Choose `legacy_json` for a complete text copy.

Install `mcp_documents.py` as `document_metadata_mcp.py` alongside the gateway
modules and apply `mcp_gateway.conf.in` with the local gateway root. Restart
the gateway after all of its LightRAG backends advertise the new operations.
