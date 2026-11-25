"""Workspace-scoped bibliographic lookup and literal source search."""

import asyncio
import json
import re
from typing import Annotated, Any, Literal

import asyncpg
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


DocumentId = Annotated[str, StringConstraints(min_length=1, max_length=255)]
FilterText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)
]
FieldPath = Annotated[str, StringConstraints(min_length=1, max_length=220)]
SearchSource = Literal["content", "metadata", "both"]
SearchMatch = Literal["literal", "regex"]

_FIELD_PATTERN = re.compile(r"[\w-]+(?:\.[\w-]+){0,11}\Z")
_METADATA_FIELDS = (
    "title, document_type, document_subtype, contributors, published_date, publisher, "
    "identifiers, source_urls, language, subjects, summary, overall_confidence, evidence_strength, "
    "limitations, evidence, review_status, assessment_source, type_details, domain_details"
)


class DocumentSearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: FilterText | None = Field(
        None, description="Case-insensitive title substring."
    )
    author: FilterText | None = Field(
        None, description="Case-insensitive author name substring."
    )
    doi: FilterText | None = Field(
        None,
        description="Exact DOI, optionally prefixed with doi: or https://doi.org/.",
    )
    document_type: FilterText | None = Field(
        None, description="Stored document type, for example paper or book."
    )
    year: int | None = Field(None, ge=1, le=9999)
    research_task: FilterText | None = Field(
        None,
        description="Stored task, for example RUL_prediction in domain_details.prediction_tasks.",
    )
    overall_confidence: Literal["high", "medium", "low"] | None = None


class DocumentSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        None,
        min_length=1,
        max_length=256,
        description="Source text pattern. Omit for filter-only lookup.",
    )
    match: SearchMatch = Field(
        "literal",
        description="Use literal text or a PostgreSQL advanced regular expression.",
    )
    search_in: SearchSource = Field(
        "both", description="Search the parsed document, its metadata, or both."
    )
    case_sensitive: bool = False
    document_ids: list[DocumentId] | None = Field(None, min_length=1, max_length=100)
    filters: DocumentSearchFilters = Field(default_factory=DocumentSearchFilters)
    limit: int = Field(10, ge=1, le=50)
    cursor: DocumentId | None = Field(
        None,
        description="Next cursor from the previous page with the same search and filters.",
    )

    @model_validator(mode="after")
    def reject_nul(self):
        # PostgreSQL text parameters cannot contain a NUL character.
        if any(
            "\x00" in value
            for value in (
                self.query or "",
                self.cursor or "",
                *(self.document_ids or ()),
                *(
                    item
                    for item in self.filters.model_dump().values()
                    if isinstance(item, str)
                ),
            )
        ):
            raise ValueError("Search values may not contain a NUL character")
        return self


class DocumentMatch(BaseModel):
    source: Literal["content", "metadata"]
    snippet: str
    field: str | None = None
    line: int | None = Field(None, description="One-based line in the parsed document.")


class DocumentSearchHit(BaseModel):
    document_id: str
    file_path: str | None
    metadata: dict[str, Any]
    matches: list[DocumentMatch]


class DocumentSearchResult(BaseModel):
    documents: list[DocumentSearchHit]
    next_cursor: str | None


class MetadataField(BaseModel):
    path: str
    type: Literal["object", "array", "string", "integer", "number", "boolean", "null"]


class DocumentMetadataResult(BaseModel):
    document_id: str
    file_path: str | None
    metadata: dict[str, Any] | None = None
    fields: list[MetadataField] | None = None


_SEARCH_SQL = r"""
WITH request AS (
    SELECT $1::text AS workspace, $2::text AS cursor, $3::text[] AS ids,
           $4::text AS pattern, $5::text AS flags, $6::boolean AS scan_content,
           $7::boolean AS scan_metadata, $8::text AS title, $9::text AS author,
           $10::text AS doi, $11::text AS doc_type, $12::text AS year,
           $13::text AS task, $14::text AS confidence, $15::int AS row_limit
), page AS MATERIALIZED (
    SELECT s.id, s.file_path, m.value AS metadata, body.value AS body,
           match_body.position AS body_position, match_metadata.*
    FROM lightrag_doc_status AS s
    CROSS JOIN request AS r
    LEFT JOIN lightrag_doc_full AS d
      ON r.scan_content AND d.workspace = s.workspace AND d.id = s.id
      AND d.parse_format IS DISTINCT FROM 'pending_parse'
    CROSS JOIN LATERAL (VALUES (s.echem_document_metadata)) AS m(value)
    CROSS JOIN LATERAL (VALUES (
        CASE WHEN d.parse_format = 'lightrag' AND starts_with(d.content, '{{LRdoc}}')
             THEN substr(d.content, 10) ELSE d.content END
    )) AS body(value)
    CROSS JOIN LATERAL (VALUES (
        CASE WHEN r.scan_content AND body.value IS NOT NULL
             THEN regexp_instr(body.value, r.pattern, 1, 1, 0, r.flags) END
    )) AS match_body(position)
    LEFT JOIN LATERAL (
        SELECT item.field AS match_field, item.value AS match_value, item.position AS match_position
        FROM (
            SELECT metadata.priority, metadata.field, metadata.value,
                   regexp_instr(metadata.value, r.pattern, 1, 1, 0, r.flags) AS position
            FROM (VALUES
                (1, 'title', m.value->>'title'),
                (2, 'file_path', s.file_path),
                (3, 'contributors', (m.value->'contributors')::text),
                (4, 'identifiers', (m.value->'identifiers')::text),
                (5, 'summary', m.value->>'summary'),
                (6, 'metadata', m.value::text)
            ) AS metadata(priority, field, value)
            WHERE r.scan_metadata
              AND metadata.value IS NOT NULL
        ) AS item
        WHERE item.position > 0
        ORDER BY item.priority
        LIMIT 1
    ) AS match_metadata ON true
    WHERE s.workspace = r.workspace AND s.status = 'processed'
      AND (r.cursor IS NULL OR s.id > r.cursor)
      AND (r.ids IS NULL OR s.id = ANY(r.ids))
      AND (r.title IS NULL OR strpos(lower(m.value->>'title'), lower(r.title)) > 0)
      AND (r.author IS NULL OR EXISTS (
          SELECT FROM jsonb_array_elements(CASE WHEN jsonb_typeof(m.value->'contributors') = 'array'
            THEN m.value->'contributors' ELSE '[]'::jsonb END) AS author(value)
          WHERE lower(author.value->>'role') = 'author'
            AND strpos(lower(author.value->>'name'), lower(r.author)) > 0
      ))
      AND (r.doi IS NULL OR EXISTS (
          SELECT FROM jsonb_array_elements(CASE WHEN jsonb_typeof(m.value->'identifiers') = 'array'
            THEN m.value->'identifiers' ELSE '[]'::jsonb END) AS identifier(value)
          WHERE lower(identifier.value->>'type') = 'doi'
            AND regexp_replace(lower(trim(identifier.value->>'value')), '^(https?://(dx\.)?doi\.org/|doi:[[:space:]]*)', '')
              = regexp_replace(lower(r.doi), '^(https?://(dx\.)?doi\.org/|doi:[[:space:]]*)', '')
      ))
      AND (r.doc_type IS NULL OR lower(m.value->>'document_type') = lower(r.doc_type))
      AND (r.year IS NULL OR left(m.value->>'published_date', 4) = r.year)
      AND (r.task IS NULL OR EXISTS (
          SELECT FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(m.value#>'{domain_details,prediction_tasks}') = 'array'
            THEN m.value#>'{domain_details,prediction_tasks}' ELSE '[]'::jsonb END) AS task(value)
          WHERE lower(task.value) = lower(r.task)
      ))
      AND (r.confidence IS NULL OR lower(m.value->>'overall_confidence') = r.confidence)
      AND (r.pattern IS NULL OR match_body.position > 0 OR match_metadata.match_position > 0)
    ORDER BY s.id
    LIMIT (SELECT row_limit FROM request)
)
SELECT id, file_path, jsonb_strip_nulls(jsonb_build_object(
           'title', left(metadata->>'title', 512),
           'document_type', metadata->'document_type',
           'published_date', metadata->'published_date',
           'overall_confidence', metadata->'overall_confidence'
       )) AS metadata,
       CASE WHEN body_position > 0 THEN substr(body, greatest(1, body_position - 60), 220) END AS body_snippet,
       CASE WHEN body_position > 0 THEN 1 + length(left(body, body_position - 1))
         - length(replace(left(body, body_position - 1), E'\n', '')) END AS body_line,
       match_field,
       CASE WHEN match_position > 0 THEN substr(match_value, greatest(1, match_position - 60), 220) END AS metadata_snippet
FROM page
ORDER BY id
"""


def _json_object(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else None


def _field_inventory(value, prefix=""):
    for key, item in value.items():
        path = f"{prefix}{key}"
        if "." in key or len(path) > 220 or not _FIELD_PATTERN.fullmatch(path):
            # Whole parent objects still include opaque, non-addressable keys.
            continue
        kind = {
            type(None): "null",
            bool: "boolean",
            int: "integer",
            float: "number",
            str: "string",
            list: "array",
            dict: "object",
        }[type(item)]
        yield MetadataField(path=path, type=kind)
        if kind == "object":
            yield from _field_inventory(item, f"{path}.")


def _select_fields(metadata, fields):
    output = {}
    # Selecting a parent also selects every child.
    selected = set(fields)
    for field in dict.fromkeys(fields):
        parts = field.split(".")
        if any(".".join(parts[:index]) in selected for index in range(1, len(parts))):
            continue
        value, target = metadata, output
        for part in parts[:-1]:
            value = value.get(part) if isinstance(value, dict) else None
        if not isinstance(value, dict) or parts[-1] not in value:
            continue
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value[parts[-1]]
    return output


async def _search(storage, request):
    flags = "q" if request.match == "literal" else "n"
    if not request.case_sensitive:
        flags += "i"
    filters = request.filters
    params = (
        storage.workspace,
        request.cursor,
        request.document_ids,
        request.query,
        flags,
        request.query is not None and request.search_in in ("content", "both"),
        request.query is not None and request.search_in in ("metadata", "both"),
        filters.title,
        filters.author,
        filters.doi,
        filters.document_type,
        f"{filters.year:04d}" if filters.year is not None else None,
        filters.research_task,
        filters.overall_confidence,
        request.limit + 1,
    )

    async def query(connection):
        async with connection.transaction(readonly=True):
            await connection.execute("SET LOCAL statement_timeout = '3000ms'")
            if request.query is not None and request.match == "regex":
                await connection.fetchval(
                    "SELECT regexp_instr('', $1, 1, 1, 0, $2)", request.query, flags
                )
            return await connection.fetch(_SEARCH_SQL, *params)

    try:
        async with asyncio.timeout(5):
            rows = await storage.db._run_with_retry(query)
    except asyncpg.InvalidRegularExpressionError as error:
        raise HTTPException(422, "Invalid search regular expression.") from error
    except (TimeoutError, asyncpg.QueryCanceledError) as error:
        raise HTTPException(
            408, "Document search timed out. Narrow the query or the document scope."
        ) from error

    documents = []
    for row in rows[: request.limit]:
        matches = []
        if row["body_snippet"] is not None:
            matches.append(
                DocumentMatch(
                    source="content", line=row["body_line"], snippet=row["body_snippet"]
                )
            )
        if row["metadata_snippet"] is not None:
            matches.append(
                DocumentMatch(
                    source="metadata",
                    field=row["match_field"],
                    snippet=row["metadata_snippet"],
                )
            )
        documents.append(
            DocumentSearchHit(
                document_id=row["id"],
                file_path=row["file_path"],
                metadata=_json_object(row["metadata"]) or {},
                matches=matches,
            )
        )
    return DocumentSearchResult(
        documents=documents,
        next_cursor=documents[-1].document_id if len(rows) > request.limit else None,
    )


def register_document_routes(router: APIRouter, rag) -> None:
    @router.post(
        "/search",
        response_model=DocumentSearchResult,
        summary="Search documents by metadata and exact source text",
    )
    async def search_documents(request: DocumentSearchRequest):
        """Search this workspace's processed documents. Each hit shows the first relevant location per search source."""
        return await _search(rag.doc_status, request)

    @router.get(
        "/{document_id}/metadata",
        response_model=DocumentMetadataResult,
        response_model_exclude_unset=True,
        summary="Read bibliographic metadata or discover available fields",
    )
    async def read_document_metadata(
        document_id: Annotated[
            DocumentId, Path(description="Document ID in the current workspace.")
        ],
        view: Annotated[
            Literal["full", "fields"],
            Query(description="Use fields to list the stored paths and types."),
        ] = "full",
        fields: Annotated[
            list[FieldPath] | None,
            Query(
                max_length=24,
                description=f"Repeat this parameter to select dotted paths. Arrays are selected as a whole. Common fields include {_METADATA_FIELDS}.",
            ),
        ] = None,
    ):
        """Return the stored record, a projection, or its field inventory."""
        if "\x00" in document_id:
            raise HTTPException(422, "Document ID may not contain a NUL character.")
        if fields and (
            view == "fields"
            or any(not _FIELD_PATTERN.fullmatch(field) for field in fields)
        ):
            raise HTTPException(
                422,
                "Select valid dotted fields with view=full, or use view=fields alone.",
            )
        storage = rag.doc_status
        row = await storage.db.query(
            """SELECT file_path, echem_document_metadata AS metadata FROM lightrag_doc_status
               WHERE workspace = $1 AND id = $2 AND status IS DISTINCT FROM 'deleting'""",
            [storage.workspace, document_id],
        )
        if row is None:
            raise HTTPException(404, "Document not found in this workspace.")
        metadata = _json_object(row["metadata"])
        base = {"document_id": document_id, "file_path": row["file_path"]}
        if view == "fields":
            return DocumentMetadataResult(
                **base, fields=list(_field_inventory(metadata or {}))
            )
        if fields and metadata is not None:
            metadata = _select_fields(metadata, fields)
        return DocumentMetadataResult(**base, metadata=metadata)
