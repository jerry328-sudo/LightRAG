"""Workspace-scoped bibliographic lookup and literal source search."""

import asyncio
import json
import re
from datetime import date
from typing import Annotated, Any, Literal

import asyncpg
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


DocumentId = Annotated[str, StringConstraints(min_length=1, max_length=255)]
FilterText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)
]
FieldPath = Annotated[str, StringConstraints(min_length=1, max_length=220)]
ISODate = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")]
SearchSource = Literal["content", "metadata", "both"]
SearchMatch = Literal["literal", "regex"]

_FIELD_PATTERN = re.compile(r"[\w-]+(?:\.[\w-]+){0,11}\Z")
_METADATA_FIELDS = (
    "title, document_type, document_subtype, contributors, published_date, publisher, "
    "identifiers, source_urls, language, subjects, summary, overall_confidence, evidence_strength, "
    "limitations, evidence, review_status, assessment_source, field_sources, "
    "type_details (including type_details.journal), domain_details"
)
_COMPACT_FIELDS = (
    "title",
    "document_type",
    "document_subtype",
    "published_date",
    "publisher",
    "type_details.journal",
    "language",
    "summary",
    "overall_confidence",
    "evidence_strength",
    "review_status",
)
_RECORD_ID_FIELDS = {"id", "document_id"}
_PREVIEW_MAX_LIMIT = 50
_PROJECTED_PAGE_MAX_BYTES = 256 * 1024
_PAGE_ENVELOPE_BYTES = len(b'{"documents":[],"next_cursor":}')


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
    document_subtype: FilterText | None = Field(
        None,
        description="Exact document subtype, for example research_article or review_article.",
    )
    year: int | None = Field(
        None,
        ge=1,
        le=9999,
        description="Exact publication year at any available date precision.",
    )
    year_from: int | None = Field(
        None,
        ge=1,
        le=9999,
        description="Inclusive earliest publication year. Also accepts year_min.",
    )
    year_to: int | None = Field(
        None,
        ge=1,
        le=9999,
        description="Inclusive latest publication year. Also accepts year_max.",
    )
    year_min: int | None = Field(
        None, ge=1, le=9999, description="Alias for the inclusive year_from bound."
    )
    year_max: int | None = Field(
        None, ge=1, le=9999, description="Alias for the inclusive year_to bound."
    )
    published_from: ISODate | None = Field(
        None,
        description="Inclusive earliest full publication date (YYYY-MM-DD). Use year_from for partial dates.",
    )
    published_to: ISODate | None = Field(
        None,
        description="Inclusive latest full publication date (YYYY-MM-DD). Use year_to for partial dates.",
    )
    research_task: FilterText | None = Field(
        None,
        description="Stored task, for example RUL_prediction in domain_details.prediction_tasks.",
    )
    overall_confidence: Literal["high", "medium", "low"] | None = None
    evidence_strength: Literal["strong", "moderate", "weak"] | None = Field(
        None, description="Exact stored evidence strength."
    )
    cross_domain_validation: bool | None = Field(
        None,
        description="Cross-domain validation as explicitly recorded. False excludes unknown values.",
    )
    method_tag: FilterText | None = Field(
        None, description="Exact stored method tag, for example zero-shot learning."
    )
    generalization_target: FilterText | None = Field(
        None,
        description="Exact stored generalization target, for example cross_dataset.",
    )

    @model_validator(mode="after")
    def validate_publication_range(self):
        for canonical, alias in (("year_from", "year_min"), ("year_to", "year_max")):
            value, alternate = getattr(self, canonical), getattr(self, alias)
            if value is not None and alternate is not None and value != alternate:
                raise ValueError(
                    f"{canonical} and {alias} must agree when both are supplied."
                )
            if value is None:
                setattr(self, canonical, alternate)
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be no later than year_to.")
        for value in (self.published_from, self.published_to):
            if value is not None:
                date.fromisoformat(value)
        if (
            self.published_from
            and self.published_to
            and self.published_from > self.published_to
        ):
            raise ValueError("published_from must be no later than published_to.")
        return self


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
    document_ids: list[DocumentId] | None = Field(
        None,
        min_length=1,
        max_length=100,
        description="Restrict results to these IDs. Combine with fields and omit query for batch metadata lookup.",
    )
    fields: list[FieldPath] | None = Field(
        None,
        max_length=24,
        description=(
            "Project only these dotted metadata paths. An empty list returns no metadata fields. "
            "The top-level document_id and file_path are always included. id is an alias "
            "for document_id. Omit fields for the legacy preview. "
            f"Common fields include {_METADATA_FIELDS}."
        ),
    )
    filters: DocumentSearchFilters = Field(default_factory=DocumentSearchFilters)
    limit: int = Field(
        10,
        ge=1,
        le=200,
        description=(
            f"At most {_PREVIEW_MAX_LIMIT} rows for text search or the legacy preview. "
            "Filter-only requests with explicit fields accept up to 200 rows. Projected results "
            "can end sooner at a 256 KiB page budget. Follow next_cursor for more rows. "
            "Extended pages selecting only compact preview fields cap each value at 512 characters."
        ),
    )
    cursor: DocumentId | None = Field(
        None,
        description="Next cursor from the previous page with the same search and filters.",
    )

    @model_validator(mode="after")
    def validate_search(self):
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
        if self.fields is not None and any(
            not _FIELD_PATTERN.fullmatch(field) for field in self.fields
        ):
            raise ValueError("Select valid dotted metadata paths.")
        if self.limit > _PREVIEW_MAX_LIMIT and (
            self.query is not None or self.fields is None
        ):
            raise ValueError(
                f"Limit above {_PREVIEW_MAX_LIMIT} requires explicit fields and no text query."
            )
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
           $13::text AS task, $14::text AS confidence, $15::int AS row_limit,
           $16::boolean AS project_metadata, $17::text AS doc_subtype,
           $18::text AS year_from, $19::text AS year_to,
           $20::text AS published_from, $21::text AS published_to, $22::text AS strength,
           $23::boolean AS cross_domain_validation, $24::text AS method_tag,
           $25::text AS generalization_target
), page AS MATERIALIZED (
    SELECT s.id, s.file_path, m.value AS metadata, body.value AS body,
           match_body.position AS body_position, match_metadata.*
    FROM lightrag_doc_status AS s
    CROSS JOIN request AS r
    LEFT JOIN lightrag_doc_full AS d
      ON r.scan_content AND d.workspace = s.workspace AND d.id = s.id
      AND d.parse_format IS DISTINCT FROM 'pending_parse'
    CROSS JOIN LATERAL (VALUES (s.echem_document_metadata)) AS m(value)
    CROSS JOIN LATERAL (VALUES (m.value->>'published_date')) AS publication(value)
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
      AND (r.doc_subtype IS NULL OR lower(m.value->>'document_subtype') = lower(r.doc_subtype))
      AND (r.year IS NULL OR left(publication.value, 4) = r.year)
      AND ((r.year_from IS NULL AND r.year_to IS NULL) OR (
          publication.value ~ '^[0-9]{4}($|-)'
          AND (r.year_from IS NULL OR left(publication.value, 4) >= r.year_from)
          AND (r.year_to IS NULL OR left(publication.value, 4) <= r.year_to)
      ))
      AND ((r.published_from IS NULL AND r.published_to IS NULL) OR (
          publication.value ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
          AND (r.published_from IS NULL OR publication.value >= r.published_from)
          AND (r.published_to IS NULL OR publication.value <= r.published_to)
      ))
      AND (r.task IS NULL OR EXISTS (
          SELECT FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(m.value#>'{domain_details,prediction_tasks}') = 'array'
            THEN m.value#>'{domain_details,prediction_tasks}' ELSE '[]'::jsonb END) AS task(value)
          WHERE lower(task.value) = lower(r.task)
      ))
      AND (r.confidence IS NULL OR lower(m.value->>'overall_confidence') = r.confidence)
      AND (r.strength IS NULL OR lower(m.value->>'evidence_strength') = r.strength)
      AND (r.cross_domain_validation IS NULL
           OR m.value#>'{domain_details,evaluation,cross_domain_validation}'
              = to_jsonb(r.cross_domain_validation))
      AND (r.method_tag IS NULL OR EXISTS (
          SELECT FROM jsonb_array_elements_text(CASE
              WHEN jsonb_typeof(m.value#>'{domain_details,ml_contribution,method_tags}') = 'array'
              THEN m.value#>'{domain_details,ml_contribution,method_tags}' ELSE '[]'::jsonb END) AS tag(value)
          WHERE lower(tag.value) = lower(r.method_tag)
      ))
      AND (r.generalization_target IS NULL OR EXISTS (
          SELECT FROM jsonb_array_elements_text(CASE
              WHEN jsonb_typeof(m.value#>'{domain_details,ml_contribution,generalization_target}') = 'array'
              THEN m.value#>'{domain_details,ml_contribution,generalization_target}' ELSE '[]'::jsonb END) AS target(value)
          WHERE lower(target.value) = lower(r.generalization_target)
      ))
      AND (r.pattern IS NULL OR match_body.position > 0 OR match_metadata.match_position > 0)
    ORDER BY s.id
    LIMIT (SELECT row_limit FROM request)
)
SELECT id, file_path, CASE WHEN (SELECT project_metadata FROM request) THEN metadata
       ELSE jsonb_strip_nulls(jsonb_build_object(
           'title', left(metadata->>'title', 512),
           'document_type', metadata->'document_type',
           'published_date', metadata->'published_date',
           'overall_confidence', metadata->'overall_confidence'
       )) END AS metadata,
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


def _select_fields(metadata, fields, *, text_limit=None):
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
        item = value[parts[-1]]
        if text_limit is not None and item is not None:
            if not isinstance(item, str):
                continue
            item = item[:text_limit]
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = item
    return output


async def _search(storage, request):
    flags = "q" if request.match == "literal" else "n"
    if not request.case_sensitive:
        flags += "i"
    filters = request.filters
    fields = (
        None
        if request.fields is None
        else [field for field in request.fields if field not in _RECORD_ID_FIELDS]
    )
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
        bool(fields),
        filters.document_subtype,
        f"{filters.year_from:04d}" if filters.year_from is not None else None,
        f"{filters.year_to:04d}" if filters.year_to is not None else None,
        filters.published_from,
        filters.published_to,
        filters.evidence_strength,
        filters.cross_domain_validation,
        filters.method_tag,
        filters.generalization_target,
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
    page_bytes = 0
    text_limit = (
        512
        if request.limit > _PREVIEW_MAX_LIMIT
        and fields is not None
        and set(fields).issubset(_COMPACT_FIELDS)
        else None
    )
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
        metadata = _json_object(row["metadata"]) or {}
        if fields is not None:
            metadata = _select_fields(metadata, fields, text_limit=text_limit)
        hit = DocumentSearchHit(
            document_id=row["id"],
            file_path=row["file_path"],
            metadata=metadata,
            matches=matches,
        )
        if fields is not None:
            # Reserve room to continue after this hit if the next one does not fit.
            cursor_bytes = len(
                json.dumps(hit.document_id, ensure_ascii=False).encode("utf-8")
            )
            size = (
                page_bytes
                + len(hit.model_dump_json().encode("utf-8"))
                + bool(documents)
            )
            if (
                size + _PAGE_ENVELOPE_BYTES + max(cursor_bytes, len(b"null"))
                > _PROJECTED_PAGE_MAX_BYTES
            ):
                if not documents:
                    raise HTTPException(
                        413,
                        "One projected record exceeds the response budget. Select fewer fields or read its single-document metadata.",
                    )
                break
            page_bytes = size
        documents.append(hit)
    return DocumentSearchResult(
        documents=documents,
        next_cursor=documents[-1].document_id if len(rows) > len(documents) else None,
    )


def register_document_routes(router: APIRouter, rag) -> None:
    @router.post(
        "/search",
        response_model=DocumentSearchResult,
        summary="Search documents by metadata and exact source text",
        description=(
            "Search processed documents in this workspace. Use fields for a compact catalogue, "
            "or combine fields with document_ids for batch metadata lookup. Projected pages can "
            "end at the byte budget before the requested limit. Follow next_cursor to continue. Omit query to filter "
            "without scanning text. Filters include exact or ranged publication dates, document "
            "type and subtype, confidence, evidence strength, research task, cross-domain "
            "validation, method tag and generalization target. Omitted metadata values remain unknown."
        ),
    )
    async def search_documents(request: DocumentSearchRequest):
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
