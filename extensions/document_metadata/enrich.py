"""Fill missing bibliographic fields from a reviewed, source-backed batch."""

from collections import Counter
from copy import deepcopy
from datetime import date
import json
import re


_TEXT_FIELDS = {
    "title",
    "published_date",
    "publisher",
    "language",
    "type_details.journal",
    "type_details.conference",
    "type_details.volume",
    "type_details.issue",
    "type_details.pages",
    "type_details.article_number",
    "type_details.edition",
}
_LIST_FIELDS = {
    "contributors": "name",
    "identifiers": "value",
    "source_urls": None,
}
_SOURCE_KINDS = {"pdf_title_page", "okf_json", "doi_registry", "publisher"}
_DOCUMENT_ID = re.compile(r"[A-Za-z0-9_-]{1,512}\Z")


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _valid_date(value):
    if not re.fullmatch(r"[0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?", value):
        return False
    parts = [int(part) for part in value.split("-")]
    try:
        date(*(parts + [1] * (3 - len(parts))))
    except ValueError:
        return False
    return True


def _valid_value(field, value):
    if field in _TEXT_FIELDS:
        return _nonempty_text(value) and (
            field != "published_date" or _valid_date(value)
        )
    if field not in _LIST_FIELDS or not isinstance(value, list) or not value:
        return False
    key = _LIST_FIELDS[field]
    if key is None:
        return all(
            _nonempty_text(item) and item.startswith(("https://", "http://"))
            for item in value
        )
    return all(
        isinstance(item, dict)
        and _nonempty_text(item.get(key))
        and all(_nonempty_text(text) for text in item.values())
        and (field != "identifiers" or _nonempty_text(item.get("type")))
        for item in value
    )


def validate_batch(batch):
    """Reject ambiguous target rows, fields, and provenance before any query."""
    entries = batch.get("records")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Supply a nonempty records list.")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each enrichment entry must be an object.")
        workspace = entry.get("workspace")
        document_id = entry.get("document_id")
        file_path = entry.get("file_path")
        fields = entry.get("fields")
        sources = entry.get("sources")
        if (
            not _nonempty_text(workspace)
            or not isinstance(document_id, str)
            or not _DOCUMENT_ID.fullmatch(document_id)
            or not _nonempty_text(file_path)
            or not isinstance(fields, dict)
            or not fields
            or not isinstance(sources, dict)
            or sources.keys() != fields.keys()
        ):
            raise ValueError(
                "Each row needs a workspace, document ID, file, fields and sources."
            )
        key = (workspace, document_id)
        if key in seen:
            raise ValueError(f"Repeated metadata enrichment target: {key!r}")
        seen.add(key)
        for field, value in fields.items():
            source = sources[field]
            if not _valid_value(field, value):
                raise ValueError(
                    f"Invalid bibliographic value for {document_id}: {field}"
                )
            if (
                not isinstance(source, dict)
                or source.get("kind") not in _SOURCE_KINDS
                or not _nonempty_text(source.get("locator"))
            ):
                raise ValueError(f"Invalid evidence locator for {document_id}: {field}")
    return entries


def _field_at(record, path):
    current = record
    for part in path:
        if not isinstance(current, dict):
            return current
        current = current.get(part)
    return current


def _missing(value):
    return (
        value is None
        or value == []
        or value == {}
        or (isinstance(value, str) and not value.strip())
    )


def _fill(record, path, value):
    current = record
    for part in path[:-1]:
        if _missing(current.get(part)):
            current[part] = {}
        current = current[part]
    current[path[-1]] = value


def enrich_record(original, entry):
    updated = deepcopy(original)
    added = []
    for field, value in entry["fields"].items():
        path = field.split(".")
        if not _missing(_field_at(updated, path)):
            continue
        parent = _field_at(updated, path[:-1])
        if not isinstance(parent, dict) and not _missing(parent):
            continue
        source_path = ["field_sources", *path]
        source_parent = _field_at(updated, source_path[:-1])
        if not isinstance(source_parent, dict) and not _missing(source_parent):
            raise ValueError(f"Invalid field provenance container for {field}")
        _fill(updated, path, value)
        _fill(
            updated,
            source_path,
            {
                "kind": entry["sources"][field]["kind"],
                "locator": entry["sources"][field]["locator"],
            },
        )
        added.append(field)
    return updated, added


async def enrich(db, workspace, batch, *, dry_run):
    entries = [
        entry for entry in validate_batch(batch) if entry["workspace"] == workspace
    ]
    found = Counter()
    documents = 0
    async with db.transaction(readonly=dry_run):
        for entry in entries:
            key = (workspace, entry["document_id"], entry["file_path"])
            record = await db.fetchval(
                """SELECT echem_document_metadata FROM lightrag_doc_status
                   WHERE workspace = $1 AND id = $2 AND file_path = $3
                     AND status = 'processed'
                     AND metadata->>'is_duplicate' IS DISTINCT FROM 'true'""",
                *key,
            )
            original = json.loads(record) if record is not None else None
            if not isinstance(original, dict):
                raise ValueError(
                    f"Processed document with metadata not found: {key[1]}"
                )
            updated, added = enrich_record(original, entry)
            if not added:
                continue
            if not dry_run:
                written = await db.fetchval(
                    """UPDATE lightrag_doc_status SET echem_document_metadata = $5::jsonb
                       WHERE workspace = $1 AND id = $2 AND file_path = $3
                         AND status = 'processed'
                         AND metadata->>'is_duplicate' IS DISTINCT FROM 'true'
                         AND echem_document_metadata = $4::jsonb
                       RETURNING id""",
                    *key,
                    record,
                    json.dumps(updated, ensure_ascii=False),
                )
                if written != key[1]:
                    raise RuntimeError(f"Metadata changed during enrichment: {key[1]}")
            documents += 1
            found.update(added)
    return {
        "workspace": workspace,
        "dry_run": dry_run,
        "requested_documents": len(entries),
        "documents_with_additions": documents,
        "fields_added": dict(found),
    }
