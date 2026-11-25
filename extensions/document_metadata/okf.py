"""Map an original OKF JSON record to LightRAG document metadata."""

from typing import Any


_PAPER_SUBTYPES = frozenset(
    {
        "research_article",
        "review_article",
        "perspective",
        "dataset_or_benchmark_article",
    }
)
_MISSING_LABELS = frozenset({"", "not_reported", "not_specified", "unknown"})


def _without_missing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, child in value.items()
            if (cleaned := _without_missing(child)) is not None
        } or None
    if isinstance(value, list):
        return [
            cleaned
            for child in value
            if (cleaned := _without_missing(child)) is not None
        ] or None
    if isinstance(value, str):
        return value if value.strip().casefold() not in _MISSING_LABELS else None
    return value


def okf_document_file(record: dict[str, Any]) -> str:
    """Resolve a record to the PDF basename used in the imported workspace."""
    bibliography = record.get("bibliography") or {}
    index = bibliography.get("index")
    if type(index) is not int or index < 1:
        raise ValueError("OKF record must have a positive integer bibliography.index")
    if bibliography.get("pdf_file") != f"papers/{index:03d}/paper.pdf":
        raise ValueError(f"OKF record {index:03d} has an inconsistent PDF path")
    return f"{index:03d}.pdf"


def map_okf_record(
    record: dict[str, Any], *, source_revision: str | None = None
) -> dict[str, Any]:
    """Copy fields with source values and retain the provenance of assessments."""
    bibliography = record.get("bibliography") or {}
    source_type = record.get("document_type")
    summary = record.get("summary") or {}
    limitations = record.get("limitations") or {}
    quality = record.get("quality_control") or {}
    index = bibliography.get("index")
    year = bibliography.get("year")
    doi = bibliography.get("doi")
    source_url = bibliography.get("url")

    domain_details = {
        **(record.get("research_scope") or {}),
        **(record.get("study_details") or {}),
        **{
            key: summary.get(key)
            for key in (
                "research_question",
                "approach",
                "main_findings",
                "key_contributions",
                "practical_significance",
            )
        },
        "reproducibility": record.get("reproducibility"),
    }
    return _without_missing(
        {
            "document_type": "paper" if source_type in _PAPER_SUBTYPES else source_type,
            "document_subtype": (
                source_type
                if source_type in _PAPER_SUBTYPES
                else record.get("document_subtype")
            ),
            "title": bibliography.get("title"),
            "contributors": [
                {"name": name, "role": "author"}
                for name in (_without_missing(bibliography.get("authors")) or [])
            ],
            "published_date": str(year) if year is not None else None,
            "publisher": bibliography.get("publisher"),
            "identifiers": [{"type": "doi", "value": doi}]
            if _without_missing(doi)
            else None,
            "source_urls": [source_url] if source_url else None,
            "language": bibliography.get("language"),
            "subjects": summary.get("keywords_normalized"),
            "summary": summary.get("one_sentence"),
            "overall_confidence": limitations.get("overall_confidence"),
            "evidence_strength": limitations.get("evidence_strength"),
            "limitations": {
                key: limitations.get(key)
                for key in ("author_stated", "assessor_identified", "scope_limitations")
            },
            "evidence": record.get("evidence"),
            "review_status": quality.get("review_status"),
            "assessment_source": {
                "kind": "okf_json",
                "record_path": (
                    f"sources/records/{index:03d}.json" if type(index) is int else None
                ),
                "schema_version": record.get("schema_version"),
                "revision": source_revision,
                "reviewer": quality.get("reviewer"),
                "review_date": quality.get("review_date"),
            },
            "type_details": {
                key: bibliography.get(key)
                for key in (
                    "journal",
                    "conference",
                    "volume",
                    "issue",
                    "pages",
                    "article_number",
                )
            },
            "domain_details": domain_details,
        }
    )
