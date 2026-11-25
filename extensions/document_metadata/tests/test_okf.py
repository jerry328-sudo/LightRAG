import pytest

from extensions.document_metadata.okf import map_okf_record, okf_document_file


pytestmark = pytest.mark.offline


def test_preserves_bibliography_and_source_assessment():
    record = {
        "schema_version": "1.0.0",
        "document_type": "review_article",
        "bibliography": {
            "index": 27,
            "title": "A review of test cells",
            "authors": ["A. Jones", "Z. Smith"],
            "doi": "10.1234/paper",
            "year": 2017,
            "journal": "Journal of Test Cells",
            "publisher": "University Press",
            "url": "https://example.org/article",
        },
        "summary": {
            "one_sentence": "Review summary",
            "keywords_normalized": ["battery"],
        },
        "research_scope": {"chemistries": ["LFP"]},
        "study_details": {"evaluation": {"cross_domain_validation": False}},
        "limitations": {
            "assessor_identified": ["Small sample"],
            "overall_confidence": "medium",
            "evidence_strength": "weak",
        },
        "quality_control": {
            "review_status": "machine_checked",
            "reviewer": "reviewer-12",
            "review_date": "2023-04-05",
        },
        "evidence": [
            {
                "claim": "Small sample",
                "confidence": "high",
                "source_location": {"page": 14, "section": "Methods"},
            }
        ],
    }
    assert map_okf_record(record, source_revision="example-revision") == {
        "document_type": "paper",
        "document_subtype": "review_article",
        "title": "A review of test cells",
        "contributors": [
            {"name": "A. Jones", "role": "author"},
            {"name": "Z. Smith", "role": "author"},
        ],
        "identifiers": [{"type": "doi", "value": "10.1234/paper"}],
        "published_date": "2017",
        "publisher": "University Press",
        "source_urls": ["https://example.org/article"],
        "subjects": ["battery"],
        "summary": "Review summary",
        "type_details": {"journal": "Journal of Test Cells"},
        "domain_details": {
            "chemistries": ["LFP"],
            "evaluation": {"cross_domain_validation": False},
        },
        "limitations": {"assessor_identified": ["Small sample"]},
        "overall_confidence": "medium",
        "evidence_strength": "weak",
        "review_status": "machine_checked",
        "evidence": [
            {
                "claim": "Small sample",
                "confidence": "high",
                "source_location": {"page": 14, "section": "Methods"},
            }
        ],
        "assessment_source": {
            "kind": "okf_json",
            "record_path": "sources/records/027.json",
            "schema_version": "1.0.0",
            "revision": "example-revision",
            "reviewer": "reviewer-12",
            "review_date": "2023-04-05",
        },
    }


def test_missing_values_are_omitted_without_inventing_rating_or_identifiers():
    record = {
        "bibliography": {"index": 39, "year": None, "title": " ", "doi": ""},
        "summary": {"one_sentence": ""},
        "research_scope": {"battery_scale": "not_reported"},
        "limitations": {"overall_confidence": "unknown", "author_stated": []},
        "evidence": [
            {
                "claim": "Test conditions are described",
                "source_location": {"page": None, "section": "Methods"},
                "short_quote": None,
            }
        ],
    }
    assert map_okf_record(record) == {
        "assessment_source": {
            "kind": "okf_json",
            "record_path": "sources/records/039.json",
        },
        "evidence": [
            {
                "claim": "Test conditions are described",
                "source_location": {"section": "Methods"},
            }
        ],
    }


def test_resolves_pdf_basename_from_consistent_okf_bibliography():
    record = {"bibliography": {"index": 18, "pdf_file": "papers/018/paper.pdf"}}
    assert okf_document_file(record) == "018.pdf"


def test_refuses_to_map_inconsistent_record_to_an_imported_pdf():
    record = {"bibliography": {"index": 18, "pdf_file": "papers/081/paper.pdf"}}
    with pytest.raises(ValueError, match="inconsistent PDF path"):
        okf_document_file(record)
