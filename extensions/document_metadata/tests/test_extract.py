import json

import pytest

from extensions.document_metadata.extract import extract_metadata


pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_full_text_retains_grounded_metadata_and_leaves_unknowns_empty():
    body = (
        "# The Circuit Archive\n"
        "Érica Xu, editor. Published June 2023 by Alba Editions."
        "\nISBN 9781234567897. https://example.org/archive\n"
        "# Method\nAll measurements used a single\nreference device."
    )

    async def complete(system, user):
        assert json.loads(user) == {"stage": "full", "text": body}
        return json.dumps(
            {
                "title": "The Circuit Archive",
                "document_type": "book",
                "document_subtype": None,
                "contributors": [
                    {"name": "Érica Xu", "role": "editor"},
                    {"name": "Invented Author", "role": "author"},
                ],
                "published_date": "2023-06",
                "publisher": "Alba Editions",
                "identifiers": [
                    {"type": "isbn", "value": "9781234567897"},
                    {"type": "doi", "value": "10.123/invented"},
                ],
                "source_urls": [
                    "https://example.org/archive",
                    "https://example.org/invented",
                ],
                "overall_confidence": "medium",
                "evidence_strength": "weak",
                "evidence": [
                    {
                        "claim": "Measurements used a single device.",
                        "short_quote": "used a single reference device",
                        "supports_field": "overall_confidence",
                        "source_location": {"section": "Method", "page": None},
                    },
                    {
                        "claim": "The cataloguer is verified.",
                        "short_quote": "Approved by the assessor",
                        "supports_field": "evidence_strength",
                    },
                ],
                "review_status": "manually_verified",
                "domain_details": {"external_validation": False, "failures": 0},
            }
        )

    result = await extract_metadata(body, complete)
    assert result == {
        "title": "The Circuit Archive",
        "document_type": "book",
        "contributors": [{"name": "Érica Xu", "role": "editor"}],
        "published_date": "2023-06",
        "publisher": "Alba Editions",
        "identifiers": [{"type": "isbn", "value": "9781234567897"}],
        "source_urls": ["https://example.org/archive"],
        "overall_confidence": "medium",
        "evidence": [
            {
                "claim": "Measurements used a single device.",
                "short_quote": "used a single reference device",
                "supports_field": "overall_confidence",
                "source_location": {"section": "Method"},
            }
        ],
        "domain_details": {"external_validation": False, "failures": 0},
        "review_status": "machine_assessed",
        "assessment_source": {"kind": "model"},
    }


@pytest.mark.asyncio
async def test_long_document_every_utf8_section_contributes_to_result():
    body = "# 内容\n" + "核🌲⼏文本" * 15 + "\n\n# 章节\n" + "参考📖文档" * 18
    sections = []
    combined = []

    async def complete(system, user):
        request = json.loads(user)
        if request["stage"] == "section":
            sections.append(request["text"])
            assert len(request["text"].encode("utf-8")) <= 33
            return json.dumps({"parts": [request["part"]]})
        parts = [number for record in request["records"] for number in record["parts"]]
        if request["stage"] == "combine":
            combined.append(parts)
            return json.dumps({"parts": parts})
        assert request["stage"] == "final"
        return json.dumps({"domain_details": {"parts_reviewed": parts}})

    result = await extract_metadata(body, complete, max_input_bytes=33)
    assert "".join(sections) == body
    assert combined
    assert result["domain_details"]["parts_reviewed"] == list(
        range(1, len(sections) + 1)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", ["broken", "[]", '{"domain_details":{"missing":NaN}}']
)
async def test_invalid_model_output_is_reported_for_retry(response):
    async def complete(system, user):
        return response

    with pytest.raises(ValueError, match="not a valid JSON object"):
        await extract_metadata("# Parsed contents", complete)
