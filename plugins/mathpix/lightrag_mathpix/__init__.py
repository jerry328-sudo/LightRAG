"""Registration for the Mathpix PDF parser."""

import os

from lightrag.parser.registry import ParserSpec, register_parser

CREDENTIAL_NAMES = ("MATHPIX_APP_ID", "MATHPIX_APP_KEY")


def credentials_configured() -> bool:
    return all(os.getenv(name, "").strip() for name in CREDENTIAL_NAMES)


def register() -> None:
    register_parser(
        ParserSpec(
            engine_name="mathpix",
            impl="lightrag_mathpix.parser:MathpixParser",
            suffixes=frozenset({"pdf"}),
            queue_group="mathpix",
            concurrency=max(1, int(os.getenv("MAX_PARALLEL_PARSE_MATHPIX", "2"))),
            endpoint_configured=credentials_configured,
            endpoint_requirement=lambda: "MATHPIX_APP_ID and MATHPIX_APP_KEY",
        )
    )
