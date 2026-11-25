"""Operate the document metadata extension using a LightRAG deployment .env."""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

from . import store
from .okf import map_okf_record, okf_document_file


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    commands.add_parser("run")
    backfill = commands.add_parser("backfill")
    backfill.add_argument("records", type=Path)
    backfill.add_argument("--revision", required=True)
    backfill.add_argument("--dry-run", action="store_true")
    enrich = commands.add_parser("enrich")
    enrich.add_argument("batch", type=Path)
    enrich.add_argument("--dry-run", action="store_true")
    import_one = commands.add_parser("import-one")
    import_one.add_argument("file_path")
    import_one.add_argument("metadata", type=Path)
    commands.add_parser("show").add_argument("document_id")
    commands.add_parser("retry").add_argument("document_id")
    return parser.parse_args()


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a nonempty JSON object in {path.name}")
    return value


async def backfill_okf(db, workspace, records, revision, dry_run):
    source = {}
    for path in sorted(records.glob("*.json")):
        record = read_json(path)
        name = okf_document_file(record)
        if name != path.with_suffix(".pdf").name or name in source:
            raise ValueError(
                f"Invalid or duplicate OKF document reference in {path.name}"
            )
        source[name] = map_okf_record(record, source_revision=revision)
    if not source:
        raise ValueError("No OKF JSON records found")

    written = 0
    async with db.transaction():
        documents = await store.legacy_documents(db, workspace)
        matches = source.keys() & documents.keys()
        if not dry_run:
            for name in sorted(matches):
                result = await store.import_metadata(
                    db, workspace, documents[name], source[name]
                )
                written += result is not None
    return {
        "workspace": workspace,
        "dry_run": dry_run,
        "records": len(source),
        "matched": len(matches),
        "imported": written,
        "documents_without_records": len(documents.keys() - source.keys()),
        "records_without_documents": len(source.keys() - documents.keys()),
    }


async def run(args):
    load_dotenv(args.env)
    if (
        os.getenv("LIGHTRAG_KV_STORAGE") != "PGKVStorage"
        or os.getenv("LIGHTRAG_DOC_STATUS_STORAGE") != "PGDocStatusStorage"
    ):
        raise ValueError("PGKVStorage and PGDocStatusStorage must be configured")
    logical_workspace = re.sub(r"[^A-Za-z0-9_]", "_", os.getenv("WORKSPACE", ""))
    workspace = os.getenv("POSTGRES_WORKSPACE") or logical_workspace or "default"
    db = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DATABASE", "postgres"),
        ssl=os.getenv("POSTGRES_SSL_MODE") or None,
        command_timeout=30,
    )
    try:
        if args.command == "migrate":
            await store.migrate(db)
            return {"migration": "complete"}
        if args.command == "backfill":
            return await backfill_okf(
                db, workspace, args.records, args.revision, args.dry_run
            )
        if args.command == "enrich":
            from .enrich import enrich

            return await enrich(
                db, workspace, read_json(args.batch), dry_run=args.dry_run
            )
        if args.command == "import-one":
            metadata = read_json(args.metadata)
            async with db.transaction():
                documents = await store.legacy_documents(db, workspace)
                document_id = documents.get(args.file_path)
                if document_id is None:
                    raise ValueError(
                        "The file does not match a legacy document in this workspace"
                    )
                imported = await store.import_metadata(
                    db, workspace, document_id, metadata
                )
            return {"id": document_id, "imported": imported is not None}
        if args.command == "show":
            return await store.show_metadata(db, workspace, args.document_id)
        if args.command == "retry":
            document_id = await store.retry_document(db, workspace, args.document_id)
            return {"id": args.document_id, "requeued": document_id is not None}

        from .worker import run as run_worker

        input_dir = Path(os.getenv("INPUT_DIR", "./inputs")).resolve()
        if logical_workspace:
            input_dir /= logical_workspace
        await run_worker(db, workspace, input_dir)
    finally:
        await db.close()


def main():
    logging.basicConfig(level=logging.INFO)
    args = arguments()
    try:
        result = asyncio.run(run(args))
    except KeyboardInterrupt:
        return
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
