"""Generate metadata from the saved document without coupling to indexing."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from openai import AsyncOpenAI

from . import store
from .extract import extract_metadata


logger = logging.getLogger(__name__)
ATTEMPT_LIMIT = 3
POLL_SECONDS = 15


def parsed_text(document, input_dir):
    body = document["content"]
    if document["parse_format"] == "lightrag":
        body = body.removeprefix("{{LRdoc}}").lstrip("\n")
    uri = urlsplit(document["sidecar_location"] or "")
    if uri.scheme != "file" or uri.netloc not in ("", "localhost"):
        return body
    directory = Path(unquote(uri.path)).resolve()
    if (
        directory.parent not in (input_dir, input_dir / "__parsed__")
        or not directory.is_dir()
    ):
        return body

    blocks = sorted(directory.glob("*.blocks.jsonl"))
    if len(blocks) != 1:
        return body
    stem = blocks[0].name.removesuffix(".blocks.jsonl")
    context = []
    for modality in ("drawings", "tables", "equations"):
        path = directory / f"{stem}.{modality}.json"
        if not path.is_file() or path.resolve().parent != directory:
            continue
        items = json.loads(path.read_text(encoding="utf-8")).get(modality, {})
        for item in items.values():
            analysis = item.get("llm_analyze_result")
            if (
                not isinstance(analysis, dict)
                or analysis.get("status") != "success"
                or not analysis.get("description")
            ):
                continue
            entry = {
                key: item[key]
                for key in ("heading", "parent_headings", "caption", "id")
                if item.get(key)
            }
            entry.update(description=analysis["description"], modality=modality)
            context.append(json.dumps(entry, ensure_ascii=False))
    if context:
        return (
            body
            + "\n\n[Supplementary machine-generated figure/table descriptions]\n\n"
            + "\n\n".join(context)
        )
    return body


def openai_settings():
    binding = os.getenv("EXTRACT_LLM_BINDING") or os.getenv("LLM_BINDING", "openai")
    if binding != "openai":
        raise ValueError("Document metadata requires the OpenAI extraction binding")
    model = os.getenv("EXTRACT_LLM_MODEL") or os.getenv("LLM_MODEL")
    api_key = (
        os.getenv("EXTRACT_LLM_BINDING_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    host = os.getenv("EXTRACT_LLM_BINDING_HOST") or os.getenv("LLM_BINDING_HOST")
    if not model or not api_key:
        raise ValueError("The configured extraction model or credential is missing")
    return model, host, api_key


async def handle_document(db, workspace, document, input_dir, complete, model):
    try:
        body = await asyncio.to_thread(parsed_text, document, input_dir)
        metadata = await extract_metadata(body, complete)
        metadata["assessment_source"]["model"] = model
    except Exception as error:
        attempt = document["attempts"] + 1
        state = "failed" if attempt >= ATTEMPT_LIMIT else "pending"
        job = {"status": state, "attempts": attempt, "error": type(error).__name__}
        if state == "pending":
            job["after"] = int(time.time()) + 60 * 2 ** (attempt - 1)
        await store.finish_document(db, workspace, document, None, job)
        # Provider messages could contain material from the document.
        logger.warning(
            "Metadata attempt %s for %s: %s", attempt, document["id"], job["error"]
        )
        return
    saved = await store.finish_document(
        db, workspace, document, metadata, {"status": "done"}
    )
    if saved:
        logger.info("Generated metadata for %s", saved)


async def run(db, workspace, input_dir):
    model, host, api_key = openai_settings()
    logger.info("Metadata worker started for workspace %s using %s", workspace, model)
    async with AsyncOpenAI(
        api_key=api_key, base_url=host, timeout=180, max_retries=0
    ) as client:

        async def complete(system, prompt):
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            if choice.finish_reason != "stop" or not choice.message.content:
                raise ValueError("Metadata completion did not return complete JSON")
            # Keep the extension at one in-flight call across both workspaces.
            await asyncio.sleep(1)
            return choice.message.content

        while True:
            if await db.fetchval("SELECT pg_try_advisory_lock($1)", store.WORKER_LOCK):
                try:
                    document = await store.next_document(db, workspace)
                    if document:
                        await handle_document(
                            db, workspace, document, input_dir, complete, model
                        )
                finally:
                    await db.fetchval(
                        "SELECT pg_advisory_unlock($1)", store.WORKER_LOCK
                    )
            await asyncio.sleep(POLL_SECONDS)
