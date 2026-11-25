"""Apply a reviewed graph merge plan using the configured offline LightRAG instance.

Run from the target service's working directory with its usual environment and
the service stopped. Use one private report for the baseline, apply, and verify
phases. A report with a started or failed operation requires inspection before
any additional apply. Noise names are inventoried for review.
"""

import argparse
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


READ_BATCH_SIZE = 200


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_report(path, report):
    # The journal and retrieval results can contain internal names and paths.
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_plan(path):
    from lightrag.utils import normalize_entity_name

    plan = read_json(path)
    if not isinstance(plan, dict) or not isinstance(plan.get("groups"), list):
        raise ValueError("The plan must contain a groups array.")
    seen_names = set()
    seen_targets = set()
    for group in plan["groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("aliases"), list):
            raise ValueError("Each merge group must contain an aliases array.")
        canonical = group.get("canonical")
        names = [canonical, *group["aliases"]]
        if len(names) < 2 or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError(
                "A merge group needs a canonical name and at least one non-empty alias."
            )
        target_key = normalize_entity_name(canonical)
        if (
            not target_key
            or target_key in seen_targets
            or len(names) != len(set(names))
            or seen_names.intersection(names)
        ):
            raise ValueError(f"Overlapping or invalid merge group at {canonical!r}.")
        seen_names.update(names)
        seen_targets.add(target_key)
    noise = plan.get("noise", [])
    if not isinstance(noise, list) or any(
        not isinstance(name, str) or not name.strip() for name in noise
    ):
        raise ValueError("Noise entries must be names in an array.")
    return plan


@asynccontextmanager
async def configured_rag(expected_workspace):
    # Server configuration should read the service environment and its .env,
    # without interpreting operator flags, even during eager module imports.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        from lightrag.api import lightrag_server
        from lightrag.api.config import initialize_config
        from lightrag.kg.shared_storage import finalize_share_data

        args = initialize_config()
    finally:
        sys.argv = original_argv
    if args.workspace != expected_workspace:
        raise ValueError(
            f"Configured workspace {args.workspace!r} differs from --expect-workspace."
        )

    graphs = lightrag_server.create_graph_routes
    instances = []

    def capture(rag, *args, **kwargs):
        instances.append(rag)
        return graphs(rag, *args, **kwargs)

    lightrag_server.create_graph_routes = capture
    try:
        lightrag_server.create_app(args)
    finally:
        lightrag_server.create_graph_routes = graphs
    if len(instances) != 1:
        raise RuntimeError(
            "The native server did not construct exactly one graph instance."
        )
    rag = instances[0]

    # Opening the native stores suffices for graph operations. The server
    # lifespan also runs startup migration, which is outside the reviewed plan.
    try:
        await rag.initialize_storages()
        yield rag
    finally:
        try:
            await rag.finalize_storages()
        finally:
            finalize_share_data()


def identity(rag):
    result = {
        "workspace": rag.workspace,
        "graph_workspace": rag.chunk_entity_relation_graph.workspace,
        "working_dir": str(Path(rag.working_dir).resolve()),
        "graph_storage": rag.graph_storage,
        "kv_storage": rag.kv_storage,
        "vector_storage": rag.vector_storage,
        "doc_status_storage": rag.doc_status_storage,
        "config_storage": rag.config_storage,
        "embedding_model": rag.embedding_func.model_name,
        "embedding_dimension": rag.embedding_func.embedding_dim,
    }
    database = getattr(rag.chunk_entity_relation_graph, "db", None)
    if database is not None and all(
        hasattr(database, field) for field in ("host", "port", "database")
    ):
        result["database"] = {
            "host": database.host,
            "port": database.port,
            "name": database.database,
        }
    return result


def merge_mapping(plan, snapshot):
    from lightrag.utils import normalize_entity_name

    mapping = {}
    for group in plan["groups"]:
        canonical = group["canonical"]
        target = (
            canonical
            if canonical in snapshot["nodes"]
            else normalize_entity_name(canonical)
        )
        if target in group["aliases"]:
            raise ValueError(f"Canonical target {target!r} resolves to its own source.")
        mapping.update(dict.fromkeys(group["aliases"], target))
    if set(mapping).intersection(mapping.values()):
        raise ValueError("A resolved canonical target is also a source in the plan.")
    return mapping


def plan_names(plan):
    from lightrag.utils import normalize_entity_name

    names = set()
    for group in plan["groups"]:
        canonical = group["canonical"]
        names.update([canonical, normalize_entity_name(canonical), *group["aliases"]])
    return sorted(names)


async def read_records(read, keys):
    records = {}
    for offset in range(0, len(keys), READ_BATCH_SIZE):
        batch = keys[offset : offset + READ_BATCH_SIZE]
        values = await read(batch)
        if len(batch) != len(values):
            raise RuntimeError(
                "A native storage read did not return one result per key."
            )
        records.update(zip(batch, values, strict=True))
    return records


def provenance(record, tracked):
    from lightrag.constants import GRAPH_FIELD_SEP, RELATION_NO_EVIDENCE_SOURCE_IDS
    from lightrag.utils import has_chunk_tracking_row

    graph_sources = set((record.get("source_id") or "").split(GRAPH_FIELD_SEP))
    tracking_known = has_chunk_tracking_row(tracked)
    tracked_sources = set(tracked["chunk_ids"]) if tracking_known else graph_sources
    placeholders = {"", *RELATION_NO_EVIDENCE_SOURCE_IDS}
    return {
        "tracked_sources": sorted(tracked_sources - placeholders),
        "graph_sources": sorted(graph_sources - placeholders),
        "tracking_known": tracking_known,
    }


async def snapshot_graph(rag, names):
    from lightrag.utils import make_relation_chunk_key

    graph = rag.chunk_entity_relation_graph
    nodes = await graph.get_nodes_batch(names)
    incidence = await graph.get_nodes_edges_batch(sorted(nodes))
    pairs = sorted(
        {tuple(sorted(pair)) for edges in incidence.values() for pair in edges}
    )
    edges = await graph.get_edges_batch(
        [{"src": source, "tgt": target} for source, target in pairs]
    )
    if set(edges) != set(pairs):
        raise RuntimeError(
            "Graph edge data changed while reading the offline workspace."
        )
    nodes_tracking = await read_records(rag.entity_chunks.get_by_ids, sorted(nodes))
    relation_keys = {pair: make_relation_chunk_key(*pair) for pair in pairs}
    edges_tracking = await read_records(
        rag.relation_chunks.get_by_ids, list(relation_keys.values())
    )
    return {
        "nodes": {
            name: {
                **provenance(node, nodes_tracking[name]),
                "entity_type": node.get("entity_type"),
                "description": node.get("description"),
            }
            for name, node in sorted(nodes.items())
        },
        "edges": [
            {
                "pair": list(pair),
                **provenance(edges[pair], edges_tracking[relation_keys[pair]]),
            }
            for pair in pairs
        ],
    }


def expected_graph(snapshot, mapping):
    nodes = defaultdict(set)
    relations = defaultdict(set)
    collapsed = []
    targets = set(mapping.values())
    for name, node in snapshot["nodes"].items():
        nodes[mapping.get(name, name)].update(node["tracked_sources"])
    for edge in snapshot["edges"]:
        source, target = edge["pair"]
        next_source, next_target = (
            mapping.get(source, source),
            mapping.get(target, target),
        )
        if next_source == next_target and next_source in targets:
            collapsed.append(edge["pair"])
            continue
        relations[tuple(sorted((next_source, next_target)))].update(
            edge["tracked_sources"]
        )
    return nodes, relations, collapsed


def check_graph(before, after, mapping):
    nodes, edges, collapsed = expected_graph(before, mapping)
    actual_edges = {tuple(edge["pair"]): edge for edge in after["edges"]}
    if set(nodes) != set(after["nodes"]) or set(edges) != set(actual_edges):
        raise RuntimeError(
            "The post-merge node or edge set differs from the approved remapping."
        )
    for name, sources in nodes.items():
        if sources != set(after["nodes"][name]["tracked_sources"]):
            raise RuntimeError(f"Entity {name!r} changed its expected source chunks.")
    for pair, sources in edges.items():
        if sources != set(actual_edges[pair]["tracked_sources"]):
            raise RuntimeError(
                f"Relationship {pair!r} changed its expected source chunks."
            )
    return {"nodes": len(nodes), "relationships": len(edges), "collapsed": collapsed}


async def anchor_requirements(rag, snapshot, mapping):
    # Use both views for document anchors, as the native integrity audit does.
    # The reverse index is a superset and a later purge filters by provenance.
    combined = {
        "nodes": {
            name: with_all_sources(node) for name, node in snapshot["nodes"].items()
        },
        "edges": [with_all_sources(edge) for edge in snapshot["edges"]],
    }
    nodes, edges, _ = expected_graph(combined, mapping)
    names = set(mapping.values())
    nodes = {name: sources for name, sources in nodes.items() if name in names}
    edges = {
        pair: sources for pair, sources in edges.items() if names.intersection(pair)
    }
    source_ids = sorted(
        {source for sources in [*nodes.values(), *edges.values()] for source in sources}
    )
    chunks = await read_records(rag.text_chunks.get_by_ids, source_ids)
    documents = {
        source: (row or {}).get("full_doc_id") for source, row in chunks.items()
    }
    anchors = defaultdict(lambda: {"nodes": set(), "edges": set()})
    for kind, entries in [("nodes", nodes), ("edges", edges)]:
        for key, sources in entries.items():
            for source in sources:
                if doc_id := documents[source]:
                    anchors[doc_id][kind].add(key)
    unresolved = sorted(
        source for source, document in documents.items() if not document
    )
    return anchors, unresolved


def with_all_sources(record):
    sources = set(record["tracked_sources"]) | set(record["graph_sources"])
    return {**record, "tracked_sources": sorted(sources)}


async def missing_anchors(rag, anchors):
    documents = sorted(anchors)
    entity_rows = await read_records(rag.full_entities.get_by_ids, documents)
    relation_rows = await read_records(rag.full_relations.get_by_ids, documents)
    gaps = {}
    for document, expected in anchors.items():
        entities = set((entity_rows[document] or {}).get("entity_names") or [])
        relations = {
            tuple(pair)
            for pair in (relation_rows[document] or {}).get("relation_pairs") or []
        }
        missing_entities = expected["nodes"] - entities
        missing_relations = expected["edges"] - relations
        if missing_entities or missing_relations:
            gaps[document] = {
                "nodes": sorted(missing_entities),
                "edges": [list(pair) for pair in sorted(missing_relations)],
            }
    return gaps


async def verify_vectors(rag, before, mapping):
    from lightrag.utils import compute_mdhash_id

    nodes, edges, _ = expected_graph(before, mapping)
    node_ids = {
        compute_mdhash_id(name, prefix="ent-"): name in nodes
        for name in before["nodes"].keys() | nodes.keys()
    }
    before_edges = {tuple(edge["pair"]) for edge in before["edges"]}
    edge_ids = {
        compute_mdhash_id("".join(pair), prefix="rel-"): pair in edges
        for pair in before_edges | edges.keys()
    }
    for storage, expected in [
        (rag.entities_vdb, node_ids),
        (rag.relationships_vdb, edge_ids),
    ]:
        records = await read_records(storage.get_by_ids_strict, list(expected))
        found = {
            record_id for record_id, record in records.items() if record is not None
        }
        wanted = {record_id for record_id, present in expected.items() if present}
        if found != wanted:
            raise RuntimeError(
                f"Vector records in {storage.namespace} do not match the verified graph."
            )


def merged_fields(group, snapshot, target):
    from lightrag.constants import GRAPH_FIELD_SEP

    nodes = snapshot["nodes"]
    ordered = [target, *group["aliases"]]
    descriptions = dict.fromkeys(
        fragment
        for name in ordered
        for fragment in (nodes.get(name, {}).get("description") or "").split(
            GRAPH_FIELD_SEP
        )
        if fragment
    )
    # Preserve each distinct text fragment and the canonical node's existing
    # type. Keep legacy spellings searchable in the same entity vector.
    aliases_text = "Aliases: " + ", ".join(group["aliases"])
    descriptions[aliases_text] = None
    result = {"description": GRAPH_FIELD_SEP.join(descriptions)}
    if entity_type := nodes.get(target, {}).get("entity_type"):
        result["entity_type"] = entity_type
    return result


async def evaluate(rag, path):
    from lightrag import QueryParam

    if path is None:
        return []
    queries = read_json(path)
    if (
        not isinstance(queries, list)
        or len(queries) > 12
        or any(
            not isinstance(query, str) or not 3 <= len(query.strip()) <= 1000
            for query in queries
        )
    ):
        raise ValueError("Queries must be an array of up to 12 non-empty queries.")
    results = []
    for query in queries:
        response = await rag.aquery_data(
            query,
            QueryParam(
                mode="mix",
                top_k=20,
                chunk_top_k=8,
                max_entity_tokens=2048,
                max_relation_tokens=2048,
                max_total_tokens=12000,
                enable_rerank=False,
            ),
        )
        data = response.get("data") or {}
        results.append(
            {
                "query": query,
                "status": response.get("status"),
                "entities": [item["entity_name"] for item in data.get("entities", [])],
                "relations": [
                    [item["src_id"], item["tgt_id"]]
                    for item in data.get("relationships", [])
                ],
                "chunks": [item.get("chunk_id") for item in data.get("chunks", [])],
                "references": data.get("references", []),
            }
        )
    return results


async def apply_plan(rag, plan, names, report, report_path):
    if report["operations"]:
        raise RuntimeError(
            "This report already records an apply attempt. Inspect it before continuing."
        )
    current = await snapshot_graph(rag, names)
    if current != report["baseline"]:
        raise RuntimeError(
            "The graph has changed since the baseline. Review a new baseline before proceeding."
        )
    await verify_vectors(rag, current, {})
    for group in plan["groups"]:
        group_plan = {"groups": [group]}
        group_names = plan_names(group_plan)
        before = await snapshot_graph(rag, group_names)
        mapping = merge_mapping(group_plan, before)
        aliases = group["aliases"]
        target = mapping[aliases[0]]
        anchors, unresolved = await anchor_requirements(rag, before, mapping)
        gaps = await missing_anchors(rag, anchors)
        operation = {
            "canonical": target,
            "aliases": aliases,
            "status": "started",
            "missing_anchors": gaps,
            "unresolved_source_chunks": unresolved,
        }
        report["operations"].append(operation)
        save_report(report_path, report)
        try:
            # Publish the candidate superset first. If the merge fails, a
            # future purge can safely skip candidates with no contribution.
            for document, gap in sorted(gaps.items()):
                await rag._union_doc_recovery_anchors(
                    document, gap["nodes"], gap["edges"]
                )
            await rag.amerge_entities(
                aliases, target, target_entity_data=merged_fields(group, before, target)
            )
            after = await snapshot_graph(rag, group_names)
            operation["graph"] = check_graph(before, after, mapping)
            await verify_vectors(rag, before, mapping)
            if await missing_anchors(rag, anchors):
                raise RuntimeError(
                    "Document anchor verification failed after native merge."
                )
        except BaseException as error:
            operation.update(status="failed", error=type(error).__name__)
            save_report(report_path, report)
            raise
        operation["status"] = "verified"
        save_report(report_path, report)


async def run(options, plan):
    async with configured_rag(options.expect_workspace) as rag:
        if options.phase == "audit":
            from lightrag.tools.kg_integrity_repair import audit_kg_integrity

            save_report(
                options.report,
                {"identity": identity(rag), "audit": await audit_kg_integrity(rag)},
            )
            return
        names = plan_names(plan)
        if options.phase == "baseline":
            snapshot = await snapshot_graph(rag, names)
            mapping = merge_mapping(plan, snapshot)
            absent = set(mapping) - snapshot["nodes"].keys()
            if absent:
                raise ValueError(
                    f"Merge aliases missing under their exact stored names: {sorted(absent)!r}."
                )
            anchors, unresolved = await anchor_requirements(rag, snapshot, mapping)
            noise = await rag.chunk_entity_relation_graph.get_nodes_batch(
                plan.get("noise", [])
            )
            save_report(
                options.report,
                {
                    "identity": identity(rag),
                    "plan": plan,
                    "baseline": snapshot,
                    "mapping": mapping,
                    "noise_present": sorted(noise),
                    "operations": [],
                    "predicted_anchor_gaps": await missing_anchors(rag, anchors),
                    "unresolved_source_chunks": unresolved,
                    "baseline_queries": await evaluate(rag, options.queries),
                },
            )
            return
        report = read_json(options.report)
        if report["identity"] != identity(rag) or report["plan"] != plan:
            raise ValueError(
                "The report must match the supplied plan and configured workspace."
            )
        if options.phase == "evaluate":
            if report["operations"]:
                raise RuntimeError("Baseline queries can only be rerun before apply.")
            report["baseline_queries"] = await evaluate(rag, options.queries)
            save_report(options.report, report)
            return
        if options.phase == "apply":
            await apply_plan(rag, plan, names, report, options.report)
            return
        if len(report["operations"]) != len(plan["groups"]) or any(
            operation["status"] != "verified" for operation in report["operations"]
        ):
            raise RuntimeError(
                "The journal does not show a verified merge for every group."
            )
        before = report["baseline"]
        mapping = report["mapping"]
        after = await snapshot_graph(rag, names)
        graph_result = check_graph(before, after, mapping)
        await verify_vectors(rag, before, mapping)
        anchors, unresolved = await anchor_requirements(rag, before, mapping)
        if await missing_anchors(rag, anchors):
            raise RuntimeError("The verified graph has missing document anchors.")
        if options.curated_query:
            from extensions.graph_curation.rules import attach_graph_curation

            attach_graph_curation(rag, options.curated_query)
        report["verification"] = {
            "graph": graph_result,
            "unresolved_source_chunks": unresolved,
            "queries": await evaluate(rag, options.queries),
        }
        save_report(options.report, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["baseline", "evaluate", "apply", "verify", "audit"]
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--queries", type=Path, help="JSON array of example retrieval queries"
    )
    parser.add_argument(
        "--curated-query", type=Path, help="Apply these rules to verify queries"
    )
    parser.add_argument(
        "--expect-workspace", default="", help="Expected configured workspace name"
    )
    options = parser.parse_args()
    if options.phase != "audit" and options.plan is None:
        parser.error("--plan is required except for audit")
    if options.phase == "evaluate" and options.queries is None:
        parser.error("--queries is required for evaluate")
    if options.curated_query and (options.phase != "verify" or options.queries is None):
        parser.error("--curated-query requires verify and --queries")
    if options.phase in {"baseline", "audit"} and options.report.exists():
        parser.error("Choose a new report path for baseline or audit")
    plan = None if options.phase == "audit" else read_plan(options.plan)
    asyncio.run(run(options, plan))


if __name__ == "__main__":
    main()
