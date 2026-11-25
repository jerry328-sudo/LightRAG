"""Apply reviewed names and keep ambiguous global labels out of retrieval."""

import json
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
from inspect import isawaitable
from pathlib import Path


def load_rules(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases = {}
    canonicals = set()
    for group in plan["groups"]:
        canonical = group["canonical"]
        if not isinstance(canonical, str) or not canonical or canonical in canonicals:
            raise ValueError(f"Invalid or repeated canonical entity: {canonical!r}")
        canonicals.add(canonical)
        for alias in group["aliases"]:
            if not isinstance(alias, str) or not alias or alias in aliases:
                raise ValueError(f"Invalid or repeated entity alias: {alias!r}")
            aliases[alias] = canonical

    if aliases.keys() & canonicals:
        raise ValueError(
            "Entity aliases must point directly to unambiguous canonical names"
        )
    noise = frozenset(name.casefold() for name in plan.get("noise", []))
    if "" in noise or noise & {name.casefold() for name in (*aliases, *canonicals)}:
        raise ValueError(
            "The noise list overlaps with curated entities or is malformed"
        )
    return aliases, noise


def _normalize_records(aliases, nodes, edges):
    node_records = defaultdict(list)
    edge_records = defaultdict(list)
    for name, records in nodes.items():
        canonical = aliases.get(name, name)
        if canonical != name:
            records = [dict(record, entity_name=canonical) for record in records]
        node_records[canonical].extend(records)

    for (source, target), records in edges.items():
        canonical_source = aliases.get(source, source)
        canonical_target = aliases.get(target, target)
        if canonical_source == canonical_target:
            continue
        if (canonical_source, canonical_target) != (source, target):
            records = [
                dict(record, src_id=canonical_source, tgt_id=canonical_target)
                for record in records
            ]
        edge_records[(canonical_source, canonical_target)].extend(records)
    return dict(node_records), dict(edge_records)


def _during_retrieval(query, retrieving):
    @wraps(query)
    async def run(*args, **kwargs):
        token = retrieving.set(True)
        try:
            return await query(*args, **kwargs)
        finally:
            retrieving.reset(token)

    return run


def _filtered_vector_query(original, is_hidden, retrieving, reserve):
    @wraps(original)
    async def run(query, top_k, *args, **kwargs):
        if not retrieving.get() or top_k < 1:
            return await original(query, top_k, *args, **kwargs)
        matches = await original(query, top_k + reserve, *args, **kwargs)
        return [row for row in matches if not is_hidden(row)][:top_k]

    return run


def attach_graph_curation(rag, rules_path):
    aliases, noise = load_rules(rules_path)
    graph = rag.chunk_entity_relation_graph
    previous_validator = rag.kg_extraction_validator

    def is_noise(name):
        return isinstance(name, str) and name.casefold() in noise

    async def validate(chunk_id, chunk_text, nodes, edges):
        if previous_validator is not None:
            validated = previous_validator(chunk_id, chunk_text, nodes, edges)
            if isawaitable(validated):
                validated = await validated
            nodes, edges = validated
        return _normalize_records(aliases, nodes, edges)

    rag.kg_extraction_validator = validate
    original_search = graph.search_labels
    original_subgraph = rag.get_knowledge_graph

    @wraps(original_search)
    async def search_labels(query, limit=50):
        canonical = aliases.get(query.strip())
        if canonical and not await graph.has_node(canonical):
            canonical = None
        results = await original_search(canonical or query, limit)
        if canonical is None or canonical not in results:
            return results
        return [canonical, *(name for name in results if name != canonical)][:limit]

    @wraps(original_subgraph)
    async def get_knowledge_graph(node_label, *args, **kwargs):
        canonical = aliases.get(node_label, node_label)
        if canonical != node_label and not await graph.has_node(canonical):
            canonical = node_label
        result = await original_subgraph(canonical, *args, **kwargs)
        if not noise or is_noise(canonical):
            return result
        hidden = {node.id for node in result.nodes if is_noise(node.id)}
        if not hidden:
            return result
        return result.model_copy(
            update={
                "nodes": [node for node in result.nodes if node.id not in hidden],
                "edges": [
                    edge
                    for edge in result.edges
                    if edge.source not in hidden and edge.target not in hidden
                ],
            }
        )

    graph.search_labels = search_labels
    rag.get_knowledge_graph = get_knowledge_graph
    if not noise:
        return

    retrieving = ContextVar("graph_curation_retrieval", default=False)
    reserve = min(len(noise), 24)
    rag.aquery_data = _during_retrieval(rag.aquery_data, retrieving)
    rag.aquery_llm = _during_retrieval(rag.aquery_llm, retrieving)
    rag.entities_vdb.query = _filtered_vector_query(
        rag.entities_vdb.query,
        lambda row: is_noise(row.get("entity_name")),
        retrieving,
        reserve,
    )
    rag.relationships_vdb.query = _filtered_vector_query(
        rag.relationships_vdb.query,
        lambda row: is_noise(row.get("src_id")) or is_noise(row.get("tgt_id")),
        retrieving,
        reserve,
    )

    original_neighbors = graph.get_nodes_edges_batch
    original_popular = graph.get_popular_labels

    @wraps(original_neighbors)
    async def get_nodes_edges_batch(node_ids):
        neighbors = await original_neighbors(node_ids)
        if not retrieving.get():
            return neighbors
        return {
            name: [pair for pair in pairs if not any(map(is_noise, pair))]
            for name, pairs in neighbors.items()
        }

    @wraps(original_popular)
    async def get_popular_labels(limit=300):
        labels = await original_popular(limit + reserve)
        return [name for name in labels if not is_noise(name)][:limit]

    graph.get_nodes_edges_batch = get_nodes_edges_batch
    graph.get_popular_labels = get_popular_labels
