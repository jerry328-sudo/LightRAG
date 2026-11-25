import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extensions.graph_curation.rules import attach_graph_curation, load_rules
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode


pytestmark = pytest.mark.offline


class RulesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.plan = Path(directory.name, "plan.json")
        self.plan.write_text(
            json.dumps(
                {
                    "groups": [
                        {
                            "canonical": "Lithium-Ion Battery",
                            "aliases": ["Li-Ion Battery", "Lithium-Ion Batteries"],
                        }
                    ],
                    "noise": ["Figure 1"],
                }
            ),
            encoding="utf-8",
        )
        canonical = "Lithium-Ion Battery"
        self.graph = SimpleNamespace(
            has_node=AsyncMock(return_value=True),
            search_labels=AsyncMock(return_value=["Sibling", canonical]),
            get_nodes_edges_batch=AsyncMock(
                return_value={canonical: [(canonical, "Figure 1"), (canonical, "LFP")]}
            ),
            get_popular_labels=AsyncMock(return_value=["FIGURE 1", canonical, "LFP"]),
        )
        self.rag = SimpleNamespace(
            chunk_entity_relation_graph=self.graph,
            kg_extraction_validator=None,
            get_knowledge_graph=AsyncMock(
                return_value=KnowledgeGraph(
                    nodes=[
                        KnowledgeGraphNode(id=name, labels=[name], properties={})
                        for name in (canonical, "Figure 1", "LFP")
                    ],
                    edges=[
                        KnowledgeGraphEdge(
                            id="edge-" + other,
                            type=None,
                            source=canonical,
                            target=other,
                            properties={},
                        )
                        for other in ("Figure 1", "LFP")
                    ],
                )
            ),
            aquery_llm=AsyncMock(),
            entities_vdb=SimpleNamespace(
                query=AsyncMock(
                    return_value=[
                        {"entity_name": "figure 1"},
                        {"entity_name": canonical},
                        {"entity_name": "LFP"},
                    ]
                )
            ),
            relationships_vdb=SimpleNamespace(
                query=AsyncMock(
                    return_value=[
                        {"src_id": canonical, "tgt_id": "FIGURE 1"},
                        {"src_id": canonical, "tgt_id": "LFP"},
                    ]
                )
            ),
        )

    async def test_normalizes_node_and_edge_records_after_existing_validator(self):
        source = "Li-Ion Battery"
        canonical = "Lithium-Ion Battery"
        destination = "LFP"
        raw_record = {"entity_name": source, "source_id": "chunk-original"}
        raw_edge = {"src_id": source, "tgt_id": destination, "source_id": "chunk-edge"}
        prior = AsyncMock(
            return_value=(
                {source: [raw_record], canonical: [{"entity_name": canonical}]},
                {
                    (source, destination): [raw_edge],
                    (source, "Lithium-Ion Batteries"): [{}],
                },
            )
        )
        self.rag.kg_extraction_validator = prior
        self.rag.aquery_data = AsyncMock()
        attach_graph_curation(self.rag, self.plan)
        nodes, edges = await self.rag.kg_extraction_validator("chunk-s", "text", {}, {})
        prior.assert_awaited_once_with("chunk-s", "text", {}, {})
        self.assertEqual(list(nodes), [canonical])
        self.assertEqual(nodes[canonical][0], dict(raw_record, entity_name=canonical))
        self.assertEqual(len(nodes[canonical]), 2)
        self.assertEqual(
            edges, {(canonical, destination): [dict(raw_edge, src_id=canonical)]}
        )
        self.assertEqual(raw_record["entity_name"], source)
        self.assertEqual(raw_edge["src_id"], source)

    async def test_filters_graph_retrieval_without_leaking_into_concurrent_admin_reads(
        self,
    ):
        canonical = "Lithium-Ion Battery"
        ready = asyncio.Event()
        proceed = asyncio.Event()

        async def fetch():
            ready.set()
            await proceed.wait()
            return (
                await self.rag.entities_vdb.query(query="topic", top_k=2),
                await self.rag.relationships_vdb.query("topic", 1),
                await self.graph.get_nodes_edges_batch([canonical]),
            )

        self.rag.aquery_data = fetch
        entity_query = self.rag.entities_vdb.query
        attach_graph_curation(self.rag, self.plan)

        task = asyncio.create_task(self.rag.aquery_data())
        await asyncio.wait_for(ready.wait(), timeout=1)
        try:
            self.assertIn(
                (canonical, "Figure 1"),
                (await self.graph.get_nodes_edges_batch([canonical]))[canonical],
            )
            self.assertEqual(
                (await self.rag.entities_vdb.query("admin", 1))[0]["entity_name"],
                "figure 1",
            )
        finally:
            proceed.set()
        nodes, edges, neighbors = await asyncio.wait_for(task, timeout=1)
        self.assertEqual([row["entity_name"] for row in nodes], [canonical, "LFP"])
        self.assertEqual([row["tgt_id"] for row in edges], ["LFP"])
        self.assertEqual(neighbors, {canonical: [(canonical, "LFP")]})
        entity_query.assert_awaited_with("topic", 3)

    async def test_old_label_reaches_canonical_and_noise_is_hidden_from_popular_labels(
        self,
    ):
        self.rag.aquery_data = AsyncMock()
        native_subgraph = self.rag.get_knowledge_graph
        attach_graph_curation(self.rag, self.plan)
        canonical = "Lithium-Ion Battery"
        self.assertEqual(
            await self.graph.search_labels("Li-Ion Battery", 1), [canonical]
        )
        self.assertEqual(await self.graph.get_popular_labels(2), [canonical, "LFP"])
        displayed = await self.rag.get_knowledge_graph(
            node_label="Li-Ion Battery", max_depth=1
        )
        native_subgraph.assert_awaited_once_with(canonical, max_depth=1)
        self.assertEqual([node.id for node in displayed.nodes], [canonical, "LFP"])
        self.assertEqual([edge.target for edge in displayed.edges], ["LFP"])
        explicit_noise = await self.rag.get_knowledge_graph(node_label="Figure 1")
        self.assertIn("Figure 1", [node.id for node in explicit_noise.nodes])

    def test_rejects_alias_chains(self):
        self.plan.write_text(
            json.dumps(
                {
                    "groups": [
                        {"canonical": "Three", "aliases": ["Two"]},
                        {"canonical": "Two", "aliases": ["One"]},
                    ]
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "aliases must point directly"):
            load_rules(self.plan)
