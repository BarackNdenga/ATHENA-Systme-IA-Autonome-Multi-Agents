#!/usr/bin/env python3
"""
brain/knowledge_graph.py - Graphe de connaissances
Relie les concepts entre eux avec des relations typées et des poids.
Complète la mémoire vectorielle (sémantique) avec une structure relationnelle.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import networkx as nx

logger = logging.getLogger(__name__)

GRAPH_FILE = Path("./knowledge_graph.json")


class KnowledgeGraph:
    """
    Graphe de connaissances orienté et pondéré.
    Nœuds = concepts. Arêtes = relations typées (cause, implique, similaire, contient...).
    """

    def __init__(self, filepath: Path = GRAPH_FILE):
        self.filepath = filepath
        self.graph: nx.DiGraph = nx.DiGraph()

    async def initialize(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load)
        logger.info(f"KnowledgeGraph chargé: {self.graph.number_of_nodes()} nœuds, {self.graph.number_of_edges()} arêtes.")

    def _load(self):
        if not self.filepath.exists():
            return
        try:
            data = json.loads(self.filepath.read_text(encoding="utf-8"))
            self.graph = nx.node_link_graph(data)
        except Exception as e:
            logger.warning(f"Impossible de charger le graphe: {e}")

    async def add_concept(self, concept: str, metadata: Dict[str, Any] = None):
        """Ajoute un nœud concept."""
        self.graph.add_node(concept, **(metadata or {}))
        await self._save()

    async def add_relation(self, source: str, target: str, relation: str, weight: float = 1.0):
        """
        Ajoute une relation typée entre deux concepts.
        relation: 'cause', 'implique', 'similaire', 'contient', 'contredit', 'utilise'
        """
        if not self.graph.has_node(source):
            self.graph.add_node(source)
        if not self.graph.has_node(target):
            self.graph.add_node(target)
        self.graph.add_edge(source, target, relation=relation, weight=weight)
        await self._save()

    async def extract_and_add(self, text: str, source_label: str = "auto"):
        """
        Extrait des concepts simples d'un texte et les relie au nœud source.
        Heuristique légère: mots de plus de 5 lettres, non-stopwords.
        """
        stopwords = {"pour", "dans", "avec", "cette", "sont", "mais", "plus",
                     "aussi", "comme", "donc", "ainsi", "entre", "leurs", "their",
                     "that", "this", "with", "from", "have", "been", "will"}
        words = [w.strip(".,;:!?\"'()[]").lower() for w in text.split()]
        concepts = list({w for w in words if len(w) > 5 and w not in stopwords})[:10]

        if not concepts:
            return

        if not self.graph.has_node(source_label):
            self.graph.add_node(source_label, type="query")

        for concept in concepts:
            if not self.graph.has_node(concept):
                self.graph.add_node(concept, type="extracted")
            self.graph.add_edge(source_label, concept, relation="contient", weight=0.5)

        await self._save()

    async def get_related(self, concept: str, depth: int = 2) -> List[Tuple[str, str, float]]:
        """Retourne les concepts reliés jusqu'à une profondeur donnée."""
        if not self.graph.has_node(concept):
            return []
        related = []
        for node in nx.ego_graph(self.graph, concept, radius=depth).nodes():
            if node != concept and self.graph.has_edge(concept, node):
                edge = self.graph[concept][node]
                related.append((node, edge.get("relation", "?"), edge.get("weight", 1.0)))
        return sorted(related, key=lambda x: -x[2])

    async def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Trouve le chemin le plus court entre deux concepts."""
        try:
            return nx.shortest_path(self.graph, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    async def get_central_concepts(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Retourne les concepts les plus centraux (PageRank)."""
        if self.graph.number_of_nodes() == 0:
            return []
        scores = nx.pagerank(self.graph, weight="weight")
        return sorted(scores.items(), key=lambda x: -x[1])[:top_n]

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            "density": round(nx.density(self.graph), 4)
        }

    async def _save(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_disk)

    def _write_disk(self):
        data = nx.node_link_data(self.graph)
        self.filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
