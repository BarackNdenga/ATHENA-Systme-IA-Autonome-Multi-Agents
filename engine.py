#!/usr/bin/env python3
"""
core/engine.py - AthenaEngine : Coordonne les modules (S.E.T.H.)
Le cœur du système qui orchestre brain, vision, security de manière asynchrone.
"""

import asyncio
import logging
import time
from typing import Dict, Any
from pathlib import Path

from brain.llm_handler import LLMHandler
from brain.memory import VectorMemory
from brain.episodic_memory import EpisodicMemory
from brain.knowledge_graph import KnowledgeGraph
from vision.scanner import FileScanner
from vision.web_spyder import WebSpyder
from core.security import SecurityMonitor
from core.agentic_framework import AgenticSwarm
from core.react_planner import ReActPlanner
from core.observability import ObservabilityHub
from core.feedback_loop import FeedbackLoop

logger = logging.getLogger(__name__)


class AthenaEngine:
    """Orchestrateur principal: coordonne tous les modules de manière asynchrone."""

    def __init__(self):
        self.modules: Dict[str, Any] = {}
        self.brain = None
        self.vision = None
        self.security = None
        self.swarm = None
        self.planner = None
        self.episodic_memory = None
        self.knowledge_graph = None
        self.obs = None
        self.feedback_loop = None

    async def initialize(self):
        logger.info("Initialisation d'AthenaEngine...")

        self.obs = ObservabilityHub()

        span = self.obs.start_span("init_security", "engine")
        self.security = SecurityMonitor()
        await self.security.start_monitoring()
        await self.obs.finish_span(span)

        span = self.obs.start_span("init_brain", "engine")
        self.brain = {
            'llm': LLMHandler(),
            'memory': VectorMemory()
        }
        await self.brain['llm'].initialize()
        await self.brain['memory'].initialize()
        await self.obs.finish_span(span)

        self.episodic_memory = EpisodicMemory()
        await self.episodic_memory.initialize()
        self.knowledge_graph = KnowledgeGraph()
        await self.knowledge_graph.initialize()

        span = self.obs.start_span("init_vision", "engine")
        self.vision = {
            'scanner': FileScanner(),
            'spyder': WebSpyder()
        }
        await self.vision['scanner'].initialize()
        await self.vision['spyder'].initialize()
        await self.obs.finish_span(span)

        # Swarm + Planner
        self.swarm = AgenticSwarm(self)
        self.planner = ReActPlanner(self)

        # Boucle fermée: enregistrer tous les agents + charger profils persistés
        self.feedback_loop = FeedbackLoop(self)
        for agent_name, agent in self.swarm.agents.items():
            self.feedback_loop.register_agent(agent_name, agent.role.value)
        await self.feedback_loop.load_profiles()

        self.modules = {
            'brain': self.brain,
            'vision': self.vision,
            'security': self.security,
            'swarm': self.swarm,
            'episodic_memory': self.episodic_memory,
            'knowledge_graph': self.knowledge_graph,
            'observability': self.obs,
            'feedback_loop': self.feedback_loop,
        }

        logger.info("AthenaEngine initialisée — boucle fermée active.")

    async def process_query(self, query: str) -> str:
        """Traite une requête via ReAct + swarm + boucle fermée."""
        span = self.obs.start_span("process_query", "engine", {"query": query[:80]})
        start = time.time()

        try:
            # Enrichir le contexte LLM avec la mémoire épisodique
            episodic_context = ""
            if self.episodic_memory:
                past_episodes = await self.episodic_memory.recall_similar(query, n=3)
                if past_episodes:
                    episodic_context = "\n".join(
                        f"- [{e.confidence:.0%}] {e.query[:60]} → {e.result[:80]}"
                        for e in past_episodes
                    )

            # Enrichir avec le graphe de connaissances
            graph_context = ""
            if self.knowledge_graph:
                central = await self.knowledge_graph.get_central_concepts(top_n=5)
                if central:
                    graph_context = "Concepts clés connus: " + ", ".join(c[0] for c in central)

            # Injecter dans le LLM handler
            self.brain['llm'].set_context(
                episodic=episodic_context,
                graph=graph_context
            )

            plan = await self.planner.plan_and_execute(query)
            response = plan.final_synthesis if plan.final_synthesis else "Aucun résultat."

            await self.brain['memory'].store(query, response)
            await self.episodic_memory.record(
                query=query,
                agent="AthenaEngine",
                action_taken="react_plan",
                result=response,
                confidence=0.8 if plan.success else 0.3,
                duration_ms=(time.time() - start) * 1000
            )

            # Enrichir le graphe avec les concepts de la réponse
            await self.knowledge_graph.extract_and_add(response, source_label=query[:40])

            trace = self.planner.format_plan_trace(plan)
            logger.info(f"\n{trace}")

            await self.obs.finish_span(span)
            return response

        except Exception as e:
            await self.obs.finish_span(span, status="error", error=str(e))
            logger.error(f"process_query échoué: {e}")
            return await self.brain['llm'].generate(query)

    async def shutdown(self):
        logger.info("Arrêt d'AthenaEngine...")
        for module in self.modules.values():
            if hasattr(module, 'shutdown'):
                try:
                    await module.shutdown()
                except Exception:
                    pass
        logger.info("AthenaEngine arrêtée.")

