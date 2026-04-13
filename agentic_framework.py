#!/usr/bin/env python3
"""
core/agentic_framework.py - Framework Agentique avec boucle ReAct réelle
Chaque agent Pense (Thought) -> Agit (Action) -> Observe (Observation)
et s'auto-évalue via l'agent Critic avant de répondre.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    CODER = "coder"
    CRITIC = "critic"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class ReActStep:
    """Une étape du cycle Thought -> Action -> Observation."""
    thought: str
    action: str
    observation: str
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


class AutonomousAgent:
    """Agent autonome avec boucle ReAct réelle et mémoire épisodique."""

    def __init__(self, name: str, role: AgentRole, capabilities: List[str]):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.episode_history: List[ReActStep] = []
        self.knowledge_base: List[Dict[str, Any]] = []

    async def process_message(self, engine, message: AgentMessage) -> List[AgentMessage]:
        """Boucle ReAct: Thought -> Action -> Observation -> Feedback -> réponse."""

        # FEEDBACK PRÉ-ACTION: récupérer le contexte enrichi par la boucle fermée
        feedback_context = ""
        if hasattr(engine, 'feedback_loop') and engine.feedback_loop:
            feedback_context = await engine.feedback_loop.get_agent_context(
                self.name, message.content
            )

        # KNOWLEDGE GRAPH: enrichir avec les concepts reliés
        graph_context = ""
        if hasattr(engine, 'knowledge_graph') and engine.knowledge_graph:
            concepts = await engine.knowledge_graph.get_related(message.content[:40], depth=1)
            if concepts:
                graph_context = "Concepts reliés: " + ", ".join(f"{c[0]}({c[1]})" for c in concepts[:5])

        # THOUGHT: raisonner avec le contexte enrichi
        thought = await self._think(engine, message.content, feedback_context, graph_context)

        # ACTION: exécuter selon le rôle
        action_result, action_desc = await self._act(engine, message.content, feedback_context)

        # OBSERVATION: évaluer le résultat
        confidence = self._evaluate(action_result)

        step = ReActStep(
            thought=thought,
            action=action_desc,
            observation=action_result,
            confidence=confidence
        )
        self.episode_history.append(step)
        self.knowledge_base.append({
            "task": message.content,
            "result": action_result,
            "confidence": confidence,
            "timestamp": step.timestamp
        })

        # FEEDBACK POST-ACTION: notifier la boucle fermée du résultat
        if hasattr(engine, 'feedback_loop') and engine.feedback_loop:
            await engine.feedback_loop.process_episode(
                agent_name=self.name,
                query=message.content,
                result=action_result,
                confidence=confidence
            )

        resp = AgentMessage(
            sender=self.name,
            recipient=message.sender,
            role=self.role,
            content=action_result,
            metadata={"thought": thought, "confidence": confidence, "action": action_desc},
            task_id=message.task_id
        )
        return [resp]

    async def _think(self, engine, content: str, feedback_ctx: str = "", graph_ctx: str = "") -> str:
        """Phase Thought: raisonne avec contexte feedback + graphe de connaissances."""
        if not engine.brain or not engine.brain.get('llm'):
            return f"[{self.role.value}] Analyse de: {content[:80]}"
        extra = ""
        if feedback_ctx:
            extra += f"\n{feedback_ctx}"
        if graph_ctx:
            extra += f"\n{graph_ctx}"
        prompt = (
            f"Tu es l'agent {self.name} ({self.role.value}).{extra}\n"
            f"Tâche: {content}\n"
            f"Explique en 1 phrase comment tu vas aborder cette tâche."
        )
        return await engine.brain['llm'].generate(prompt)

    async def _act(self, engine, content: str, feedback_ctx: str = "") -> Tuple[str, str]:
        """Phase Action: exécute avec contexte feedback injecté dans le prompt."""
        ctx_prefix = f"{feedback_ctx}\n\n" if feedback_ctx else ""

        if self.role == AgentRole.ANALYST:
            prompt = f"{ctx_prefix}Analyse experte en tant que {self.name}: {content}"
            result = await engine.brain['llm'].generate(prompt) if engine.brain else content
            return result, "llm_analysis"

        elif self.role == AgentRole.RESEARCHER:
            if content.startswith("http"):
                try:
                    data = await engine.vision['spyder'].fetch_page(content)
                    return json.dumps(data, ensure_ascii=False)[:500], "web_fetch"
                except Exception as e:
                    return f"Erreur fetch: {e}", "web_fetch_failed"
            prompt = f"{ctx_prefix}Recherche et synthèse approfondie sur: {content}"
            result = await engine.brain['llm'].generate(prompt) if engine.brain else content
            return result, "llm_research"

        elif self.role == AgentRole.CODER:
            prompt = f"{ctx_prefix}Génère du code Python propre et commenté pour: {content}"
            result = await engine.brain['llm'].generate(prompt) if engine.brain else content
            return result, "code_generation"

        elif self.role == AgentRole.CRITIC:
            prompt = (
                f"{ctx_prefix}Critique et améliore cette réponse de manière constructive. "
                f"Donne un score /10 et des suggestions précises: {content}"
            )
            result = await engine.brain['llm'].generate(prompt) if engine.brain else content
            return result, "critique"

        return content, "passthrough"

    def _evaluate(self, result: str) -> float:
        """Calcule un score de confiance basé sur la qualité de la réponse."""
        if not result or "Erreur" in result:
            return 0.2
        length_score = min(len(result) / 200, 1.0) * 0.4
        keyword_score = 0.6 if any(k in result.lower() for k in ["parce que", "donc", "ainsi", "car", "because", "therefore"]) else 0.3
        return round(length_score + keyword_score, 2)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les stats de performance de l'agent."""
        if not self.episode_history:
            return {"episodes": 0, "avg_confidence": 0.0}
        avg_conf = sum(s.confidence for s in self.episode_history) / len(self.episode_history)
        return {
            "agent": self.name,
            "episodes": len(self.episode_history),
            "avg_confidence": round(avg_conf, 3),
            "knowledge_items": len(self.knowledge_base)
        }


class AgenticSwarm:
    """Essaim d'agents avec dispatch parallèle et validation par Critic."""

    def __init__(self, engine):
        self.engine = engine
        self.agents: Dict[str, AutonomousAgent] = {}
        self._create_agents()

    def _create_agents(self):
        self.agents = {
            "Athena-Analyst": AutonomousAgent("Athena-Analyst", AgentRole.ANALYST, ["analyst", "nlp", "sentiment"]),
            "Athena-Researcher": AutonomousAgent("Athena-Researcher", AgentRole.RESEARCHER, ["researcher", "web", "data"]),
            "Athena-Coder": AutonomousAgent("Athena-Coder", AgentRole.CODER, ["coder", "python", "codegen"]),
            "Athena-Critic": AutonomousAgent("Athena-Critic", AgentRole.CRITIC, ["critic", "review", "optimize"]),
        }

    async def dispatch_task(self, task: str) -> List[str]:
        """Dispatch parallèle vers tous les agents + validation Critic + feedback loop."""
        task_id = f"task_{hash(task) % 100000}"

        worker_agents = [a for name, a in self.agents.items() if a.role != AgentRole.CRITIC]
        messages = [
            AgentMessage(sender="user", recipient=a.name, role=a.role,
                         content=task, task_id=task_id)
            for a in worker_agents
        ]

        try:
            agent_tasks = [
                asyncio.wait_for(
                    agent.process_message(self.engine, msg), timeout=45
                )
                for agent, msg in zip(worker_agents, messages)
            ]
            results_list = await asyncio.gather(*agent_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Swarm dispatch error: {e}")
            return []

        # Collecter réponses valides + scores de confiance par agent
        raw_results = []
        agent_confidences: Dict[str, float] = {}
        for res in results_list:
            if isinstance(res, Exception):
                logger.warning(f"Agent error: {res}")
                continue
            for msg in res:
                raw_results.append(f"[{msg.sender}] {msg.content}")
                agent_confidences[msg.sender] = msg.metadata.get("confidence", 0.5)

        if not raw_results:
            return []

        # Phase Critic: valider, scorer et synthétiser
        critic = self.agents["Athena-Critic"]
        synthesis_input = "\n".join(raw_results)
        critic_msg = AgentMessage(
            sender="orchestrator", recipient="Athena-Critic",
            role=AgentRole.CRITIC, content=synthesis_input, task_id=task_id
        )
        try:
            critic_responses = await asyncio.wait_for(
                critic.process_message(self.engine, critic_msg), timeout=30
            )
            final = [r.content for r in critic_responses]
            critic_confidence = critic_responses[0].metadata.get("confidence", 0.5) if critic_responses else 0.5
        except Exception as e:
            logger.warning(f"Critic timeout: {e}")
            final = raw_results
            critic_confidence = 0.4

        # BOUCLE FERMÉE: le score Critic est propagé à chaque agent worker
        if hasattr(self.engine, 'feedback_loop') and self.engine.feedback_loop:
            for agent_name, agent_conf in agent_confidences.items():
                await self.engine.feedback_loop.process_episode(
                    agent_name=agent_name,
                    query=task,
                    result=raw_results[0] if raw_results else "",
                    confidence=agent_conf,
                    critic_score=critic_confidence  # Le Critic juge les workers
                )

        return final

    def get_swarm_stats(self) -> List[Dict[str, Any]]:
        """Retourne les stats de performance de tous les agents."""
        return [agent.get_performance_stats() for agent in self.agents.values()]

