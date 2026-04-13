#!/usr/bin/env python3
"""
core/react_planner.py - Planificateur ReAct multi-étapes
Décompose une tâche complexe en sous-tâches avec dépendances,
les exécute dans l'ordre et fusionne les résultats.
Cycle: Plan -> Think -> Act -> Observe -> Reflect -> Réponse finale
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """Une sous-tâche avec son statut et ses dépendances."""
    task_id: str
    description: str
    agent_role: str          # "analyst" | "researcher" | "coder" | "critic"
    depends_on: List[str] = field(default_factory=list)
    result: Optional[str] = None
    status: str = "pending"  # pending | running | done | failed
    confidence: float = 0.0
    duration_ms: float = 0.0


@dataclass
class ExecutionPlan:
    """Plan d'exécution complet pour une tâche complexe."""
    original_query: str
    subtasks: List[SubTask]
    final_synthesis: str = ""
    total_duration_ms: float = 0.0
    success: bool = False


class ReActPlanner:
    """
    Planificateur ReAct: décompose, ordonne, exécute et synthétise.
    Respecte les dépendances entre sous-tâches (DAG d'exécution).
    """

    # Mots-clés qui déclenchent chaque type d'agent
    ROLE_KEYWORDS = {
        "researcher": ["cherche", "trouve", "recherche", "crawl", "web", "http", "url", "info", "search"],
        "coder":      ["code", "programme", "script", "fonction", "python", "génère", "écris", "implémente"],
        "analyst":    ["analyse", "explique", "compare", "évalue", "résume", "synthèse", "comprends"],
        "critic":     ["critique", "améliore", "optimise", "vérifie", "valide", "corrige"],
    }

    def __init__(self, engine):
        self.engine = engine

    async def plan_and_execute(self, query: str) -> ExecutionPlan:
        """Point d'entrée principal: planifie et exécute une tâche complexe."""
        start = time.time()

        # 1. PLAN: décomposer la tâche
        subtasks = await self._decompose(query)
        plan = ExecutionPlan(original_query=query, subtasks=subtasks)

        # 2. EXECUTE: respecter les dépendances
        await self._execute_dag(plan)

        # 3. REFLECT: synthèse finale via LLM
        plan.final_synthesis = await self._synthesize(plan)
        plan.total_duration_ms = (time.time() - start) * 1000
        plan.success = any(t.status == "done" for t in plan.subtasks)

        # 4. Enrichir le graphe de connaissances
        if hasattr(self.engine, 'knowledge_graph') and self.engine.knowledge_graph:
            await self.engine.knowledge_graph.extract_and_add(query, source_label=query[:40])

        return plan

    async def _decompose(self, query: str) -> List[SubTask]:
        """
        Décompose la query en sous-tâches selon les rôles détectés.
        Toujours: Analyst en premier, Critic en dernier.
        """
        subtasks = []
        query_lower = query.lower()

        # Analyst toujours présent (tâche 0)
        subtasks.append(SubTask(
            task_id="t0",
            description=f"Analyser et comprendre: {query}",
            agent_role="analyst",
            depends_on=[]
        ))

        # Researcher si besoin de données externes
        if any(kw in query_lower for kw in self.ROLE_KEYWORDS["researcher"]):
            subtasks.append(SubTask(
                task_id="t1",
                description=f"Rechercher des informations sur: {query}",
                agent_role="researcher",
                depends_on=["t0"]
            ))

        # Coder si besoin de code
        if any(kw in query_lower for kw in self.ROLE_KEYWORDS["coder"]):
            deps = ["t1"] if any(t.task_id == "t1" for t in subtasks) else ["t0"]
            subtasks.append(SubTask(
                task_id="t2",
                description=f"Générer le code pour: {query}",
                agent_role="coder",
                depends_on=deps
            ))

        # Critic toujours en dernier
        last_ids = [t.task_id for t in subtasks]
        subtasks.append(SubTask(
            task_id="tc",
            description="Valider, critiquer et améliorer les résultats",
            agent_role="critic",
            depends_on=last_ids
        ))

        logger.info(f"Plan décomposé: {[t.task_id + '/' + t.agent_role for t in subtasks]}")
        return subtasks

    async def _execute_dag(self, plan: ExecutionPlan):
        """Exécute les sous-tâches en respectant les dépendances (DAG)."""
        completed: Dict[str, str] = {}  # task_id -> result

        max_rounds = len(plan.subtasks) + 2
        rounds = 0

        while True:
            rounds += 1
            if rounds > max_rounds:
                break

            ready = [
                t for t in plan.subtasks
                if t.status == "pending"
                and all(dep in completed for dep in t.depends_on)
            ]

            if not ready:
                break

            # Exécuter les tâches prêtes en parallèle
            await asyncio.gather(*[self._run_subtask(t, completed) for t in ready])

            for t in ready:
                if t.status == "done":
                    completed[t.task_id] = t.result

    async def _run_subtask(self, subtask: SubTask, context: Dict[str, str]):
        """Exécute une sous-tâche via l'agent correspondant."""
        subtask.status = "running"
        start = time.time()

        # Enrichir la description avec les résultats des dépendances
        dep_context = "\n".join(
            f"[{dep_id}]: {context[dep_id][:200]}"
            for dep_id in subtask.depends_on
            if dep_id in context
        )
        full_prompt = subtask.description
        if dep_context:
            full_prompt += f"\n\nContexte des étapes précédentes:\n{dep_context}"

        try:
            agent = self._get_agent(subtask.agent_role)
            if agent:
                from core.agentic_framework import AgentMessage, AgentRole
                role_map = {
                    "analyst": AgentRole.ANALYST,
                    "researcher": AgentRole.RESEARCHER,
                    "coder": AgentRole.CODER,
                    "critic": AgentRole.CRITIC,
                }
                msg = AgentMessage(
                    sender="planner",
                    recipient=agent.name,
                    role=role_map[subtask.agent_role],
                    content=full_prompt,
                    task_id=subtask.task_id
                )
                responses = await asyncio.wait_for(
                    agent.process_message(self.engine, msg), timeout=40
                )
                subtask.result = responses[0].content if responses else "Pas de résultat"
                subtask.confidence = responses[0].metadata.get("confidence", 0.5) if responses else 0.0
            else:
                # Fallback LLM direct
                subtask.result = await self.engine.brain['llm'].generate(full_prompt)
                subtask.confidence = 0.6

            subtask.status = "done"
        except Exception as e:
            subtask.status = "failed"
            subtask.result = f"Erreur: {e}"
            subtask.confidence = 0.0
            logger.error(f"Sous-tâche {subtask.task_id} échouée: {e}")

        subtask.duration_ms = (time.time() - start) * 1000

        # BOUCLE FERMÉE: notifier le feedback_loop du résultat de cette sous-tâche
        if hasattr(self.engine, 'feedback_loop') and self.engine.feedback_loop:
            agent_name = self._get_agent_name(subtask.agent_role)
            if agent_name:
                await self.engine.feedback_loop.process_episode(
                    agent_name=agent_name,
                    query=subtask.description,
                    result=subtask.result or "",
                    confidence=subtask.confidence
                )

    def _get_agent(self, role: str):
        """Récupère l'agent correspondant au rôle depuis le swarm."""
        if not hasattr(self.engine, 'swarm'):
            return None
        role_to_name = {
            "analyst": "Athena-Analyst",
            "researcher": "Athena-Researcher",
            "coder": "Athena-Coder",
            "critic": "Athena-Critic",
        }
        return self.engine.swarm.agents.get(role_to_name.get(role))

    def _get_agent_name(self, role: str) -> Optional[str]:
        """Retourne le nom de l'agent pour un rôle donné."""
        role_to_name = {
            "analyst": "Athena-Analyst",
            "researcher": "Athena-Researcher",
            "coder": "Athena-Coder",
            "critic": "Athena-Critic",
        }
        return role_to_name.get(role)

    async def _synthesize(self, plan: ExecutionPlan) -> str:
        """Synthèse finale: fusionne tous les résultats en une réponse cohérente."""
        done_tasks = [t for t in plan.subtasks if t.status == "done"]
        if not done_tasks:
            return "Aucun résultat disponible."

        parts = "\n\n".join(
            f"[{t.agent_role.upper()} - confiance {t.confidence:.0%}]:\n{t.result}"
            for t in done_tasks
        )

        if not self.engine.brain or not self.engine.brain.get('llm'):
            return parts

        synthesis_prompt = (
            f"Tu es ATHÉNA. Voici les analyses de tes agents pour la question: '{plan.original_query}'\n\n"
            f"{parts}\n\n"
            f"Synthétise ces résultats en une réponse finale claire, structurée et actionnable."
        )
        try:
            return await asyncio.wait_for(
                self.engine.brain['llm'].generate(synthesis_prompt), timeout=30
            )
        except Exception as e:
            logger.error(f"Synthèse échouée: {e}")
            return parts

    def format_plan_trace(self, plan: ExecutionPlan) -> str:
        """Formate la trace d'exécution pour l'observabilité."""
        lines = [f"📋 Plan pour: {plan.original_query[:60]}"]
        for t in plan.subtasks:
            icon = "✅" if t.status == "done" else "❌" if t.status == "failed" else "⏳"
            lines.append(
                f"  {icon} [{t.task_id}] {t.agent_role} | "
                f"{t.duration_ms:.0f}ms | confiance: {t.confidence:.0%}"
            )
        lines.append(f"⏱ Total: {plan.total_duration_ms:.0f}ms | Succès: {plan.success}")
        return "\n".join(lines)
