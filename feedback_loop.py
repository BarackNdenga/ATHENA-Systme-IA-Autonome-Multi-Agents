#!/usr/bin/env python3
"""
core/feedback_loop.py - Boucle de rétroaction fermée
C'est ici qu'ATHÉNA apprend vraiment de ses actions passées.

Cycle complet:
  Action → Résultat → Évaluation → Ajustement → Prochaine action améliorée

Concrètement:
  - Le score Critic de chaque épisode ajuste les poids des agents
  - Les agents sous-performants reçoivent plus de contexte historique
  - Les stratégies gagnantes sont renforcées automatiquement
  - Un profil d'amélioration est maintenu par agent
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

PROFILES_FILE = Path("./feedback_profiles.json")


@dataclass
class AgentPerformanceProfile:
    """Profil évolutif d'un agent — mis à jour après chaque épisode."""
    agent_name: str
    role: str
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=50))
    success_rate: float = 1.0
    context_boost: int = 0        # Nb d'épisodes historiques injectés en plus
    prompt_adjustments: List[str] = field(default_factory=list)  # Instructions adaptatives
    total_episodes: int = 0
    consecutive_failures: int = 0

    def update(self, confidence: float, success: bool):
        self.confidence_history.append(confidence)
        self.total_episodes += 1
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        if len(self.confidence_history) >= 3:
            self.success_rate = round(
                sum(1 for c in self.confidence_history if c >= 0.6) / len(self.confidence_history), 3
            )

    @property
    def avg_confidence(self) -> float:
        if not self.confidence_history:
            return 0.5
        return round(sum(self.confidence_history) / len(self.confidence_history), 3)

    @property
    def trend(self) -> str:
        """Tendance récente: improving / declining / stable."""
        if len(self.confidence_history) < 4:
            return "stable"
        recent = list(self.confidence_history)[-4:]
        if recent[-1] > recent[0] + 0.1:
            return "improving"
        if recent[-1] < recent[0] - 0.1:
            return "declining"
        return "stable"


class FeedbackLoop:
    """
    Boucle fermée d'auto-amélioration.
    Observe les performances → ajuste les stratégies → améliore les prochaines actions.
    """

    # Seuils d'adaptation
    LOW_CONFIDENCE_THRESHOLD = 0.55
    HIGH_CONFIDENCE_THRESHOLD = 0.80
    BOOST_TRIGGER_FAILURES = 2       # Nb d'échecs consécutifs avant boost
    MAX_CONTEXT_BOOST = 5            # Max épisodes historiques injectés

    def __init__(self, engine):
        self.engine = engine
        self.profiles: Dict[str, AgentPerformanceProfile] = {}
        self._adjustment_log: List[Dict[str, Any]] = []

    def register_agent(self, agent_name: str, role: str):
        """Enregistre un agent — charge son profil persisté si disponible."""
        if agent_name not in self.profiles:
            self.profiles[agent_name] = AgentPerformanceProfile(
                agent_name=agent_name, role=role
            )

    async def load_profiles(self):
        """Charge les profils persistés depuis le disque au démarrage."""
        if not PROFILES_FILE.exists():
            logger.info("FeedbackLoop: aucun profil persisté, démarrage à zéro.")
            return
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, lambda: json.loads(PROFILES_FILE.read_text(encoding="utf-8")))
            for name, p in data.items():
                profile = AgentPerformanceProfile(
                    agent_name=p["agent_name"],
                    role=p["role"],
                    success_rate=p.get("success_rate", 1.0),
                    context_boost=p.get("context_boost", 0),
                    prompt_adjustments=p.get("prompt_adjustments", []),
                    total_episodes=p.get("total_episodes", 0),
                    consecutive_failures=p.get("consecutive_failures", 0),
                )
                for c in p.get("confidence_history", []):
                    profile.confidence_history.append(c)
                self.profiles[name] = profile
            logger.info(f"FeedbackLoop: {len(self.profiles)} profils chargés depuis {PROFILES_FILE}.")
        except Exception as e:
            logger.warning(f"FeedbackLoop: impossible de charger les profils: {e}")

    async def save_profiles(self):
        """Persiste tous les profils sur le disque."""
        data = {
            name: {
                "agent_name": p.agent_name,
                "role": p.role,
                "success_rate": p.success_rate,
                "context_boost": p.context_boost,
                "prompt_adjustments": p.prompt_adjustments,
                "total_episodes": p.total_episodes,
                "consecutive_failures": p.consecutive_failures,
                "confidence_history": list(p.confidence_history),
            }
            for name, p in self.profiles.items()
        }
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: PROFILES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        )
        logger.info(f"FeedbackLoop: {len(self.profiles)} profils persistés dans {PROFILES_FILE}.")

    async def shutdown(self):
        """Sauvegarde les profils avant l'arrêt."""
        await self.save_profiles()

    async def process_episode(
        self,
        agent_name: str,
        query: str,
        result: str,
        confidence: float,
        critic_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Traite un épisode terminé et ajuste le profil de l'agent.
        Retourne les ajustements appliqués.
        """
        if agent_name not in self.profiles:
            self.register_agent(agent_name, "unknown")

        profile = self.profiles[agent_name]
        effective_confidence = critic_score if critic_score is not None else confidence
        success = effective_confidence >= self.LOW_CONFIDENCE_THRESHOLD

        profile.update(effective_confidence, success)
        adjustments = await self._compute_adjustments(profile, query, result, effective_confidence)

        # Stocker dans le graphe de connaissances si disponible
        if hasattr(self.engine, 'knowledge_graph') and self.engine.knowledge_graph:
            await self.engine.knowledge_graph.add_relation(
                source=agent_name,
                target=f"perf_{int(effective_confidence * 100)}pct",
                relation="performance",
                weight=effective_confidence
            )

        log_entry = {
            "timestamp": time.time(),
            "agent": agent_name,
            "confidence": effective_confidence,
            "success": success,
            "trend": profile.trend,
            "adjustments": adjustments
        }
        self._adjustment_log.append(log_entry)

        if adjustments:
            logger.info(f"[FeedbackLoop] {agent_name} → {adjustments}")

        return log_entry

    async def _compute_adjustments(
        self,
        profile: AgentPerformanceProfile,
        query: str,
        result: str,
        confidence: float
    ) -> List[str]:
        """Calcule et applique les ajustements nécessaires."""
        adjustments = []

        # Cas 1: Agent en difficulté → boost de contexte historique
        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            if profile.context_boost < self.MAX_CONTEXT_BOOST:
                profile.context_boost = min(profile.context_boost + 1, self.MAX_CONTEXT_BOOST)
                adjustments.append(f"context_boost+1 → {profile.context_boost} épisodes historiques")

        # Cas 2: Agent performant → réduire le contexte superflu
        elif confidence >= self.HIGH_CONFIDENCE_THRESHOLD and profile.context_boost > 0:
            profile.context_boost = max(0, profile.context_boost - 1)
            adjustments.append(f"context_boost-1 → {profile.context_boost} épisodes historiques")

        # Cas 3: Échecs consécutifs → injecter une instruction corrective
        if profile.consecutive_failures >= self.BOOST_TRIGGER_FAILURES:
            instruction = await self._generate_corrective_instruction(profile, query, result)
            if instruction and instruction not in profile.prompt_adjustments:
                profile.prompt_adjustments.append(instruction)
                if len(profile.prompt_adjustments) > 5:
                    profile.prompt_adjustments.pop(0)  # Garder les 5 plus récentes
                adjustments.append(f"instruction_corrective: {instruction[:60]}...")

        # Cas 4: Tendance déclinante → alerte et reset partiel
        if profile.trend == "declining" and profile.total_episodes > 5:
            adjustments.append("tendance_déclinante: surveillance renforcée")

        return adjustments

    async def _generate_corrective_instruction(
        self, profile: AgentPerformanceProfile, query: str, result: str
    ) -> Optional[str]:
        """Génère une instruction corrective via LLM pour améliorer l'agent."""
        if not self.engine.brain or not self.engine.brain.get('llm'):
            return None
        prompt = (
            f"L'agent '{profile.agent_name}' (rôle: {profile.role}) a un taux de succès de "
            f"{profile.success_rate:.0%} sur {profile.total_episodes} épisodes.\n"
            f"Dernière tâche: '{query[:100]}'\n"
            f"Résultat obtenu: '{result[:150]}'\n"
            f"En UNE phrase courte, donne une instruction pour améliorer ses prochaines réponses."
        )
        try:
            instruction = await asyncio.wait_for(
                self.engine.brain['llm'].generate(prompt), timeout=15
            )
            return instruction.strip()[:200]
        except Exception:
            return None

    async def get_agent_context(self, agent_name: str, query: str) -> str:
        """
        Retourne le contexte enrichi pour un agent avant qu'il agisse.
        Inclut: épisodes similaires passés + instructions correctives accumulées.
        """
        if agent_name not in self.profiles:
            return ""

        profile = self.profiles[agent_name]
        context_parts = []

        # Instructions correctives accumulées
        if profile.prompt_adjustments:
            context_parts.append(
                "Instructions issues de tes performances passées:\n" +
                "\n".join(f"- {instr}" for instr in profile.prompt_adjustments[-3:])
            )

        # Épisodes similaires depuis la mémoire épisodique
        if hasattr(self.engine, 'episodic_memory') and self.engine.episodic_memory:
            n = profile.context_boost + 2  # Minimum 2, plus si boost actif
            episodes = await self.engine.episodic_memory.recall_similar(query, n=n)
            agent_episodes = [e for e in episodes if e.agent == agent_name]
            if agent_episodes:
                ep_lines = []
                for ep in agent_episodes[:profile.context_boost + 2]:
                    ep_lines.append(
                        f"  • [{ep.confidence:.0%}] '{ep.query[:60]}' → '{ep.result[:80]}'"
                    )
                context_parts.append("Tes expériences similaires passées:\n" + "\n".join(ep_lines))

        return "\n\n".join(context_parts)

    def get_system_health(self) -> Dict[str, Any]:
        """Rapport de santé global du système basé sur les performances des agents."""
        if not self.profiles:
            return {"status": "no_data"}

        avg_success = sum(p.success_rate for p in self.profiles.values()) / len(self.profiles)
        struggling = [
            p.agent_name for p in self.profiles.values()
            if p.success_rate < self.LOW_CONFIDENCE_THRESHOLD
        ]
        improving = [
            p.agent_name for p in self.profiles.values()
            if p.trend == "improving"
        ]

        return {
            "system_success_rate": round(avg_success, 3),
            "total_adjustments": len(self._adjustment_log),
            "struggling_agents": struggling,
            "improving_agents": improving,
            "agent_profiles": {
                name: {
                    "success_rate": p.success_rate,
                    "avg_confidence": p.avg_confidence,
                    "trend": p.trend,
                    "context_boost": p.context_boost,
                    "total_episodes": p.total_episodes,
                    "active_instructions": len(p.prompt_adjustments)
                }
                for name, p in self.profiles.items()
            }
        }
