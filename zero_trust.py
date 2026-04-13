#!/usr/bin/env python3
"""
core/zero_trust.py - Zero Trust Architecture réelle
Vérification continue, scoring comportemental, isolation des entités suspectes.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List
from collections import deque

import psutil

logger = logging.getLogger(__name__)


class BehavioralProfile:
    """Profil comportemental d'une entité avec historique glissant."""

    def __init__(self, entity_id: str, window: int = 20):
        self.entity_id = entity_id
        self.request_times: deque = deque(maxlen=window)
        self.anomaly_count: int = 0
        self.trust_score: float = 1.0
        self.isolated: bool = False

    def record_request(self):
        self.request_times.append(time.time())

    def compute_request_rate(self) -> float:
        """Requêtes par seconde sur la fenêtre glissante."""
        if len(self.request_times) < 2:
            return 0.0
        elapsed = self.request_times[-1] - self.request_times[0]
        return len(self.request_times) / elapsed if elapsed > 0 else 0.0


class ZeroTrustSentinel:
    """Sentinel Zero Trust: scoring comportemental continu + isolation."""

    # Seuils configurables
    MAX_REQUEST_RATE = 10.0   # req/s
    MAX_CPU_PERCENT = 90.0
    MIN_TRUST_SCORE = 0.4
    ANOMALY_DECAY = 0.05      # récupération progressive du score

    def __init__(self):
        self.profiles: Dict[str, BehavioralProfile] = {}
        self.system_baseline: Dict[str, float] = {}
        self.anomaly_detector_active = False
        self._monitor_task: asyncio.Task = None

    async def activate_anomaly_detection(self):
        """Démarre la surveillance système en arrière-plan."""
        self.system_baseline = await self._capture_baseline()
        self.anomaly_detector_active = True
        self._monitor_task = asyncio.create_task(self._system_monitor_loop())
        logger.info("Zero Trust: détection d'anomalies activée.")

    async def continuous_verification(self, entity_id: str, behavior: Dict[str, Any]) -> float:
        """Vérifie et met à jour le score de confiance d'une entité."""
        if entity_id not in self.profiles:
            self.profiles[entity_id] = BehavioralProfile(entity_id)

        profile = self.profiles[entity_id]
        profile.record_request()

        score = self._compute_trust_score(profile, behavior)
        profile.trust_score = score

        if score < self.MIN_TRUST_SCORE:
            profile.anomaly_count += 1
            logger.warning(f"Zero Trust VIOLATION: {entity_id} score={score:.2f} anomalies={profile.anomaly_count}")
            await self._isolate_entity(entity_id)
        else:
            # Récupération progressive
            profile.trust_score = min(1.0, score + self.ANOMALY_DECAY)

        return profile.trust_score

    def _compute_trust_score(self, profile: BehavioralProfile, behavior: Dict[str, Any]) -> float:
        """Score multi-facteurs: taux de requêtes + CPU + anomalies passées."""
        score = 1.0

        # Facteur 1: taux de requêtes
        rate = profile.compute_request_rate()
        if rate > self.MAX_REQUEST_RATE:
            score -= min((rate - self.MAX_REQUEST_RATE) / self.MAX_REQUEST_RATE, 0.4)

        # Facteur 2: CPU système
        cpu = psutil.cpu_percent(interval=None)
        if cpu > self.MAX_CPU_PERCENT:
            score -= 0.2

        # Facteur 3: pénalité historique
        score -= profile.anomaly_count * 0.05

        # Facteur 4: comportement explicite passé en paramètre
        if behavior.get("failed_auth", False):
            score -= 0.3
        if behavior.get("unusual_hour", False):
            score -= 0.1

        return round(max(0.0, min(1.0, score)), 3)

    async def _isolate_entity(self, entity_id: str):
        """Isole une entité suspecte (log + flag)."""
        if entity_id in self.profiles:
            self.profiles[entity_id].isolated = True
        logger.critical(f"ISOLATION: entité {entity_id} mise en quarantaine.")

    async def _capture_baseline(self) -> Dict[str, float]:
        """Capture les métriques système de référence."""
        return {
            "cpu": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory().percent,
            "timestamp": time.time()
        }

    async def _system_monitor_loop(self):
        """Boucle de surveillance système toutes les 30s."""
        while self.anomaly_detector_active:
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                if cpu > self.MAX_CPU_PERCENT:
                    logger.warning(f"Anomalie CPU: {cpu:.1f}%")
                if mem > 95.0:
                    logger.warning(f"Anomalie mémoire: {mem:.1f}%")
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            await asyncio.sleep(30)

    def get_trust_report(self) -> List[Dict[str, Any]]:
        """Rapport de confiance de toutes les entités."""
        return [
            {
                "entity": p.entity_id,
                "trust_score": p.trust_score,
                "anomalies": p.anomaly_count,
                "isolated": p.isolated
            }
            for p in self.profiles.values()
        ]

    async def shutdown(self):
        self.anomaly_detector_active = False
        if self._monitor_task:
            self._monitor_task.cancel()

