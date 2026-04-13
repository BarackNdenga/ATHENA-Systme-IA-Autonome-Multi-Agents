#!/usr/bin/env python3
"""
brain/episodic_memory.py - Mémoire épisodique : ce qu'ATHÉNA a FAIT
Séparée de la mémoire sémantique (ce qu'elle SAIT).
Stocke les épisodes horodatés avec contexte, action, résultat et score.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

EPISODES_FILE = Path("./episodic_memory.jsonl")


@dataclass
class Episode:
    """Un épisode = une interaction complète avec contexte et résultat."""
    episode_id: str
    timestamp: float
    query: str
    agent: str
    action_taken: str
    result: str
    confidence: float
    duration_ms: float
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        return cls(**d)


class EpisodicMemory:
    """
    Mémoire épisodique persistante sur disque (JSONL).
    Permet de retrouver des épisodes similaires par mots-clés ou agent.
    """

    def __init__(self, filepath: Path = EPISODES_FILE):
        self.filepath = filepath
        self._episodes: List[Episode] = []

    async def initialize(self):
        """Charge les épisodes existants depuis le disque."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_from_disk)
        logger.info(f"EpisodicMemory chargée: {len(self._episodes)} épisodes.")

    def _load_from_disk(self):
        if not self.filepath.exists():
            return
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._episodes.append(Episode.from_dict(json.loads(line)))
                    except Exception:
                        pass

    async def record(
        self,
        query: str,
        agent: str,
        action_taken: str,
        result: str,
        confidence: float,
        duration_ms: float,
        tags: List[str] = None
    ) -> Episode:
        """Enregistre un nouvel épisode en mémoire et sur disque."""
        episode = Episode(
            episode_id=f"ep_{int(time.time() * 1000)}",
            timestamp=time.time(),
            query=query,
            agent=agent,
            action_taken=action_taken,
            result=result,
            confidence=confidence,
            duration_ms=duration_ms,
            tags=tags or []
        )
        self._episodes.append(episode)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._append_to_disk, episode)
        return episode

    def _append_to_disk(self, episode: Episode):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    async def recall_similar(self, query: str, n: int = 5) -> List[Episode]:
        """Retrouve les épisodes dont la query contient des mots communs."""
        query_words = set(query.lower().split())
        scored = []
        for ep in self._episodes:
            ep_words = set(ep.query.lower().split())
            overlap = len(query_words & ep_words)
            if overlap > 0:
                scored.append((overlap, ep))
        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [ep for _, ep in scored[:n]]

    async def recall_by_agent(self, agent: str, n: int = 10) -> List[Episode]:
        """Retourne les derniers épisodes d'un agent spécifique."""
        filtered = [ep for ep in self._episodes if ep.agent == agent]
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)[:n]

    async def get_stats(self) -> Dict[str, Any]:
        """Statistiques globales de la mémoire épisodique."""
        if not self._episodes:
            return {"total": 0}
        avg_conf = sum(e.confidence for e in self._episodes) / len(self._episodes)
        agents = {}
        for ep in self._episodes:
            agents[ep.agent] = agents.get(ep.agent, 0) + 1
        return {
            "total_episodes": len(self._episodes),
            "avg_confidence": round(avg_conf, 3),
            "episodes_by_agent": agents,
            "oldest": self._episodes[0].timestamp,
            "newest": self._episodes[-1].timestamp
        }
