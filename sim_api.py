#!/usr/bin/env python3
"""
simulation/sim_api.py — Serveur FastAPI + WebSocket pour la simulation navigateur.

Usage:
    uvicorn simulation.sim_api:app --reload --port 8001
    # puis ouvrir simulation/sim_web.html dans le navigateur
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ATHÉNA Simulation", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AGENTS_DEF = [
    {"name": "Athena-Analyst",    "role": "analyst",    "emoji": "🔍"},
    {"name": "Athena-Researcher", "role": "researcher", "emoji": "🌐"},
    {"name": "Athena-Coder",      "role": "coder",      "emoji": "💻"},
    {"name": "Athena-Critic",     "role": "critic",     "emoji": "⚖️"},
]

QUERIES = [
    "Analyse l'architecture microservices d'un système bancaire",
    "Génère un script Python pour parser du JSON imbriqué",
    "Recherche les meilleures pratiques Zero Trust 2025",
    "Implémente un algorithme de tri rapide optimisé",
    "Analyse les vulnérabilités d'une API REST publique",
    "Génère un pipeline CI/CD pour un projet FastAPI",
    "Explique la cryptographie post-quantique Kyber1024",
    "Crée un système de cache Redis avec TTL adaptatif",
]


@dataclass
class AgentState:
    name: str
    role: str
    emoji: str
    episodes: int = 0
    avg_confidence: float = 0.5
    success_rate: float = 1.0
    context_boost: int = 0
    trend: str = "stable"
    status: str = "idle"
    confidence_history: List[float] = field(default_factory=list)

    def update(self, confidence: float):
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 50:
            self.confidence_history.pop(0)
        self.episodes += 1
        self.avg_confidence = round(sum(self.confidence_history) / len(self.confidence_history), 2)
        self.success_rate = round(
            sum(1 for c in self.confidence_history if c >= 0.55) / len(self.confidence_history), 2
        )
        if len(self.confidence_history) >= 4:
            r = self.confidence_history[-4:]
            if r[-1] > r[0] + 0.08:
                self.trend = "improving"
            elif r[-1] < r[0] - 0.08:
                self.trend = "declining"
            else:
                self.trend = "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "role": self.role, "emoji": self.emoji,
            "episodes": self.episodes, "avg_confidence": self.avg_confidence,
            "success_rate": self.success_rate, "context_boost": self.context_boost,
            "trend": self.trend, "status": self.status,
        }


# État global partagé entre connexions
_global_states: Dict[str, AgentState] = {
    a["name"]: AgentState(**a) for a in AGENTS_DEF
}
_cycle = 0
_active_connections: List[WebSocket] = []


async def broadcast(msg: Dict[str, Any]):
    dead = []
    for ws in _active_connections:
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _active_connections.remove(ws)


async def simulate_agent(state: AgentState, query: str, delay: float = 0.0):
    await asyncio.sleep(delay)
    state.status = "running"
    await broadcast({"type": "agent_status", "agent": state.name, "status": "running", "query": query})

    await asyncio.sleep(random.uniform(0.4, 0.9))
    await broadcast({"type": "thought", "agent": state.name, "emoji": state.emoji,
                     "text": f"Analyse de la tâche en cours ({state.role})…"})

    await asyncio.sleep(random.uniform(0.5, 1.1))
    await broadcast({"type": "action", "agent": state.name, "emoji": state.emoji,
                     "text": f"Exécution de l'action principale ({state.role})…"})

    base = 0.45 + (state.episodes * 0.012)
    confidence = round(min(max(base + random.uniform(-0.15, 0.22), 0.1), 0.99), 2)
    state.update(confidence)

    feedback_msg = None
    if confidence < 0.55:
        state.context_boost = min(state.context_boost + 1, 5)
        feedback_msg = {"type": "feedback", "agent": state.name, "kind": "boost",
                        "text": f"context_boost +1 → {state.context_boost}", "confidence": confidence}
    elif confidence >= 0.80 and state.context_boost > 0:
        state.context_boost -= 1
        feedback_msg = {"type": "feedback", "agent": state.name, "kind": "reduce",
                        "text": f"context_boost -1 → {state.context_boost}", "confidence": confidence}

    state.status = "done" if confidence >= 0.55 else "failed"
    await broadcast({"type": "observation", "agent": state.name, "emoji": state.emoji,
                     "confidence": confidence, "status": state.status, "trend": state.trend})
    if feedback_msg:
        await broadcast(feedback_msg)
    await broadcast({"type": "agent_update", "agent": state.to_dict()})


async def simulation_loop():
    global _cycle
    await asyncio.sleep(1)
    while True:
        if not _active_connections:
            await asyncio.sleep(1)
            continue

        _cycle += 1
        query = random.choice(QUERIES)

        for s in _global_states.values():
            s.status = "idle"

        await broadcast({"type": "cycle_start", "cycle": _cycle, "query": query,
                         "agents": [s.to_dict() for s in _global_states.values()]})

        # Analyst en premier
        await simulate_agent(_global_states["Athena-Analyst"], query)

        # Researcher + Coder en parallèle
        await asyncio.gather(
            simulate_agent(_global_states["Athena-Researcher"], query, delay=0.0),
            simulate_agent(_global_states["Athena-Coder"], query, delay=0.3),
        )

        # Critic en dernier
        await simulate_agent(_global_states["Athena-Critic"], query, delay=0.2)

        system_health = {
            "avg_confidence": round(
                sum(s.avg_confidence for s in _global_states.values()) / len(_global_states), 2
            ),
            "total_episodes": sum(s.episodes for s in _global_states.values()),
            "improving": [s.name for s in _global_states.values() if s.trend == "improving"],
        }
        await broadcast({"type": "cycle_end", "cycle": _cycle, "health": system_health})
        await asyncio.sleep(3)


@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "sim_web.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>sim_web.html introuvable</h1>")


@app.get("/status")
async def status():
    return {
        "cycle": _cycle,
        "connections": len(_active_connections),
        "agents": [s.to_dict() for s in _global_states.values()],
    }


@app.websocket("/ws/sim")
async def websocket_sim(websocket: WebSocket):
    await websocket.accept()
    _active_connections.append(websocket)
    # Envoyer l'état courant immédiatement
    await websocket.send_text(json.dumps({
        "type": "init",
        "cycle": _cycle,
        "agents": [s.to_dict() for s in _global_states.values()],
    }))
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        _active_connections.remove(websocket)
