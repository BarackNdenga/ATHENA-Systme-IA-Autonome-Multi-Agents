#!/usr/bin/env python3
"""
interfaces/api.py - API FastAPI pour intégration externe
Supporte REST classique + WebSocket streaming pour réponses en temps réel.
"""

import asyncio
import json
import time
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional

from core.engine import AthenaEngine

app = FastAPI(title="ATHÉNA API", version="2.1.0")
_engine: AthenaEngine = None


def set_engine(engine: AthenaEngine):
    global _engine
    _engine = engine


class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    global _engine
    if _engine is None:
        _engine = AthenaEngine()
        await _engine.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    if _engine:
        await _engine.shutdown()


# ─── REST classique ──────────────────────────────────────────────────────────

@app.post("/query")
async def process_query(request: QueryRequest):
    """Traite une requête vers ATHÉNA (réponse complète)."""
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine non initialisé")
    try:
        response = await _engine.process_query(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    if not _engine:
        return {"status": "not_initialized"}
    swarm_stats = _engine.swarm.get_swarm_stats() if hasattr(_engine, 'swarm') else []
    return {
        "status": "active",
        "modules": list(_engine.modules.keys()),
        "swarm_stats": swarm_stats
    }


@app.get("/memory/{limit}")
async def get_memory(limit: int = 5):
    if _engine and _engine.brain and _engine.brain.get('memory'):
        memories = await _engine.brain['memory'].retrieve_relevant("recent", limit)
        return {"memories": memories.split('\n') if memories else []}
    return {"error": "Mémoire non disponible"}


@app.get("/trust-report")
async def trust_report():
    if _engine and _engine.security:
        return {"report": _engine.security.zero_trust.get_trust_report()}
    return {"error": "Sécurité non disponible"}


@app.get("/health")
async def health():
    if not _engine:
        return {"status": "not_initialized"}
    health_data = {"status": "active"}
    if hasattr(_engine, 'feedback_loop') and _engine.feedback_loop:
        health_data["feedback_loop"] = _engine.feedback_loop.get_system_health()
    if hasattr(_engine, 'episodic_memory') and _engine.episodic_memory:
        health_data["episodic_memory"] = await _engine.episodic_memory.get_stats()
    if hasattr(_engine, 'knowledge_graph') and _engine.knowledge_graph:
        health_data["knowledge_graph"] = await _engine.knowledge_graph.get_stats()
    if hasattr(_engine, 'obs') and _engine.obs:
        health_data["observability"] = _engine.obs.get_metrics_json()
    return health_data


# ─── WebSocket streaming ─────────────────────────────────────────────────────

@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket streaming: envoie les étapes du plan en temps réel.

    Protocole client:
      → Envoyer: {"query": "ta question"}
      ← Recevoir: flux de messages JSON jusqu'à {"type": "done"}

    Types de messages reçus:
      {"type": "start",    "data": {"query": "..."}}
      {"type": "plan",     "data": {"subtasks": [...]}}
      {"type": "step",     "data": {"task_id": "t0", "agent": "analyst", "status": "running"}}
      {"type": "result",   "data": {"task_id": "t0", "agent": "analyst", "confidence": 0.8, "result": "..."}}
      {"type": "feedback", "data": {"agent": "...", "adjustments": [...]}}
      {"type": "final",    "data": {"response": "...", "duration_ms": 1200}}
      {"type": "done"}
      {"type": "error",    "data": {"message": "..."}}
    """
    await websocket.accept()

    if not _engine:
        await websocket.send_json({"type": "error", "data": {"message": "Engine non initialisé"}})
        await websocket.close()
        return

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30)
        payload = json.loads(raw)
        query = payload.get("query", "").strip()
        if not query:
            await websocket.send_json({"type": "error", "data": {"message": "query manquante"}})
            await websocket.close()
            return
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "data": {"message": "timeout attente query"}})
        await websocket.close()
        return
    except Exception as e:
        await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        await websocket.close()
        return

    start = time.time()

    try:
        # Étape 1: annonce du début
        await websocket.send_json({"type": "start", "data": {"query": query}})

        # Étape 2: décomposition du plan
        subtasks = await _engine.planner._decompose(query)
        await websocket.send_json({
            "type": "plan",
            "data": {"subtasks": [
                {"task_id": t.task_id, "agent": t.agent_role, "depends_on": t.depends_on}
                for t in subtasks
            ]}
        })

        # Étape 3: exécution avec streaming par sous-tâche
        from core.react_planner import ExecutionPlan
        plan = ExecutionPlan(original_query=query, subtasks=subtasks)
        completed = {}
        max_rounds = len(subtasks) + 2

        for _ in range(max_rounds):
            ready = [
                t for t in plan.subtasks
                if t.status == "pending"
                and all(dep in completed for dep in t.depends_on)
            ]
            if not ready:
                break

            for subtask in ready:
                # Signal "en cours"
                await websocket.send_json({
                    "type": "step",
                    "data": {"task_id": subtask.task_id, "agent": subtask.agent_role, "status": "running"}
                })

                await _engine.planner._run_subtask(subtask, completed)

                # Résultat de la sous-tâche
                await websocket.send_json({
                    "type": "result",
                    "data": {
                        "task_id": subtask.task_id,
                        "agent": subtask.agent_role,
                        "status": subtask.status,
                        "confidence": subtask.confidence,
                        "duration_ms": subtask.duration_ms,
                        "result": (subtask.result or "")[:500]
                    }
                })

                # Feedback loop: envoyer les ajustements en temps réel
                if hasattr(_engine, 'feedback_loop') and _engine.feedback_loop:
                    agent_name = _engine.planner._get_agent_name(subtask.agent_role)
                    if agent_name and agent_name in _engine.feedback_loop.profiles:
                        profile = _engine.feedback_loop.profiles[agent_name]
                        await websocket.send_json({
                            "type": "feedback",
                            "data": {
                                "agent": agent_name,
                                "success_rate": profile.success_rate,
                                "trend": profile.trend,
                                "context_boost": profile.context_boost,
                                "active_instructions": len(profile.prompt_adjustments)
                            }
                        })

                if subtask.status == "done":
                    completed[subtask.task_id] = subtask.result

        # Étape 4: synthèse finale
        plan.final_synthesis = await _engine.planner._synthesize(plan)
        plan.success = any(t.status == "done" for t in plan.subtasks)
        duration_ms = (time.time() - start) * 1000

        await websocket.send_json({
            "type": "final",
            "data": {
                "response": plan.final_synthesis,
                "duration_ms": round(duration_ms, 1),
                "success": plan.success
            }
        })

        # Stocker en mémoire
        await _engine.brain['memory'].store(query, plan.final_synthesis)
        await _engine.episodic_memory.record(
            query=query, agent="WebSocket",
            action_taken="ws_react_plan",
            result=plan.final_synthesis,
            confidence=0.8 if plan.success else 0.3,
            duration_ms=duration_ms
        )

        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket déconnecté pendant la requête: {query[:50]}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


import logging
logger = logging.getLogger(__name__)

