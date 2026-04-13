#!/usr/bin/env python3
"""
tests/test_core.py - Tests de base pour valider le système ATHÉNA
Lance avec: pytest tests/ -v
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ─── Mémoire épisodique ─────────────────────────────────────────────────────

class TestEpisodicMemory:

    @pytest.fixture
    def episodic(self, tmp_path):
        from brain.episodic_memory import EpisodicMemory
        return EpisodicMemory(filepath=tmp_path / "episodes.jsonl")

    @pytest.mark.asyncio
    async def test_record_and_recall(self, episodic):
        await episodic.initialize()
        ep = await episodic.record(
            query="analyser le code",
            agent="Athena-Analyst",
            action_taken="llm_analysis",
            result="Le code est propre.",
            confidence=0.85,
            duration_ms=120.0,
            tags=["code", "analyse"]
        )
        assert ep.episode_id.startswith("ep_")
        assert ep.confidence == 0.85
        similar = await episodic.recall_similar("analyser le code Python")
        assert len(similar) >= 1

    @pytest.mark.asyncio
    async def test_stats(self, episodic):
        await episodic.initialize()
        await episodic.record("test", "agent-x", "action", "result", 0.9, 50.0)
        stats = await episodic.get_stats()
        assert stats["total_episodes"] == 1
        assert stats["avg_confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        from brain.episodic_memory import EpisodicMemory
        path = tmp_path / "ep.jsonl"
        em1 = EpisodicMemory(filepath=path)
        await em1.initialize()
        await em1.record("query1", "agent", "action", "result", 0.7, 100.0)
        em2 = EpisodicMemory(filepath=path)
        await em2.initialize()
        assert len(em2._episodes) == 1


# ─── Graphe de connaissances ─────────────────────────────────────────────────

class TestKnowledgeGraph:

    @pytest.fixture
    def kg(self, tmp_path):
        from brain.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph(filepath=tmp_path / "graph.json")

    @pytest.mark.asyncio
    async def test_add_and_relate(self, kg):
        await kg.initialize()
        await kg.add_concept("intelligence artificielle", {"type": "domaine"})
        await kg.add_concept("machine learning", {"type": "sous-domaine"})
        await kg.add_relation("intelligence artificielle", "machine learning", "contient", 0.9)
        related = await kg.get_related("intelligence artificielle")
        assert any(r[0] == "machine learning" for r in related)

    @pytest.mark.asyncio
    async def test_find_path(self, kg):
        await kg.initialize()
        await kg.add_relation("A", "B", "implique", 1.0)
        await kg.add_relation("B", "C", "cause", 1.0)
        path = await kg.find_path("A", "C")
        assert path == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_extract_and_add(self, kg):
        await kg.initialize()
        await kg.extract_and_add("analyse sémantique du langage naturel", "query_test")
        stats = await kg.get_stats()
        assert stats["nodes"] > 1

    @pytest.mark.asyncio
    async def test_central_concepts(self, kg):
        await kg.initialize()
        for i in range(5):
            await kg.add_relation("hub", f"node_{i}", "contient", 1.0)
        central = await kg.get_central_concepts(top_n=3)
        assert len(central) <= 3


# ─── Observabilité ───────────────────────────────────────────────────────────

class TestObservability:

    @pytest.fixture
    def hub(self, tmp_path):
        from core.observability import ObservabilityHub
        return ObservabilityHub(trace_file=tmp_path / "traces.jsonl")

    @pytest.mark.asyncio
    async def test_span_lifecycle(self, hub):
        span = hub.start_span("test_op", "engine", {"key": "val"})
        assert span.end_time == 0.0
        await hub.finish_span(span, status="ok")
        assert span.duration_ms > 0

    @pytest.mark.asyncio
    async def test_error_span(self, hub):
        span = hub.start_span("failing_op", "llm")
        await hub.finish_span(span, status="error", error="timeout")
        errors = hub.get_recent_errors()
        assert len(errors) == 1
        assert errors[0]["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_metrics_summary(self, hub):
        for _ in range(5):
            span = hub.start_span("query", "engine")
            await asyncio.sleep(0.001)
            await hub.finish_span(span)
        summary = hub.metrics.get_summary()
        assert "query" in summary
        assert summary["query"]["count"] == 5
        assert summary["query"]["error_rate"] == 0.0

    def test_dashboard_format(self, hub):
        dashboard = hub.get_dashboard()
        assert "ATHÉNA" in dashboard


# ─── ReAct Planner ───────────────────────────────────────────────────────────

class TestReActPlanner:

    def _make_engine(self):
        engine = MagicMock()
        engine.brain = {"llm": AsyncMock()}
        engine.brain["llm"].generate = AsyncMock(return_value="Résultat LLM simulé")
        engine.swarm = MagicMock()
        engine.swarm.agents = {}
        engine.knowledge_graph = None
        return engine

    @pytest.mark.asyncio
    async def test_decompose_analyst_only(self):
        from core.react_planner import ReActPlanner
        planner = ReActPlanner(self._make_engine())
        subtasks = await planner._decompose("explique ce concept")
        roles = [t.agent_role for t in subtasks]
        assert "analyst" in roles
        assert "critic" in roles

    @pytest.mark.asyncio
    async def test_decompose_with_coder(self):
        from core.react_planner import ReActPlanner
        planner = ReActPlanner(self._make_engine())
        subtasks = await planner._decompose("génère un script Python pour trier une liste")
        roles = [t.agent_role for t in subtasks]
        assert "coder" in roles

    @pytest.mark.asyncio
    async def test_decompose_with_researcher(self):
        from core.react_planner import ReActPlanner
        planner = ReActPlanner(self._make_engine())
        subtasks = await planner._decompose("recherche les dernières news sur l'IA")
        roles = [t.agent_role for t in subtasks]
        assert "researcher" in roles

    @pytest.mark.asyncio
    async def test_plan_and_execute(self):
        from core.react_planner import ReActPlanner
        planner = ReActPlanner(self._make_engine())
        plan = await planner.plan_and_execute("analyse ce texte")
        assert plan.original_query == "analyse ce texte"
        assert plan.total_duration_ms > 0

    def test_format_plan_trace(self):
        from core.react_planner import ReActPlanner, ExecutionPlan, SubTask
        planner = ReActPlanner(self._make_engine())
        plan = ExecutionPlan(
            original_query="test",
            subtasks=[SubTask("t0", "desc", "analyst", status="done", duration_ms=50, confidence=0.8)],
            total_duration_ms=50,
            success=True
        )
        trace = planner.format_plan_trace(plan)
        assert "analyst" in trace
        assert "✅" in trace


# ─── AgenticSwarm ────────────────────────────────────────────────────────────

class TestAgenticSwarm:

    def _make_engine(self):
        engine = MagicMock()
        engine.brain = {"llm": AsyncMock()}
        engine.brain["llm"].generate = AsyncMock(return_value="Analyse complète simulée.")
        engine.vision = {"spyder": AsyncMock()}
        engine.vision["spyder"].fetch_page = AsyncMock(return_value={"title": "Test", "text": "contenu"})
        return engine

    @pytest.mark.asyncio
    async def test_dispatch_returns_list(self):
        from core.agentic_framework import AgenticSwarm
        swarm = AgenticSwarm(self._make_engine())
        results = await swarm.dispatch_task("Analyse ce projet Python")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_swarm_stats(self):
        from core.agentic_framework import AgenticSwarm
        swarm = AgenticSwarm(self._make_engine())
        stats = swarm.get_swarm_stats()
        assert len(stats) == 4
        assert all("agent" in s for s in stats)

    @pytest.mark.asyncio
    async def test_react_steps_recorded(self):
        from core.agentic_framework import AgenticSwarm, AgentMessage, AgentRole
        engine = self._make_engine()
        swarm = AgenticSwarm(engine)
        agent = swarm.agents["Athena-Analyst"]
        msg = AgentMessage("user", "Athena-Analyst", AgentRole.ANALYST, "test query", task_id="t1")
        await agent.process_message(engine, msg)
        assert len(agent.episode_history) == 1
        assert agent.episode_history[0].confidence >= 0


# ─── Feedback Loop ───────────────────────────────────────────────────────────

class TestFeedbackLoop:

    def _make_engine(self):
        engine = MagicMock()
        engine.brain = {"llm": AsyncMock()}
        engine.brain["llm"].generate = AsyncMock(return_value="Instruction corrective simulée.")
        engine.knowledge_graph = None
        engine.episodic_memory = None
        return engine

    @pytest.mark.asyncio
    async def test_register_and_process(self):
        from core.feedback_loop import FeedbackLoop
        engine = self._make_engine()
        fl = FeedbackLoop(engine)
        fl.register_agent("Athena-Analyst", "analyst")
        result = await fl.process_episode(
            agent_name="Athena-Analyst",
            query="analyse ce code",
            result="Le code est bien structuré.",
            confidence=0.85
        )
        assert result["agent"] == "Athena-Analyst"
        assert result["confidence"] == 0.85
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_context_boost_on_low_confidence(self):
        from core.feedback_loop import FeedbackLoop
        engine = self._make_engine()
        fl = FeedbackLoop(engine)
        fl.register_agent("Athena-Coder", "coder")
        # 3 épisodes avec faible confiance → context_boost doit augmenter
        for _ in range(3):
            await fl.process_episode("Athena-Coder", "génère du code", "résultat court", 0.3)
        profile = fl.profiles["Athena-Coder"]
        assert profile.context_boost > 0

    @pytest.mark.asyncio
    async def test_context_boost_decreases_on_high_confidence(self):
        from core.feedback_loop import FeedbackLoop
        engine = self._make_engine()
        fl = FeedbackLoop(engine)
        fl.register_agent("Athena-Analyst", "analyst")
        fl.profiles["Athena-Analyst"].context_boost = 3
        await fl.process_episode("Athena-Analyst", "analyse", "réponse très détaillée et complète car bien structurée donc pertinente", 0.95)
        assert fl.profiles["Athena-Analyst"].context_boost < 3

    @pytest.mark.asyncio
    async def test_critic_score_propagation(self):
        from core.feedback_loop import FeedbackLoop
        engine = self._make_engine()
        fl = FeedbackLoop(engine)
        fl.register_agent("Athena-Researcher", "researcher")
        result = await fl.process_episode(
            agent_name="Athena-Researcher",
            query="recherche IA",
            result="résultat de recherche",
            confidence=0.7,
            critic_score=0.4   # Critic juge mal → doit déclencher boost
        )
        assert result["confidence"] == 0.4
        assert fl.profiles["Athena-Researcher"].context_boost > 0

    @pytest.mark.asyncio
    async def test_get_agent_context_empty(self):
        from core.feedback_loop import FeedbackLoop
        engine = self._make_engine()
        fl = FeedbackLoop(engine)
        fl.register_agent("Athena-Analyst", "analyst")
        ctx = await fl.get_agent_context("Athena-Analyst", "test query")
        assert isinstance(ctx, str)

    def test_system_health(self):
        from core.feedback_loop import FeedbackLoop
        engine = self._make_engine()
        fl = FeedbackLoop(engine)
        fl.register_agent("Athena-Analyst", "analyst")
        fl.register_agent("Athena-Coder", "coder")
        health = fl.get_system_health()
        assert "system_success_rate" in health
        assert "agent_profiles" in health
        assert len(health["agent_profiles"]) == 2

    @pytest.mark.asyncio
    async def test_consecutive_failures_trigger_instruction(self):
        from core.feedback_loop import FeedbackLoop
        engine = self._make_engine()
        fl = FeedbackLoop(engine)
        fl.register_agent("Athena-Coder", "coder")
        # 3 échecs consécutifs → génération d'instruction corrective
        for _ in range(3):
            await fl.process_episode("Athena-Coder", "génère algo", "err", 0.2)
        profile = fl.profiles["Athena-Coder"]
        assert profile.consecutive_failures >= 2 or len(profile.prompt_adjustments) > 0

    @pytest.mark.asyncio
    async def test_trend_detection(self):
        from core.feedback_loop import FeedbackLoop, AgentPerformanceProfile
        profile = AgentPerformanceProfile("test-agent", "analyst")
        # Tendance croissante
        for conf in [0.3, 0.4, 0.5, 0.7]:
            profile.update(conf, conf >= 0.55)
        assert profile.trend == "improving"
        # Tendance déclinante
        for conf in [0.8, 0.7, 0.5, 0.3]:
            profile.update(conf, conf >= 0.55)
        assert profile.trend == "declining"


# ─── Persistance FeedbackLoop ────────────────────────────────────────────────

class TestFeedbackLoopPersistence:

    def _make_engine(self):
        engine = MagicMock()
        engine.brain = {"llm": AsyncMock()}
        engine.brain["llm"].generate = AsyncMock(return_value="instruction")
        engine.knowledge_graph = None
        engine.episodic_memory = None
        return engine

    @pytest.mark.asyncio
    async def test_save_and_load_profiles(self, tmp_path):
        import core.feedback_loop as fl_module
        original = fl_module.PROFILES_FILE
        fl_module.PROFILES_FILE = tmp_path / "profiles.json"

        try:
            from core.feedback_loop import FeedbackLoop
            engine = self._make_engine()

            fl1 = FeedbackLoop(engine)
            fl1.register_agent("Athena-Analyst", "analyst")
            fl1.profiles["Athena-Analyst"].context_boost = 3
            fl1.profiles["Athena-Analyst"].prompt_adjustments = ["Sois plus précis"]
            fl1.profiles["Athena-Analyst"].total_episodes = 10
            await fl1.save_profiles()

            fl2 = FeedbackLoop(engine)
            fl2.register_agent("Athena-Analyst", "analyst")
            await fl2.load_profiles()

            p = fl2.profiles["Athena-Analyst"]
            assert p.context_boost == 3
            assert p.prompt_adjustments == ["Sois plus précis"]
            assert p.total_episodes == 10
        finally:
            fl_module.PROFILES_FILE = original

    @pytest.mark.asyncio
    async def test_load_missing_file(self, tmp_path):
        import core.feedback_loop as fl_module
        original = fl_module.PROFILES_FILE
        fl_module.PROFILES_FILE = tmp_path / "nonexistent.json"
        try:
            from core.feedback_loop import FeedbackLoop
            fl = FeedbackLoop(self._make_engine())
            fl.register_agent("Athena-Coder", "coder")
            await fl.load_profiles()
            assert "Athena-Coder" in fl.profiles
        finally:
            fl_module.PROFILES_FILE = original

    @pytest.mark.asyncio
    async def test_shutdown_saves(self, tmp_path):
        import core.feedback_loop as fl_module
        original = fl_module.PROFILES_FILE
        fl_module.PROFILES_FILE = tmp_path / "shutdown_profiles.json"
        try:
            from core.feedback_loop import FeedbackLoop
            fl = FeedbackLoop(self._make_engine())
            fl.register_agent("Athena-Researcher", "researcher")
            fl.profiles["Athena-Researcher"].total_episodes = 5
            await fl.shutdown()
            assert fl_module.PROFILES_FILE.exists()
            data = json.loads(fl_module.PROFILES_FILE.read_text())
            assert "Athena-Researcher" in data
            assert data["Athena-Researcher"]["total_episodes"] == 5
        finally:
            fl_module.PROFILES_FILE = original


# ─── ReActPlanner + FeedbackLoop intégration ─────────────────────────────────

class TestReActPlannerFeedback:

    def _make_engine(self):
        engine = MagicMock()
        engine.brain = {"llm": AsyncMock()}
        engine.brain["llm"].generate = AsyncMock(return_value="résultat simulé")
        engine.swarm = MagicMock()
        engine.swarm.agents = {}
        engine.knowledge_graph = None

        from core.feedback_loop import FeedbackLoop
        engine.feedback_loop = FeedbackLoop(engine)
        engine.feedback_loop.register_agent("Athena-Analyst", "analyst")
        engine.feedback_loop.register_agent("Athena-Critic", "critic")
        return engine

    @pytest.mark.asyncio
    async def test_subtask_notifies_feedback_loop(self):
        from core.react_planner import ReActPlanner, SubTask
        engine = self._make_engine()
        planner = ReActPlanner(engine)

        subtask = SubTask("t0", "analyse ce texte", "analyst")
        await planner._run_subtask(subtask, {})

        profile = engine.feedback_loop.profiles.get("Athena-Analyst")
        assert profile is not None
        assert profile.total_episodes >= 1

    @pytest.mark.asyncio
    async def test_failed_subtask_updates_profile(self):
        from core.react_planner import ReActPlanner, SubTask
        engine = self._make_engine()
        engine.brain["llm"].generate = AsyncMock(side_effect=Exception("LLM down"))
        planner = ReActPlanner(engine)

        subtask = SubTask("t0", "analyse", "analyst")
        await planner._run_subtask(subtask, {})

        assert subtask.status == "failed"
        profile = engine.feedback_loop.profiles.get("Athena-Analyst")
        assert profile is not None
        assert profile.consecutive_failures >= 1


# ─── WebSocket ───────────────────────────────────────────────────────────────

class TestWebSocket:

    @pytest.mark.asyncio
    async def test_websocket_protocol_messages(self):
        """Vérifie que le WebSocket envoie les bons types de messages."""
        from fastapi.testclient import TestClient
        from interfaces.api import app, set_engine

        engine = MagicMock()
        engine.brain = {"llm": AsyncMock(), "memory": AsyncMock()}
        engine.brain["llm"].generate = AsyncMock(return_value="réponse finale")
        engine.brain["memory"].store = AsyncMock()
        engine.episodic_memory = AsyncMock()
        engine.episodic_memory.record = AsyncMock()

        from core.react_planner import ReActPlanner, ExecutionPlan, SubTask
        mock_planner = MagicMock()
        mock_planner._decompose = AsyncMock(return_value=[
            SubTask("t0", "analyse", "analyst"),
            SubTask("tc", "critique", "critic", depends_on=["t0"])
        ])
        mock_planner._run_subtask = AsyncMock(side_effect=lambda t, c: setattr(t, 'status', 'done') or setattr(t, 'result', 'ok') or setattr(t, 'confidence', 0.8))
        mock_planner._synthesize = AsyncMock(return_value="synthèse finale")
        mock_planner._get_agent_name = MagicMock(return_value=None)
        engine.planner = mock_planner
        engine.feedback_loop = None

        set_engine(engine)

        with TestClient(app) as client:
            with client.websocket_connect("/ws/query") as ws:
                ws.send_json({"query": "test query"})
                messages = []
                for _ in range(10):
                    try:
                        msg = ws.receive_json()
                        messages.append(msg)
                        if msg.get("type") in ("done", "error"):
                            break
                    except Exception:
                        break

        types = [m["type"] for m in messages]
        assert "start" in types
        assert "plan" in types
