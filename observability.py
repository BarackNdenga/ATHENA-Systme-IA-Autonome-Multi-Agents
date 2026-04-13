#!/usr/bin/env python3
"""
core/observability.py - Observabilité complète du système ATHÉNA
Trace chaque décision, mesure les latences, expose les métriques.
Sans observabilité, un système autonome est une boîte noire.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACE_FILE = Path("./traces.jsonl")


@dataclass
class Span:
    """Une unité de trace: une opération avec début, fin et métadonnées."""
    span_id: str
    operation: str
    component: str          # engine | agent | llm | memory | vision
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"      # ok | error | timeout
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def finish(self, status: str = "ok", error: str = None):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Collecte des métriques agrégées en mémoire (fenêtre glissante)."""

    def __init__(self, window: int = 100):
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._counts: Dict[str, int] = defaultdict(int)
        self._errors: Dict[str, int] = defaultdict(int)

    def record(self, operation: str, duration_ms: float, success: bool):
        self._latencies[operation].append(duration_ms)
        self._counts[operation] += 1
        if not success:
            self._errors[operation] += 1

    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        for op, latencies in self._latencies.items():
            if not latencies:
                continue
            lat_list = list(latencies)
            lat_list.sort()
            p50 = lat_list[len(lat_list) // 2]
            p95 = lat_list[int(len(lat_list) * 0.95)]
            summary[op] = {
                "count": self._counts[op],
                "errors": self._errors[op],
                "error_rate": round(self._errors[op] / max(self._counts[op], 1), 3),
                "avg_ms": round(sum(lat_list) / len(lat_list), 1),
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "min_ms": round(lat_list[0], 1),
                "max_ms": round(lat_list[-1], 1),
            }
        return summary


class ObservabilityHub:
    """
    Hub central d'observabilité: traces + métriques + dashboard terminal.
    Toutes les décisions importantes du système passent par ici.
    """

    def __init__(self, trace_file: Path = TRACE_FILE):
        self.trace_file = trace_file
        self.metrics = MetricsCollector()
        self._recent_spans: deque = deque(maxlen=200)
        self._span_counter = 0

    def start_span(self, operation: str, component: str, metadata: Dict[str, Any] = None) -> Span:
        """Démarre une nouvelle trace."""
        self._span_counter += 1
        span = Span(
            span_id=f"span_{self._span_counter}_{int(time.time() * 1000)}",
            operation=operation,
            component=component,
            start_time=time.time(),
            metadata=metadata or {}
        )
        return span

    async def finish_span(self, span: Span, status: str = "ok", error: str = None):
        """Termine une trace et persiste."""
        span.finish(status=status, error=error)
        self.metrics.record(span.operation, span.duration_ms, status == "ok")
        self._recent_spans.append(span)
        await self._persist_span(span)
        if status == "error":
            logger.warning(f"[TRACE] {span.component}/{span.operation} ERREUR {span.duration_ms:.0f}ms: {error}")
        else:
            logger.debug(f"[TRACE] {span.component}/{span.operation} {span.duration_ms:.0f}ms")

    async def _persist_span(self, span: Span):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_span, span)

    def _write_span(self, span: Span):
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")

    def get_dashboard(self) -> str:
        """Génère un rapport texte lisible pour le terminal."""
        lines = ["=" * 55, "  📊 ATHÉNA - Tableau de bord observabilité", "=" * 55]
        summary = self.metrics.get_summary()
        if not summary:
            lines.append("  Aucune métrique disponible.")
        else:
            lines.append(f"  {'Opération':<25} {'Appels':>6} {'Err%':>6} {'Moy ms':>8} {'P95 ms':>8}")
            lines.append("  " + "-" * 53)
            for op, m in sorted(summary.items()):
                err_pct = f"{m['error_rate']*100:.0f}%"
                lines.append(
                    f"  {op:<25} {m['count']:>6} {err_pct:>6} {m['avg_ms']:>8.1f} {m['p95_ms']:>8.1f}"
                )
        lines.append("=" * 55)
        return "\n".join(lines)

    def get_recent_errors(self, n: int = 10) -> List[Dict[str, Any]]:
        """Retourne les N dernières erreurs tracées."""
        errors = [s for s in self._recent_spans if s.status == "error"]
        return [s.to_dict() for s in list(errors)[-n:]]

    def get_metrics_json(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.get_summary(),
            "total_spans": self._span_counter,
            "recent_errors": len(self.get_recent_errors())
        }
