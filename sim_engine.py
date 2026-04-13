#!/usr/bin/env python3
"""
simulation/sim_engine.py — Simulation ATHÉNA en terminal (Rich)
Lance une simulation visuelle des agents sans LLM requis.

Usage:
    python simulation/sim_engine.py
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.columns import Columns
from rich.text import Text
from rich import box

console = Console()

AGENTS = [
    {"name": "Athena-Analyst",    "role": "analyst",    "emoji": "🔍"},
    {"name": "Athena-Researcher", "role": "researcher", "emoji": "🌐"},
    {"name": "Athena-Coder",      "role": "coder",      "emoji": "💻"},
    {"name": "Athena-Critic",     "role": "critic",     "emoji": "⚖️ "},
]

QUERIES = [
    "Analyse l'architecture microservices d'un système bancaire",
    "Génère un script Python pour parser du JSON imbriqué",
    "Recherche les meilleures pratiques Zero Trust 2025",
    "Implémente un algorithme de tri rapide optimisé",
    "Analyse les vulnérabilités d'une API REST publique",
    "Génère un pipeline CI/CD pour un projet FastAPI",
]


@dataclass
class AgentProfile:
    name: str
    role: str
    emoji: str
    confidence_history: List[float] = field(default_factory=list)
    episodes: int = 0
    context_boost: int = 0
    status: str = "idle"

    @property
    def avg_confidence(self) -> float:
        if not self.confidence_history:
            return 0.5
        return round(sum(self.confidence_history) / len(self.confidence_history), 2)

    @property
    def trend(self) -> str:
        if len(self.confidence_history) < 4:
            return "stable"
        recent = self.confidence_history[-4:]
        if recent[-1] > recent[0] + 0.08:
            return "↑ improving"
        if recent[-1] < recent[0] - 0.08:
            return "↓ declining"
        return "→ stable"

    @property
    def success_rate(self) -> float:
        if not self.confidence_history:
            return 1.0
        return round(sum(1 for c in self.confidence_history if c >= 0.55) / len(self.confidence_history), 2)


def make_dashboard(profiles: Dict[str, AgentProfile], query: str, cycle: int, log: List[str]) -> Table:
    """Construit le tableau de bord Rich."""
    grid = Table.grid(expand=True)
    grid.add_column()

    # Header
    header = Panel(
        Text(f"  ATHÉNA — Simulation Multi-Agents  |  Cycle #{cycle}  |  Query: {query[:55]}…",
             style="bold cyan", justify="center"),
        style="cyan", box=box.DOUBLE
    )

    # Table agents
    tbl = Table(
        "Agent", "Rôle", "Statut", "Confiance", "Succès", "Boost ctx", "Tendance", "Épisodes",
        box=box.SIMPLE_HEAVY, style="white", header_style="bold magenta",
        expand=True
    )
    status_colors = {"idle": "dim", "running": "yellow", "done": "green", "failed": "red"}
    for p in profiles.values():
        conf_bar = "█" * int(p.avg_confidence * 10) + "░" * (10 - int(p.avg_confidence * 10))
        conf_color = "green" if p.avg_confidence >= 0.7 else "yellow" if p.avg_confidence >= 0.5 else "red"
        tbl.add_row(
            f"{p.emoji} {p.name}",
            p.role,
            Text(p.status, style=status_colors.get(p.status, "white")),
            Text(f"{conf_bar} {p.avg_confidence:.0%}", style=conf_color),
            f"{p.success_rate:.0%}",
            f"+{p.context_boost}" if p.context_boost else "0",
            p.trend,
            str(p.episodes),
        )

    # Log panel
    log_text = "\n".join(log[-8:]) if log else "En attente…"
    log_panel = Panel(log_text, title="[bold yellow]📋 Journal[/]", style="dim white", box=box.ROUNDED)

    grid.add_row(header)
    grid.add_row(tbl)
    grid.add_row(log_panel)
    return grid


async def simulate_agent(profile: AgentProfile, query: str, log: List[str], delay: float = 0.0):
    """Simule l'exécution d'un agent avec boucle ReAct."""
    await asyncio.sleep(delay)
    profile.status = "running"

    # Thought
    await asyncio.sleep(random.uniform(0.3, 0.7))
    log.append(f"[cyan]{profile.emoji} {profile.name}[/] → [italic]Thought:[/] analyse de la tâche…")

    # Action
    await asyncio.sleep(random.uniform(0.5, 1.2))
    log.append(f"[cyan]{profile.emoji} {profile.name}[/] → [italic]Action:[/] exécution ({profile.role})…")

    # Observation + confidence simulée avec légère amélioration progressive
    base = 0.45 + (profile.episodes * 0.015)
    noise = random.uniform(-0.15, 0.20)
    confidence = round(min(max(base + noise, 0.1), 0.99), 2)
    profile.confidence_history.append(confidence)
    profile.episodes += 1

    # Feedback loop: boost si échec
    if confidence < 0.55:
        profile.context_boost = min(profile.context_boost + 1, 5)
        log.append(f"[red]⚠ FeedbackLoop:[/] {profile.name} → context_boost +1 ({profile.context_boost})")
    elif confidence >= 0.80 and profile.context_boost > 0:
        profile.context_boost -= 1
        log.append(f"[green]✓ FeedbackLoop:[/] {profile.name} → context_boost -1 ({profile.context_boost})")

    status_icon = "✅" if confidence >= 0.55 else "❌"
    log.append(
        f"[cyan]{profile.emoji} {profile.name}[/] → [italic]Observation:[/] "
        f"confiance {confidence:.0%} {status_icon}"
    )
    profile.status = "done" if confidence >= 0.55 else "failed"


async def run_simulation():
    profiles = {a["name"]: AgentProfile(**a) for a in AGENTS}
    log: List[str] = []
    cycle = 0

    console.print(Panel(
        "[bold cyan]ATHÉNA — Simulation Terminal[/]\n"
        "[dim]Appuie sur Ctrl+C pour arrêter[/]",
        box=box.DOUBLE, style="cyan"
    ))
    await asyncio.sleep(1)

    with Live(console=console, refresh_per_second=8, screen=True) as live:
        while True:
            cycle += 1
            query = random.choice(QUERIES)
            log.append(f"\n[bold white]━━ Cycle #{cycle} ━━[/] [yellow]{query[:60]}[/]")

            # Reset statuts
            for p in profiles.values():
                p.status = "idle"

            live.update(make_dashboard(profiles, query, cycle, log))
            await asyncio.sleep(0.3)

            # Analyst en premier
            analyst = profiles["Athena-Analyst"]
            await simulate_agent(analyst, query, log)
            live.update(make_dashboard(profiles, query, cycle, log))

            # Researcher + Coder en parallèle
            await asyncio.gather(
                simulate_agent(profiles["Athena-Researcher"], query, log, delay=0.0),
                simulate_agent(profiles["Athena-Coder"], query, log, delay=0.2),
            )
            live.update(make_dashboard(profiles, query, cycle, log))

            # Critic en dernier
            await simulate_agent(profiles["Athena-Critic"], query, log, delay=0.1)
            live.update(make_dashboard(profiles, query, cycle, log))

            log.append(f"[bold green]✔ Cycle #{cycle} terminé[/] — durée simulée")
            live.update(make_dashboard(profiles, query, cycle, log))

            await asyncio.sleep(2.5)


if __name__ == "__main__":
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        console.print("\n[bold cyan]ATHÉNA simulation arrêtée.[/]")
