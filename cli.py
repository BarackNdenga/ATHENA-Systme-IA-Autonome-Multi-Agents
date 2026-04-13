#!/usr/bin/env python3
"""
interfaces/cli.py - Interface CLI interactive avec Rich
"""

import asyncio
import logging
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

from core.engine import AthenaEngine

console = Console()
logging.basicConfig(level=logging.WARNING)


async def cli_app(engine: Optional[AthenaEngine] = None):
    """Lance l'interface CLI d'ATHÉNA (coroutine async)."""
    if engine is None:
        engine = AthenaEngine()
        await engine.initialize()

    console.print(Panel.fit(
        Text("🦉 ATHÉNA - Système Omniscient Actif", style="bold cyan"),
        title="Statut: [green]PRÊT[/green]"
    ))
    console.print("[dim]Commandes: quit | exit | stats[/dim]\n")

    loop = asyncio.get_event_loop()

    while True:
        try:
            query = await loop.run_in_executor(None, lambda: Prompt.ask("[bold blue]ATHÉNA>[/bold blue]"))

            if query.lower() in ['quit', 'exit', 'bye']:
                console.print("\n👋 ATHÉNA s'endort...")
                break

            if query.lower() == 'stats':
                if hasattr(engine, 'swarm'):
                    stats = engine.swarm.get_swarm_stats()
                    for s in stats:
                        console.print(f"  [cyan]{s['agent']}[/cyan] | épisodes: {s['episodes']} | confiance moy: {s['avg_confidence']}")
                continue

            console.print("[bold yellow]🤖 Swarm en cours...[/bold yellow]")
            response = await engine.process_query(query)
            console.print(Panel(response, title="[bold green]Réponse ATHÉNA[/bold green]"))

        except KeyboardInterrupt:
            console.print("\n👋 ATHÉNA s'endort...")
            break
        except Exception as e:
            console.print(f"[red]Erreur: {e}[/red]")

