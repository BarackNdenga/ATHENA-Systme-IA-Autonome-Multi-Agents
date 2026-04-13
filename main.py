#!/usr/bin/env python3
"""
main.py - Point d'entrée ATHÉNA
Lance CLI et API en parallèle de manière compatible asyncio.
"""

import asyncio
import os
import sys
import threading
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from core.engine import AthenaEngine
from interfaces.cli import cli_app


def run_api(engine: AthenaEngine):
    """Lance l'API FastAPI dans un thread séparé (uvicorn est bloquant)."""
    import uvicorn
    from interfaces.api import app, set_engine
    set_engine(engine)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


async def main():
    load_dotenv()

    engine = AthenaEngine()
    await engine.initialize()

    print("\U0001f989 ATHÉNA s'éveille...")

    # API dans un thread daemon (non bloquant)
    api_thread = threading.Thread(target=run_api, args=(engine,), daemon=True)
    api_thread.start()

    # CLI dans la boucle asyncio principale
    await cli_app(engine)

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

