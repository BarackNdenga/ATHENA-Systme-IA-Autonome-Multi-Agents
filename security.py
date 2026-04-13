#!/usr/bin/env python3
"""
core/security.py - Auto-surveillance et intégrité du système.
Surveille les fichiers critiques et détecte les modifications suspectes.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Dict, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from core.zero_trust import ZeroTrustSentinel
from core.quantum_crypto import QuantumSecureChannel


logger = logging.getLogger(__name__)

CRITICAL_FILES = [
    'main.py',
    'core/engine.py',
    'core/security.py',
    'brain/llm_handler.py',
    'requirements.txt'
]

class IntegrityChecker:
    """Vérifie l'intégrité des fichiers critiques via hash."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.file_hashes: Dict[str, str] = {}
    
    def compute_hashes(self):
        """Calcule les hashes des fichiers critiques."""
        self.file_hashes.clear()
        for rel_path in CRITICAL_FILES:
            file_path = self.project_root / rel_path
            if file_path.exists():
                hash_md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
                self.file_hashes[rel_path] = hash_md5
                logger.info(f"Hash calculé pour {rel_path}: {hash_md5[:8]}...")
    
    def is_integrity_ok(self) -> bool:
        """Vérifie si tous les hashes correspondent."""
        self.compute_hashes()
        for rel_path, expected_hash in self.file_hashes.items():
            file_path = self.project_root / rel_path
            if not file_path.exists():
                logger.warning(f"Fichier manquant: {rel_path}")
                return False
            current_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
            if current_hash != expected_hash:
                logger.error(f"Intégrité compromise: {rel_path}")
                return False
        return True

class SecurityEventHandler(FileSystemEventHandler):
    """Handler pour les événements de surveillance."""
    
    def __init__(self, checker: IntegrityChecker):
        self.checker = checker
    
    def on_modified(self, event):
        if any(critical in event.src_path for critical in CRITICAL_FILES):
            logger.warning(f"Modification détectée: {event.src_path}")
            if not self.checker.is_integrity_ok():
                logger.critical("ALERTE SÉCURITÉ: Intégrité compromise!")

class SecurityMonitor:
    """Module de surveillance de sécurité."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.checker = IntegrityChecker(self.project_root)
        self.observer = None
    
    async def start_monitoring(self):
        """Démarre surveillance ULTRA-SOPHISTIQUÉE."""
        self.checker.compute_hashes()
        if not self.checker.is_integrity_ok():
            logger.error("Échec initial intégrité!")
            return
        
        # Zero Trust + Quantum Crypto
        self.zero_trust = ZeroTrustSentinel()
        await self.zero_trust.activate_anomaly_detection()
        self.quantum_channel = QuantumSecureChannel()
        
        event_handler = SecurityEventHandler(self.checker)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.project_root), recursive=True)
        self.observer.start()
        logger.info("🔒 Zero Trust + Quantum Security ACTIVÉ")
    
    async def shutdown(self):
        """Arrête la surveillance."""
        if self.observer:
            self.observer.stop()
            self.observer.join()

