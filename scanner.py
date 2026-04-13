#!/usr/bin/env python3
"""
vision/scanner.py - Scan de fichiers et flux locaux
Scan asynchrone de répertoires et analyse de contenus.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class FileScanner:
    """Scanner de fichiers avec support image/texte."""
    
    def __init__(self):
        self.scanned_cache: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialisation du scanner."""
        logger.info("FileScanner initialisé.")
    
    async def scan_directory(self, path: str, extensions: List[str] = None) -> Dict[str, Any]:
        """Scan asynchrone d'un répertoire."""
        if extensions is None:
            extensions = ['.txt', '.py', '.md', '.json']
        
        path_obj = Path(path)
        files_data = {}
        
        for file_path in path_obj.rglob('*'):
            if file_path.is_file() and any(file_path.suffix.lower() in extensions):
                content = await self._read_file_async(file_path)
                files_data[str(file_path)] = {
                    'size': file_path.stat().st_size,
                    'content_preview': content[:500] + '...' if len(content) > 500 else content
                }
        
        self.scanned_cache[path] = files_data
        return files_data
    
    async def _read_file_async(self, file_path: Path) -> str:
        """Lecture asynchrone d'un fichier."""
        loop = asyncio.get_event_loop()
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return await loop.run_in_executor(None, f.read)
    
    async def get_context(self) -> str:
        """Retourne le contexte scanné récent."""
        if self.scanned_cache:
            recent = list(self.scanned_cache.values())[-1]
            return f"Scanned {len(recent)} files: {list(recent.keys())}"
        return "Aucun scan récent."
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyse basique d'image avec OpenCV."""
        loop = asyncio.get_event_loop()
        def _cv_analysis():
            img = cv2.imread(image_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return {
                    'dimensions': img.shape[:2],
                    'mean_intensity': np.mean(gray)
                }
            return {'error': 'Image non chargée'}
        return await loop.run_in_executor(None, _cv_analysis)

