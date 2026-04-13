#!/usr/bin/env python3
"""
brain/memory.py - Mémoire vectorielle sémantique (ChromaDB + sentence-transformers)
Stockage et retrieval vectoriel asynchrone avec vrais embeddings.
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class VectorMemory:
    """Mémoire vectorielle persistante avec embeddings sémantiques réels."""

    def __init__(self):
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    async def initialize(self):
        """Initialise la collection ChromaDB avec embeddings sémantiques."""
        self.collection = self.client.get_or_create_collection(
            name="athena_memory",
            embedding_function=self._ef,
            metadata={"description": "Mémoire sémantique d'ATHÉNA"}
        )
        logger.info("VectorMemory initialisée avec sentence-transformers.")

    async def store(self, query: str, response: str, metadata: Dict[str, Any] = None):
        """Stocke une paire query-réponse avec embedding sémantique réel."""
        doc_id = hashlib.sha256(f"{query}{time.time()}".encode()).hexdigest()[:16]
        meta = {"query": query, "type": "semantic", "timestamp": time.time()}
        if metadata:
            meta.update(metadata)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.collection.add(
            documents=[response],
            metadatas=[meta],
            ids=[doc_id]
        ))
        logger.debug(f"Stocké en mémoire sémantique: {query[:50]}...")

    async def retrieve_relevant(self, query: str, n_results: int = 3) -> str:
        """Récupère les souvenirs sémantiquement proches via embeddings."""
        try:
            count = self.collection.count()
            if count == 0:
                return ""
            n = min(n_results, count)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: self.collection.query(
                query_texts=[query],
                n_results=n
            ))
            memories = results['documents'][0] if results['documents'] else []
            return "\n".join(memories)
        except Exception as e:
            logger.warning(f"Retrieval mémoire échoué: {e}")
            return ""

    async def shutdown(self):
        """Fermeture propre."""
        self.client.close()

