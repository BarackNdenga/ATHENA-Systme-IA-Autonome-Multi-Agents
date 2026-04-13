#!/usr/bin/env python3
"""
brain/llm_handler.py - Connexion aux modèles LLM (Ollama/GPT)
Gestion asynchrone des appels LLM avec fallback.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

logger = logging.getLogger(__name__)

class LLMHandler:
    """Gestionnaire de modèles LLM avec support Ollama et OpenAI."""

    def __init__(self):
        self.ollama = None
        self.openai = None
        self.active_model = None
        self.chain = None
        self._episodic_context: str = ""
        self._graph_context: str = ""

    def set_context(self, episodic: str = "", graph: str = ""):
        """Injecte le contexte épisodique et graphe avant génération."""
        self._episodic_context = episodic
        self._graph_context = graph
    
    async def initialize(self):
        """Initialise les clients LLM."""
        # Ollama (local)
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            self.ollama = Ollama(
                model="llama3.1",  # ou autre modèle local
                base_url=ollama_url
            )
            self.active_model = self.ollama
            logger.info("Ollama LLM initialisé.")
        except Exception as e:
            logger.warning(f"Ollama non disponible: {e}")
        
        # OpenAI fallback
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai_key,
                temperature=0.1
            )
            if not self.active_model:
                self.active_model = self.openai
                logger.info("OpenAI LLM initialisé (fallback).")
        
        if self.active_model:
            prompt = ChatPromptTemplate.from_template(
                """Tu es ATHÉNA, un système IA autonome qui apprend de ses expériences.

Expériences passées similaires:
{episodic}

Connaissances structurées:
{graph}

Contexte actuel: {context}
Mémoire sémantique: {memory}
Question: {query}

Réponse concise, actionnable et enrichie par tes expériences passées:"""
            )
            self.chain = prompt | self.active_model | StrOutputParser()
    
    async def generate(self, query: str, context: str = "", memory=None) -> str:
        """Génère une réponse enrichie par le contexte épisodique et le graphe."""
        if not self.chain:
            return "Erreur: Aucun modèle LLM disponible."
        try:
            memory_str = await memory.retrieve_relevant(query) if memory else ""
            result = await self.chain.ainvoke({
                "context": context,
                "memory": memory_str,
                "query": query,
                "episodic": self._episodic_context or "Aucune expérience similaire.",
                "graph": self._graph_context or "Aucun concept connu."
            })
            return result
        except Exception as e:
            logger.error(f"Erreur génération LLM: {e}")
            return "Erreur lors de la génération."

