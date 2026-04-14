#!/usr/bin/env python3
"""
domains/education.py — Domaine Éducation
Tuteur IA personnalisé: adaptation au niveau, génération d'exercices,
évaluation, plans d'apprentissage, correction détaillée.
"""

import time
from typing import Dict, Any


LEVELS = {
    "primaire": "enfant de 6-11 ans, vocabulaire simple, exemples concrets",
    "collège":  "adolescent de 11-15 ans, notions intermédiaires",
    "lycée":    "adolescent de 15-18 ans, niveau baccalauréat",
    "université": "étudiant universitaire, niveau académique avancé",
    "professionnel": "adulte en formation professionnelle, approche pratique",
    "expert":   "expert du domaine, niveau recherche et développement",
}

SUBJECTS = {
    "maths": "mathématiques", "physique": "physique-chimie",
    "histoire": "histoire-géographie", "bio": "biologie",
    "info": "informatique", "langue": "langues",
    "philo": "philosophie", "éco": "économie",
}


class EducationDomain:
    """Module tuteur IA pour ATHÉNA."""

    def __init__(self, engine):
        self.engine = engine
        self._student_profiles: Dict[str, Dict] = {}

    async def analyze(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()

        if any(k in query_lower for k in ["explique", "c'est quoi", "définition", "comprendre", "qu'est-ce"]):
            result = await self._explain(query)
        elif any(k in query_lower for k in ["exercice", "problème", "entraînement", "quiz", "qcm"]):
            result = await self._generate_exercise(query)
        elif any(k in query_lower for k in ["corrige", "correction", "erreur", "vérifie ma réponse"]):
            result = await self._correct(query)
        elif any(k in query_lower for k in ["plan", "programme", "apprendre", "formation", "parcours"]):
            result = await self._learning_plan(query)
        elif any(k in query_lower for k in ["résume", "synthèse", "fiche", "mémo"]):
            result = await self._summarize(query)
        else:
            result = await self._tutor(query)

        return {"domain": "education", "analysis": result, "timestamp": time.time()}

    def _detect_level(self, query: str) -> str:
        q = query.lower()
        for level, desc in LEVELS.items():
            if level in q:
                return desc
        return LEVELS["lycée"]  # niveau par défaut

    async def _explain(self, query: str) -> str:
        level = self._detect_level(query)
        prompt = (
            f"Tu es un tuteur pédagogue expert. Ton élève est: {level}.\n"
            "Explique le concept demandé avec:\n"
            "1. Définition claire et simple\n"
            "2. Analogie ou exemple concret du quotidien\n"
            "3. Les points clés à retenir (3 max)\n"
            "4. Une question pour vérifier la compréhension\n\n"
            f"Concept à expliquer: {query}"
        )
        return await self._llm(prompt)

    async def _generate_exercise(self, query: str) -> str:
        level = self._detect_level(query)
        prompt = (
            f"Tu es un professeur créant des exercices pour: {level}.\n"
            "Génère 3 exercices progressifs (facile → difficile) avec:\n"
            "- Énoncé clair\n"
            "- Données nécessaires\n"
            "- Indication de la méthode à utiliser\n"
            "- Réponse complète avec étapes détaillées\n\n"
            f"Sujet: {query}"
        )
        return await self._llm(prompt)

    async def _correct(self, query: str) -> str:
        prompt = (
            "Tu es un correcteur bienveillant et pédagogue. Analyse cette réponse:\n"
            "1. Ce qui est correct ✅\n"
            "2. Les erreurs identifiées ❌ avec explication du pourquoi\n"
            "3. La correction complète et détaillée\n"
            "4. Conseil pour éviter cette erreur à l'avenir\n"
            "5. Note estimée et encouragement\n\n"
            f"Réponse à corriger: {query}"
        )
        return await self._llm(prompt)

    async def _learning_plan(self, query: str) -> str:
        level = self._detect_level(query)
        prompt = (
            f"Tu es un expert en ingénierie pédagogique pour: {level}.\n"
            "Crée un plan d'apprentissage structuré:\n"
            "1. Prérequis nécessaires\n"
            "2. Objectifs pédagogiques (SMART)\n"
            "3. Programme semaine par semaine (4 semaines)\n"
            "4. Ressources recommandées (livres, vidéos, exercices)\n"
            "5. Méthode d'évaluation des progrès\n\n"
            f"Objectif d'apprentissage: {query}"
        )
        return await self._llm(prompt)

    async def _summarize(self, query: str) -> str:
        level = self._detect_level(query)
        prompt = (
            f"Tu es un expert créant des fiches de révision pour: {level}.\n"
            "Crée une fiche synthétique avec:\n"
            "- Titre et mots-clés\n"
            "- Définitions essentielles\n"
            "- Formules ou règles importantes\n"
            "- Schéma ou tableau récapitulatif\n"
            "- 3 points à retenir absolument\n\n"
            f"Sujet: {query}"
        )
        return await self._llm(prompt)

    async def _tutor(self, query: str) -> str:
        level = self._detect_level(query)
        prompt = (
            f"Tu es ATHÉNA, tuteur IA bienveillant et expert pour: {level}.\n"
            "Réponds de façon pédagogique, encourageante et adaptée au niveau.\n\n"
            f"Question: {query}"
        )
        return await self._llm(prompt)

    async def _llm(self, prompt: str) -> str:
        if self.engine.brain and self.engine.brain.get("llm"):
            return await self.engine.brain["llm"].generate(prompt)
        return "[LLM non disponible]"
