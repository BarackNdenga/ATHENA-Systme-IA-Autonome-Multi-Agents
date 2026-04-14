#!/usr/bin/env python3
"""
domains/enterprise.py — Domaine Entreprise
Assistant IA interne: analyse de données business, rapports,
aide à la décision, gestion de projets, RH, finance.
"""

import time
from typing import Dict, Any


class EnterpriseDomain:
    """Module assistant entreprise pour ATHÉNA."""

    def __init__(self, engine):
        self.engine = engine

    async def analyze(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()

        if any(k in query_lower for k in ["finance", "budget", "coût", "revenu", "p&l", "bilan"]):
            result = await self._financial_analysis(query)
        elif any(k in query_lower for k in ["projet", "planning", "gantt", "délai", "milestone"]):
            result = await self._project_management(query)
        elif any(k in query_lower for k in ["rh", "recrutement", "employé", "talent", "formation"]):
            result = await self._hr_analysis(query)
        elif any(k in query_lower for k in ["marché", "concurrence", "stratégie", "swot", "pestel"]):
            result = await self._strategic_analysis(query)
        elif any(k in query_lower for k in ["rapport", "dashboard", "kpi", "métrique", "performance"]):
            result = await self._business_report(query)
        elif any(k in query_lower for k in ["contrat", "juridique", "légal", "clause", "conformité"]):
            result = await self._legal_review(query)
        else:
            result = await self._general_enterprise(query)

        return {"domain": "enterprise", "analysis": result, "timestamp": time.time()}

    async def _financial_analysis(self, query: str) -> str:
        prompt = (
            "Tu es un analyste financier senior. Fournis une analyse structurée:\n"
            "1. Indicateurs clés (ROI, EBITDA, marge nette, liquidité)\n"
            "2. Tendances et variations significatives\n"
            "3. Points de vigilance et risques financiers\n"
            "4. Recommandations d'optimisation\n"
            "5. Benchmarks sectoriels si pertinent\n\n"
            f"Données financières: {query}"
        )
        return await self._llm(prompt)

    async def _project_management(self, query: str) -> str:
        prompt = (
            "Tu es un chef de projet certifié PMP/Prince2. Analyse et fournis:\n"
            "1. Évaluation de l'avancement et des risques\n"
            "2. Identification des dépendances critiques (chemin critique)\n"
            "3. Ressources nécessaires et goulots d'étranglement\n"
            "4. Plan de mitigation des risques\n"
            "5. Recommandations pour respecter les délais et le budget\n\n"
            f"Projet: {query}"
        )
        return await self._llm(prompt)

    async def _hr_analysis(self, query: str) -> str:
        prompt = (
            "Tu es un DRH expert. Analyse la situation RH et fournis:\n"
            "1. Évaluation des besoins en compétences\n"
            "2. Stratégie de recrutement ou de développement interne\n"
            "3. Indicateurs RH clés (turnover, engagement, absentéisme)\n"
            "4. Plan d'action recommandé\n"
            "5. Conformité légale et réglementaire\n\n"
            f"Contexte RH: {query}"
        )
        return await self._llm(prompt)

    async def _strategic_analysis(self, query: str) -> str:
        prompt = (
            "Tu es un consultant en stratégie d'entreprise. Réalise une analyse:\n"
            "1. SWOT détaillé (Forces, Faiblesses, Opportunités, Menaces)\n"
            "2. Analyse concurrentielle (5 forces de Porter si pertinent)\n"
            "3. Positionnement stratégique recommandé\n"
            "4. Options stratégiques avec avantages/inconvénients\n"
            "5. Plan d'action prioritaire à 3-12 mois\n\n"
            f"Contexte: {query}"
        )
        return await self._llm(prompt)

    async def _business_report(self, query: str) -> str:
        prompt = (
            "Génère un rapport business exécutif structuré:\n"
            "- Résumé exécutif (3 lignes max)\n"
            "- KPIs principaux avec statut (vert/orange/rouge)\n"
            "- Faits marquants de la période\n"
            "- Points d'attention et risques\n"
            "- Décisions requises et recommandations\n\n"
            f"Données: {query}"
        )
        return await self._llm(prompt)

    async def _legal_review(self, query: str) -> str:
        prompt = (
            "Tu es un juriste d'entreprise. Analyse ce document/situation:\n"
            "1. Points clés et clauses importantes\n"
            "2. Risques juridiques identifiés\n"
            "3. Points de négociation recommandés\n"
            "4. Conformité réglementaire (RGPD, droit du travail, etc.)\n"
            "⚠ Rappelle qu'une validation par un avocat est recommandée.\n\n"
            f"Document/situation: {query}"
        )
        return await self._llm(prompt)

    async def _general_enterprise(self, query: str) -> str:
        prompt = (
            "Tu es un assistant IA d'entreprise polyvalent et expert. "
            f"Réponds de façon professionnelle, structurée et actionnable à: {query}"
        )
        return await self._llm(prompt)

    async def _llm(self, prompt: str) -> str:
        if self.engine.brain and self.engine.brain.get("llm"):
            return await self.engine.brain["llm"].generate(prompt)
        return "[LLM non disponible]"
