#!/usr/bin/env python3
"""
domains/medical.py — Domaine Médical
Analyse de dossiers patients, aide au diagnostic différentiel,
détection d'interactions médicamenteuses, résumé de littérature médicale.

⚠ AVERTISSEMENT: Outil d'aide à la décision uniquement.
  Ne remplace pas un professionnel de santé qualifié.
"""

import re
from typing import Dict, Any, List


DISCLAIMER = (
    "⚠ AIDE À LA DÉCISION UNIQUEMENT — Ne remplace pas un médecin qualifié. "
    "Toute décision clinique doit être validée par un professionnel de santé."
)

# Symptômes critiques nécessitant une alerte immédiate
CRITICAL_SYMPTOMS = [
    "douleur thoracique", "chest pain", "dyspnée sévère", "perte de conscience",
    "avc", "stroke", "hémorragie", "sepsis", "choc", "détresse respiratoire",
    "infarctus", "arrêt cardiaque", "convulsions", "coma",
]

# Interactions médicamenteuses connues (simplifiées)
DRUG_INTERACTIONS = {
    ("warfarine", "aspirine"):    "Risque hémorragique majeur — surveillance INR obligatoire",
    ("ibuprofen", "lithium"):     "Augmentation toxicité lithium — éviter ou surveiller",
    ("metformine", "alcool"):     "Risque d'acidose lactique — contre-indiqué",
    ("ssri", "tramadol"):         "Risque syndrome sérotoninergique — contre-indiqué",
    ("statine", "clarithromycine"): "Risque rhabdomyolyse — réduire dose statine",
}


class MedicalDomain:
    """Module d'analyse médicale pour ATHÉNA."""

    def __init__(self, engine):
        self.engine = engine

    async def analyze(self, query: str) -> Dict[str, Any]:
        """Point d'entrée principal — route vers la bonne analyse."""
        query_lower = query.lower()

        # Détection alerte critique
        critical = self._check_critical(query_lower)

        # Routing selon le type de requête
        if any(k in query_lower for k in ["interaction", "médicament", "drug", "posologie"]):
            result = await self._drug_interaction(query)
        elif any(k in query_lower for k in ["diagnostic", "symptôme", "symptom", "douleur", "fièvre"]):
            result = await self._differential_diagnosis(query)
        elif any(k in query_lower for k in ["dossier", "patient", "antécédent", "anamnèse"]):
            result = await self._analyze_record(query)
        elif any(k in query_lower for k in ["littérature", "étude", "recherche", "pubmed"]):
            result = await self._literature_summary(query)
        else:
            result = await self._general_medical(query)

        return {
            "domain": "medical",
            "critical_alert": critical,
            "disclaimer": DISCLAIMER,
            "analysis": result,
        }

    def _check_critical(self, text: str) -> List[str]:
        return [s for s in CRITICAL_SYMPTOMS if s in text]

    async def _differential_diagnosis(self, query: str) -> str:
        prompt = (
            "Tu es un assistant médical d'aide au diagnostic différentiel. "
            "Analyse les symptômes décrits et propose:\n"
            "1. Les 3-5 diagnostics différentiels les plus probables (du plus au moins probable)\n"
            "2. Les examens complémentaires recommandés pour chaque hypothèse\n"
            "3. Les signes d'alarme à surveiller\n"
            "4. La conduite à tenir immédiate\n\n"
            f"Cas clinique: {query}\n\n"
            "Rappelle systématiquement qu'une consultation médicale est indispensable."
        )
        return await self._llm(prompt)

    async def _drug_interaction(self, query: str) -> str:
        # Vérification base locale d'abord
        local_alerts = []
        q_lower = query.lower()
        for (drug1, drug2), warning in DRUG_INTERACTIONS.items():
            if drug1 in q_lower and drug2 in q_lower:
                local_alerts.append(f"⚠ {drug1.capitalize()} + {drug2.capitalize()}: {warning}")

        prompt = (
            "Tu es un pharmacologue expert. Analyse les interactions médicamenteuses suivantes:\n"
            "- Niveau de risque (mineur/modéré/majeur/contre-indiqué)\n"
            "- Mécanisme pharmacologique\n"
            "- Conduite à tenir clinique\n"
            "- Alternatives thérapeutiques si nécessaire\n\n"
            f"Requête: {query}"
        )
        llm_result = await self._llm(prompt)
        if local_alerts:
            return "🚨 ALERTES DÉTECTÉES:\n" + "\n".join(local_alerts) + "\n\n" + llm_result
        return llm_result

    async def _analyze_record(self, query: str) -> str:
        prompt = (
            "Tu es un médecin assistant analysant un dossier patient. Fournis:\n"
            "1. Résumé structuré des antécédents pertinents\n"
            "2. Facteurs de risque identifiés\n"
            "3. Points de vigilance clinique\n"
            "4. Recommandations de suivi\n\n"
            f"Dossier: {query}"
        )
        return await self._llm(prompt)

    async def _literature_summary(self, query: str) -> str:
        prompt = (
            "Tu es un médecin expert en médecine basée sur les preuves (EBM). "
            "Synthétise les données de la littérature médicale sur:\n"
            "- Niveau de preuve disponible (grade A/B/C)\n"
            "- Recommandations des sociétés savantes\n"
            "- Limites et biais des études\n\n"
            f"Sujet: {query}"
        )
        return await self._llm(prompt)

    async def _general_medical(self, query: str) -> str:
        prompt = f"Tu es un assistant médical expert. Réponds de façon précise et structurée à: {query}"
        return await self._llm(prompt)

    async def _llm(self, prompt: str) -> str:
        if self.engine.brain and self.engine.brain.get("llm"):
            return await self.engine.brain["llm"].generate(prompt)
        return "[LLM non disponible — configurez OLLAMA_BASE_URL ou OPENAI_API_KEY]"
