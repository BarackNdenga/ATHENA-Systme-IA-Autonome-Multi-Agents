#!/usr/bin/env python3
"""
domains/industry.py — Domaine Industrie
Monitoring équipements, détection d'anomalies, maintenance prédictive,
analyse de capteurs IoT, optimisation de production.
"""

import time
import statistics
from typing import Dict, Any, List, Optional


# Seuils d'anomalie par type de capteur
SENSOR_THRESHOLDS = {
    "temperature":  {"warning": 80,  "critical": 95,  "unit": "°C"},
    "vibration":    {"warning": 5.0, "critical": 8.0, "unit": "mm/s"},
    "pressure":     {"warning": 8.0, "critical": 10.0,"unit": "bar"},
    "current":      {"warning": 90,  "critical": 110, "unit": "A"},
    "humidity":     {"warning": 85,  "critical": 95,  "unit": "%"},
    "rpm":          {"warning": 3200,"critical": 3600,"unit": "tr/min"},
    "noise":        {"warning": 85,  "critical": 95,  "unit": "dB"},
}

ANOMALY_PATTERNS = {
    "spike":    "Pic soudain — possible court-circuit ou surcharge",
    "drift":    "Dérive progressive — usure ou dégradation lente",
    "flatline": "Signal plat — capteur défaillant ou arrêt machine",
    "oscillation": "Oscillations — résonance mécanique ou instabilité",
}


class IndustryDomain:
    """Module de monitoring industriel pour ATHÉNA."""

    def __init__(self, engine):
        self.engine = engine
        self._sensor_history: Dict[str, List[float]] = {}

    async def analyze(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()

        if any(k in query_lower for k in ["capteur", "sensor", "mesure", "valeur", "données"]):
            result = await self._analyze_sensor_data(query)
        elif any(k in query_lower for k in ["anomalie", "alerte", "défaut", "panne", "erreur"]):
            result = await self._detect_anomaly(query)
        elif any(k in query_lower for k in ["maintenance", "prédictif", "usure", "remplacement"]):
            result = await self._predictive_maintenance(query)
        elif any(k in query_lower for k in ["production", "rendement", "efficacité", "optimis"]):
            result = await self._optimize_production(query)
        elif any(k in query_lower for k in ["rapport", "report", "bilan", "kpi"]):
            result = await self._generate_report(query)
        else:
            result = await self._general_industry(query)

        return {"domain": "industry", "analysis": result, "timestamp": time.time()}

    async def ingest_sensor(self, sensor_id: str, sensor_type: str, value: float) -> Dict[str, Any]:
        """Ingère une valeur capteur et retourne une alerte si nécessaire."""
        if sensor_id not in self._sensor_history:
            self._sensor_history[sensor_id] = []
        self._sensor_history[sensor_id].append(value)
        if len(self._sensor_history[sensor_id]) > 100:
            self._sensor_history[sensor_id].pop(0)

        alert = None
        thresholds = SENSOR_THRESHOLDS.get(sensor_type, {})
        if thresholds:
            if value >= thresholds.get("critical", float("inf")):
                alert = {"level": "CRITICAL", "sensor": sensor_id, "value": value,
                         "threshold": thresholds["critical"], "unit": thresholds["unit"]}
            elif value >= thresholds.get("warning", float("inf")):
                alert = {"level": "WARNING", "sensor": sensor_id, "value": value,
                         "threshold": thresholds["warning"], "unit": thresholds["unit"]}

        pattern = self._detect_pattern(self._sensor_history[sensor_id])
        return {"sensor_id": sensor_id, "value": value, "alert": alert, "pattern": pattern}

    def _detect_pattern(self, history: List[float]) -> Optional[str]:
        if len(history) < 5:
            return None
        recent = history[-10:]
        mean = statistics.mean(recent)
        stdev = statistics.stdev(recent) if len(recent) > 1 else 0

        if stdev < 0.01 * mean:
            return ANOMALY_PATTERNS["flatline"]
        if max(recent) > mean + 3 * stdev:
            return ANOMALY_PATTERNS["spike"]
        if len(history) >= 20:
            old_mean = statistics.mean(history[-20:-10])
            if abs(mean - old_mean) > 0.15 * old_mean:
                return ANOMALY_PATTERNS["drift"]
        return None

    async def _analyze_sensor_data(self, query: str) -> str:
        prompt = (
            "Tu es un ingénieur industriel expert en analyse de données capteurs. "
            "Analyse les données suivantes et fournis:\n"
            "1. Interprétation des valeurs mesurées\n"
            "2. Détection d'anomalies ou tendances préoccupantes\n"
            "3. Comparaison aux normes industrielles (ISO 10816, etc.)\n"
            "4. Actions correctives recommandées\n\n"
            f"Données: {query}"
        )
        return await self._llm(prompt)

    async def _detect_anomaly(self, query: str) -> str:
        prompt = (
            "Tu es un expert en diagnostic industriel. Analyse cette situation:\n"
            "1. Classification de l'anomalie (mécanique/électrique/thermique/process)\n"
            "2. Causes racines probables (méthode 5 Pourquoi)\n"
            "3. Impact sur la production et la sécurité\n"
            "4. Plan d'action correctif et préventif (PDCA)\n"
            "5. Délai d'intervention recommandé\n\n"
            f"Situation: {query}"
        )
        return await self._llm(prompt)

    async def _predictive_maintenance(self, query: str) -> str:
        prompt = (
            "Tu es un expert en maintenance prédictive (CBM/PdM). Fournis:\n"
            "1. Estimation de la durée de vie résiduelle (RUL)\n"
            "2. Indicateurs de dégradation à surveiller\n"
            "3. Fenêtre d'intervention optimale\n"
            "4. Pièces de rechange à préparer\n"
            "5. Coût estimé vs coût d'une panne non planifiée\n\n"
            f"Équipement/situation: {query}"
        )
        return await self._llm(prompt)

    async def _optimize_production(self, query: str) -> str:
        prompt = (
            "Tu es un expert en optimisation industrielle (Lean, Six Sigma). Analyse:\n"
            "1. Goulots d'étranglement identifiés\n"
            "2. Pertes (mudas) détectées\n"
            "3. KPIs: OEE, TRS, MTBF, MTTR\n"
            "4. Plan d'amélioration priorisé\n"
            "5. ROI estimé des améliorations\n\n"
            f"Contexte: {query}"
        )
        return await self._llm(prompt)

    async def _generate_report(self, query: str) -> str:
        prompt = (
            "Génère un rapport industriel structuré avec:\n"
            "- Résumé exécutif\n- État des équipements\n- Incidents et anomalies\n"
            "- KPIs de performance\n- Recommandations prioritaires\n\n"
            f"Données: {query}"
        )
        return await self._llm(prompt)

    async def _general_industry(self, query: str) -> str:
        prompt = f"Tu es un expert industriel. Réponds de façon technique et structurée à: {query}"
        return await self._llm(prompt)

    async def _llm(self, prompt: str) -> str:
        if self.engine.brain and self.engine.brain.get("llm"):
            return await self.engine.brain["llm"].generate(prompt)
        return "[LLM non disponible]"
