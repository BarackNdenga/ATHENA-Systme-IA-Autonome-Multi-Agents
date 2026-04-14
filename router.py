#!/usr/bin/env python3
"""
domains/router.py — Routeur de domaines
Détecte automatiquement le secteur d'une requête et route
vers le module spécialisé approprié.
"""

import logging
from typing import Dict, Any, Tuple

from domains.medical import MedicalDomain
from domains.industry import IndustryDomain
from domains.enterprise import EnterpriseDomain
from domains.education import EducationDomain
from domains.cybersecurity import CybersecurityDomain

logger = logging.getLogger(__name__)

# Mots-clés par domaine (ordre = priorité)
DOMAIN_KEYWORDS = {
    "medical": [
        "patient", "diagnostic", "symptôme", "symptom", "médicament", "drug",
        "maladie", "disease", "traitement", "treatment", "médecin", "doctor",
        "hôpital", "hospital", "clinique", "douleur", "fièvre", "tension",
        "chirurgie", "ordonnance", "posologie", "interaction médicamenteuse",
        "antécédent médical", "anamnèse", "pathologie", "thérapie",
    ],
    "industry": [
        "capteur", "sensor", "machine", "équipement", "usine", "factory",
        "maintenance", "panne", "anomalie", "vibration", "température capteur",
        "production", "rendement", "iot", "plc", "scada", "automate",
        "convoyeur", "moteur industriel", "pompe", "compresseur", "turbine",
        "monitoring", "alarme industrielle", "oee", "trs", "mtbf",
    ],
    "cybersecurity": [
        "vulnérabilité", "vulnerability", "hack", "exploit", "injection sql",
        "xss", "csrf", "pentest", "audit sécurité", "firewall", "intrusion",
        "malware", "ransomware", "phishing", "owasp", "cve", "patch",
        "chiffrement", "authentification", "zero trust", "soc", "siem",
        "log sécurité", "incident sécurité", "hardening", "iso 27001",
    ],
    "education": [
        "explique-moi", "c'est quoi", "apprendre", "cours", "leçon",
        "exercice scolaire", "devoir", "examen", "bac", "lycée", "collège",
        "université", "étudiant", "professeur", "tuteur", "pédagogie",
        "mathématiques", "physique chimie", "histoire géo", "biologie",
        "fiche de révision", "résumé de cours", "comprendre le concept",
    ],
    "enterprise": [
        "entreprise", "business", "société", "startup", "kpi", "roi",
        "budget", "finance", "comptabilité", "bilan", "p&l", "chiffre d'affaires",
        "stratégie", "management", "rh", "recrutement", "projet d'entreprise",
        "client", "fournisseur", "contrat", "conformité rgpd", "rapport annuel",
        "conseil d'administration", "actionnaire", "investissement",
    ],
}


class DomainRouter:
    """
    Routeur intelligent qui détecte le domaine d'une requête
    et délègue au module spécialisé.
    """

    def __init__(self, engine):
        self.engine = engine
        self.domains = {
            "medical":      MedicalDomain(engine),
            "industry":     IndustryDomain(engine),
            "enterprise":   EnterpriseDomain(engine),
            "education":    EducationDomain(engine),
            "cybersecurity": CybersecurityDomain(engine),
        }

    def detect_domain(self, query: str) -> Tuple[str, float]:
        """
        Détecte le domaine le plus probable pour une requête.
        Retourne (domain_name, confidence_score).
        """
        query_lower = query.lower()
        scores: Dict[str, int] = {d: 0 for d in DOMAIN_KEYWORDS}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[domain] += 1

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        if best_score == 0:
            return "general", 0.0

        confidence = min(best_score / 3, 1.0)
        return best_domain, round(confidence, 2)

    async def route(self, query: str) -> Dict[str, Any]:
        """
        Route la requête vers le bon domaine et retourne le résultat enrichi.
        Si aucun domaine détecté, utilise le moteur général.
        """
        domain_name, confidence = self.detect_domain(query)

        logger.info(f"[DomainRouter] Domaine détecté: {domain_name} (confiance: {confidence:.0%})")

        if domain_name == "general" or confidence < 0.2:
            return {
                "domain": "general",
                "confidence": confidence,
                "routed": False,
                "analysis": None,
            }

        domain_module = self.domains[domain_name]
        result = await domain_module.analyze(query)
        result["confidence"] = confidence
        result["routed"] = True

        # Stocker dans le graphe de connaissances
        if hasattr(self.engine, "knowledge_graph") and self.engine.knowledge_graph:
            await self.engine.knowledge_graph.add_relation(
                source=domain_name,
                target=query[:40],
                relation="handled_query",
                weight=confidence,
            )

        return result

    def get_domains_info(self) -> Dict[str, Any]:
        return {
            "available_domains": list(self.domains.keys()),
            "keywords_per_domain": {d: len(kws) for d, kws in DOMAIN_KEYWORDS.items()},
        }
