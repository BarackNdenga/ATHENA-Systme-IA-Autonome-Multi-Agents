#!/usr/bin/env python3
"""
domains/cybersecurity.py — Domaine Cybersécurité DÉFENSIVE
Détection de vulnérabilités, audit de code, analyse de logs,
recommandations de hardening, conformité (OWASP, NIST, ISO 27001).

⚠ Usage défensif uniquement — protection des systèmes.
"""

import re
import time
from typing import Dict, Any, List


DISCLAIMER = (
    "⚠ USAGE DÉFENSIF UNIQUEMENT — Cet outil est destiné à protéger vos systèmes. "
    "Toute utilisation offensive ou non autorisée est illégale."
)

# Patterns de vulnérabilités communes (détection statique basique)
VULN_PATTERNS = {
    "sql_injection": {
        "patterns": [r"SELECT.*WHERE.*=.*'", r"OR\s+1=1", r"UNION\s+SELECT", r"--\s*$"],
        "severity": "CRITICAL",
        "owasp": "A03:2021",
        "fix": "Utiliser des requêtes paramétrées (prepared statements)",
    },
    "xss": {
        "patterns": [r"<script>", r"javascript:", r"onerror=", r"onload=", r"eval\("],
        "severity": "HIGH",
        "owasp": "A03:2021",
        "fix": "Encoder les sorties HTML, utiliser Content-Security-Policy",
    },
    "hardcoded_secret": {
        "patterns": [r"password\s*=\s*['\"][^'\"]{4,}", r"api_key\s*=\s*['\"][^'\"]{8,}",
                     r"secret\s*=\s*['\"][^'\"]{4,}", r"sk-[a-zA-Z0-9]{20,}"],
        "severity": "CRITICAL",
        "owasp": "A02:2021",
        "fix": "Utiliser des variables d'environnement ou un gestionnaire de secrets (Vault)",
    },
    "path_traversal": {
        "patterns": [r"\.\./", r"\.\.\\", r"%2e%2e%2f"],
        "severity": "HIGH",
        "owasp": "A01:2021",
        "fix": "Valider et normaliser tous les chemins de fichiers",
    },
    "command_injection": {
        "patterns": [r"os\.system\(", r"subprocess\.call\(.*shell=True", r"exec\(", r"eval\("],
        "severity": "CRITICAL",
        "owasp": "A03:2021",
        "fix": "Éviter shell=True, utiliser subprocess avec liste d'arguments",
    },
    "weak_crypto": {
        "patterns": [r"\bmd5\b", r"\bsha1\b", r"\bdes\b", r"\brc4\b", r"ECB"],
        "severity": "MEDIUM",
        "owasp": "A02:2021",
        "fix": "Utiliser SHA-256+, AES-256-GCM, bcrypt/argon2 pour les mots de passe",
    },
}

# Ports à risque
RISKY_PORTS = {
    21: "FTP — non chiffré", 23: "Telnet — non chiffré",
    3389: "RDP — exposé", 445: "SMB — risque ransomware",
    1433: "MSSQL — exposé", 3306: "MySQL — exposé",
    6379: "Redis — sans auth", 27017: "MongoDB — sans auth",
}


class CybersecurityDomain:
    """Module cybersécurité défensive pour ATHÉNA."""

    def __init__(self, engine):
        self.engine = engine

    async def analyze(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()

        if any(k in query_lower for k in ["code", "script", "fonction", "sql", "html", "python"]):
            result = await self._code_audit(query)
        elif any(k in query_lower for k in ["log", "journal", "événement", "incident", "intrusion"]):
            result = await self._log_analysis(query)
        elif any(k in query_lower for k in ["réseau", "network", "port", "firewall", "ip", "scan"]):
            result = await self._network_audit(query)
        elif any(k in query_lower for k in ["conformité", "iso", "nist", "rgpd", "gdpr", "audit"]):
            result = await self._compliance_check(query)
        elif any(k in query_lower for k in ["hardening", "durcissement", "sécurisation", "config"]):
            result = await self._hardening_guide(query)
        elif any(k in query_lower for k in ["phishing", "social engineering", "email", "arnaque"]):
            result = await self._threat_analysis(query)
        else:
            result = await self._general_security(query)

        # Scan statique si du code est détecté
        static_findings = self._static_scan(query)

        return {
            "domain": "cybersecurity",
            "disclaimer": DISCLAIMER,
            "static_scan": static_findings,
            "analysis": result,
            "timestamp": time.time(),
        }

    def _static_scan(self, text: str) -> List[Dict]:
        """Scan statique basique sur le texte fourni."""
        findings = []
        for vuln_name, vuln_data in VULN_PATTERNS.items():
            for pattern in vuln_data["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    findings.append({
                        "vulnerability": vuln_name,
                        "severity": vuln_data["severity"],
                        "owasp": vuln_data["owasp"],
                        "fix": vuln_data["fix"],
                    })
                    break
        return findings

    async def _code_audit(self, query: str) -> str:
        prompt = (
            "Tu es un expert en sécurité applicative (OWASP, SANS). Audite ce code:\n"
            "1. Vulnérabilités détectées avec niveau de criticité (CVSS)\n"
            "2. Classification OWASP Top 10 2021\n"
            "3. Explication du vecteur d'attaque pour chaque vulnérabilité\n"
            "4. Code corrigé et sécurisé\n"
            "5. Bonnes pratiques à appliquer\n\n"
            f"Code à auditer:\n{query}"
        )
        return await self._llm(prompt)

    async def _log_analysis(self, query: str) -> str:
        prompt = (
            "Tu es un analyste SOC (Security Operations Center). Analyse ces logs:\n"
            "1. Événements suspects ou malveillants identifiés\n"
            "2. Indicateurs de compromission (IoC) détectés\n"
            "3. Timeline de l'incident reconstituée\n"
            "4. Niveau de menace (faible/moyen/élevé/critique)\n"
            "5. Actions de réponse à incident recommandées\n\n"
            f"Logs: {query}"
        )
        return await self._llm(prompt)

    async def _network_audit(self, query: str) -> str:
        # Vérification ports risqués
        risky = []
        for port, desc in RISKY_PORTS.items():
            if str(port) in query:
                risky.append(f"Port {port}: {desc}")

        prompt = (
            "Tu es un expert en sécurité réseau. Analyse cette configuration/situation:\n"
            "1. Exposition et surface d'attaque\n"
            "2. Vulnérabilités réseau identifiées\n"
            "3. Recommandations de segmentation et filtrage\n"
            "4. Configuration firewall recommandée\n"
            "5. Protocoles à désactiver ou remplacer\n\n"
            f"Configuration réseau: {query}"
        )
        result = await self._llm(prompt)
        if risky:
            result = "🚨 PORTS À RISQUE DÉTECTÉS:\n" + "\n".join(f"  • {r}" for r in risky) + "\n\n" + result
        return result

    async def _compliance_check(self, query: str) -> str:
        prompt = (
            "Tu es un expert en conformité cybersécurité. Évalue la conformité selon:\n"
            "- ISO 27001/27002\n- NIST Cybersecurity Framework\n"
            "- RGPD/GDPR si données personnelles\n- OWASP ASVS\n\n"
            "Fournis:\n"
            "1. Niveau de conformité actuel estimé\n"
            "2. Écarts (gaps) identifiés par priorité\n"
            "3. Plan de remédiation avec délais\n"
            "4. Contrôles compensatoires possibles\n\n"
            f"Contexte: {query}"
        )
        return await self._llm(prompt)

    async def _hardening_guide(self, query: str) -> str:
        prompt = (
            "Tu es un expert en durcissement système (CIS Benchmarks). Fournis:\n"
            "1. Checklist de hardening prioritaire\n"
            "2. Configurations spécifiques recommandées\n"
            "3. Services/ports à désactiver\n"
            "4. Politiques de sécurité à appliquer\n"
            "5. Outils de vérification recommandés\n\n"
            f"Système/contexte: {query}"
        )
        return await self._llm(prompt)

    async def _threat_analysis(self, query: str) -> str:
        prompt = (
            "Tu es un expert en threat intelligence défensive. Analyse cette menace:\n"
            "1. Type d'attaque et vecteur utilisé\n"
            "2. Indicateurs de compromission (IoC)\n"
            "3. Groupes de menaces associés (si connu)\n"
            "4. Mesures de protection immédiates\n"
            "5. Sensibilisation des utilisateurs recommandée\n\n"
            f"Menace: {query}"
        )
        return await self._llm(prompt)

    async def _general_security(self, query: str) -> str:
        prompt = (
            "Tu es ATHÉNA, expert en cybersécurité défensive. "
            f"Réponds de façon technique, précise et orientée protection à: {query}"
        )
        return await self._llm(prompt)

    async def _llm(self, prompt: str) -> str:
        if self.engine.brain and self.engine.brain.get("llm"):
            return await self.engine.brain["llm"].generate(prompt)
        return "[LLM non disponible]"
