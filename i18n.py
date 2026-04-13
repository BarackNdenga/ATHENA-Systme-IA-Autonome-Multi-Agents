#!/usr/bin/env python3
"""
internationalization/i18n.py - Support Multi-Langues Global
Traduction auto via LLM + Locale detection.
"""

import locale
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ['en', 'fr', 'es', 'zh', 'ar', 'ru', 'de', 'ja', 'pt', 'it', 'ko', 'hi']

LANGUAGE_NAMES = {
    'en': 'English', 'fr': 'Français', 'es': 'Español',
    'zh': 'Chinese', 'ar': 'Arabic', 'ru': 'Russian',
    'de': 'Deutsch', 'ja': 'Japanese', 'pt': 'Português',
    'it': 'Italiano', 'ko': 'Korean', 'hi': 'Hindi'
}


class GlobalI18n:
    """Adaptation linguistique: détection automatique + traduction via LLM."""

    def __init__(self, llm):
        self.llm = llm
        self.current_locale = os.getenv('LOCALE') or self._detect_system_locale()
        logger.info(f"I18n initialisé: locale={self.current_locale}")

    def _detect_system_locale(self) -> str:
        """Détecte la locale système de manière robuste."""
        try:
            loc = locale.getlocale()[0] or locale.getdefaultlocale()[0]
            if loc:
                return loc.split('_')[0].lower()
        except Exception:
            pass
        return 'fr'

    async def detect_language(self, text: str) -> str:
        """Détecte la langue d'un texte via le LLM."""
        prompt = (
            f"Détecte la langue de ce texte et réponds UNIQUEMENT avec le code ISO 2 lettres "
            f"(ex: fr, en, es, zh). Texte: '{text[:200]}'"
        )
        try:
            result = await self.llm.generate(prompt)
            detected = result.strip().lower()[:2]
            return detected if detected in SUPPORTED_LANGUAGES else self.current_locale
        except Exception:
            return self.current_locale

    async def translate(self, text: str, target_lang: Optional[str] = None) -> str:
        """Traduit un texte vers la langue cible."""
        target = target_lang or self.current_locale
        if target not in SUPPORTED_LANGUAGES:
            logger.warning(f"Langue non supportée: {target}, fallback fr")
            target = 'fr'
        lang_name = LANGUAGE_NAMES.get(target, target.upper())
        prompt = (
            f"Traduis ce texte en {lang_name}. "
            f"Réponds UNIQUEMENT avec la traduction, sans explication:\n{text}"
        )
        return await self.llm.generate(prompt)

    async def detect_and_translate(self, text: str, target_lang: Optional[str] = None) -> str:
        """Détecte la langue source puis traduit vers la cible."""
        source = await self.detect_language(text)
        target = target_lang or self.current_locale
        if source == target:
            return text
        return await self.translate(text, target_lang=target)

    def set_locale(self, lang_code: str):
        """Change la locale active."""
        if lang_code in SUPPORTED_LANGUAGES:
            self.current_locale = lang_code
            logger.info(f"Locale changée: {lang_code}")
        else:
            logger.warning(f"Locale inconnue: {lang_code}")

    @property
    def supported(self) -> list:
        return SUPPORTED_LANGUAGES

