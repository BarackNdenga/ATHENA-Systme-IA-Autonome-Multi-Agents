#!/usr/bin/env python3
"""
vision/web_spyder.py - Extraction de données web asynchrone
Web scraping avec aiohttp et BeautifulSoup.
"""

import asyncio
import logging
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebSpyder:
    """Spyder web asynchrone pour extraction de données."""

    def __init__(self):
        self._session: aiohttp.ClientSession = None
        self.extracted_data: Dict[str, Any] = {}

    async def initialize(self):
        """Initialise la session aiohttp (persistante)."""
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(timeout=timeout)
        logger.info("WebSpyder initialisé.")

    async def fetch_page(self, url: str) -> Dict[str, Any]:
        """Récupère le contenu d'une seule page."""
        if not self._session or self._session.closed:
            await self.initialize()
        try:
            async with self._session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                return {
                    'url': url,
                    'title': soup.title.string.strip() if soup.title else 'No title',
                    'text': soup.get_text(separator=' ', strip=True)[:1000],
                    'links': [a.get('href') for a in soup.find_all('a', href=True)][:10],
                    'status': response.status
                }
        except Exception as e:
            logger.warning(f"Erreur fetch {url}: {e}")
            return {'url': url, 'error': str(e)}

    async def crawl_url(self, url: str, max_depth: int = 1) -> Dict[str, Any]:
        """Crawle une URL récursivement sans fermer la session."""
        visited = set()
        to_visit = [(url, 0)]
        results = {}

        while to_visit:
            current_url, depth = to_visit.pop(0)
            if current_url in visited or depth > max_depth:
                continue
            visited.add(current_url)

            data = await self.fetch_page(current_url)
            results[current_url] = data

            if depth < max_depth and 'links' in data:
                for link in data['links'][:5]:
                    if link and urlparse(link).netloc:
                        to_visit.append((link, depth + 1))

        self.extracted_data[url] = results
        return results

    async def get_context(self) -> str:
        """Contexte web récent."""
        if self.extracted_data:
            recent = list(self.extracted_data.values())[-1]
            urls = list(recent.keys())[:3]
            return f"Web data from: {', '.join(urls)}"
        return "Aucun web scraping récent."

    async def shutdown(self):
        """Fermeture propre de la session."""
        if self._session and not self._session.closed:
            await self._session.close()

