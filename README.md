# 🦉 ATHÉNA — Système IA Autonome Multi-Agents

> Développé par **Barack Ndenga** 🧠 — Intelligence Suprême 2026

---

```text
╔══════════════════════════════════════════════════════════╗
║           ATHÉNA — Système IA Autonome                  ║
║                                                          ║
║         Développé par Barack Ndenga 🧠                  ║
║              Intelligence Suprême 2026                   ║
╚══════════════════════════════════════════════════════════╝
```

ATHÉNA est un système d'intelligence artificielle autonome basé sur une architecture multi-agents avec planification ReAct, boucle fermée d'auto-amélioration, mémoire duale (sémantique + épisodique), graphe de connaissances, observabilité complète et sécurité Zero Trust.

---

## Architecture complète

```text
athena_project/
├── main.py                          # Point d'entrée (CLI + API en parallèle)
├── requirements.txt
├── .env                             # Clés API (ne pas committer)
│
├── core/                            # Cerveau système (S.E.T.H.)
│   ├── engine.py                    # Orchestrateur principal
│   ├── agentic_framework.py         # Swarm d'agents ReAct autonomes
│   ├── react_planner.py             # Planificateur multi-étapes (DAG)
│   ├── feedback_loop.py             # Boucle fermée d'auto-amélioration ★
│   ├── observability.py             # Traces + métriques + dashboard
│   ├── security.py                  # Surveillance intégrité fichiers
│   ├── zero_trust.py                # Zero Trust: scoring comportemental continu
│   └── quantum_crypto.py            # Crypto post-quantique (AES-256-GCM + Ed25519 / Kyber+Dilithium si OQS)
│
├── brain/                           # Intelligence (J.A.R.V.I.S.)
│   ├── llm_handler.py               # LLM: Ollama local + OpenAI fallback
│   ├── memory.py                    # Mémoire sémantique (ChromaDB + sentence-transformers)
│   ├── episodic_memory.py           # Mémoire épisodique (ce qu'ATHÉNA a FAIT, JSONL)
│   ├── knowledge_graph.py           # Graphe de connaissances (NetworkX, relations typées)
│   └── embeddings.py                # Embeddings avancés (all-MiniLM-L6-v2)
│
├── vision/                          # Perception (Œil de Dieu)
│   ├── scanner.py                   # Scan fichiers + analyse images (OpenCV)
│   └── web_spyder.py                # Web scraping asynchrone (aiohttp + BeautifulSoup)
│
├── interfaces/
│   ├── cli.py                       # CLI interactive (Rich, async)
│   └── api.py                       # API REST (FastAPI)
│
├── internationalization/
│   └── i18n.py                      # Détection langue + traduction via LLM (12 langues)
│
└── tests/
    └── test_core.py                 # Tests unitaires (pytest + pytest-asyncio)
```

Modules actifs dans l'engine au démarrage :

```text
├── brain: LLMHandler + VectorMemory
├── vision: FileScanner + WebSpyder
├── security: SecurityMonitor
├── swarm: AgenticSwarm
├── episodic_memory: EpisodicMemory
├── knowledge_graph: KnowledgeGraph
├── observability: ObservabilityHub
└── feedback_loop: FeedbackLoop  ★
```

---

## Installation

```bash
cd athena_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Optionnel — Cryptographie post-quantique réelle (Kyber1024 + Dilithium3)

```bash
pip install liboqs-python
```

Sans ce package, ATHÉNA utilise automatiquement AES-256-GCM + Ed25519.

---

## Configuration `.env`

```env
OPENAI_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=./chroma_db
LOCALE=fr
```

---

## Lancement

```bash
# Lance CLI + API simultanément
python main.py

# API seule
uvicorn interfaces.api:app --reload --port 8000

# Tests
pytest tests/ -v
```

---

## Usage CLI

```text
ATHÉNA> Analyse ce répertoire Python
ATHÉNA> Génère un script pour parser du JSON
ATHÉNA> Recherche https://example.com
ATHÉNA> stats          ← performance de chaque agent + feedback loop
ATHÉNA> quit
```

---

## API Endpoints

| Méthode | Route | Description |
| ------- | ----- | ----------- |
| `POST` | `/query` | Envoyer une requête à ATHÉNA |
| `GET` | `/status` | Statut système + stats du swarm |
| `GET` | `/memory/{limit}` | Derniers souvenirs sémantiques |
| `GET` | `/trust-report` | Rapport Zero Trust de toutes les entités |
| `GET` | `/health` | Santé système + profils agents (feedback loop) |
| `WS` | `/ws/query` | **Streaming temps réel** — étapes du plan en direct |

---

## ★ Boucle fermée d'auto-amélioration

C'est ce qui distingue ATHÉNA d'un simple système multi-agents.  
Après chaque action, le résultat influence directement le comportement futur :

```text
┌─────────────────────────────────────────────────────────┐
│                   BOUCLE FERMÉE                         │
│                                                         │
│  Action agent → Résultat → Score Critic                 │
│       ↑                         │                       │
│       │                         ▼                       │
│  Contexte enrichi ← FeedbackLoop.process_episode()      │
│  (épisodes passés +             │                       │
│   instructions correctives)     ▼                       │
│                        Ajustements automatiques:        │
│                        • context_boost si échec         │
│                        • instruction corrective LLM     │
│                        • réduction contexte si succès   │
└─────────────────────────────────────────────────────────┘
```

**Concrètement :**

- Un agent qui échoue reçoit automatiquement plus d'historique contextuel (`context_boost`)
- Après 2 échecs consécutifs, le LLM génère une instruction corrective spécifique à cet agent
- Cette instruction est injectée dans tous ses prochains prompts
- Un agent performant voit son contexte allégé (moins de bruit, plus de vitesse)
- Le score du Critic est propagé à tous les agents workers — pas seulement leur auto-évaluation
- Le LLMHandler reçoit le contexte épisodique + graphe avant chaque génération
- **Les profils sont persistés sur disque** (`feedback_profiles.json`) — l'apprentissage survit aux redémarrages
- **Le ReActPlanner notifie la boucle** après chaque sous-tâche, pas seulement le swarm

**Résultat :** ATHÉNA s'améliore à chaque requête, en production, sans intervention humaine.

---

## Architecture des agents — Boucle ReAct

Chaque requête passe par le **ReActPlanner** qui :

1. **Décompose** la tâche en sous-tâches selon les rôles détectés
2. **Ordonne** les sous-tâches en respectant les dépendances (DAG)
3. **Exécute** en parallèle les agents disponibles
4. **Valide** via l'agent Critic
5. **Synthétise** une réponse finale cohérente via LLM

```text
Requête utilisateur
      │
      ▼
 ReActPlanner
  ├── t0: Analyst    (toujours en premier)
  ├── t1: Researcher (si recherche web détectée)
  ├── t2: Coder      (si génération de code détectée)
  └── tc: Critic     (toujours en dernier — validation + feedback)
      │
      ▼
  Synthèse LLM → Réponse finale
      │
      ▼
  FeedbackLoop → Ajustement profils agents → persistés sur disque
```

Chaque agent suit le cycle **Thought → Action → Observation → Feedback** :

- Avant d'agir : consulte son contexte enrichi (épisodes passés + instructions correctives)
- Après avoir agi : notifie la boucle fermée de son résultat

---

## WebSocket streaming

Connexion : `ws://localhost:8000/ws/query`

```json
// Client envoie:
{"query": "Génère un script Python pour trier une liste"}

// ATHÉNA répond en flux:
{"type": "start",    "data": {"query": "..."}}
{"type": "plan",     "data": {"subtasks": [{"task_id": "t0", "agent": "analyst"}, ...]}}
{"type": "step",     "data": {"task_id": "t0", "agent": "analyst", "status": "running"}}
{"type": "result",   "data": {"task_id": "t0", "confidence": 0.82, "result": "..."}}
{"type": "feedback", "data": {"agent": "Athena-Analyst", "trend": "improving", "context_boost": 0}}
{"type": "step",     "data": {"task_id": "t2", "agent": "coder", "status": "running"}}
{"type": "result",   "data": {"task_id": "t2", "confidence": 0.75, "result": "..."}}
{"type": "final",    "data": {"response": "...", "duration_ms": 1243.5, "success": true}}
{"type": "done"}
```

---

## Persistance de l'apprentissage

Fichiers générés automatiquement au runtime :

| Fichier | Contenu |
| ------- | ------- |
| `./chroma_db/` | Mémoire sémantique (ChromaDB) |
| `./episodic_memory.jsonl` | Historique de toutes les actions |
| `./knowledge_graph.json` | Graphe de connaissances |
| `./feedback_profiles.json` | Profils d'apprentissage des agents |
| `./traces.jsonl` | Traces d'observabilité |

Tout est persisté sur disque. Un redémarrage ne perd aucun apprentissage.

---

## Mémoire triple

| Type | Fichier | Ce qu'elle stocke |
| ---- | ------- | ----------------- |
| Sémantique | ChromaDB (`./chroma_db`) | Ce qu'ATHÉNA **sait** — embeddings vectoriels |
| Épisodique | JSONL (`./episodic_memory.jsonl`) | Ce qu'ATHÉNA **a fait** — actions, résultats, confiance |
| Graphe | JSON (`./knowledge_graph.json`) | Relations entre concepts — typées et pondérées |

---

## Observabilité

Chaque opération est tracée automatiquement :

```text
📊 ATHÉNA - Tableau de bord observabilité
═══════════════════════════════════════════════════════
  Opération                 Appels   Err%   Moy ms   P95 ms
  ─────────────────────────────────────────────────────
  process_query                 12     0%    843.2   1205.0
  init_brain                     1     0%    312.4    312.4
  init_vision                    1     0%     45.1     45.1
═══════════════════════════════════════════════════════
```

Traces persistées dans `./traces.jsonl`.

---

## Sécurité

- **Intégrité fichiers** : hash SHA-256 des fichiers critiques au démarrage, surveillance watchdog en temps réel
- **Zero Trust** : scoring comportemental multi-facteurs (taux de requêtes, CPU, historique d'anomalies), isolation automatique des entités suspectes
- **Cryptographie** : AES-256-GCM + Ed25519 par défaut, Kyber1024 + Dilithium3 si `liboqs` installé

---

## Internationalisation

Détection automatique de la langue + traduction via LLM.  
12 langues supportées : `fr`, `en`, `es`, `zh`, `ar`, `ru`, `de`, `ja`, `pt`, `it`, `ko`, `hi`

```python
i18n = GlobalI18n(llm)
await i18n.detect_and_translate("Analyse this code", target_lang="fr")
# → "Analyse ce code"
```

---

## Tests

```bash
pytest tests/ -v

# Résultat attendu:
# tests/test_core.py::TestEpisodicMemory::test_record_and_recall          PASSED
# tests/test_core.py::TestEpisodicMemory::test_persistence                PASSED
# tests/test_core.py::TestKnowledgeGraph::test_add_and_relate             PASSED
# tests/test_core.py::TestKnowledgeGraph::test_find_path                  PASSED
# tests/test_core.py::TestObservability::test_span_lifecycle              PASSED
# tests/test_core.py::TestObservability::test_metrics_summary             PASSED
# tests/test_core.py::TestReActPlanner::test_decompose_with_coder         PASSED
# tests/test_core.py::TestAgenticSwarm::test_dispatch_returns_list        PASSED
# tests/test_core.py::TestFeedbackLoop::test_register_and_process         PASSED
# tests/test_core.py::TestFeedbackLoop::test_context_boost_on_low_conf    PASSED
# tests/test_core.py::TestFeedbackLoop::test_critic_score_propagation     PASSED
# tests/test_core.py::TestFeedbackLoop::test_trend_detection              PASSED
# ...
```

---

## Stack technique

| Couche | Technologies |
| ------ | ------------ |
| LLM | Ollama (local) + OpenAI GPT-4o-mini (fallback) |
| Embeddings | `sentence-transformers` all-MiniLM-L6-v2 |
| Mémoire vectorielle | ChromaDB |
| Graphe | NetworkX |
| Auto-amélioration | FeedbackLoop (stdlib Python — zero dépendance) |
| Web | aiohttp + BeautifulSoup |
| Vision | OpenCV |
| API REST | FastAPI + Uvicorn |
| API Streaming | WebSocket (FastAPI natif) |
| CLI | Rich |
| Crypto | cryptography + liboqs (optionnel) |
| Tests | pytest + pytest-asyncio |

---

```text
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ATHÉNA — Conçu et développé par Barack Ndenga 🧠   ║
║          Tous droits réservés © 2026                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```
