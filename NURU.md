# NURU — Assistant IA personnel

> Agent conversationnel intelligent, local sur Apple Silicon, avec escalade Cloud.
> Projet conçu et piloté par Leblanc BAHIGA Mudarhi.

---

## Table des matières

1. [Architecture](#1-architecture)
2. [Niveaux de routage](#2-niveaux-de-routage)
3. [ComplexityClassifier](#3-complexityclassifier)
4. [Prompts système](#4-prompts-système)
5. [Signaux d'escalade](#5-signaux-d-escalade)
6. [Recherche Web](#6-recherche-web)
7. [RAG (Retrieval-Augmented Generation)](#7-rag)
8. [Mémoire](#8-mémoire)
9. [Fichiers du projet](#9-fichiers-du-projet)
10. [Configuration](#10-configuration)
11. [Interfaces](#11-interfaces)
12. [Dépendances](#12-dépendances)
13. [Paramètres du classifieur à calibrer](#13-paramètres-du-classifieur-à-calibrer)

---

## 1. Architecture

```
Requête utilisateur
       │
       ▼
┌─────────────────────────────┐
│  Interceptions prioritaires │
│  • "qui suis-je"            │
│  • Corrections prioritaires │
│  • Cache sémantique         │
└──────────┬──────────────────┘
           │ (aucune interception)
           ▼
┌─────────────────────────────┐
│  Recherche RAG (ChromaDB)   │
│  → meilleur score hybride   │
│  → rag_score pour classifieur│
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  ComplexityClassifier       │
│  4 critères pondérés        │
│  → N1 / N2 / N3            │
└──────────┬──────────────────┘
           │
     ┌─────┼─────────┐
     │     │         │
     ▼     ▼         ▼
     N1     N2        N3
  RAG    Général   Search→Cloud
  local  local     Deepseek
```

## 2. Niveaux de routage

### Niveau 1 — RAG local (documents)
- **Modèle** : Qwen 2.5-3B-Instruct-4bit (MLX, Apple Silicon)
- **Contexte** : Documents RAG + mémoire structurée
- **Prompt** : Mode RAG local — répond uniquement des extraits fournis
- **Escalade possible** : `[[ESCALADE:INTERNET]]` ou `[[ESCALADE:NIVEAU3]]`

### Niveau 2 — LLM local (connaissances générales)
- **Modèle** : Qwen 2.5-3B-Instruct-4bit
- **Contexte** : Aucun document disponible
- **Prompt** : Connaissances générales avec mention explicite
- **Escalade possible** : `[[ESCALADE:INTERNET]]` ou `[[ESCALADE:NIVEAU3]]`

### Niveau 3 — Cloud (Deepseek v4 Flash)
- **Modèle** : `deepseek-v4-flash` (API) — fallback OpenRouter (`google/gemma-4-31b-it:free`)
- **Contexte** : Résultats recherche web (Tavily ou Brave) + documents RAG si disponibles
- **Prompt** : Signaux de confiance ✅🔶⚠️❌
- **Escalade possible** : `[[ESCALADE:INCONNU]]`

### Paramètres par niveau

| Paramètre | N1 (RAG local) | N2 (Général local) | N3 (Cloud) |
|-----------|:--------------:|:------------------:|:----------:|
| Température | 0.3 | 0.7 | 0.7 |
| top_p | 0.9 | 0.9 | — |
| max_tokens | 2048 | 2048 | 4096 |
| Repetition penalty | 1.15 | 1.15 | — |
| Modèle | Qwen 3B | Qwen 3B | Deepseek v4 Flash |

---

## 3. ComplexityClassifier

Fichier : `src/complexity_classifier.py`

### Critères (4)

| Critère | Poids | Description |
|---------|:-----:|-------------|
| `s_rag` | 25% | Score RAG (meilleure similarité ChromaDB, 0.0–1.0) |
| `s_keywords` | 40% | Mots-clés complexes vs simples dans la requête |
| `s_length` | 20% | Longueur de la requête (nb mots) |
| `s_structure` | 15% | Connecteurs logiques, phrases, points d'interrogation |

### Score final

```
Score = 0.25 × s_rag + 0.40 × s_keywords + 0.20 × s_length + 0.15 × s_structure
```

### Seuils de décision

| Score | Niveau | Action |
|:-----:|:------:|--------|
| < 0.36 | **N1** | RAG local (Qwen + documents) |
| 0.36 – 0.42 | **N2** | LLM local (Qwen, connaissances générales) |
| > 0.42 | **N3** | Cloud (Tavily Search → Deepseek) |

### Surcharge directe

Si des mots-clés web sont détectés (`prix actuel`, `actualité`, `météo`, `cours de`, `taux`, `élection`, `président`, etc.), le classifieur force **N3** quel que soit le score.

### Mots-clés par catégorie

**Complexes** (poussent vers N3) : `analyse`, `compare`, `synthétise`, `évalue`, `explique pourquoi`, `étape par étape`, `rédige`, `2024`, `2025`, `aujourd'hui`...

**Simples** (poussent vers N1) : `qu'est-ce que`, `c'est quoi`, `définis`, `quand`, `où`, `qui`, `liste`...

**Web** (surcharge N3) : `prix actuel`, `cours de`, `météo`, `actualité`, `aujourd'hui`, `récemment`...

---

## 4. Prompts système

Fichier : `src/router.py`

### 4.1 Local avec RAG (N1)

```
## Identité
Tu es NURU, un assistant fiable opérant en mode RAG local.
Tu réponds à partir des extraits de documents fournis ci-dessous.

## Règles de réponse
- Tu utilises uniquement les informations des documents fournis.
- Tu ne complètes jamais avec des connaissances inventées.
- Si l'information est absente des documents, tu n'inventes pas.

## Signaux d'action (priorité absolue)
Si la question nécessite des informations récentes ou absentes des documents,
tu réponds UNIQUEMENT avec : [[ESCALADE:INTERNET]]
Si la question est trop complexe : [[ESCALADE:NIVEAU3]]
```

### 4.2 Local sans RAG (N2)

```
## Identité
Tu es NURU, un assistant fiable opérant sur tes connaissances générales.
Aucun document n'est disponible pour cette requête.

## Signaux d'action
Données récentes (post-2024) → [[ESCALADE:INTERNET]]
Raisonnement complexe → [[ESCALADE:NIVEAU3]]

## Règles de réponse
- "D'après mes connaissances générales (non vérifiées par un document)..."
- Pas de chiffres précis si incertain
```

### 4.3 Cloud Deepseek (N3)

```
## Identité
Tu es NURU, un assistant fiable opérant en mode cloud.

## Signaux de confiance
✅ [CONFIRMÉ]   🔶 [DÉDUIT]   ⚠️ [INCERTAIN]   ❌ [HORS CONTEXTE]

## Signal d'action
[[ESCALADE:INCONNU]]
```

---

## 5. Signaux d'escalade

Signaux émis par le modèle local pour demander de l'aide au routeur :

| Signal | Déclencheur | Action routeur |
|--------|-------------|----------------|
| `[[ESCALADE:INTERNET]]` | Infos récentes / absentes des docs | Recherche Web → Deepseek |
| `[[ESCALADE:NIVEAU3]]` | Trop complexe pour le local | Deepseek direct |
| `[[ESCALADE:INCONNU]]` | Hors périmètre | Message "Info insuffisante" |

Le modèle répond **uniquement** avec le signal, aucun autre texte.
La correspondance est **exacte** (`response.strip() == "[[ESCALADE:...]]"`).

---

## 6. Recherche Web

### Architecture

Deux moteurs de recherche en cascade :

```
Requête → Tavily Search (primaire, cloud)
         → Brave Search (fallback si Tavily échoue)
```

Les fichiers :
- `src/tavily_search.py` — Client Tavily Search (API cloud)
- `src/brave_search.py` — Client Brave Search (API gratuite, cache TTL 5 min)

### Tavily Search

Fichier : `src/tavily_search.py`

```python
from tavily_search import TavilySearchClient

client = TavilySearchClient(api_key="VOTRE_CLE")
response = client.search("prix du riz Kampala")
# → TavilySearchResponse(success, results, raw_context, error)
```

### Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `api_key` | `""` | Clé API Tavily (via keychain macOS) |
| `max_results` | 5 | Nombre de résultats |
| `search_depth` | `"basic"` | Profondeur de recherche |
| `timeout` | 8.0 | Timeout en secondes |

### Brave Search

Fichier : `src/brave_search.py`

| Fonctionnalité | Détail |
|---------------|--------|
| API | Brave Search API (gratuite, 2000 req/mois) |
| Cache | TTL 5 min intégré (évite les appels redondants) |
| Retry | ×2 avec backoff exponentiel (1.5s, 2.25s) |
| Paramètres | `country: "fr"`, `language: "fr"` (configurable) |

### Format d'injection

Les résultats sont formatés automatiquement pour le prompt N3 :
```
Résultats de recherche web pour : « prix du riz Kampala »
[1] Titre
    URL : https://...
    Résumé : ...
```

---

## 7. RAG

Fichier : `src/rag.py`

### Pipeline

```
Documents → Ingestion (chunking 512 tokens, overlap 64)
          → Embedding (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
          → ChromaDB (384 dim, cosine similarity)

Requête → Query Rewriting → Embedding → ChromaDB query (k×3)
       → Scoring hybride (0.6 embedding + 0.2 keyword + 0.2 recency)
       → [HyDE : facultatif, désactivé par défaut — cf. config]
       → [Reranking : facultatif, désactivé par défaut — cf. config]
       → Top-k (5 résultats)
```

### Paramètres

| Paramètre | Valeur | Fichier |
|-----------|--------|---------|
| chunk_size | 512 tokens | config.yaml |
| chunk_overlap | 64 tokens | config.yaml |
| Embedding model | paraphrase-multilingual-MiniLM-L12-v2 | config.yaml |
| HyDE | Désactivé par défaut (gain ~15s) | config.yaml `rag.hyde: false` |
| Reranker | Cross-encoder (désactivé par défaut, gain ~45s) | config.yaml `rag.reranker: false` |
| Top-k | 5 (retour), 15 (recherche) | router.py / rag.py |
| Seuil hybride | 0.50 | config.yaml |
| Max contexte RAG | 1500 tokens | config.yaml |
| Query Rewriting | Activé (réécriture locale avant embedding) | router.py |

### État actuel

- **6 074 chunks** indexés
- Sources : CV, rapports, contrats, documents agricoles
- **Corrections prioritaires** : 0

---

## 8. Mémoire

### Mémoire de session
- Fichier : `src/memory.py` (classe `SessionMemory`)
- Stocke les N derniers échanges (configurable, défaut: 5)
- Injectée dans le prompt comme historique

### Mémoire structurée
- Fichier : `src/structured_memory.py` (classe `StructuredMemory`)
- Stocke les faits connus sur l'utilisateur (nom, profession, lieu, projet)
- Extraite automatiquement des conversations
- Injectée dans le prompt système

### Cache sémantique
- Fichier : `src/semantic_cache.py`
- Évite de re-générer des réponses identiques
- Basé sur la similarité cosinus des embeddings

### Dataset (auto-amélioration)
- Fichier : `src/dataset_collector.py` + `src/fine_tune.py`
- Chaque échange → `~/.nuru/dataset/conversations.jsonl`
- Pipeline MLX LoRA : `python3 src/fine_tune.py --train`
- Objectif : 100-200 échanges de qualité pour un premier fine-tuning

---

## 9. Fichiers du projet

```
Assistant IA/
├── data/
│   ├── chroma_db/               # Base vectorielle ChromaDB
│   ├── logs/                    # Logs
│   └── complexity_log.json      # Historique des classifications
├── docs/
│   ├── V2_ROADMAP.md             # Roadmap des améliorations V2
│   └── nuru.iconset/             # Icônes pour l'application macOS
├── assets/
│   └── nuru_icon.png             # Icône NURU (PNG + ICNS)
├── src/
│   ├── router.py                # Routeur principal (stream_route)
│   ├── complexity_classifier.py # Classifieur à 4 critères
│   ├── tavily_search.py         # Client Tavily Search (primaire)
│   ├── brave_search.py          # Client Brave Search (fallback)
│   ├── rag.py                   # Pipeline RAG (ChromaDB)
│   ├── ingestion.py             # Ingestion de documents
│   ├── indexer.py               # Indexation watchdog
│   ├── memory.py                # Mémoire de session
│   ├── structured_memory.py     # Mémoire structurée (faits)
│   ├── semantic_cache.py        # Cache sémantique
│   ├── chat.py                  # Interface CLI
│   ├── nuru_overlay.py          # Interface PySide6 (Cyber-HUD)
│   ├── nuru_daemon.py           # Daemon barre de menus
│   ├── webui.py                 # Interface web (FastAPI)
│   ├── audio_tts.py             # Synthèse vocale (say / Piper)
│   ├── audio_stt.py             # Reconnaissance vocale (Whisper)
│   ├── monitor.py               # Monitoring RAM/performance
│   ├── transparency.py          # Journal de transparence
│   ├── action_engine.py         # Moteur d'actions
│   ├── dataset_collector.py     # Collecte pour fine-tuning
│   ├── fine_tune.py             # Fine-tuning MLX LoRA
│   ├── feedback.py              # Gestion des retours
│   ├── diag_rag.py              # Diagnostic RAG
│   ├── validate_env.py          # Validation environnement
│   └── keychain_utils.py        # Accès au keychain macOS
├── NURU.app/                    # Bundle macOS natif
├── NURU.md                      # Ce fichier
├── deploy_nuru.command          # Script de déploiement one-click
├── config/
│   ├── config.yaml              # Configuration principale
│   ├── com.nuru.daemon.plist    # LaunchAgent (auto-start)
│   └── nuru_hammerspoon.lua     # Raccourci Option+Space
├── requirements.txt             # Dépendances Python
├── pyproject.toml               # Métadonnées projet
└── tests/                       # Tests unitaires
```

---

## 10. Configuration

Fichier : `config/config.yaml`

```yaml
models:
  local:
    llm:
      repo_id: "mlx-community/Qwen2.5-3B-Instruct-4bit"
  cloud:
    provider: deepseek
    deepseek_model: deepseek-v4-flash
    gemini_model: gemini-2.0-flash

audio:
  stt_model: tiny
  tts_speed: 200

# --- Moteurs de Recherche Web ---
search:
  primary: "tavily"       # "tavily" | "brave"
  backup: "brave"         # Moteur de secours
  tavily:
    depth: "basic"
    max_results: 5
    include_answer: true
  brave:
    max_results: 5

rag:
  chunk_size: 512
  chunk_overlap: 64
  similarity_threshold: 0.50
  reranker: false         # Désactivé par défaut pour la vitesse
  hyde: false             # Désactivé par défaut
  max_context_tokens: 1500
  watch_dirs:
    - ~/Documents
    - ~/Google Drive
  extensions:
    - .pdf
    - .docx
    - .txt
    - .md
    - .csv

memory:
  session_buffer_size: 5
  auto_summarize: true

system:
  max_ram_gb: 3.0
  low_battery_threshold: 20
  log_level: INFO
  keychain_service: com.nuru.assistant
```

### Clés API (keychain macOS)

Service : `com.nuru.assistant`
- `deepseek` — API Deepseek (obligatoire)
- `openrouter` — API OpenRouter (fallback si Deepseek échoue)
- `tavily` — API Tavily Search (recherche web primaire)
- `brave` — API Brave Search (recherche web fallback, gratuit 2000 req/mois)

---

## 11. Interfaces

### CLI (chat.py)
```bash
python3 src/chat.py                              # Mode local (auto)
python3 src/chat.py --cloud                       # Forcer cloud Deepseek
python3 src/chat.py --web                         # Activer recherche web
python3 src/chat.py --no-cache                    # Désactiver cache sémantique
```

### Overlay PySide6 (nuru_overlay.py)
```bash
python3 src/nuru_overlay.py                       # Interface Cyber-HUD (V1)
python3 src/nuru_overlay.py --v2                  # Interface Cyber-HUD avec modules V2
```
- ~200 lignes (contre 968 en V1), streaming temps réel via pipeline V2
- Overlay flottant translucide, pas de chrome window
- Menu système ⚙ intégré (vidage cache, redémarrage routeur, health check, quitter)
- Dashboard 3 colonnes : barre RAM, cercle THINKING, matrice RAG temps réel
- Console défilante + streaming des réponses
- Touches : `Entrée` pour envoyer, `Échap` pour cacher
- App bundle : `open /Applications/NURU.app --args --overlay` (V1)
  ou `open /Applications/NURU.app --args --overlay-v2` (V2)
- Nécessite : PySide6, écran macOS.

### Daemon barre de menus (nuru_daemon.py)
```bash
python3 src/nuru_daemon.py --menubar              # Icône ⚪ dans la barre de menus
```
- Menu : Forcer Cloud / Ouvrir le Chat / Stats / Quitter
- Sans `--menubar` : mode CLI silencieux (attente stdin)

### WebUI (webui.py)
```bash
python3 src/webui.py                              # http://localhost:8080
python3 src/webui.py --cloud                      # Forcer cloud
python3 src/webui.py --port 9090                  # Port personnalisé
```

### Déploiement natif macOS (NURU.app)
```bash
./deploy_nuru.command                             # One-click : installe NURU.app, LaunchAgent, Hammerspoon
open /Applications/NURU.app                       # Lancer depuis /Applications
```
- Bundle manuel (pas PyInstaller) — utilise `.venv` du projet
- LaunchAgent : démarrage automatique au login
- Hammerspoon Option+Space : toggle overlay

---

## 12. Dépendances

| Package | Usage |
|---------|-------|
| `mlx` / `mlx-lm` | Inférence LLM locale (Apple Silicon) |
| `PySide6` | Interface graphique overlay |
| `chromadb` | Base vectorielle RAG |
| `sentence-transformers` | Embeddings + reranking |
| `openai` | Client Deepseek API |
| `httpx` | Client Tavily Search |
| `tavily-python` | SDK Tavily Search |
| `keyring` | Accès keychain macOS |
| `pyyaml` | Lecture config.yaml |
| `faster-whisper` | STT (speech-to-text) |
| `rumps` | Menu bar macOS (daemon) |
| `fastapi` / `uvicorn` | Interface web |
| `PyMuPDF` | Parsing PDF |
| `python-docx` | Parsing DOCX |
| `watchdog` | Surveillance fichiers (indexation auto) |
| `Pillow` | Génération d'icônes |
| `piper-tts` | TTS haute qualité (optionnel) |

---

## 13. Paramètres du classifieur à calibrer

### Poids (doivent sommer à 1.0)
```python
w_rag       = 0.25   # Poids du score RAG
w_keywords  = 0.40   # Poids des mots-clés
w_length    = 0.20   # Poids de la longueur
w_structure = 0.15   # Poids de la structure
```

### Seuils de décision
```python
threshold_n1 = 0.36   # Score max pour N1 (RAG local)
threshold_n3 = 0.42   # Score min pour N3 (Cloud)
```

### Seuil RAG
```python
rag_threshold = 0.50  # Seuil de similarité pour docs pertinents
```

### Méthode de calibration recommandée

1. Logger toutes les classifications pendant 1-2 semaines
2. Analyser les faux positifs (requêtes simples routées vers N3)
3. Analyser les faux négatifs (requêtes complexes routées vers N1)
4. Ajuster les poids et seuils en conséquence
5. Itérer

---

## Historique des modifications

| Date | Changement |
|------|-----------|
| 08/05/2026 | Mise à jour NURU.md : stats RAG 6074 chunks, recherche Web (Tavily+Brave), HyDE/reranker optionnel, menus overlay, déploiement macOS, dépendances |
| 07/05/2026 | Menu système ⚙ dans l'overlay (12 actions) |
| 07/05/2026 | HyDE et Re-ranker désactivés par défaut (config.yaml) |
| 07/05/2026 | Correction BUG RAG critique : le modèle local ne recevait aucun document |
| 07/05/2026 | Prompts système refaits (3 variantes) + signaux d'escalade `[[ESCALADE:...]]` |
| 07/05/2026 | Nouveau ComplexityClassifier à 4 critères -- modules Tavily + Brave Search |
| 07/05/2026 | Routeur refait (flux N1/N2/N3) -- fallback OpenRouter |
| 07/05/2026 | Préchargement embedder en arrière-plan |
| 07/05/2026 | Déploiement macOS natif : NURU.app, LaunchAgent, Hammerspoon |
| 05/05/2026 | Correction bug "qui suis-je" (prompt compacté + interception routeur) |
| 05/05/2026 | DatasetCollector + pipeline auto-amélioration (fine_tune.py MLX LoRA) |
| 03/05/2026 | Phases 0-2 terminées |
