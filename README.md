# NURU — Assistant IA Personnel Local

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-V1%20Production-brightgreen)

> **Assistant conversationnel intelligent avec RAG, I/O vocale et optimisation Apple Silicon M1.** 
> Local-first avec escalade cloud intelligente. Zéro dépendance propriétaire.

---

## 🎯 Vue d'ensemble

**NURU** est un assistant IA personnel qui fonctionne **entièrement localement** sur Apple Silicon (M1+), avec une base vectorielle ChromaDB de **6 074 chunks**, un routage intelligent à 3 niveaux, et une interface vocale complète.

### Caractéristiques principales

- **🚀 Local par défaut** — Modèle LLM Qwen 2.5-3B-Instruct sur Apple Silicon (40-55 tok/s)
- **🧠 RAG intelligent** — Retrieval-Augmented Generation sur 6 074 chunks (CV, rapports, contrats, documents agricoles)
- **🗣️ I/O vocal complet** — STT (Whisper) + TTS (Piper) en streaming
- **🌐 Escalade cloud** — Deepseek v4 Flash avec fallback OpenRouter
- **📚 3 niveaux de routage** — N1 (RAG local) → N2 (LLM local) → N3 (Cloud + Web search)
- **💾 Mémoire contextuelle** — Session + mémoire structurée des faits utilisateur
- **🖥️ Multiples interfaces** — CLI, PySide6 overlay (Cyber-HUD), Web UI (FastAPI), daemon barre de menus
- **⚡ Optimisé M1 8 Go** — <2.5 Go RAM actif, zéro swap, streaming temps réel

---

## ⚡ Démarrage rapide

### Prérequis

- **macOS 12.0+** avec Apple Silicon (M1/M2/M3)
- **Python 3.11+** (via Homebrew : `brew install python3`)
- **Git**

### Installation (30 secondes)

```bash
git clone https://github.com/leblancbahiga/NURU-Personal-AI.git
cd NURU-Personal-AI

# Installation automatique (macOS natif)
./deploy_nuru.command
```

**Ou installation manuelle :**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/chat.py
```

### Configuration des clés API

NURU utilise le **Keychain macOS** pour les API keys. Pour ajouter une clé :

```bash
# Deepseek (obligatoire pour N3)
security add-generic-password -s "com.nuru.assistant" -a "deepseek" -w "YOUR_API_KEY"

# Tavily Search (web search primaire)
security add-generic-password -s "com.nuru.assistant" -a "tavily" -w "YOUR_API_KEY"

# OpenRouter (fallback si Deepseek échoue)
security add-generic-password -s "com.nuru.assistant" -a "openrouter" -w "YOUR_API_KEY"

# Brave Search (fallback web search gratuit)
security add-generic-password -s "com.nuru.assistant" -a "brave" -w "YOUR_API_KEY"
```

---

## 🎮 Utilisation

### Interface CLI

```bash
# Mode local (auto-routing)
python3 src/chat.py

# Forcer le mode cloud
python3 src/chat.py --cloud

# Activer la recherche web
python3 src/chat.py --web

# Désactiver le cache sémantique
python3 src/chat.py --no-cache
```

### Interface Graphique (Overlay PySide6)

```bash
# Cyber-HUD V2 (streaming temps réel)
python3 src/nuru_overlay.py

# Raccourci clavier (après déploiement)
Option + Space  # Toggle overlay on/off
```

**Contrôles :**
- `Entrée` — Envoyer un message
- `Échap` — Masquer l'overlay
- `⚙️` Menu — Cache, stats, santé du système

### Interface Web

```bash
python3 src/webui.py
# http://localhost:8080
```

### Daemon barre de menus

```bash
python3 src/nuru_daemon.py --menubar
```

---

## 🏗️ Architecture

### Pipeline de routage (3 niveaux)

```
Requête utilisateur
        ↓
┌─────────────────────────────────┐
│  Interceptions prioritaires      │
│  • "Qui suis-je ?" → contexte   │
│  • Cache sémantique             │
└──────────┬──────────────────────┘
           ↓
┌─────────────────────────────────┐
│  ComplexityClassifier (4 critères)
│  • Score RAG (25%)              │
│  • Mots-clés (40%)              │
│  • Longueur (20%)               │
│  • Structure (15%)              │
└──────────┬──────────────────────┘
           ↓
    ┌──────┼──────┐
    ↓      ↓      ↓
   N1     N2     N3
  RAG    LLM   Cloud
 Local  Local  Deepseek
```

### Niveaux de routage

| Niveau | Modèle | Source | Usage | Temp | max_tokens |
|--------|--------|--------|-------|------|-----------|
| **N1** | Qwen 3B Q4 (MLX) | Documents RAG | Réponses contextualisées | 0.3 | 2048 |
| **N2** | Qwen 3B Q4 (MLX) | Connaissances générales | Questions ouvertes | 0.7 | 2048 |
| **N3** | Deepseek v4 Flash | Cloud + Web search | Données récentes | 0.7 | 4096 |

### Seuils de classification

- **Score < 0.36** → N1 (RAG local)
- **0.36 – 0.42** → N2 (LLM local)
- **Score > 0.42** → N3 (Cloud)
- **Mots-clés web détectés** → Force N3 (prix, météo, actualité, cours, élection, etc.)

---

## 📊 Performance

### Benchmarks M1 MacBook Pro (8 Go RAM)

| Métrique | Valeur | Notes |
|----------|--------|-------|
| **TTFT (N1)** | ~450ms | RAG + Qwen 3B |
| **TTFT (N2)** | ~280ms | Qwen 3B seul |
| **TTFT (N3)** | ~1200ms | API Deepseek |
| **Throughput** | 40-55 tok/s | Qwen 3B Q4 |
| **RAM idle** | ~230 MB | Sans modèle chargé |
| **RAM N1 actif** | ~2.3 Go | Overlay + Qwen + LanceDB |
| **RAM N2 actif** | ~1.8 Go | Overlay + Qwen |
| **Latence TTS** | <500ms | Piper, phrase par phrase |

### Chunks RAG indexés

- **6 074 chunks** répartis sur 512 tokens avec overlap 64
- **Sources** : CV, rapports IITA, contrats, documents agricoles
- **Embedding** : paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
- **Scoring hybride** : 0.6 embedding + 0.2 keyword + 0.2 recency

---

## 🧠 Composants clés

### ComplexityClassifier

**Fichier** : `src/complexity_classifier.py`

Classifier à 4 critères pondérés décidant du niveau de routage :
- Score RAG hybride (ChromaDB)
- Présence de mots-clés complexes vs simples
- Longueur de la requête
- Structure syntaxique (connecteurs, questions)

Entrée calibration : `config/config.yaml`

### RAG Pipeline

**Fichier** : `src/rag.py`

```
Documents → Chunking (512 tokens, overlap 64)
         → Embedding (sentence-transformers)
         → ChromaDB indexing (384 dims, cosine)
         → Query rewriting (local, MLX)
         → Top-k retrieval (k=5)
         → Hybrid scoring (embedding + keyword + recency)
```

### Mémoire

**Fichiers** :
- `src/memory.py` — Session buffer (5 derniers échanges)
- `src/structured_memory.py` — Faits extraits (JSON)
- `src/semantic_cache.py` — Cache par similarité

### Recherche Web

**Fichiers** :
- `src/tavily_search.py` — Tavily Search (primaire, cloud)
- `src/brave_search.py` — Brave Search (fallback, gratuit, cache TTL 5min)

### I/O Vocal

**Fichiers** :
- `src/audio_stt.py` — STT (Whisper tiny)
- `src/audio_tts.py` — TTS (Piper fr_FR)

### Fine-tuning auto

**Fichiers** :
- `src/dataset_collector.py` — Collecte échanges → `~/.nuru/dataset/conversations.jsonl`
- `src/fine_tune.py` — Pipeline MLX LoRA pour amélioration continue

---

## ⚙️ Configuration

**Fichier** : `config/config.yaml`

```yaml
models:
  local:
    llm:
      repo_id: "mlx-community/Qwen2.5-3B-Instruct-4bit"
  cloud:
    provider: deepseek
    deepseek_model: deepseek-v4-flash

search:
  primary: "tavily"       # Tavily Search (cloud)
  backup: "brave"         # Brave Search (fallback)

rag:
  chunk_size: 512
  chunk_overlap: 64
  similarity_threshold: 0.50
  reranker: false         # Désactivé pour la vitesse
  hyde: false             # Désactivé par défaut
  max_context_tokens: 1500

memory:
  session_buffer_size: 5
  auto_summarize: true

system:
  max_ram_gb: 3.0
  low_battery_threshold: 20
  log_level: INFO
```

---

## 📁 Structure du projet

```
NURU-Personal-AI/
├── src/
│   ├── router.py                # Routeur principal (stream_route)
│   ├── complexity_classifier.py # Classifieur 4 critères
│   ├── rag.py                   # Pipeline RAG (ChromaDB)
│   ├── chat.py                  # Interface CLI
│   ├── nuru_overlay.py          # Overlay PySide6 (Cyber-HUD)
│   ├── nuru_daemon.py           # Daemon barre de menus
│   ├── webui.py                 # FastAPI web UI
│   ├── audio_stt.py             # STT (Whisper)
│   ├── audio_tts.py             # TTS (Piper)
│   ├── memory.py                # Mémoire session
│   ├── structured_memory.py     # Mémoire structurée (faits)
│   ├── semantic_cache.py        # Cache sémantique
│   ├── tavily_search.py         # Tavily Search API
│   ├── brave_search.py          # Brave Search API
│   ├── dataset_collector.py     # Collecte pour fine-tuning
│   ├── fine_tune.py             # Pipeline MLX LoRA
│   ├── action_engine.py         # Moteur d'actions
│   ├── transparency.py          # Journal de transparence
│   ├── monitor.py               # Monitoring RAM/perf
│   └── keychain_utils.py        # Accès Keychain macOS
├── data/
│   ├── chroma_db/               # Base vectorielle ChromaDB
│   ├── logs/                    # Logs d'exécution
│   └── complexity_log.json      # Historique classifications
├── config/
│   ├── config.yaml              # Configuration principale
│   ├── com.nuru.daemon.plist    # LaunchAgent auto-start
│   └── nuru_hammerspoon.lua     # Raccourci Option+Space
├── docs/                        # Documentation (architecture, roadmap)
├── assets/                      # Ressources (icônes, etc.)
├── NURU.app/                    # Bundle macOS natif
├── requirements.txt             # Dépendances Python
├── pyproject.toml               # Métadonnées projet
├── deploy_nuru.command          # Script déploiement one-click
└── README.md                    # Ce fichier
```

---

## 📦 Dépendances

| Package | Usage | Version |
|---------|-------|---------|
| `mlx` / `mlx-lm` | Inférence LLM locale (Apple Silicon) | ≥0.20 |
| `chromadb` | Base vectorielle RAG | ≥0.5 |
| `sentence-transformers` | Embeddings RAG | ≥3.0 |
| `PySide6` | Interface graphique overlay | ≥6.0 |
| `openai` | Client Deepseek API | ≥1.0 |
| `tavily-python` | Tavily Search SDK | ≥0.7 |
| `httpx` | HTTP client (Brave Search) | ≥0.27 |
| `faster-whisper` | STT (Whisper) | ≥1.0 |
| `fastapi` / `uvicorn` | Web UI | ≥0.110 |
| `PyMuPDF` | Parsing PDF | ≥1.24 |
| `python-docx` | Parsing DOCX | ≥1.1 |
| `watchdog` | Surveillance fichiers | ≥4.0 |
| `keyring` | Keychain macOS | ≥25.0 |
| `rumps` | Menu bar macOS | ≥0.4 |
| `pyyaml` | Config YAML | ≥6.0 |
| `Pillow` | Génération icônes | ≥10.0 |

---

## 🔌 Intégrations

### APIs Cloud

- **Deepseek v4 Flash** — Requêtes complexes + web search (N3)
- **OpenRouter** — Fallback si Deepseek échoue (Gemma 4 free)
- **Tavily Search** — Recherche web primaire
- **Brave Search** — Fallback web (gratuit, 2000 req/mois)

### Stockage local

- **ChromaDB** — Vecteurs RAG (6 074 chunks)
- **SQLite** — Logs, memory, fine-tuning dataset
- **JSON** — Cache, configuration, logs

---

## 🚀 Roadmap V2 (Planification)

La **Roadmap V2** propose 9 améliorations critiques pour réduire TTFT de 40-60%, RAM de 30-40%, et améliorer la précision RAG de 15-25%.

### Phases (13 semaines)

| Phase | Semaines | Objectif | Fichiers clés |
|-------|----------|----------|---------------|
| **V2.0 Alpha** | 1-6 | Fondations (pipeline async, intent pre-classifier, LanceDB, model pool) | `pipeline_v2.py`, `intent_classifier.py`, `vector_store_v2.py`, `model_pool_v2.py` |
| **V2.1 Beta** | 7-10 | Intelligence (mémoire hiérarchique, small-to-big chunker, audio) | `memory_v2.py`, `chunker_v2.py` |
| **V2.2 Stable** | 11-13 | Finition (phi-4-mini reasoning, wake word, déploiement) | Stabilisation + tests |

Voir `Roadmap_NURU_V2.md` pour les détails complets.

---

## 🐛 Dépannage

### NURU démarre lentement (TTFT > 2 secondes)

**Cause probable** : le modèle Qwen 3B n'est pas chargé.

```bash
# Solution : pré-charger le modèle en arrière-plan
python3 -c "from mlx_lm import load; load('mlx-community/Qwen2.5-3B-Instruct-4bit')"
```

### Erreur "Insufficient memory"

**Cause** : swap détecté (RAM < 1.5 Go libre).

```bash
# Vérifier les processus gourmands
top -l 1 | grep Memory

# Solution : réduire max_ram_gb dans config.yaml
max_ram_gb: 2.5
```

### API Deepseek échoue

**Cause** : clé API invalide ou quotas dépassés.

```bash
# Vérifier la clé API
security find-generic-password -s "com.nuru.assistant" -a "deepseek" -w

# Vérifier les logs
tail -f ~/.nuru/logs/nuru.log
```

### Overlay PySide6 ne s'affiche pas

**Cause** : dépendance manquante ou X11.

```bash
# Réinstaller PySide6
pip install --upgrade --force-reinstall PySide6

# Vérifier Qt disponible
python3 -c "import PySide6.QtWidgets; print('OK')"
```

Voir `calibration_notes.md` pour des notes de calibration détaillées.

---

## 📚 Documentation

- **`NURU.md`** — Architecture détaillée, configuration, dépendances
- **`NURU_V2_Architecture.md`** — Analyse critique de 9 améliorations V2
- **`Roadmap_NURU_V2.md`** — Plan développement 13 semaines, stack finale
- **`calibration_notes.md`** — Notes de calibration (classifieur, RAG, performance)

---

## 📝 Licence

MIT License — Voir `LICENSE` pour détails.

---

## 👤 Auteur

**Leblanc BAHIGA Mudarhi**

- GitHub : [@leblancbahiga](https://github.com/leblancbahiga)
- Projet : Assistant IA personnel pour Apple Silicon

---

## 🤝 Contribution

NURU est un projet personnel en production. Les contributions sont bienvenues via issues et pull requests.

### Processus de contribution

1. Fork le repository
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

---

## 📞 Support

- **Issues** — Signaler un bug ou demander une fonctionnalité
- **Discussions** — Questions générales et brainstorming
- **Email** — Pour les questions urgentes

---

## 🎯 Prochaines étapes

Après l'installation :

1. **Ajouter vos documents** → `~/.nuru/documents/`
2. **Calibrer le classifieur** → Voir `calibration_notes.md`
3. **Activer les APIs optionnelles** → Deepseek, Tavily, Brave
4. **Lancer le déploiement macOS** → `./deploy_nuru.command`
5. **Utiliser le raccourci clavier** → `Option + Space` (après déploiement)

---

**Made with ❤️ for Apple Silicon. Local-first. Privacy-first. Open-source.**

*Dernière mise à jour : 09/05/2026*
