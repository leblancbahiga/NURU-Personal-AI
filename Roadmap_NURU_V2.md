# Roadmap NURU V2 — Analyse Critique & Architecture Cible

> Analyse comparative de 3 propositions IA (Claude Sonnet 4.6, Deepseek V4, Gemini 3.1)
> Cible : MacBook Pro M1 · 8 Go RAM unifiée · Production locale
> Auteur : Analyse architecte logiciel senior — IA locale & Apple Silicon
> Date : 8 mai 2026

---

## TABLE DES MATIÈRES

1. [Vision globale NURU V2](#1-vision-globale-nuru-v2)
2. [Analyse comparative des 3 propositions](#2-analyse-comparative)
3. [Contradictions techniques et résolution](#3-contradictions-techniques)
4. [Stack technique — Verdict final](#4-stack-technique)
5. [Architecture détaillée](#5-architecture-détaillée)
6. [Tableau des fonctionnalités retenues](#6-fonctionnalités-retenues)
7. [Plan de développement par phases](#7-plan-de-développement)
8. [Recommandations modèles](#8-recommandations-modèles)
9. [Pièges à éviter sur M1 8 Go](#9-pièges-à-éviter)

---

## 1. Vision Globale NURU V2

### Philosophie système

NURU V2 n'est pas une réécriture. C'est une **optimisation chirurgicale** de V1 en ciblant les 7 goulets d'étranglement qui plafonnent les performances sur M1 8 Go :

1. **Pipeline synchrone** — chaque étape bloque la suivante → TTFT inutilement long
2. **Escalade par tokens LLM** — le modèle 3B doit générer `[[ESCALADE:...]]` → 800-2500ms de perte
3. **Embeddings PyTorch** — sentence-transformers charge tout PyTorch (~400MB) pour 384 dims
4. **ChromaDB surdimensionné** — DuckDB + SQLite pour 6000 chunks → ~300MB RAM inutile
5. **Modèle unique pour tout** — 3B pour les salutations comme pour le raisonnement → gaspillage
6. **Pas de gestion mémoire adaptative** — `max_ram_gb: 3.0` non enforcee, swap fréquent
7. **Mémoire plate** — FIFO 5 échanges sans compression ni hiérarchie

### Principes directeurs

| Principe | Application |
|----------|-------------|
| **Async-first** | RAG + classification + memory retrieval en parallèle avant l'appel LLM |
| **Pre-classifier < 50ms** | Déterminer le routage sans toucher au LLM |
| **MLX-native** | Zéro PyTorch. MLX pour LLM + embeddings. Garder ce qui marche déjà. |
| **Lazy loading strict** | Un seul modèle chargé à la fois. Les autres → del + gc + mx.metal.clear_cache() |
| **RAM budget : 4.5 Go max** | macOS ~2.5 + NURU ~2.0 = 4.5. Toujours sous 8. Jamais de swap. |
| **Évolution incrémentale** | Modifier un composant à la fois. V1 continue de fonctionner entre chaque étape. |

---

## 2. Analyse Comparative

### 2.1 Tableau de synthèse

| Critère | Claude Sonnet | Deepseek V4 | Gemini 3.1 | Mon verdict |
|---------|--------------|-------------|------------|-------------|
| **Stack LLM** | MLX + Ollama | llama.cpp | MLX + GC manuel | ✅ **MLX** (zéro réécriture) |
| **Routing** | Intent Pre-classifier (regex) | DistilBERT 80 Mo | LLM 1 token | ✅ **Pre-classifier** (sub-50ms, no LLM) |
| **Architecture** | Orchestrator model pool | Event Bus + agents spécialisés | Mono-modèle à rôles | ✅ **Model pool** (Claude) |
| **Vector DB** | LanceDB | ChromaDB (amélioré) | SQLite-VSS | ✅ **LanceDB** |
| **Embeddings** | nomic-embed-text (768d) | bge-small (384d) | nomic-embed Matryoshka | ✅ **nomic-embed** (768d, Matryoshka) |
| **Reranking** | BM25 + cosine (< 10ms) | Cross-encoder ONNX | ColBERTv2 | ✅ **FlashRerank BM25** zéro modèle |
| **Chunking** | Small-to-Big (128→512) | Sliding window 256 | Non mentionné | ✅ **Small-to-Big** |
| **Mémoire** | 3 niveaux hiérarchiques | Unifiée + résumé | Non détaillé | ✅ **Hiérarchique 3 niveaux** |
| **RAM management** | Resource manager LRU | ModelPool + idle timeout | Time-multiplexing strict | ✅ **LRU + RAM monitor** |
| **STT** | mlx-whisper base | whisper.cpp tiny/small | Distil-whisper | ✅ **mlx-whisper base** |
| **TTS** | Kokoro (ONNX) | Piper | Piper | ✅ **Piper** (déjà installé) |
| **Wake word** | Porcupine | Porcupine/openWakeWord | Non mentionné | ✅ **Porcupine** |
| **Vision** | moondream2 (Ollama) | Non mentionné | Qwen2-VL-2B | ⏸️ **Remis à V3** |

### 2.2 Analyse détaillée par proposition

#### Proposition A — Claude Sonnet 4.6 (NURU_V2_Architecture.md)

**Forces :**
- Analyse systémique la plus complète (9 améliorations interconnectées)
- Code prêt à l'emploi pour chaque composant (intent_classifier, model_pool, memory_v2, reranker_v2, resource_manager, chunker_v2)
- Budget RAM réaliste et chiffré (4.6-5.1 Go avec 3B, 5.5-6.0 Go avec phi-4-mini)
- Recommandation MLX + Ollama hybride pragmatique
- Roadmap progressive crédible (V2.0 → V2.1 → V2.2 → V3)
- Pièges documentés (8 pièges spécifiques M1 8 Go)

**Faiblesses :**
- Certains codes sont conceptuels (notamment le pipeline async complet)
- Small-to-Big chunker = réindexation complète des 6074 chunks (coût réel 2-3 jours)
- Model pool avec 3 modèles = complexité de test/maintenance

**Verdict global :** ⭐⭐⭐⭐⭐ **Référence principale** — le document le plus complet et réaliste.

---

#### Proposition B — Deepseek V4

**Forces :**
- Analyse fine de la stack (comparaison détaillée des frameworks)
- Prompt caching et KV cache réutilisable bien expliqués
- Gestion thermique (limitation threads, MLX_GPU_MEMORY_LIMIT)
- Swap intelligent via mlock et mémoire unifiée
- Agent multi-tâche avec Planner/Executor/Memory

**Faiblesses :**
- Recommande **llama.cpp** comme stack principale → incompatible avec MLX déjà intégré. Signifierait réécrire load/generate/stream + format GGUF → perte de la compatibilité mlx-lm LoRA
- Propose DistilBERT (80 Mo) pour l'intent classifier → overhead mémoire inutile vs regex < 50ms
- Propose T5-small quantifié pour la compression de chunks → complexité injustifiée
- Architecture Event Bus (Trio/anyio) inutile pour un assistant mono-utilisateur
- Certaines optimisations (K-Quants, mlock) ne sont pas compatibles MLX

**Verdict global :** ⭐⭐⭐ **Bonnes idées isolées** — mais l'orientation llama.cpp est un non catégorique pour NURU.

---

#### Proposition C — Gemini 3.1

**Forces :**
- Analyse la plus réaliste du problème RAM (macOS = 2-3 Go → 4.5-5.5 Go restants)
- Time-multiplexing strict : déchargement agressif entre chaque étape
- `mx.metal.clear_cache()` critique pour Apple Silicon bien identifié
- Recommandation de ne JAMAIS dépasser 3B paramètres (pragmatique)
- ColBERTv2 mentionné (intéressant mais complexe)
- Logging RSS du daemon (watchdog RAM)

**Faiblesses :**
- Propose un **routage via le LLM lui-même** (1 token) → 150ms de perte vs pre-classifier < 50ms, et nécessite de contraindre les logits
- Document le plus court (4 pages) — moins de détails d'implémentation
- Pas de roadmap claire, pas de budget RAM chiffré
- ColBERTv2 nécessite une intégration MLX complexe
- "Ne pas utiliser threading pour MLX" est trop strict — `asyncio.to_thread()` fonctionne bien

**Verdict global :** ⭐⭐⭐ **Meilleure analyse RAM** — complément idéal sur la partie gestion mémoire.

---

## 3. Contradictions Techniques et Résolution

### 3.1 Stack LLM : MLX vs llama.cpp vs Ollama

| Source | Recommandation | Justification |
|--------|---------------|---------------|
| Claude | **MLX + Ollama** hybride | MLX pour le quotidien, Ollama pour les modèles rares |
| Deepseek | **llama.cpp** (via llama-cpp-python) | KV cache, prompt cache, K-Quants, mlock |
| Gemini | **MLX** avec GC manuel agressif | Déjà intégré, zéro réécriture |

**Verdict : MLX.** Pour 3 raisons qui l'emportent sur tout le reste :

1. **Zéro réécriture** — router.py charge déjà `mlx_lm.load()`. Passer à llama.cpp = réécrire load, generate, stream_generate, tokenizer, prompt formatting.
2. **LoRA fine-tuning** — `mlx_lm.lora.train_model()` est natif. Pas d'équivalent simple en llama.cpp.
3. **Performance suffisante** — Qwen 3B Q4 fait 40-55 tok/s en MLX. Le gain marginal de llama.cpp (38-52 tok/s) n'est pas significatif.

**Les avantages llama.cpp (KV cache, K-Quants) ne justifient pas la réécriture.**

### 3.2 Routage : Pre-classifier vs DistilBERT vs LLM

| Source | Méthode | Coût | Latence |
|--------|---------|------|---------|
| Claude | Regex pur, LRU cache, < 50ms | 0 MB | < 50ms |
| Deepseek | DistilBERT quantifié, ~80 Mo | 80 MB | ~20ms |
| Gemini | LLM 1 token avec logit bias | 0 MB (utilise Qwen) | ~150ms |

**Verdict : Pre-classifier regex (Claude).**
- Le complexity_classifier actuel fait déjà 80% du travail. Il suffit de l'étendre avec les patterns du pre-classifier.
- DistilBERT ajoute 80 Mo de RAM permanent → inacceptable sur 8 Go.
- LLM 1 token = mobilise le modèle 3B pour une tâche qu'un script Python fait en 1ms → gaspillage.

**Extension du classifier existant :** enrichir `complexity_classifier.py` avec les patterns de `intent_classifier.py` plutôt que de créer un nouveau module.

### 3.3 Reranking : BM25 vs ONNX vs ColBERTv2

| Source | Méthode | RAM | Latence | Précision |
|--------|---------|-----|---------|-----------|
| Claude | BM25 + cosine hybride | 0 MB | < 10ms | ⭐⭐⭐⭐ (améliore le cosinus seul) |
| Deepseek | Cross-encoder ONNX (ms-marco) | ~100 MB | ~100ms | ⭐⭐⭐⭐⭐ |
| Gemini | ColBERTv2 Late Interaction | ~100 MB | ~50ms | ⭐⭐⭐⭐⭐ |

**Verdict : FlashRerank BM25 (Claude).** Zéro modèle chargé, maths pures, < 10ms. La précision BM25 + cosine est suffisante pour 6074 chunks sur un assistant personnel. Le cross-encoder et ColBERTv2 ajoutent 100 Mo + complexité d'intégration pour un gain marginal sur ce volume de données.

### 3.4 Vector DB : LanceDB vs ChromaDB amélioré vs SQLite-VSS

| Source | Solution | RAM | Setup |
|--------|----------|-----|-------|
| Claude | LanceDB (Arrow natif) | ~50 MB | Migration des 6074 chunks |
| Deepseek | ChromaDB existant amélioré | ~300 MB | Aucune migration |
| Gemini | SQLite-VSS (extension C) | ~20 MB | Réindexation complète |

**Verdict : LanceDB (Claude).** SQLite-VSS est plus léger mais moins mature (pas d'ANN index, recherche linéaire seulement). LanceDB offre IVF+PQ, requêtes hybrides BM25 natives, et zéro dépendance lourde (Arrow contre DuckDB). La migration (2-3 jours) est un investissement unique qui libère ~200 MB de RAM permanent.

### 3.5 Multi-agent : Orchestrateur vs Event Bus vs Mono-modèle

| Source | Approche | RAM | Complexité |
|--------|----------|-----|-----------|
| Claude | Model pool + orchestrateur | Faible (1 modèle chargé) | Moyenne |
| Deepseek | Event Bus + agents spécialisés | Élevée (plusieurs modèles) | Haute |
| Gemini | Mono-modèle à rôles multiples | Faible (1 seul modèle) | Faible |

**Verdict : Model Pool (Claude) avec des rôles.** L'Event Bus (Deepseek) est trop lourd pour un assistant mono-utilisateur. Le mono-modèle (Gemini) est trop limité. Le Model Pool de Claude avec 3 niveaux (nano/default/reasoning) + sélection par complexité = le meilleur compromis RAM/qualité.

---

## 4. Stack Technique

### 4.1 Recommandation finale

| Couche | Technologie | Justification |
|--------|------------|---------------|
| **LLM local** | MLX (mlx-lm) | Déjà intégré, LoRA natif, 40-55 tok/s, zéro réécriture |
| **LLM cloud** | Deepseek v4 Flash + fallback OpenRouter | Existant, fonctionne |
| **Embeddings** | nomic-embed-text-v1.5 via MLX | 768 dims, Matryoshka, multilingue, +15-20% qualité RAG |
| **Vector DB** | LanceDB | Arrow natif, IVF+PQ, BM25 natif, -200 MB vs ChromaDB |
| **Reranking** | FlashRerank BM25 (rank_bm25) | Zéro modèle, < 10ms, améliore le cosinus de 15-25% |
| **Chunking** | Small-to-Big (128→512) | Précision recherche +20-30%, contexte complet |
| **STT** | mlx-whisper base | 74 MB, WER ~10%, natif MLX, upgrade vs tiny actuel |
| **TTS** | Piper (inchangé) | Déjà installé, 50 MB, temps réel |
| **Wake word** | Porcupine (Picovoice) | < 1 MB, < 1% CPU, offline |
| **Memory** | SQLite (faits) + LanceDB (épisodes) + deque (working) | 3 niveaux, ~100 MB total |
| **Orchestration** | asyncio natif (pas de Trio/anyio) | Déjà compatible Python 3.11+, zéro dépendance |
| **RAM monitor** | psutil + mx.metal.clear_cache() + pmset (batterie) | Existant à enrichir |

### 4.2 Budget RAM cible

```
Composant                  Idle      Actif (nano)  Actif (3B)   Actif (phi-4)
────────────────────────────────────────────────────────────────────────────
macOS système              2.5 Go    2.5 Go        2.5 Go        2.5 Go
Intent classifier          0 Mo      0 Mo          0 Mo          0 Mo
Model pool (nano)          —         0.9 Go        —             —
Model pool (default)       —         —             1.8 Go        —
Model pool (phi-4-mini)    —         —             —             2.3 Go
Embedder nomic             —         0.27 Go       —             —
LanceDB                    0.05 Go   0.05 Go       0.05 Go       0.05 Go
Memory hiérarchique        0.02 Go   0.05 Go       0.05 Go       0.05 Go
PySide6 overlay            0.15 Go   0.15 Go       0.15 Go       0.15 Go
Piper TTS                  —         0.05 Go       0.05 Go       0.05 Go
────────────────────────────────────────────────────────────────────────────
TOTAL NURU                 ~0.2 Go   ~1.5 Go       ~2.1 Go       ~2.6 Go
TOTAL SYSTÈME + NURU       ~2.7 Go   ~4.0 Go       ~4.6 Go       ~5.1 Go
Marge disponible           5.3 Go    4.0 Go        3.4 Go        2.9 Go ✅
```

**Règle d'or :** jamais plus de **2 modèles** chargés simultanément dans la RAM. Les modèles STT/TTS sont chargés à la demande et immédiatement déchargés. L'embedder est déchargé après chaque recherche RAG.

---

## 5. Architecture Détaillée

### 5.1 Schéma logique complet

```
╔══════════════════════════════════════════════════════════════════════╗
║                     NURU V2 — Architecture Cible                    ║
╚══════════════════════════════════════════════════════════════════════╝

                                  [ENTRÉES]
  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
  │  CLI    │  │ PySide6  │  │ FastAPI  │  │ Porcupine    │
  │  chat   │  │ Overlay  │  │ WebUI    │  │ Wake Word    │
  └────┬────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘
       │            │             │               │
       └────────────┴─────────────┴───────────────┘
                              │
                              ▼
  ╔═══════════════════════════════════════════════════╗
  ║           ASYNCIO PIPELINE                        ║
  ║   (asyncio.create_task + Queue + gather)          ║
  ╚══════════╦═══════════════════════════════╦════════╝
             │                               │
  ┌──────────▼────────────┐    ┌─────────────▼──────────┐
  │  INTENT CLASSIFIER    │    │   RESOURCE MANAGER      │
  │  (extend complexity_ │    │   • RAM (psutil, 5s)    │
  │   classifier.py)      │    │   • Power (pmset)       │
  │  • Intent: chat/rag/  │    │   • LRU éviction        │
  │    web/code/action    │    │   • Thermal threads     │
  │  • Complexity: simple │    └─────────────────────────┘
  │    /medium/complex    │
  │  • Route → N1/N2/N3  │
  └──────────┬────────────┘
             │
  ┌──────────▼────────────────────────────────────────────┐
  │           ORCHESTRATOR (pipeline_v2.py)                │
  │                                                        │
  │   # Étape 1 : Parallèle (asyncio.gather)               │
  │   ╔══════════════╗  ╔══════════════════╗              │
  │   ║ RAG search   ║  ║ Memory retrieval║              │
  │   ║ (LanceDB)    ║  ║ (3 niveaux)     ║              │
  │   ╚══════════════╝  ╚══════════════════╝              │
  │                                                        │
  │   # Étape 2 : Select model (ModelPool)                 │
  │   # Étape 3 : Build prompt (dynamic context)           │
  │   # Étape 4 : Stream generate → token_queue            │
  │   # Étape 5 : Parallèle (display + TTS pipeline)       │
  └──────────┬────────────────────────────────────────────┘
             │
    ┌────────┼──────────────────────┐
    │        │                      │
    ▼        ▼                      ▼
  [N1]     [N2]                   [N3]
  RAG      LLM Local              CLOUD
  local    (no docs)              ┌─────────────────┐
    │        │                    │ Web Search       │
    │        │                    │ Tavily → Brave   │
    │        │                    │         ↓        │
    │        │                    │ Deepseek API     │
    │        │                    │ (ou OpenRouter)  │
    │        │                    └─────────────────┘
    │        │                      │
    └────────┴──────────────────────┘
             │
  ┌──────────▼────────────────────────────────────────────┐
  │               MODEL POOL (RAM-Aware + LRU)             │
  │   nano: Qwen1.5B Q4  │  default: Qwen3B Q4           │
  │   Lazy load on demand │  Toujours en cache            │
  │   reasoning: phi-4-mini Q4 (sur RAM dispo)            │
  │   Éviction: del model → gc → mx.metal.clear_cache()   │
  └──────────┬────────────────────────────────────────────┘
             │
  ┌──────────▼────────────────────────────────────────────┐
  │              MEMORY MANAGER (3 niveaux)                │
  │   Working (3 échanges verbatim)                       │
  │   Episodic (résumés → LanceDB)                       │
  │   Semantic (faits → SQLite FTS)                      │
  │   Auto-extract faits périodique                       │
  └───────────────────────────────────────────────────────┘
             │
  ┌──────────▼────────────────────────────────────────────┐
  │              RAG PIPELINE V2                           │
  │   SmallToBig Chunker (128→512)                        │
  │   nomic-embed-text v1.5 (768d, Matryoshka)           │
  │   LanceDB (IVF+PQ, BM25 natif)                       │
  │   FlashRerank BM25 + cosine (< 10ms)                  │
  └───────────────────────────────────────────────────────┘
             │
  ┌──────────▼────────────────────────────────────────────┐
  │              OUTPUT PIPELINE                           │
  │   Token streaming → Sentence boundary detection        │
  │   → Piper TTS (phrase par phrase)                     │
  │   → PySide6 overlay streaming                         │
  │   → Dataset collector → conversations.jsonl           │
  └───────────────────────────────────────────────────────┘
```

### 5.2 Flux de données détaillé (séquence temporelle)

```
t=0ms    User: "Quel est le prix actuel du riz à Kinshasa ?"
         │
t=1ms    CLI/Overlay → pipeline_v2.handle(query)
         │
t=2ms    Intent classifier (extend complexity_classifier):
         │  - Pattern match "prix actuel" → Intent.WEB_SEARCH
         │  - Route = "N3" (surcharge web)
         │  - Complexity = MEDIUM
         │  - Temps: < 1ms
         │
t=3ms    Orchestrator (asyncio.gather):
         ╔══════════════════════╗
         ║ Parallel Task 1:     ║   ╔══════════════════════╗
         ║ LanceDB search       ║   ║ Parallel Task 2:     ║
         ║ (embed via nomic)    ║   ║ Memory retrieval     ║
         ║ → 5-20ms             ║   ║ → 50ms               ║
         ║ → top-15 chunks      ║   ║ → faits + working    ║
         ╚══════════════════════╝   ╚══════════════════════╝
         │
t=25ms   FlashRerank BM25: 15 → top-5 (< 1ms)
         │
t=30ms   Build prompt (dynamic context, ~800 tokens)
         │
t=35ms   ModelPool.get_model(complexity=MEDIUM):
         │  - default (Qwen 3B) déjà chargé → instantané
         │  - max_tokens = 1024
         │
t=40ms   Stream generate (Deepseek API en parallèle déjà lancée)
         │  via asyncio.to_thread()
         │
t=50ms   TTFT : premier token affiché
         │
t=100ms  Première phrase complète détectée → Piper TTS lancé
         │
t=1000ms Réponse complète affichée + audio terminée
         │
         Dataset collector enregistre l'échange (async, fire & forget)
```

### 5.3 Gestion mémoire — Cycle de vie des modèles

```
État IDLE (aucune requête en cours) :
  ┌─────────────────────────────────────────┐
  │  Intent classifier     0 Mo   Résident  │
  │  Memory Manager        30 Mo  Résident  │
  │  LanceDB index         50 Mo  Résident  │
  │  PySide6 overlay       150 Mo Résident  │
  │  Piper TTS             0 Mo   Déchargé  │
  │  Model pool            0 Mo   Vide      │
  │  Embedder nomic        0 Mo   Déchargé  │
  │  TOTAL IDLE :          ~230 Mo          │
  └─────────────────────────────────────────┘

État REQUÊTE N1 (RAG local) :
  1. Load embedder nomic (+270 Mo) → embed query → LanceDB search
  2. Unload embedder → gc → mx.metal.clear_cache() → -270 Mo
  3. Load Qwen 3B (+1.8 Go) → build prompt → stream generate
  4. Après réponse : unload Qwen si idle > 30s → gc → clear_cache
  PIC RAM : ~2.3 Go (overlay + LanceDB + Qwen)

État REQUÊTE N3 (Cloud Deepseek) :
  1. Skip embedder (pas de RAG si web_search_mode)
  2. Use Qwen 3B si déjà chargé, sinon charger
  3. Appel API Deepseek (pas de RAM additionnelle)
  4. Stream réponse
  PIC RAM : ~1.8 Go (overlay + Qwen seul)

État REQUÊTE COMPLEXE (phi-4-mini) :
  1. Éviction LRU : décharger le modèle actuel (si 3B) → gc → clear_cache
  2. Load phi-4-mini (+2.3 Go)
  3. Stream generate
  4. Unload phi-4-mini → reload 3B pour la prochaine requête normale
  PIC RAM : ~2.6 Go
```

---

## 6. Fonctionnalités Retenues

### 6.1 Tableau de priorisation

| # | Fonctionnalité | Source | Utilité | RAM | Latence | Priorité | Difficulté | Dépendances |
|---|---------------|--------|---------|-----|---------|----------|------------|-------------|
| 1 | **Pipeline async** | Claude | TTFT -40%, parallélise RAG/classify | +0 Mo | -200 à -600ms | 🔴 P0 | Haute | asyncio |
| 2 | **Intent pre-classifier** | Claude | Routage sans LLM, < 50ms, supprime `[[ESCALADE]]` | +0 Mo | -800 à -2500ms | 🔴 P0 | Moyenne | Extension complexity_classifier |
| 3 | **Embedder nomic-768d** | Claude+Gemini | +15-20% qualité RAG, -400MB PyTorch | +270 Mo (temporaire) | -50ms par requête | 🔴 P0 | Moyenne | Migration LanceDB |
| 4 | **LanceDB** | Claude | -200 MB RAM, BM25 natif, IVF+PQ | -200 MB permanent | -30ms/requête | 🔴 P0 | Haute | Réindexation 6074 chunks |
| 5 | **Model Pool (nano/default/reasoning)** | Claude | -900MB sur requêtes simples, qualité raisonnement | Variable | +30% vitesse | 🟠 P1 | Moyenne | Intent classifier |
| 6 | **Resource Manager RAM + thermal** | Claude+Gemini | Zéro swap, stabilité thermique | +5 Mo | Stable garanti | 🔴 P0 | Faible | Model Pool |
| 7 | **FlashRerank BM25** | Claude | +15-25% précision RAG, zéro modèle | +0 Mo | < 10ms | 🟠 P1 | Faible | Aucune |
| 8 | **Mémoire hiérarchique 3 niveaux** | Claude+Deepseek | Contexte long cohérent, -60% tokens mémoire | +30 Mo | +50ms retrieval | 🟠 P1 | Haute | LanceDB |
| 9 | **Small-to-Big chunker** | Claude | +20-30% précision retrieval | +0 Mo (storage) | +0ms | 🟠 P1 | Moyenne | Réindexation |
| 10 | **Dynamic context window** | Deepseek | -40% RAM KV cache sur requêtes simples | Variable | -100ms TTFT | 🟡 P2 | Faible | Intent classifier |
| 11 | **Prompt prefix caching (MLX)** | Deepseek+Claude | -150 à -300ms TTFT par requête | +0 Mo | -150ms TTFT | 🟡 P2 | Moyenne | MLX v0.18+ |
| 12 | **mlx-whisper base** | Claude+Gemini | WER 10% au lieu de 25-30% | +35 Mo (temporaire) | +0ms | 🟡 P2 | Faible | Téléchargement modèle |
| 13 | **Interruption intelligente** | Claude | UX fluide, annulation propre | +0 Mo | +0ms | 🟡 P2 | Faible | asyncio |
| 14 | **Wake word Porcupine** | Claude+Deepseek | Activation vocale mains-libres | +5 Mo permanent | < 50ms détection | 🟡 P2 | Moyenne | Microphone |

### 6.2 Fonctionnalités abandonnées

| Idée | Source | Raison de l'abandon |
|------|--------|---------------------|
| **Event Bus Trio/anyio** | Deepseek | Overkill pour mono-utilisateur. asyncio natif suffit. |
| **DistilBERT intent classifier** | Deepseek | 80 Mo permanent vs regex 0 Mo. Même précision sur 6 catégories. |
| **Cross-encoder ONNX** | Deepseek | 100 Mo + 100ms vs BM25 0 Mo + 10ms. Gain marginal sur 6074 chunks. |
| **ColBERTv2** | Gemini | Intégration MLX complexe, non testé, 100 Mo. Pas justifié. |
| **llama.cpp comme primary** | Deepseek | Réécriture complète, perte LoRA MLX, pas de gain significatif. |
| **Swap intelligent (mlock)** | Deepseek | Pas compatible MLX. Les allocations Metal sont déjà gérées par le driver. |
| **K-Quants / Q4_K_M** | Deepseek | MLX ne supporte pas les GGUF K-Quants. Les Q4 MLX sont déjà performants. |
| **T5-small pour compression** | Deepseek | Modèle supplémentaire à charger, gain marginal sur les chunks déjà petits. |
| **Qwen-7B raisonnement** | Claude | 3.8 Go. Trop risqué sur 8 Go. phi-4-mini (2.3 Go) offre 90% de la qualité. |
| **Gemini fallback** | NURU V1 | Clé non configurée, doublon avec OpenRouter. Supprimer. |
| **Vision (moondream2/Qwen2-VL)** | Claude+Gemini | 1.1 Go à charger à la demande. Pas prioritaire. Remis à V3. |
| **Graph memory (KùzuDB)** | Gemini | Complexité inutile pour un assistant mono-utilisateur. SQLite + LanceDB suffisent. |

---

## 7. Plan de Développement par Phases

### Phase 0 — Préparation (1 semaine)

**Avant de coder :**
```
✅ Valider cette roadmap avec Leblanc
✅ Sauvegarder l'état actuel de V1 (git tag v1.0)
✅ Lister toutes les API keys disponibles (Deepseek, Tavily, Brave, OpenRouter)
✅ Vérifier les chemins et dépendances actuelles
✅ Calibrer : mesurer les latences réelles de V1 pour benchmark
```

---

### Phase V2.0 Alpha — Fondations critiques (4-6 semaines)

**Objectif : résoudre les 3 goulets les plus coûteux.**

| Semaine | Module | Description | Fichiers |
|---------|--------|-------------|----------|
| **S1** | **Intent classifier** | Étendre `complexity_classifier.py` avec les 6 intents + patterns web enrichis | `complexity_classifier.py` |
| **S2** | **Pipeline async** | Refacto `router.py` → `pipeline_v2.py` avec asyncio.gather() + token_queue | `pipeline_v2.py`, `router.py` |
| **S3** | **LanceDB + nomic-embed** | Migration ChromaDB → LanceDB. Embedder nomic-768d. Réindexation 6074 chunks. | `vector_store_v2.py`, `embedder_v2.py` |
| **S4** | **FlashRerank BM25** | `reranker_v2.py` — intégré dans le pipeline RAG | `reranker_v2.py` |
| **S5** | **Model Pool** | `model_pool_v2.py` — nano/default/reasoning + LRU eviction | `model_pool_v2.py` |
| **S6** | **Resource Manager** | `resource_manager_v2.py` — RAM + thermal + power mode. Tests intégration. | `resource_manager_v2.py` |

**Livrable V2.0 Alpha :** NURU tourne avec la nouvelle stack. 30-40% RAM en moins. TTFT réduit. Zéro swap.

---

### Phase V2.1 Beta — Intelligence & UX (3-4 semaines)

**Objectif : améliorer la qualité des réponses et l'expérience.**

| Semaine | Module | Description |
|---------|--------|-------------|
| **S7** | **Mémoire hiérarchique** | `memory_v2.py` — working (3) / episodic (résumés) / semantic (SQLite) |
| **S8** | **Small-to-Big chunker** | `chunker_v2.py` — réindexation avec chunks 128→512 |
| **S9** | **Dynamic context + Prompt caching** | Ajustement `max_tokens`, KV cache promo système |
| **S10** | **Audio pipeline** | mlx-whisper base, Piper streaming phrase, interruption |

**Livrable V2.1 Beta :** NURU comprend les conversations longues. Streaming TTS phrase par phrase. Qualité RAG améliorée.

---

### Phase V2.2 Stable — Spécialisation (2-3 semaines)

**Objectif : finition, robustesse, déploiement.**

| Semaine | Module | Description |
|---------|--------|-------------|
| **S11** | **phi-4-mini reasoning** | Intégration dans le model pool. Routage des requêtes complexes. |
| **S12** | **Wake word + mode vocal** | Porcupine "Hey NURU". Activation vocale complète. |
| **S13** | **Stabilisation + tests + déploiement** | Tests de charge 8 Go, regression, documentation, déploiement V2 |

**Livrable V2.2 Stable :** NURU V2 en production. Wake word, streaming, model pool, zéro swap.

---

### Phase V3.0 — Autonomie (trimestre suivant)

```
V3 Objectifs :
  □ Tool calling structuré (JSON schema + Outlines)
  □ Agent email via Gmail/Himalaya (déjà connecté)
  □ Agent fichiers (surveillance + synthèse automatique)
  □ Fine-tuning LoRA mensuel sur dataset collecté
  □ Vision à la demande (moondream2 via Ollama)
  □ Interface Shortcut macOS native (remplacement Hammerspoon)
```

---

## 8. Recommandations Modèles

### 8.1 Tableau de sélection

| Usage | Modèle | Format | RAM | tok/s | Priorité |
|-------|--------|--------|-----|-------|----------|
| **Nano (simple)** | Qwen2.5-1.5B-Instruct | MLX Q4 | 0.9 Go | 65-80 | P1 |
| **Default (général)** | Qwen2.5-3B-Instruct | MLX Q4 | 1.8 Go | 40-55 | P0 (déjà en place) |
| **Reasoning (complexe)** | phi-4-mini-instruct | MLX Q4 | 2.3 Go | 25-32 | P2 |
| **Code** | Qwen2.5-Coder-1.5B | MLX Q4 | 0.9 Go | 65-80 | P3 |
| **STT** | mlx-whisper/base | MLX | 74 MB | Temps réel | P2 |
| **TTS** | Piper (fr_FR-siwis-medium) | ONNX | 50 MB | Temps réel | P0 (déjà en place) |
| **Embeddings** | nomic-embed-text-v1.5 | MLX | 274 MB | 5-15ms/req | P0 |
| **Wake word** | Porcupine | ONNX | 1 MB | < 50ms | P2 |

### 8.2 Règle de sélection automatique (Model Pool)

```python
complexity = classify_intent(query).complexity
ram_free_gb = psutil.virtual_memory().available / (1024**3)

if complexity == SIMPLE:
    model = "nano"        # Qwen 1.5B — 0.9 Go, 70 tok/s
elif complexity == COMPLEX and ram_free_gb > 4.0:
    model = "reasoning"   # phi-4-mini — 2.3 Go, 28 tok/s
else:
    model = "default"     # Qwen 3B — 1.8 Go, 45 tok/s
```

---

## 9. Pièges à Éviter sur Apple Silicon 8 Go

### ⚠️ Critique — Bloque la V2

| Piège | Détail | Solution |
|-------|--------|----------|
| **Swap immédiat** | 2 modèles chargés + macOS > 8 Go → latence 1-2 tok/s | Model pool avec éviction stricte. Un seul modèle à la fois. |
| **PyTorch résident** | `sentence-transformers` = 400 MB même inactif | Migrer vers nomic-embed via MLX (zéro PyTorch). |
| **ChromaDB en mémoire** | DuckDB + SQLite = 200-300 MB permanent | LanceDB = 50 MB. |
| **FastAPI toujours actif** | ~150 MB même sans client WebUI | Lazy start. |
| **MLX threads par défaut** | Tous les cœurs → thermal throttling 5-10 min | `MLX_NUM_THREADS=4` sur batterie, `6` sur AC. |
| **Cache KV non géré** | 2048 tokens → ~0.5 Go KV cache par session | Dynamic context window. max_tokens adaptatif. |

### ⚠️ Important — Dégradation progressive

| Piège | Détail | Solution |
|-------|--------|----------|
| **Fuites mémoire Metal** | `mx.metal.clear_cache()` jamais appelé | Appel systématique après déchargement de modèle |
| **Accumulation cache sémantique** | Embeddings stockés indéfiniment | TTL + purge périodique |
| **Historique non résumé** | Fenêtre de 5 échanges bruts → 2000 tokens | Mémoire hiérarchique : working 3 échanges + résumé |
| **Thermal throttling batterie** | Inférence intensive sans limite | Power mode detection (pmset) + réduction threads |

### ⚠️ UX — Expérience utilisateur

| Piège | Détail | Solution |
|-------|--------|----------|
| **TTFT long** | 2000ms avant de voir le premier mot | Intent pre-classifier + pipeline async + prompt caching |
| **TTS après la fin** | L'utilisateur lit puis entend | Streaming phrase par phrase : audio en parallèle du texte |
| **Impossible d'interrompre** | Nouvelle question pendant génération → confusion | InterruptibleGenerator avec cancellation token |
| **Swap pendant utilisation** | Latence qui passe de 45 tok/s à 2 tok/s | Resource manager avec seuil d'éviction à 1.5 Go libre |

---

## Résumé Exécutif

```
V2.0 Alpha (semaines 1-6) :
├── Intent classifier pre-LLM (50ms, 0 Mo)
├── Pipeline async (TTFT -40%)
├── LanceDB + nomic-embed (-200 Mo RAM, +15% RAG)
├── FlashRerank BM25 (0 Mo, <10ms)
├── Model Pool (nano/default/reasoning)
└── Resource Manager (swap zéro)

V2.1 Beta (semaines 7-10) :
├── Mémoire hiérarchique 3 niveaux
├── Small-to-Big chunker (+20% précision)
├── Dynamic context + prompt caching
└── Audio pipeline (mlx-whisper, Piper streaming)

V2.2 Stable (semaines 11-13) :
├── phi-4-mini reasoning
├── Wake word Porcupine
└── Tests + déploiement V2

Stack finale : MLX (LLM + embeddings) + LanceDB (vector) + BM25 (rerank)
Budget RAM (idle/actif) : 0.2 / 2.1 Go
Gains estimés : TTFT -40-60%, RAM -30-40%, Précision RAG +15-25%
```

---

*Document généré le 8 mai 2026 — Analyse critique de 3 propositions IA*
*Basé sur NURU V1 — Projet conçu et piloté par Leblanc BAHIGA Mudarhi*
