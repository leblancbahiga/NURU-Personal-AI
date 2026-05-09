# NURU V2 — Roadmap des améliorations

> Mise à jour : 5 mai 2026 (v2)
> Sources : Propositions d'amélioration.pdf + suggestions séance du 5 mai

---

## ✅ V1 déjà appliqué (ce jour)

| # | Amélioration | Fichiers | Statut |
|---|-------------|----------|--------|
| 1 | Signaux d'escalade Regex + Stop Words MLX | `router.py` | ✅ |
| 2 | Cache sémantique avec TTL (N3=1h, N1/N2=30j) | `semantic_cache.py`, `router.py` | ✅ |
| 3 | Query rewriting pour le RAG | `router.py` | ✅ |
| 4 | Mots-clés multilingues EN + SW | `complexity_classifier.py` | ✅ |
| 5 | Cache TTL + Retry Brave Search | `brave_search.py` | ✅ |

## ✅ Vérifié : déjà dans le code

Ces points suggérés existent déjà — rien à faire :

| Suggestion | Où | Vérification |
|-----------|-----|-------------|
| Poids classifier synchronisés doc/code | `complexity_classifier.py:108-117` | 0.25/0.40/0.20/0.15 identiques au PDF |
| Historique {history} injecté dans N1 et N2 | `router.py:_format_local_prompt()` lignes 625-627, 666-668 | `self.memory.get_context()` appelé dans les deux branches |

---

## 🔴 Immédiat — Correctifs critiques

### I1. Mots-clés multilingues (EN + SW minimum)
**Statut : ✅ FAIT (5 mai 2026)**
**Où :** `src/complexity_classifier.py`
**Quoi :** Listes EN + SW ajoutées (`KEYWORDS_COMPLEX_EN/SW`, `KEYWORDS_SIMPLE_EN/SW`,
`KEYWORDS_WEB_EN/SW`). Les trois langues sont scorées simultanément — pas de
détection de langue nécessaire. ~160 mots-clés au total.

### I2. Fallback OpenRouter si Deepseek échoue
**Statut : ✅ FAIT (5 mai 2026)**
**Où :** `src/router.py` — `_execute_cloud()` et nouveau `_call_openrouter()`
**Quoi :** Si Deepseek retourne une erreur (timeout, clé manquante, rate-limit),
OpenRouter est appelé automatiquement en fallback. Supporte aussi le provider
`"openrouter"` direct dans la config.
**Config :** `config.yaml` → `models.cloud.openrouter_model` (défaut : `google/gemma-4-31b-it:free`)
**Clé API :** à mettre dans le trousseau macOS avec
```
security add-generic-password -s 'com.nuru.assistant' -a 'openrouter' -w 'ta_clé'
```

---

## 🟠 Court terme — Améliorations légères

### C1. Retry logic + TTL cache pour Brave Search
**Statut : ✅ FAIT (5 mai 2026)**
**Où :** `src/brave_search.py` — méthode `BraveSearchClient.search()`
**Quoi :**
1. Cache TTL (5 min) intégré directement dans `BraveSearchClient` — évite les appels
   redondants pour la même requête
2. Retry ×2 avec backoff exponentiel (1.5s, 2.25s) sur timeouts et erreurs HTTP
3. Nettoyage auto des entrées expirées à chaque nouveau `search()`

### C2. Documenter action_engine.py
**Où :** `docs/action_engine.md` (nouveau fichier)
**Pourquoi :** `action_engine.py` existe dans l'arborescence (non commité) mais pas
documenté. C'est le moteur qui exécute des actions locales (créer dossier, exécuter
scripts) en parsant la réponse du LLM.
**Quoi :** Documentation de l'interface `Tool`, du registry, et comment ajouter
une nouvelle action.
**Effort :** ~20 min

### C3. Documenter la boucle feedback → fine-tuning LoRA
**Où :** `docs/auto_amelioration.md` (nouveau fichier)
**Pourquoi :** `dataset_collector.py` et `feedback.py` existent mais le flux complet
n'est documenté nulle part. C'est le cœur de l'auto-amélioration.
**Quoi :** Documenter le pipeline :
```
Réponse NURU → feedback utilisateur → priority_corrections (ChromaDB)
→ dataset_collector.py → conversations.jsonl → fine-tuning MLX LoRA
```
Inclure : format du dataset, fréquence de fine-tuning recommandée, comment lancer
le fine-tuning, comment évaluer les résultats.
**Effort :** ~30 min

---

## 🟡 Moyen terme — Améliorations significatives

*(les items P1, P2 de la version précédente sont renumérotés ici)*

### M1. Context compression pour le RAG
**Gain estimé :** −40% tokens, +20% vitesse, +30% précision
**Où :** `src/rag.py` — après `search()`, avant `format_context()`
**Quoi :** Résumer les chunks RAG avant injection dans le prompt LLM. Le Qwen 3B peut
générer un résumé de 200 tokens à partir de 500 tokens de contexte.
**Risque :** Perte d'information si le résumé est trop agressif — garder le chunk
original en fallback si le score de confiance du résumé est faible.

```python
def compress_context(chunks: str, llm_model, llm_tokenizer) -> str:
    prompt = f"Résume les informations suivantes en gardant uniquement les faits utiles:\n{chunks}"
    return generate(model=llm_model, tokenizer=llm_tokenizer, prompt=prompt, max_tokens=256)
```

### M2. Détection de langue (5ᵉ critère du classifier)
**Où :** `src/complexity_classifier.py`
**Quoi :** Ajouter un 5ᵉ score basé sur la langue détectée (via `langdetect` ou
`lingua-py`). Coût quasi nul (~1ms). La langue influence les mots-clés actifs :
- FR → mots-clés français
- EN → mots-clés anglais
- SW → mots-clés swahili
- Autre → mode « mots-clés désactivés », se fier aux autres critères

**Mécanisme :**
```python
def score_langue(query: str) -> tuple[float, str]:
    lang = detect(query)  # 'fr', 'en', 'sw', ...
    # Assigner les listes de mots-clés selon la langue détectée
    keywords = LANG_KEYWORDS.get(lang, KEYWORDS_FR)
    return compute_keyword_score(query, keywords), lang
```

**Effort :** ~1h (incluant installation de lingua-py)

### M3. Self-reflection / Double-pass LLM (N3 uniquement)
**Gain estimé :** +25% qualité sans changer de modèle
**Où :** `src/router.py` — après `_execute_cloud()`
**Quoi :** Pour les réponses N3 (Deepseek), faire un second appel avec un prompt
de vérification : « Vérifie cette réponse et corrige les erreurs. »
**Risque :** Double la latence N3 (3s → 6s). N'activer que si la latence n'est pas
critique (pas en mode vocal/overlay temps réel).

### M4. Évaluateur automatique de réponses
**Gain estimé :** Dataset de qualité pour le fine-tuning LoRA
**Où :** Nouveau module `src/evaluator.py`
**Quoi :** Après chaque réponse, le modèle Deepseek attribue un score 0-10.
Les réponses < 4 sont taguées [ERROR] et archivées pour le dataset.
Utile pour détecter les dérives et nourrir le pipeline `fine_tune.py`.
**Dépend :** `dataset_collector.py` (existe déjà, non commité)

```python
def evaluate_response(query, response, llm_client):
    prompt = f"Évalue cette réponse (0 à 10):\nQuestion: {query}\nRéponse: {response}\nScore:"
    result = llm_client.generate(prompt)
    return extract_score(result)
```

### M5. Stratégie RAM explicite
**Où :** `src/monitor.py` + `src/router.py` — logique de throttling
**Pourquoi :** `max_ram_gb = 3.0` est configuré dans `config.yaml` mais personne ne
vérifie ce qu'il se passe quand la RAM dépasse. Actuellement : rien.
**Quoi :** Définir une stratégie explicite :
1. RAM > 70% → désactiver le préchargement de l'embedder RAG
2. RAM > 80% → décharger le modèle local, forcer N3 (cloud) pour toutes les requêtes
3. RAM > 90% → vider le cache sémantique, décharger l'embedder

Ces seuils doivent être configurables dans `config.yaml` et loggés.

### M6. HyDE (Hypothetical Document Embeddings)
**Où :** `src/rag.py` — avant la recherche ChromaDB
**Quoi :** Au lieu de chercher avec l'embedding de la requête, demander au Qwen 3B
de générer une « réponse hypothétique », puis chercher avec l'embedding de cette
réponse. Améliore le rappel pour les questions abstraites.
**Note :** Complémentaire au Query Rewriting déjà implémenté en V1 (HyDE est plus
lourd mais plus puissant).

### M7. Filtrage par métadonnées temporelles dans le RAG
**Où :** `src/rag.py` — méthode `search()`
**Quoi :** Si la requête contient « 2025 », « récent », « cette année », ajouter un
filtre ChromaDB `$gt` sur `created_at` pour exclure les documents trop vieux.
**Effort :** ~30 min

### M8. Parallélisation async RAG + Classification
**Où :** `src/router.py` — flux `stream_route()`
**Quoi :** Lancer la recherche RAG et la classification de complexité en parallèle
plutôt qu'en séquence. Gain : −30% à −60% de latence sur le chemin critique.
**Problème :** L'overlay PySide6 utilise des threads Qt — l'async natif (`asyncio`)
n'est pas compatible. Solution : utiliser `concurrent.futures.ThreadPoolExecutor`
ou un pattern producteur-consommateur avec des queues.
**Effort :** 2-3h de refacto.

### M9. Mémoire active avec scoring
**Où :** `src/structured_memory.py`
**Quoi :** Ajouter un `importance_score` (0.0-1.0) et un `last_used` timestamp à
chaque fait mémoire. Le retrieval utilise `score = importance * 0.7 + recency * 0.3`.
Les faits rarement utilisés et peu importants sont purgés automatiquement.
**Note :** Complémentaire à l'existant — la structure actuelle stocke sans scoring.

### M10. Tool Use générique
**Où :** `src/action_engine.py` (existe déjà, non commité)
**Quoi :** Au-delà de Brave Search, ajouter : calculatrice, météo via API, rappels
macOS, exécution de scripts système sécurisés.
**Déjà partiellement fait :** `action_engine.py` listé dans l'arborescence.
**À faire :** Définir une interface `Tool` standard et un registry.

---

## 🔵 Vision V3 — Trop tôt ou trop lourd

| # | Idée | Pourquoi pas maintenant |
|---|------|------------------------|
| V1 | Router adaptatif (ML) | Pas assez de données (< 300 requêtes étiquetées) |
| V2 | Parent Document Retrieval | RAG actuel satisfaisant, complexifie l'ingestion |
| V3 | Planner / Raisonnement multi-step | Latence ×2-5s, pas adapté à un assistant vocal temps réel |
| V4 | Speculative decoding | Ne fonctionne bien qu'avec des paires de modèles compatibles |
| V5 | GraphRAG | Overkill pour Qwen 3B local sur Apple Silicon |
| V6 | Mixture-of-models (Qwen + Phi) | OOM garanti sur 8/16GB |

---

## Résumé exécutif

```
V1 (fait)          → 🔴 Immédiat          → 🟠 Court terme       → 🟡 Moyen terme
──────────────────────────────────────────────────────────────────────────────────
✓ Regex escalade     I1. Mots-clés EN+SW    C1. Retry Brave        M1. Context compression
✓ TTL cache          I2. Fallback Gemini    C2. Doc action_engine  M2. Détection langue
✓ Query rewriting                          C3. Doc boucle FT      M3. Self-reflection N3
                                                                   M4. Évaluateur auto
                                                                   M5. Stratégie RAM
                                                                   M6. HyDE
                                                                   M7. Filtrage temporel
                                                                   M8. Parallélisation
                                                                   M9. Mémoire scorée
                                                                   M10. Tool Use
```

La V2 peut être développée en parallèle du pipeline d'auto-amélioration
(`dataset_collector.py` + `fine_tune.py`). Les deux s'alimentent mutuellement :
l'évaluateur (M4) nourrit le dataset, et le fine-tuning améliore les réponses
évaluées.
