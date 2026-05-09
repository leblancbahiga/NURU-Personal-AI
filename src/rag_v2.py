#!/usr/bin/env python3
"""
rag_v2.py — RAG V2 composants avancés pour NURU.

Classes :
  - TemporalRAGFilter : Filtrage/re-scoring temporel des résultats RAG
  - SmallToBigSearch  : Recherche small-chunk → retour parent-chunk

Dépendances :
  - rag.py      : VectorStore, get_embedder, _compute_recency_weight
  - chunker_v2.py : SmallToBigChunker, SmallChunk, ParentChunk
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("nuru.v2.rag")

# ── Imports optionnels depuis rag.py ──
try:
    from rag import VectorStore, get_embedder, _compute_recency_weight
    RAG_V1_AVAILABLE = True
except ImportError:
    logger.warning("rag.py (V1) non disponible — mode standalone limité")
    RAG_V1_AVAILABLE = False

    def _compute_recency_weight(created_at: float, now: float = None) -> float:
        """Fallback : poids de récence identique à rag.py."""
        if now is None:
            now = time.time()
        age = now - created_at
        seven_days = 7 * 24 * 3600
        ninety_days = 90 * 24 * 3600
        if age <= seven_days:
            return 1.0
        elif age >= ninety_days:
            return 0.3
        return 1.0 - (age - seven_days) / (ninety_days - seven_days) * 0.7


# ── Imports optionnels depuis chunker_v2.py ──
try:
    from chunker_v2 import SmallToBigChunker, SmallChunk, ParentChunk, get_chunker
    CHUNKER_V2_AVAILABLE = True
except ImportError:
    logger.warning("chunker_v2.py non disponible")
    CHUNKER_V2_AVAILABLE = False


# =============================================================================
# TemporalRAGFilter
# =============================================================================

class TemporalRAGFilter:
    """
    Filtre temporel pour les résultats RAG.

    Prend les résultats bruts de VectorStore.search() et leur ajoute un score
    temporel composé. Peut soit :

      - **Fusion** (mode='rerank') : Re-score les résultats avec un hybride
        embedding + recence, puis re-trie (défaut).
      - **Filtrer** (mode='filter') : Supprime les résultats plus vieux qu'un
        seuil (max_age_days).
      - **Marquer** (mode='annotate') : Ajoute le score temporel aux métadonnées
        sans re-trier.

    Le score temporel est calculé via _compute_recency_weight() — poids maximal
    1.0 pour les 7 premiers jours, décroît linéairement jusqu'à 0.3 à 90 jours.

    Usage :
        filter = TemporalRAGFilter(temp_weight=0.25, max_age_days=365)
        results = store.search("question")
        filtered = filter.apply(results, query="question")
    """

    def __init__(
        self,
        temp_weight: float = 0.25,
        embedding_weight: float = 0.55,
        keyword_weight: float = 0.20,
        max_age_days: Optional[int] = None,
        mode: str = "rerank",
        min_recency: float = 0.0,
    ):
        """
        Args:
            temp_weight: Poids du score temporel dans le score hybride (0.0-1.0)
            embedding_weight: Poids du score d'embedding (0.0-1.0)
            keyword_weight: Poids du keyword match (0.0-1.0)
            max_age_days: Si mode='filter', supprime les docs plus vieux que N jours
            mode: 'rerank' | 'filter' | 'annotate'
            min_recency: Score de récence minimum (0.0-1.0) — supprime les docs
                         en-dessous si mode='filter'
        """
        if mode not in ("rerank", "filter", "annotate"):
            raise ValueError(f"mode doit être 'rerank', 'filter' ou 'annotate', got '{mode}'")
        self.temp_weight = temp_weight
        self.embedding_weight = embedding_weight
        self.keyword_weight = keyword_weight
        self.max_age_days = max_age_days
        self.mode = mode
        self.min_recency = min_recency

        # Normalisation : les poids doivent sommer à 1.0 si on utilise l'hybride
        total = self.temp_weight + self.embedding_weight + self.keyword_weight
        if total > 0 and abs(total - 1.0) > 0.01:
            logger.debug(
                "Poids normalisés : temp=%.2f emb=%.2f kw=%.2f (somme=%.2f)",
                self.temp_weight, self.embedding_weight, self.keyword_weight, total,
            )

    def apply(
        self,
        results: list[dict],
        query: str = "",
    ) -> list[dict]:
        """
        Applique le filtre temporel sur une liste de résultats RAG.

        Args:
            results: Liste de dicts de VectorStore.search() — chaque dict doit
                     contenir {"score": float, "metadata": {"created_at": float}, ...}
            query: Requête originale (utilisée pour le keyword match si fournie)

        Retourne:
            Liste filtrée/re-triée des résultats (même structure que l'entrée,
            avec un champ 'temporal_score' et 'hybrid_components' enrichis).
        """
        if not results:
            return []

        now = time.time()
        max_age_sec = (self.max_age_days * 86400) if self.max_age_days else None

        enriched = []

        for r in results:
            md = r.get("metadata", {})
            created_at = md.get("created_at", now)
            orig_score = r.get("score", 0.0)

            # Score temporel
            recency_score = _compute_recency_weight(created_at, now)

            # Vérifier l'âge max
            if max_age_sec is not None and (now - created_at) > max_age_sec:
                if self.mode == "filter":
                    continue  # Trop vieux, on supprime

            # Vérifier le min_recency
            if self.mode == "filter" and recency_score < self.min_recency:
                continue

            # Score hybride enrichi
            components = r.get("hybrid_components", {})
            emb_score = components.get("embedding", orig_score)

            # Keyword match si pas déjà calculé
            kw_score = components.get("keyword", 0.0)
            if not kw_score and query:
                from rag import _compute_keyword_match
                kw_score = _compute_keyword_match(query, r.get("text", ""))

            # Score hybride temporel
            total_weight = self.temp_weight + self.embedding_weight + self.keyword_weight
            if total_weight > 0:
                hybrid = (
                    self.embedding_weight * emb_score
                    + self.keyword_weight * kw_score
                    + self.temp_weight * recency_score
                ) / total_weight
            else:
                hybrid = orig_score

            r["temporal_score"] = round(recency_score, 4)
            r["hybrid_components"] = {
                "embedding": round(emb_score, 4),
                "keyword": round(kw_score, 4),
                "recency": round(recency_score, 4),
                "temporal_hybrid": round(hybrid, 4),
            }

            if self.mode in ("rerank", "filter"):
                # Remplacer le score principal par le score hybride temporel
                r["score"] = round(hybrid, 4)
                r["_orig_score"] = round(orig_score, 4)

            enriched.append(r)

        if self.mode in ("rerank",):
            enriched.sort(key=lambda x: x["score"], reverse=True)

        return enriched

    def __repr__(self) -> str:
        return (
            f"TemporalRAGFilter(mode={self.mode}, "
            f"temp_weight={self.temp_weight}, "
            f"max_age_days={self.max_age_days})"
        )


# =============================================================================
# SmallToBigSearch
# =============================================================================

class SmallToBigSearch:
    """
    Recherche Small-to-Big pour NURU V2.

    Stratégie :
      1. Les **petits chunks** (128 tokens) sont indexés dans ChromaDB pour
         une recherche de haute précision.
      2. Lors d'une requête, on cherche les petits chunks les plus pertinents.
      3. Pour chaque petit chunk trouvé, on remonte au **chunk parent**
         (512 tokens) qui contient le contexte élargi.
      4. On retourne les chunks parents dédupliqués comme contexte final.

    Le lien small → parent est assuré par le champ 'parent_id' stocké dans
    les métadonnées du petit chunk dans ChromaDB.

    Les chunks parents peuvent être stockés :
      - Soit dans une collection ChromaDB dédiée ('parent_chunks')
      - Soit dans un dictionnaire en mémoire (parent_store) passé à l'init

    Usage :
        from chunker_v2 import SmallToBigChunker
        from rag import VectorStore

        store = VectorStore()
        chunker = SmallToBigChunker()
        s2b = SmallToBigSearch(store, chunker)
        s2b.index_parent_chunks(text, source="doc.pdf")
        results = s2b.search("Qu'est-ce que NURU ?", k=3)
        context = s2b.format_context(results)
    """

    # Nom de la collection ChromaDB dédiée aux chunks parents
    PARENT_COLLECTION_NAME = "parent_chunks"

    # Métadonnée stockée dans chaque petit chunk ChromaDB
    PARENT_ID_META_KEY = "parent_id"

    def __init__(
        self,
        vector_store: object,
        chunker: Optional[object] = None,
        use_chromadb_parent_store: bool = True,
        parent_store: Optional[Dict[str, str]] = None,
        temporal_filter: Optional[TemporalRAGFilter] = None,
        enable_dedup: bool = True,
    ):
        """
        Args:
            vector_store: Instance de rag.VectorStore (ou compatible)
            chunker: Instance de SmallToBigChunker (optionnel, utilisé pour
                     chunk_document si index_parent_chunks est appelé)
            use_chromadb_parent_store: Stocker les parents dans une collection
                                       ChromaDB dédiée (True) ou en mémoire (False)
            parent_store: Dict pré-rempli {parent_id → parent_text} (optionnel)
            temporal_filter: Instance de TemporalRAGFilter pour post-filtrage
            enable_dedup: Déduplication des chunks parents (par défaut True)
        """
        self.vector_store = vector_store
        self.chunker = chunker
        self.use_chromadb_parent_store = use_chromadb_parent_store
        self.enable_dedup = enable_dedup
        self.temporal_filter = temporal_filter

        # Cache en mémoire parent_id → parent_text
        self._parent_store: Dict[str, str] = {}
        if parent_store:
            self._parent_store.update(parent_store)

        # Collection ChromaDB pour les parents (si use_chromadb_parent_store)
        self._parent_collection = None
        if use_chromadb_parent_store:
            self._init_parent_collection()

        # Lock thread-safe pour le cache mémoire
        self._lock = threading.Lock()

        logger.info(
            "SmallToBigSearch initialisé (use_chromadb=%s, dedup=%s, temporal=%s)",
            use_chromadb_parent_store, enable_dedup,
            temporal_filter is not None,
        )

    def _init_parent_collection(self):
        """Initialise la collection ChromaDB des chunks parents."""
        if not RAG_V1_AVAILABLE:
            logger.warning("ChromaDB non disponible pour le stockage des parents")
            self.use_chromadb_parent_store = False
            return
        try:
            # Accès direct au client ChromaDB pour créer/get la collection
            client = self.vector_store.client
            try:
                self._parent_collection = client.get_collection(self.PARENT_COLLECTION_NAME)
            except Exception:
                self._parent_collection = client.create_collection(
                    self.PARENT_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            logger.debug("Collection parent_chunks initialisée")
        except Exception as e:
            logger.warning("Impossible d'initialiser la collection parent : %s", e)
            self.use_chromadb_parent_store = False

    # ── Indexation des parents ──

    def index_parent_chunks(
        self,
        parent_chunks: list,
        source: str = "",
        overwrite: bool = False,
    ) -> int:
        """
        Indexe une liste de ParentChunk dans le store parent.

        Les parents peuvent être stockés :
          - En mémoire (dict parent_id → text)
          - Dans une collection ChromaDB dédiée

        Args:
            parent_chunks: Liste d'objets ParentChunk (de chunker_v2)
            source: Source du document (optionnel, pour le logging)
            overwrite: Ré-écraser les entrées existantes

        Retourne:
            Nombre de chunks parents indexés
        """
        count = 0
        for p in parent_chunks:
            parent_id = p.id if hasattr(p, 'id') else str(id(p))
            parent_text = p.text if hasattr(p, 'text') else str(p)

            with self._lock:
                if parent_id in self._parent_store and not overwrite:
                    continue
                self._parent_store[parent_id] = parent_text

            if self.use_chromadb_parent_store and self._parent_collection is not None:
                try:
                    from rag import get_embedder
                    embedder = get_embedder()
                    emb = embedder.encode(parent_text).tolist()
                    self._parent_collection.add(
                        ids=[parent_id],
                        embeddings=[emb],
                        metadatas=[{
                            "source": source,
                            "parent_id": parent_id,
                            "created_at": time.time(),
                        }],
                        documents=[parent_text],
                    )
                except Exception as e:
                    logger.debug("Indexation parent ChromaDB ignorée : %s", e)

            count += 1

        logger.debug("Indexé %d chunks parents (source=%s)", count, source)
        return count

    def index_from_chunker(
        self,
        text: str,
        source: str,
        chunker: Optional[object] = None,
    ) -> Tuple[int, int]:
        """
        Raccourci : découpe le texte via SmallToBigChunker, indexe les parents
        et retourne les stats.

        Args:
            text: Texte complet du document
            source: Identifiant de la source
            chunker: Instance de chunker (utilise self.chunker si None)

        Retourne:
            (nombre_small_chunks, nombre_parent_chunks)
        """
        if not CHUNKER_V2_AVAILABLE:
            logger.error("chunker_v2 non disponible")
            return 0, 0

        c = chunker or self.chunker
        if c is None:
            c = get_chunker()

        small_chunks, parent_chunks = c.chunk_document(text, source)
        self.index_parent_chunks(parent_chunks, source=source)

        return len(small_chunks), len(parent_chunks)

    # ── Recherche ──

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.50,
        include_corrections: bool = True,
        apply_temporal_filter: bool = True,
        dedup: Optional[bool] = None,
    ) -> list[dict]:
        """
        Recherche Small-to-Big.

        Étapes :
          1. Recherche sémantique via VectorStore.search() (= petits chunks)
          2. Pour chaque résultat, lit le parent_id dans les métadonnées
          3. Récupère le texte du chunk parent (cache mémoire ou ChromaDB)
          4. Déduplication des parents (optionnel)
          5. Filtrage temporel (optionnel)

        Args:
            query: Requête utilisateur
            k: Nombre de résultats (parents) à retourner
            threshold: Seuil de score minimum pour les petits chunks
            include_corrections: Inclure les corrections prioritaires
            apply_temporal_filter: Appliquer le TemporalRAGFilter si configuré
            dedup: Surcharge enable_dedup (None = utiliser la valeur par défaut)

        Retourne:
            Liste de dicts avec les champs standards + 'parent_text', 'parent_id',
            'small_chunks' (liste des petits chunks ayant matché pour ce parent).
        """
        dedup = self.enable_dedup if dedup is None else dedup

        # Étape 1 : Recherche des petits chunks
        small_results = self.vector_store.search(
            query=query,
            k=k * 3,  # Plus de petits chunks pour compenser la déduplication
            threshold=threshold,
            include_corrections=include_corrections,
        )

        if not small_results:
            return []

        # Étape 2 : Regroupement par parent_id
        parent_map: Dict[str, dict] = {}
        parent_order: List[str] = []

        for sr in small_results:
            md = sr.get("metadata", {})
            parent_id = md.get(self.PARENT_ID_META_KEY) or md.get("parent_id", "")

            if not parent_id:
                logger.debug("Résultat sans parent_id, ignoré")
                continue

            if parent_id not in parent_map:
                parent_map[parent_id] = {
                    "parent_id": parent_id,
                    "parent_text": self._get_parent_text(parent_id),
                    "score": 0.0,
                    "small_chunks": [],
                    "metadata": {},
                    "source": md.get("filename", md.get("source", "unknown")),
                }
                parent_order.append(parent_id)

            # Accumuler les scores des petits chunks pour ce parent
            entry = parent_map[parent_id]
            entry["small_chunks"].append(sr)
            entry["score"] = max(entry["score"], sr.get("score", 0.0))
            # Prendre les métadonnées du meilleur petit chunk
            if sr.get("score", 0.0) >= entry["score"]:
                entry["metadata"] = md

        # Étape 3 : Construire la liste des résultats parents
        results = []
        for pid in parent_order:
            entry = parent_map[pid]
            parent_text = entry["parent_text"]
            if not parent_text:
                logger.debug("Parent text introuvable pour %s, ignoré", pid)
                continue

            # Score du parent = score max des petits chunks
            score = entry["score"]
            # Bonus si plusieurs petits chunks du même parent match (diversité)
            bonus = min(0.05, len(entry["small_chunks"]) * 0.02)
            score = min(1.0, score + bonus)

            results.append({
                "text": parent_text,
                "score": round(score, 4),
                "metadata": entry["metadata"],
                "source": "parent_chunk",
                "parent_id": pid,
                "parent_text": parent_text,
                "small_chunks": entry["small_chunks"],
                "small_chunk_count": len(entry["small_chunks"]),
            })

        # Étape 4 : Trier par score descendant
        results.sort(key=lambda r: r["score"], reverse=True)

        # Étape 5 : Filtrage temporel optionnel
        if apply_temporal_filter and self.temporal_filter is not None:
            results = self.temporal_filter.apply(results, query=query)

        # Étape 6 : Limiter à k résultats
        return results[:k]

    def _get_parent_text(self, parent_id: str) -> str:
        """
        Récupère le texte d'un chunk parent.

        Ordre de recherche :
          1. Cache mémoire (_parent_store)
          2. Collection ChromaDB parent_chunks (si activée)
        """
        # 1. Cache mémoire
        with self._lock:
            text = self._parent_store.get(parent_id)
            if text:
                return text

        # 2. Collection ChromaDB
        if self.use_chromadb_parent_store and self._parent_collection is not None:
            try:
                result = self._parent_collection.get(
                    ids=[parent_id],
                    include=["documents"],
                )
                if result and result.get("documents") and result["documents"][0]:
                    text = result["documents"][0]
                    # Mettre en cache
                    with self._lock:
                        self._parent_store[parent_id] = text
                    return text
            except Exception as e:
                logger.debug("Erreur récupération parent ChromaDB : %s", e)

        return ""

    # ── Gestion du cache parent ──

    def add_parent(self, parent_id: str, parent_text: str):
        """Ajoute un parent au cache mémoire."""
        with self._lock:
            self._parent_store[parent_id] = parent_text

    def add_parents_batch(self, parent_map: Dict[str, str]):
        """Ajoute plusieurs parents au cache mémoire."""
        with self._lock:
            self._parent_store.update(parent_map)

    def get_parent_count(self) -> int:
        """Nombre de chunks parents dans le cache mémoire."""
        with self._lock:
            return len(self._parent_store)

    def clear_parent_cache(self):
        """Vide le cache mémoire des parents."""
        with self._lock:
            self._parent_store.clear()

    def save_parent_cache(self, filepath: str):
        """Sauvegarde le cache parent au format JSON."""
        with self._lock:
            data = self._parent_store.copy()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Cache parent sauvegardé (%d entrées) → %s", len(data), filepath)

    def load_parent_cache(self, filepath: str):
        """Charge le cache parent depuis un fichier JSON."""
        path = Path(filepath)
        if not path.exists():
            logger.warning("Fichier cache parent introuvable : %s", filepath)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            self._parent_store.update(data)
        logger.info("Cache parent chargé (%d entrées) depuis %s", len(data), filepath)

    # ── Formatage du contexte ──

    def format_context(
        self,
        results: list[dict],
        max_tokens: int = 2000,
        include_small_chunks: bool = False,
    ) -> str:
        """
        Formate les résultats Small-to-Big pour injection dans le prompt.

        Args:
            results: Résultats de search()
            max_tokens: Nombre max de tokens de contexte
            include_small_chunks: Inclure les extraits de petits chunks dans
                                  le contexte (pour traçabilité)

        Retourne:
            Texte formaté avec les chunks parents
        """
        from ingestion import estimate_tokens

        parts = []
        total_tokens = 0

        for r in results:
            source = r.get("source", "Inconnu")
            parent_id = r.get("parent_id", "")
            score = r.get("score", 0.0)
            n_small = r.get("small_chunk_count", 0)

            header = (
                f"[📄 {source} — {n_small} extrait(s) — pertinence: {score:.2f}]"
            )
            chunk_text = r.get("parent_text", r.get("text", ""))
            chunk_tokens = estimate_tokens(chunk_text) + estimate_tokens(header)

            if total_tokens + chunk_tokens > max_tokens:
                break

            block = f"{header}\n{chunk_text}\n"

            if include_small_chunks:
                small_parts = []
                for sc in r.get("small_chunks", []):
                    sc_text = sc.get("text", "")[:200]  # Extrait court
                    sc_score = sc.get("score", 0.0)
                    small_parts.append(f"  [extrait score={sc_score:.2f}] {sc_text}")
                if small_parts:
                    block += "\n".join(small_parts) + "\n"

            parts.append(block)
            total_tokens += chunk_tokens

        if parts:
            return "--- Contexte RAG (Small-to-Big) ---\n" + "\n".join(parts) + "\n--- Fin du contexte ---"
        return ""

    def __repr__(self) -> str:
        cache_size = self.get_parent_count()
        return (
            f"SmallToBigSearch("
            f"parents_in_cache={cache_size}, "
            f"use_chromadb={self.use_chromadb_parent_store}, "
            f"dedup={self.enable_dedup})"
        )


# =============================================================================
# Singleton helpers
# =============================================================================

_temporal_filter_instance: Optional[TemporalRAGFilter] = None
_small_to_big_instance: Optional[SmallToBigSearch] = None
_lock_instances = threading.Lock()


def get_temporal_filter(
    temp_weight: float = 0.25,
    max_age_days: Optional[int] = None,
    mode: str = "rerank",
) -> TemporalRAGFilter:
    """
    Retourne l'instance singleton du TemporalRAGFilter.

    Args:
        temp_weight: Poids temporel
        max_age_days: Âge maximum en jours (None = pas de limite)
        mode: 'rerank' | 'filter' | 'annotate'
    """
    global _temporal_filter_instance
    with _lock_instances:
        if _temporal_filter_instance is None:
            _temporal_filter_instance = TemporalRAGFilter(
                temp_weight=temp_weight,
                max_age_days=max_age_days,
                mode=mode,
            )
    return _temporal_filter_instance


def get_small_to_big_search(
    vector_store: object = None,
    chunker: object = None,
    **kwargs,
) -> SmallToBigSearch:
    """
    Retourne l'instance singleton du SmallToBigSearch.

    Args:
        vector_store: Instance de VectorStore (requis si premiere instanciation)
        chunker: Instance optionnelle de SmallToBigChunker
        **kwargs: Passés à SmallToBigSearch.__init__()
    """
    global _small_to_big_instance
    with _lock_instances:
        if _small_to_big_instance is None:
            if vector_store is None:
                if RAG_V1_AVAILABLE:
                    vector_store = VectorStore()
                else:
                    raise RuntimeError(
                        "VectorStore requis pour SmallToBigSearch. "
                        "Passez vector_store= ou installez rag.py"
                    )
            _small_to_big_instance = SmallToBigSearch(vector_store, chunker, **kwargs)
    return _small_to_big_instance


def reset_singletons():
    """Réinitialise tous les singletons (utile pour les tests)."""
    global _temporal_filter_instance, _small_to_big_instance
    with _lock_instances:
        _temporal_filter_instance = None
        _small_to_big_instance = None
    logger.debug("Singletons rag_v2 réinitialisés")
