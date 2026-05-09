#!/usr/bin/env python3
"""
reranker_v2.py — FlashRerank BM25 zero-modèle pour NURU V2.

Re-ranking hybride combinant :
  - BM25 (Okapi) : score lexical via rank_bm25
  - Cosine similarity : score sémantique via embeddings

Aucun modèle ML chargé — 100% CPU, 0 VRAM.
Idéal pour le post-filtering des résultats RAG.

Usage :
    reranker = FlashReranker(alpha=0.7)
    results = reranker.rerank(
        query="Qu'est-ce que NURU ?",
        candidates=[{"text": "NURU est un assistant...", "score": 0.85}, ...],
        top_k=5
    )
"""

import sys
import logging
from typing import Optional

logger = logging.getLogger("nuru.v2.reranker")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 non installé — pip install rank-bm25")


class FlashReranker:
    """
    Re-ranker hybride BM25 + Cosine, zéro modèle chargé.

    Args:
        alpha: Poids du score vectoriel (0.0 = BM25 only, 1.0 = vector only)
               Par défaut 0.7 (70% vectoriel, 30% BM25)
    """

    def __init__(self, alpha: float = 0.7):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha doit être entre 0.0 et 1.0, reçu {alpha}")
        self.alpha = alpha
        self._bm25_cache: dict = {}  # Cache des index BM25
        self._cache_id: int = 0
        logger.debug("FlashReranker initialisé (alpha=%.2f)", alpha)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenisation simple pour BM25."""
        import re
        return re.findall(r'\w+', text.lower())

    def _build_bm25(self, texts: list[str]):
        """Construit un index BM25 à partir d'une liste de textes."""
        tokenized = [self._tokenize(t) for t in texts]
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized)

    def _cosine_similarity(self, query_tokens: list[str], text: str) -> float:
        """
        Calcule une similarité cosinus-like sans modèle d'embedding.

        Utilise le TF-IDF maison : fréquence des tokens de la requête
        dans le texte, normalisée par la longueur.
        """
        if not query_tokens or not text:
            return 0.0

        text_lower = text.lower()
        text_tokens = set(self._tokenize(text_lower))

        # Compter les occurrences des tokens de la requête
        hits = 0
        for token in query_tokens:
            if token in text_tokens:
                hits += 1

        if not hits:
            return 0.0

        # Normalisation par la taille de la requête et du texte
        query_len = len(query_tokens)
        text_len = max(len(text_tokens), 1)
        return (hits / query_len) * (1.0 / (1.0 + text_len / 100.0))

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Re-rank une liste de candidats par score hybride BM25 + Cosine.

        Args:
            query: Requête utilisateur
            candidates: Liste de dicts, chacun doit contenir au moins 'text'
                       et peut contenir 'score' (score vectoriel existant)
            top_k: Nombre max de résultats à retourner
            min_score: Score minimum pour inclure un résultat

        Retourne:
            Liste de dicts avec 'hybrid_score' ajouté, triés par score descendant
        """
        if not candidates:
            return []

        if not BM25_AVAILABLE:
            logger.warning("BM25 non disponible — fallback sur le score cosine uniquement")
            # Fallback : scoring cosine simple
            q_tokens = self._tokenize(query)
            scored = []
            for c in candidates:
                text = c.get("text", "")
                cosine = self._cosine_similarity(q_tokens, text)
                existing = c.get("score", 0.0)
                hybrid = self.alpha * existing + (1 - self.alpha) * cosine
                scored.append({**c, "hybrid_score": round(hybrid, 4)})
            scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
            return [s for s in scored if s["hybrid_score"] >= min_score][:top_k]

        # Tokenisation de la requête
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return candidates[:top_k]

        # Extraire les textes des candidats
        texts = [c.get("text", "") for c in candidates]

        # Construire l'index BM25
        bm25 = self._build_bm25(texts)

        # Calculer les scores BM25
        bm25_scores = bm25.get_scores(query_tokens)

        # Normaliser BM25 en [0, 1] via soft max-min scaling
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores

        # Calculer les scores hybrides
        scored = []
        for i, candidate in enumerate(candidates):
            text = candidate.get("text", "")
            existing_vector_score = candidate.get("score", 0.0)

            # Cosine-like score
            cosine = self._cosine_similarity(query_tokens, text)

            # Score BM25 normalisé
            bm25_score = float(bm25_norm[i])

            # Score hybride : alpha * vectoriel + (1-alpha) * BM25 + bonus cosine
            hybrid = (
                self.alpha * existing_vector_score
                + (1.0 - self.alpha) * bm25_score
                + 0.05 * cosine  # Petit bonus cosine
            )

            scored.append({
                **candidate,
                "bm25_score": round(bm25_score, 4),
                "cosine_score": round(cosine, 4),
                "hybrid_score": round(min(hybrid, 1.0), 4),
            })

        # Trier par score hybride descendant
        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Filtrer et limiter
        results = [s for s in scored if s["hybrid_score"] >= min_score][:top_k]

        logger.debug("Re-rank : %d → %d résultats (alpha=%.2f)", len(candidates), len(results), self.alpha)
        return results


# ── Singleton partagé ──
_reranker_instance: Optional[FlashReranker] = None


def get_reranker(alpha: float = 0.7) -> FlashReranker:
    """Retourne l'instance singleton du re-ranker V2."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = FlashReranker(alpha=alpha)
    return _reranker_instance
