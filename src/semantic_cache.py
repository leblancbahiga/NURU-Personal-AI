#!/usr/bin/env python3
"""
semantic_cache.py — Cache sémantique pour NURU.

Stocke les paires question/réponse avec embedding vectoriel dans un fichier JSON.
Avant de router une question, le routeur vérifie si une question sémantiquement
proche (similarité cosinus > 0.92) existe déjà dans le cache.

Usage :
    cache = SemanticCache()
    cache.put("Quelle est la capitale de la France ?", "Paris")
    réponse = cache.get("Quelle est la capitale française ?")  # retourne "Paris"
    stats = cache.stats()
"""

import json
import os
import sys
import time
import math
from pathlib import Path
from typing import Optional

# Chemin du fichier de cache
CACHE_FILE = Path(__file__).parent.parent / "data" / "cache.json"

# Modèle d'embedding
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Seuil de similarité cosinus
SIMILARITY_THRESHOLD = 0.92

# Modèle d'embedding (cache global pour éviter rechargements)
_EMBEDDER = None


def _get_embedder():
    """Charge le modèle d'embedding (singleton)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
        except ImportError:
            print(
                "  ⚠ sentence_transformers non installé — "
                "`pip install sentence-transformers`",
                file=sys.stderr,
            )
            _EMBEDDER = False
        except Exception as e:
            print(f"  ⚠ Erreur chargement modèle embedding : {e}", file=sys.stderr)
            _EMBEDDER = False
    return _EMBEDDER if _EMBEDDER is not False else None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot = sum(av * bv for av, bv in zip(a, b))
    norm_a = math.sqrt(sum(av * av for av in a))
    norm_b = math.sqrt(sum(bv * bv for bv in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """
    Cache sémantique pour les paires question/réponse.

    Utilise sentence_transformers pour l'embedding et cosine similarity
    pour trouver les questions sémantiquement proches.
    Stockage : fichier JSON dans data/cache.json
    """

    def __init__(self, cache_path: str | Path | None = None):
        self.cache_path = Path(cache_path) if cache_path else CACHE_FILE
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []
        self._dirty = False
        self._load()

    # ── Persistance ──

    def _load(self):
        """Charge le cache depuis le fichier JSON."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = data.get("entries", [])
            except (json.JSONDecodeError, OSError) as e:
                print(f"  ⚠ Erreur lecture cache {self.cache_path} : {e}", file=sys.stderr)
                self._entries = []
        else:
            self._entries = []

    def _save(self):
        """Sauvegarde le cache dans le fichier JSON."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "entries": self._entries}, f, ensure_ascii=False, indent=2)
            self._dirty = False
        except OSError as e:
            print(f"  ⚠ Erreur écriture cache {self.cache_path} : {e}", file=sys.stderr)

    # ── API publique ──

    def put(self, query: str, response: str, niveau: str = "N2"):
        """
        Stocke une paire question/réponse dans le cache avec TTL.

        Args:
            query: La question posée
            response: La réponse correspondante
            niveau: Niveau de routage (N1/N2/N3). N3 expire après 1h, N1/N2 après 30 jours.
        """
        embedder = _get_embedder()
        if embedder is None:
            return  # Pas d'embedding disponible, on ne cache pas

        # Calcul du TTL selon le niveau
        if niveau == "N3":
            ttl_seconds = 3600  # 1 heure pour les données web volatiles
        else:
            ttl_seconds = 2592000  # 30 jours pour les connaissances stables
        expires_at = time.time() + ttl_seconds

        # Éviter les doublons exacts
        for entry in self._entries:
            if entry["query"] == query:
                entry["response"] = response
                entry["timestamp"] = time.time()
                entry["expires_at"] = expires_at
                entry["niveau"] = niveau
                self._dirty = True
                self._save()
                return

        # Calculer l'embedding
        try:
            emb = embedder.encode(query).tolist()
        except Exception as e:
            print(f"  ⚠ Erreur embedding : {e}", file=sys.stderr)
            return

        # Ajouter l'entrée avec TTL
        self._entries.append({
            "query": query,
            "response": response,
            "embedding": emb,
            "timestamp": time.time(),
            "expires_at": expires_at,
            "niveau": niveau,
            "hits": 0,
        })
        self._dirty = True
        self._save()

    def get(self, query: str) -> Optional[str]:
        """
        Cherche une réponse mise en cache pour une question sémantiquement proche.

        Args:
            query: La question posée

        Returns:
            La réponse si une question similaire (cos > seuil) existe, sinon None
        """
        if not self._entries:
            return None

        embedder = _get_embedder()
        if embedder is None:
            return None  # Pas d'embedding disponible

        # Calculer l'embedding de la requête
        try:
            q_emb = embedder.encode(query).tolist()
        except Exception as e:
            print(f"  ⚠ Erreur embedding requête : {e}", file=sys.stderr)
            return None

        # Chercher la meilleure correspondance
        best_similarity = 0.0
        best_entry = None

        for entry in self._entries:
            emb = entry.get("embedding")
            if not emb:
                continue
            sim = cosine_similarity(q_emb, emb)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        # Vérifier le seuil
        if best_entry and best_similarity >= SIMILARITY_THRESHOLD:
            # Vérifier l'expiration TTL
            expires_at = best_entry.get("expires_at")
            if expires_at is not None and time.time() > expires_at:
                # Entrée expirée — la supprimer
                try:
                    self._entries = [e for e in self._entries if e is not best_entry]
                    self._dirty = True
                    self._save()
                except Exception:
                    pass
                return None
            # Incrémenter le compteur de hits
            best_entry["hits"] = best_entry.get("hits", 0) + 1
            best_entry["last_access"] = time.time()
            self._dirty = True
            self._save()
            return best_entry["response"]

        return None

    def clear(self):
        """Vide le cache."""
        self._entries = []
        self._dirty = True
        self._save()

    def stats(self) -> dict:
        """
        Statistiques du cache.

        Returns:
            dict avec : total_entries, total_hits, total_misses, size_bytes,
                       cache_path, threshold, hit_rate (si applicable)
        """
        total_hits = sum(e.get("hits", 0) for e in self._entries)

        # Taille du fichier
        size_bytes = 0
        if self.cache_path.exists():
            try:
                size_bytes = self.cache_path.stat().st_size
            except OSError:
                pass

        return {
            "total_entries": len(self._entries),
            "total_hits": total_hits,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 1),
            "cache_path": str(self.cache_path),
            "threshold": SIMILARITY_THRESHOLD,
            "embedding_model": EMBEDDING_MODEL,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SemanticCache(entries={len(self._entries)}, path={self.cache_path})"


# ── CLI (test rapide) ──
if __name__ == "__main__":
    cache = SemanticCache()

    import argparse
    parser = argparse.ArgumentParser(description="Cache sémantique NURU")
    parser.add_argument("action", choices=["get", "put", "stats", "clear"], help="Action")
    parser.add_argument("--query", "-q", type=str, help="Requête (pour get/put)")
    parser.add_argument("--response", "-r", type=str, help="Réponse (pour put)")
    args = parser.parse_args()

    if args.action == "put":
        if not args.query or not args.response:
            print("Erreur : --query et --response requis pour put")
            sys.exit(1)
        cache.put(args.query, args.response)
        print(f"✓ Question mise en cache : {args.query[:50]}...")

    elif args.action == "get":
        if not args.query:
            print("Erreur : --query requis pour get")
            sys.exit(1)
        result = cache.get(args.query)
        if result:
            print(f"✓ Trouvé dans le cache : {result[:200]}...")
            print(f"  Similarité > {SIMILARITY_THRESHOLD}")
        else:
            print("✗ Aucune correspondance dans le cache")

    elif args.action == "stats":
        stats = cache.stats()
        print("Statistiques du cache sémantique :")
        print(f"  Fichier    : {stats['cache_path']}")
        print(f"  Entrées    : {stats['total_entries']}")
        print(f"  Hits       : {stats['total_hits']}")
        print(f"  Taille     : {stats['size_kb']} KB")
        print(f"  Seuil      : {stats['threshold']}")
        print(f"  Modèle     : {stats['embedding_model']}")

    elif args.action == "clear":
        cache.clear()
        print("✓ Cache vidé")
