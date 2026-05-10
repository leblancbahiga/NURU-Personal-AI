#!/usr/bin/env python3
"""
rag.py — Orchestrateur RAG pour NURU.

Gère :
  - 3 collections ChromaDB : documents, conversations, corrections_prioritaires
  - Embedding via sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
  - Recherche sémantique (similarité cosinus, threshold > 0.75)
  - Re-ranking via cross-encoder (ms-marco-MiniLM-L6-v2)
  - Injection de contexte dans le prompt LLM
"""

import json
import re
import time
import warnings
from pathlib import Path
from typing import Optional

# Désactiver les avertissements de HuggingFace concernant les requêtes non authentifiées
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Embedding et re-ranking sont chargés à la demande (lazy)
_embedder = None
_reranker = None
_reranker_enabled = True  # Flag global pour activer/désactiver le re-ranker


def set_reranker_enabled(enabled: bool):
    """Active ou désactive le re-ranker globalement."""
    global _reranker_enabled
    _reranker_enabled = enabled


# ── Noms des collections ──
COLL_DOCUMENTS = "documents"
COLL_CONVERSATIONS = "conversations"
COLL_CORRECTIONS = "corrections_prioritaires"

# ── Chemins par défaut ──
DEFAULT_CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma_db")
HASH_CACHE_FILE = str(Path(__file__).parent.parent / "data" / "indexed_hashes.json")


# ── Fonctions de score hybride ──

def _compute_recency_weight(created_at: float, now: float = None) -> float:
    """
    Calcule le poids de récence.

    1.0 pour les 7 premiers jours, décroît linéairement jusqu'à 0.3 après 90 jours.
    """
    if now is None:
        now = time.time()
    age = now - created_at
    seven_days = 7 * 24 * 3600       # 604 800 s
    ninety_days = 90 * 24 * 3600      # 7 776 000 s

    if age <= seven_days:
        return 1.0
    elif age >= ninety_days:
        return 0.3
    else:
        # Interpolation linéaire entre 1.0 (7j) et 0.3 (90j)
        return 1.0 - (age - seven_days) / (ninety_days - seven_days) * 0.7


def _compute_keyword_match(query: str, text: str) -> float:
    """
    Proportion de mots-clés de la requête présents dans le chunk.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    if not query_words:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for w in query_words if w in text_lower)
    return matches / len(query_words)


# ── Embedding (lazy loading) ──

def get_embedder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Charge le modèle d'embedding (lazy, une seule fois)."""
    global _embedder
    if _embedder is None:
        print(f"  🧠 Chargement du modèle d'embedding...", end=" ", flush=True)
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)
        print(f"✓ ({time.time() - t0:.1f}s)")
    return _embedder


def get_reranker(model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
    """
    Charge le modèle de re-ranking multilingue (Bug #3).
    Supporte nativement le français (14 langues).
    Retourne None si le modèle n'est pas disponible (fallback gracieux).
    """
    global _reranker
    if not _reranker_enabled:
        return None
    if _reranker is None:
        try:
            print(f"  🧠 Chargement du re-ranker multilingue...", end=" ", flush=True)
            t0 = time.time()
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(model_name)
            print(f"✓ ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"⚠ Re-ranker non disponible ({e}). Le scoring hybride sera utilisé.", flush=True)
            _reranker = False  # False = tentative échouée, ne pas réessayer
    return _reranker if _reranker is not False else None


def unload_embedding():
    """Décharge le modèle d'embedding pour libérer la RAM."""
    global _embedder
    if _embedder is not None:
        import gc
        del _embedder
        _embedder = None
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
        print("  ♻️ Embedding déchargé", file=__import__('sys').stderr)


def unload_reranker():
    """Décharge le re-ranker pour libérer la RAM."""
    global _reranker
    if _reranker is not None and _reranker is not False:
        import gc
        del _reranker
        _reranker = None
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
        print("  ♻️ Re-ranker déchargé", file=__import__('sys').stderr)
    elif _reranker is False:
        _reranker = None  # Reset pour permettre une nouvelle tentative


# ── Base vectorielle ──

class VectorStore:
    """
    Base vectorielle ChromaDB avec 3 collections.

    Usage:
        store = VectorStore()
        store.add_document_chunks(doc)
        results = store.search("Quelle est la capitale de la France ?", k=5)
    """

    def __init__(self, persist_dir: str = DEFAULT_CHROMA_PATH):
        self.persist_dir = persist_dir

        if not CHROMA_AVAILABLE:
            raise RuntimeError("ChromaDB non installé — pip3 install chromadb")

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Création / récupération des 3 collections
        self.col_documents = self._get_or_create(COLL_DOCUMENTS)
        self.col_conversations = self._get_or_create(COLL_CONVERSATIONS)
        self.col_corrections = self._get_or_create(COLL_CORRECTIONS)

        # Cache des hashs indexés
        self.hash_cache = self._load_hash_cache()
        self.embedder = None  # Initialisé à la demande

    def _get_or_create(self, name: str):
        """Crée ou récupère une collection (similarité cosinus)."""
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(
                name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity
            )

    def _load_hash_cache(self) -> dict:
        """Charge le cache des empreintes MD5 déjà indexées."""
        cache_path = Path(HASH_CACHE_FILE)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_hash_cache(self):
        """Sauvegarde le cache des empreintes MD5."""
        cache_path = Path(HASH_CACHE_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(self.hash_cache, f, indent=2)

    def is_indexed(self, filepath: str, file_hash: str) -> bool:
        """Vérifie si un fichier a déjà été indexé (par MD5)."""
        cached = self.hash_cache.get(filepath)
        return cached == file_hash

    def clear(self):
        """Vide entièrement la base vectorielle et le cache hash."""
        # Supprimer les collections
        for name in [COLL_DOCUMENTS, COLL_CONVERSATIONS, COLL_CORRECTIONS]:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        
        # Recréer les collections vides
        self.col_documents = self._get_or_create(COLL_DOCUMENTS)
        self.col_conversations = self._get_or_create(COLL_CONVERSATIONS)
        self.col_corrections = self._get_or_create(COLL_CORRECTIONS)
        
        # Vider le cache des hashs
        self.hash_cache = {}
        self._save_hash_cache()

    def mark_indexed(self, filepath: str, file_hash: str):
        """Marque un fichier comme indexé."""
        self.hash_cache[filepath] = file_hash
        self._save_hash_cache()

    # ── Indexation ──

    def add_document_chunks(self, filepath: str, filename: str, file_hash: str,
                            mime_type: str, chunks: list) -> int:
        """
        Ajoute les chunks d'un document à la collection 'documents'.

        Retourne le nombre de chunks ajoutés.
        """
        from ingestion import Chunk as ChunkType

        self.embedder = get_embedder()

        texts = [c.text for c in chunks]
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            page = chunk.page_number if hasattr(chunk, 'page_number') else None
            metadatas.append({
                "filepath": filepath,
                "filename": filename,
                "mime_type": mime_type,
                "chunk_index": i,
                "page": page or 0,
                "file_hash": file_hash,
                "n_tokens": chunk.token_estimate if hasattr(chunk, 'token_estimate') else len(chunk.text) // 4,
                "created_at": time.time(),
            })
            ids.append(f"{file_hash}_{i}")

        # Embedding par lots pour économiser la RAM
        print(f"  📦 Vectorisation de {len(texts)} chunks...", end=" ", flush=True)
        t0 = time.time()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        print(f"✓ ({time.time() - t0:.1f}s)")

        # Ajout à ChromaDB
        self.col_documents.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts,
        )

        return len(chunks)

    def count_documents(self) -> int:
        """Nombre de documents indexés."""
        return self.col_documents.count()

    def count_corrections(self) -> int:
        """Nombre de corrections stockées."""
        return self.col_corrections.count()

    # ── Recherche ──

    def search(self, query: str, hyde_doc: str = None, k: int = 5, threshold: float = 0.20,
               include_corrections: bool = True) -> list[dict]:
        """
        Recherche hybride dans ChromaDB (Bug #7: threshold augmenté).

        Score hybride = 0.6 * embedding_similarity
                      + 0.2 * keyword_match
                      + 0.2 * recency_weight

        Args:
            query: Requête utilisateur
            k: Nombre de résultats à retourner
            threshold: Seuil de score hybride minimum (Bug #7: 0.50)
            include_corrections: Inclure la collection corrections_prioritaires

        Retourne:
            Liste de dicts : {text, score, metadata, source, hybrid_components}
        """
        self.embedder = get_embedder()

        # Embedding de la requête (ou du doc HyDE)
        search_text = hyde_doc if hyde_doc else query
        query_embedding = self.embedder.encode(search_text).tolist()

        raw_results = []

        # Recherche dans les documents
        try:
            doc_results = self.col_documents.query(
                query_embeddings=[query_embedding],
                n_results=k * 4,  # Plus de résultats pour MMR et scoring hybride
            )
        except Exception as e:
            print(f"  ⚠ Erreur requête ChromaDB : {e}")
            doc_results = None

        if doc_results and doc_results.get("ids") and doc_results["ids"][0]:
            for i in range(len(doc_results["ids"][0])):
                distance = doc_results["distances"][0][i] if doc_results.get("distances") else 0
                embedding_sim = max(0, 1.0 - distance)
                raw_results.append({
                    "text": doc_results["documents"][0][i],
                    "score": embedding_sim,
                    "metadata": doc_results["metadatas"][0][i],
                    "source": "documents",
                    "id": doc_results["ids"][0][i],
                })

        # Recherche dans les corrections prioritaires
        if include_corrections:
            try:
                corr_results = self.col_corrections.query(
                    query_embeddings=[query_embedding],
                    n_results=k * 2,
                )
                if corr_results and corr_results.get("ids") and corr_results["ids"][0]:
                    for i in range(len(corr_results["ids"][0])):
                        distance = corr_results["distances"][0][i] if corr_results.get("distances") else 0
                        embedding_sim = max(0, 1.0 - distance / 2.0)
                        # Bonus léger aux corrections
                        embedding_sim = min(1.0, embedding_sim + 0.05)
                        raw_results.append({
                            "text": corr_results["documents"][0][i],
                            "score": embedding_sim,
                            "metadata": corr_results["metadatas"][0][i],
                            "source": "corrections",
                            "id": corr_results["ids"][0][i],
                        })
            except Exception:
                pass

        # Calcul du score hybride pour chaque résultat
        for r in raw_results:
            embedding_sim = r["score"]
            kw_match = _compute_keyword_match(query, r["text"])
            created_at = r["metadata"].get("created_at", time.time())
            recency = _compute_recency_weight(created_at)

            hybrid = 0.4 * embedding_sim + 0.4 * kw_match + 0.2 * recency
            r["score"] = round(hybrid, 4)
            r["hybrid_components"] = {
                "embedding": round(embedding_sim, 4),
                "keyword": round(kw_match, 4),
                "recency": round(recency, 4),
            }

        # Filtrer par seuil de score hybride (Bug #7)
        results = [r for r in raw_results if r["score"] >= threshold]

        # Trier par score hybride descendant
        results.sort(key=lambda r: r["score"], reverse=True)

        # Bug #6 : Diversité (Simple MMR-like)
        # On évite de renvoyer trop de chunks consécutifs du même document
        diverse_results = []
        seen_files = {}
        for r in results:
            filename = r["metadata"].get("filename", "unknown")
            # Max 2 chunks par fichier pour assurer la diversité (Bug #6)
            if seen_files.get(filename, 0) < 2:
                diverse_results.append(r)
                seen_files[filename] = seen_files.get(filename, 0) + 1
            if len(diverse_results) >= k * 2: # On garde un peu de marge pour le reranking
                break
        
        results = diverse_results

        # Re-ranking si assez de résultats (Bug #3)
        if len(results) >= 2:
            results = self._rerank(query, results, k)

        return results[:k]

    def _rerank(self, query: str, results: list[dict], k: int) -> list[dict]:
        """
        Re-ranking via cross-encoder.
        Retourne immédiatement si le reranker est désactivé (None).
        """
        if not results:
            return results

        try:
            reranker = get_reranker()
            if reranker is None:
                return results  # Reranker désactivé
            pairs = [(query, r["text"]) for r in results]
            scores = reranker.predict(pairs)

            for i, score in enumerate(scores):
                results[i]["rerank_score"] = float(score)

            # Re-tri par score de re-ranking
            results.sort(key=lambda r: r.get("rerank_score", r["score"]), reverse=True)
        except Exception as e:
            print(f"  ⚠ Re-ranking ignoré : {e}")

        return results

    def format_context(self, results: list[dict], max_tokens: int = 1500) -> str:
        """
        Formate les résultats RAG pour injection dans le prompt.

        Args:
            results: Résultats de search()
            max_tokens: Nombre max de tokens de contexte

        Retourne:
            Texte formaté : "[Source: filename (page X)]\ntexte..."
        """
        from ingestion import estimate_tokens

        parts = []
        total_tokens = 0

        for r in results:
            source = r["metadata"].get("filename", "Inconnu")
            page = r["metadata"].get("page", 0)
            score = r.get("rerank_score", r.get("score", 0))
            components = r.get("hybrid_components", {})

            header = f"[📄 {source}" + (f" (p.{page})" if page else "") + f" — pertinence: {score:.2f}]"
            if components:
                header += f" [emb:{components.get('embedding',0):.2f} kw:{components.get('keyword',0):.2f} rec:{components.get('recency',0):.2f}]"
            chunk_text = r["text"]
            chunk_tokens = estimate_tokens(chunk_text) + estimate_tokens(header)

            if total_tokens + chunk_tokens > max_tokens:
                break

            parts.append(f"{header}\n{chunk_text}\n")
            total_tokens += chunk_tokens

        if parts:
            return "--- Contexte RAG ---\n" + "\n".join(parts) + "--- Fin du contexte ---"
        return ""

    def add_correction(self, query: str, correction_text: str) -> bool:
        """
        Ajoute une correction à la collection prioritaire.
        """
        self.embedder = get_embedder()
        import hashlib
        query_hash = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        corr_id = f"corr_{int(time.time())}_{query_hash}"

        embedding = self.embedder.encode(query).tolist()

        self.col_corrections.add(
            ids=[corr_id],
            embeddings=[embedding],
            metadatas=[{
                "original_query": query,
                "created_at": time.time(),
                "last_used_at": time.time(),
                "confidence_score": 0.5,
                "usage_count": 0,
                "source": "user_feedback",
            }],
            documents=[correction_text],
        )
        return True

    def record_correction_use(self, corr_id: str) -> bool:
        """
        Enregistre l'utilisation d'une correction (incrémente usage_count,
        met à jour last_used_at).
        """
        try:
            self.col_corrections.update(
                ids=[corr_id],
                metadatas=[{
                    "last_used_at": time.time(),
                    "usage_count": 1,  # incrément via get + set
                }],
            )
        except Exception:
            pass
        return True

    def get_stats(self) -> dict:
        """Statistiques de la base vectorielle."""
        return {
            "documents_indexed": self.col_documents.count(),
            "corrections": self.col_corrections.count(),
            "conversations": self.col_conversations.count(),
            "chroma_path": self.persist_dir,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"VectorStore(docs={stats['documents_indexed']}, corr={stats['corrections']}, conv={stats['conversations']})"
