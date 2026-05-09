#!/usr/bin/env python3
"""
memory_v2.py — Mémoire hiérarchique à 3 niveaux pour NURU V2.

Niveaux :
  1. WorkingMemory   : buffer FIFO des 3 derniers échanges (utilise memory.SessionMemory)
  2. EpisodicMemory  : résumés automatiques persistés dans ~/.nuru/episodic.json
  3. SemanticMemory  : faits structurés (utilise structured_memory.StructuredMemory)

Classe principale :
  HierarchicalMemory : orchestre les 3 niveaux, fournit get_context(query) unifié.

Rétrocompatible V1 :
  - Interface SessionMemory : add(), get_context(), get_exchanges(), __len__()
  - Interface StructuredMemory : store_fact(), get_fact(), extract_and_store()

Usage :
    hmem = HierarchicalMemory()
    hmem.add("Bonjour", "Salut !")
    ctx = hmem.get_context("Qu'est-ce que NURU ?")
    print(ctx)
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nuru.v2.memory")

# Imports V1 rétrocompatibles
from memory import SessionMemory, Exchange
from structured_memory import StructuredMemory

# ── Constants ──
EPISODIC_STORAGE = Path.home() / ".nuru" / "episodic.json"
WORKING_WINDOW_SIZE = 3  # Nombre d'échanges dans la mémoire de travail


# ─────────────────────────────────────────────
# EpisodicMemory
# ─────────────────────────────────────────────

class EpisodicMemory:
    """
    Mémoire épisodique : résumés automatiques des sessions passées.

    Stocke des résumés dans ~/.nuru/episodic.json.
    Chaque entrée contient :
      - summary : résumé textuel
      - timestamp : horodatage
      - exchange_count : nombre d'échanges résumés
      - topics : liste de sujets (optionnel, stub)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._storage_path = storage_path or EPISODIC_STORAGE
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._episodes: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Charge les épisodes depuis le fichier JSON."""
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    self._episodes = json.load(f)
                logger.debug("Mémoire épisodique chargée (%d épisodes)", len(self._episodes))
            except (json.JSONDecodeError, PermissionError) as e:
                logger.warning("Erreur chargement mémoire épisodique : %s", e)
                self._episodes = []

    def _save(self) -> None:
        """Sauvegarde les épisodes dans le fichier JSON."""
        try:
            # Garder max 100 épisodes pour éviter la dérive
            episodes = self._episodes[-100:]
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(episodes, f, ensure_ascii=False, indent=2)
        except PermissionError as e:
            logger.warning("Erreur sauvegarde mémoire épisodique : %s", e)

    def add_episode(self, summary: str, exchange_count: int = 1, topics: Optional[list[str]] = None) -> None:
        """
        Ajoute un résumé de session.

        Args:
            summary: Résumé textuel de la session
            exchange_count: Nombre d'échanges dans cette session
            topics: Sujets abordés (optionnel)
        """
        episode = {
            "summary": summary,
            "timestamp": time.time(),
            "exchange_count": exchange_count,
            "topics": topics or [],
        }
        self._episodes.append(episode)
        self._save()
        logger.debug("Épisode ajouté : %s…", summary[:60])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Recherche dans les épisodes par similarité textuelle simple.

        Fonctionne sans modèle ML — utilise le ratio de mots communs.

        Args:
            query: Requête de recherche
            top_k: Nombre max de résultats

        Retourne:
            Liste de dicts {summary, timestamp, score, ...}
        """
        if not self._episodes:
            return []

        query_words = set(query.lower().split())
        if not query_words:
            return []

        scored = []
        for ep in self._episodes:
            summary_words = set(ep["summary"].lower().split())
            if not summary_words:
                continue
            overlap = len(query_words & summary_words)
            ratio = overlap / max(len(query_words), len(summary_words))
            scored.append((ratio, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "summary": ep["summary"],
                "timestamp": ep["timestamp"],
                "exchange_count": ep["exchange_count"],
                "topics": ep.get("topics", []),
                "score": round(score, 3),
            }
            for score, ep in scored[:top_k]
            if score > 0
        ]

    def get_context(self, query: str, top_k: int = 2) -> str:
        """
        Génère un bloc de contexte épisodique pour injection dans le prompt.

        Args:
            query: Requête actuelle
            top_k: Nombre max d'épisodes à inclure

        Retourne:
            Bloc texte formaté ou chaîne vide
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        lines = ["[Mémoire épisodique — sessions passées pertinentes]"]
        for i, r in enumerate(results, 1):
            t_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["timestamp"]))
            lines.append(f"- [{t_str}] {r['summary']}")
        lines.append("")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Statistiques de la mémoire épisodique."""
        return {
            "total_episodes": len(self._episodes),
            "storage_path": str(self._storage_path),
            "file_exists": self._storage_path.exists(),
        }

    def clear(self) -> None:
        """Vide tous les épisodes."""
        self._episodes.clear()
        self._save()


# ─────────────────────────────────────────────
# HierarchicalMemory
# ─────────────────────────────────────────────

class HierarchicalMemory:
    """
    Mémoire hiérarchique orchestrant 3 niveaux.

    Niveaux (du plus récent au plus permanent) :
      1. WorkingMemory  → SessionMemory (3 derniers échanges)
      2. EpisodicMemory → résumés de sessions passées
      3. SemanticMemory → faits structurés persistants

    Rétrocompatible V1 : expose add(), get_context(), get_exchanges(),
    store_fact(), get_fact(), extract_and_store().
    """

    def __init__(
        self,
        working_size: int = WORKING_WINDOW_SIZE,
        episodic_path: Optional[Path] = None,
        semantic_path: Optional[str] = None,
        auto_summarize: bool = True,
    ):
        # Niveau 1 : Mémoire de travail (buffer FIFO)
        self.working = SessionMemory(buffer_size=working_size)

        # Niveau 2 : Mémoire épisodique (résumés persistés)
        self.episodic = EpisodicMemory(storage_path=episodic_path)

        # Niveau 3 : Mémoire sémantique (faits structurés)
        self.semantic = StructuredMemory(storage_path=semantic_path)

        self._auto_summarize = auto_summarize
        self._last_summary_count = 0
        logger.debug(
            "HierarchicalMemory initialisée (working=%d, episodes=%d, facts=%d)",
            len(self.working),
            len(self.episodic._episodes),
            len(self.semantic),
        )

    # ── Interface SessionMemory (V1 compatible) ──

    def add(self, user_msg: str, assistant_msg: str) -> None:
        """
        Ajoute un échange à la mémoire de travail.
        Déclenche un auto-summary si le buffer est plein.
        Extrait aussi les faits sémantiques.
        """
        self.working.add(user_msg, assistant_msg)

        # Extraction sémantique de la requête utilisateur
        self.semantic.extract_and_store(user_msg)

        # Auto-summarize si le buffer est plein
        if self._auto_summarize and len(self.working) >= self.working.buffer_size:
            self._auto_summarize_session()

    def get_exchanges(self) -> list:
        """Retourne les échanges de la mémoire de travail (V1 compatible)."""
        return self.working.get_exchanges()

    def get_context(self, query: str = "") -> str:
        """
        Concatène les contextes des 3 niveaux pour injection dans le prompt.

        Args:
            query: Requête actuelle (utilisée pour la recherche épisodique)

        Retourne:
            Bloc de contexte formaté
        """
        parts = []

        # Niveau 1 : Contexte de travail (derniers échanges)
        working_ctx = self.working.get_context(include_timestamps=False)
        if working_ctx:
            parts.append(working_ctx)

        # Niveau 2 : Contexte épisodique (résumés de sessions passées)
        if query:
            episodic_ctx = self.episodic.get_context(query, top_k=2)
            if episodic_ctx:
                parts.append(episodic_ctx)

        # Niveau 3 : Contexte sémantique (faits sur l'utilisateur)
        semantic_ctx = self.semantic.get_context()
        if semantic_ctx:
            parts.append(semantic_ctx)

        return "\n\n".join(parts)

    # ── Interface StructuredMemory (V1 compatible) ──

    def store_fact(self, key: str, value: str, confidence: float = 0.8) -> dict:
        """Stocke un fait dans la mémoire sémantique (V1 compatible)."""
        return self.semantic.store_fact(key, value, confidence)

    def get_fact(self, key: str) -> Optional[dict]:
        """Récupère un fait par sa clé (V1 compatible)."""
        return self.semantic.get_fact(key)

    def extract_and_store(self, text: str) -> dict[str, str]:
        """Extrait et stocke des faits d'un texte (V1 compatible)."""
        return self.semantic.extract_and_store(text)

    def describe_user(self) -> Optional[str]:
        """Retourne une description de l'utilisateur (V1 compatible)."""
        return self.semantic.describe_user()

    # ── Auto-summary (nano model + fallback textuel) ──

    def _auto_summarize_session(self) -> None:
        """
        Résumé automatique de la session courante.

        Si le nano model (Qwen 1.5B) est disponible dans le ModelPool,
        l'utilise pour un vrai résumé ML. Sinon, résumé textuel basique.
        """
        exchanges = self.working.get_exchanges()
        if len(exchanges) <= self._last_summary_count:
            return

        new_exchanges = exchanges[self._last_summary_count:]
        self._last_summary_count = len(exchanges)

        summary = self._generate_summary_with_model(new_exchanges)
        if not summary:
            summary = self._generate_summary_text(new_exchanges)

        self.episodic.add_episode(
            summary=summary,
            exchange_count=len(new_exchanges),
            topics=self._extract_topics(new_exchanges),
        )
        logger.debug("Résumé automatique créé (%d échanges)", len(new_exchanges))

    def _generate_summary_with_model(self, exchanges: list) -> str | None:
        """Tente un résumé via le nano model (Qwen 1.5B) si disponible."""
        try:
            from model_pool_v2 import get_model_pool
            pool = get_model_pool()

            # Vérifier si le nano model est disponible sans le charger
            if not pool._check_reasoning_available():
                return None

            # Construire un prompt de résumé très court
            text = "\n".join(f"User: {e.user}\nNURU: {e.assistant}" for e in exchanges[-5:])
            prompt = (
                f"Résume cette conversation en 1-2 phrases, en français.\n\n{text}\n\nRésumé :"
            )

            # Tenter un résumé via le pool (nano model si disponible)
            model, tokenizer = pool.get_model(complexity=1)  # 1 = SIMPLE → nano
            if model is None:
                return None

            import mlx_lm
            response = mlx_lm.generate(model, tokenizer, prompt, max_tokens=100, verbose=False)
            summary = response.strip()
            return summary[:300] if summary else None

        except Exception:
            return None

    def _generate_summary_text(self, exchanges: list) -> str:
        """Résumé textuel basique (fallback)."""
        topics = self._extract_topics(exchanges)
        last_user = exchanges[-1].user[:100] if exchanges else ""
        last_assistant = exchanges[-1].assistant[:100] if exchanges else ""

        parts = []
        if topics:
            parts.append("Sujets : " + ", ".join(list(topics)[:5]))
        if last_user:
            parts.append(f"Dernière question : {last_user}…")
        if last_assistant:
            parts.append(f"Réponse : {last_assistant}…")

        return " | ".join(parts) if parts else "Échange mémorisé"

    def _extract_topics(self, exchanges: list) -> set:
        """Extrait les mots-clés significatifs des échanges."""
        stop_words = {
            "comment", "pourquoi", "qu'est-ce", "pouvez-vous", "peux-tu",
            "est-ce", "quelle", "quels", "quelles", "le", "la", "les",
            "des", "un", "une", "du", "de", "dans", "pour", "sur",
            "avec", "sans", "mais", "donc", "or", "car", "ni", "ou",
            "et", "bonjour", "merci", "salut", "bonsoir",
        }
        topics = set()
        for ex in exchanges:
            words = ex.user.lower().split()
            for w in words:
                w = w.strip(".,!?;:'\"")
                if len(w) > 3 and w not in stop_words:
                    topics.add(w[:25])
        return topics

    # ── Utilitaires ──

    def clear(self) -> None:
        """Vide la mémoire de travail (les épisodes et faits sont conservés)."""
        self.working.clear()
        self._last_summary_count = 0
        logger.info("Mémoire de travail vidée")

    def clear_all(self) -> None:
        """Vide toute la mémoire (travail + épisodes + faits)."""
        self.working.clear()
        self.episodic.clear()
        self.semantic.forget_all()
        self._last_summary_count = 0
        logger.info("Mémoire complète vidée")

    def get_stats(self) -> dict:
        """Statistiques combinées des 3 niveaux de mémoire."""
        return {
            "working": self.working.get_stats(),
            "episodic": self.episodic.get_stats(),
            "semantic": self.semantic.get_stats(),
            "total_exchanges": len(self.working),
            "total_facts": len(self.semantic),
            "total_episodes": self.episodic.get_stats()["total_episodes"],
        }

    def __len__(self) -> int:
        return len(self.working)

    def __repr__(self) -> str:
        return (
            f"HierarchicalMemory(working={len(self.working)}"
            f"/{self.working.buffer_size}, "
            f"episodes={self.episodic.get_stats()['total_episodes']}, "
            f"facts={len(self.semantic)})"
        )
