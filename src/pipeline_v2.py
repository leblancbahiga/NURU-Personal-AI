#!/usr/bin/env python3
"""
pipeline_v2.py — Orchestrateur async optionnel pour NURU V2.

Parallélise RAG search + memory retrieval via asyncio.gather().
Utilise ModelPoolV2 pour la sélection de modèle et le streaming.
Compatible avec l'interface stream_route() de router.py (mêmes yield types).
Mode synchrone disponible via handle_sync() pour compatibilité V1.

Yield types (compatibles router.py) :
  {"type": "status", "msg": "..."}
  {"type": "start", "level": ..., "level_name": ..., "reason": ...}
  {"type": "token", "token": "...", "final": True/False}
  {"type": "rag_hits", "sources": [...], "latency_ms": ...}
  {"type": "end", "model_used": "...", "latency_ms": ...}

Usage :
    pipeline = NuruPipelineV2()
    for chunk in pipeline.handle_sync("Qu'est-ce que NURU ?"):
        if chunk["type"] == "token":
            print(chunk["token"], end="")
"""

import sys
import time
import re
import gc
import threading
import logging
from typing import Optional, Generator, AsyncGenerator

logger = logging.getLogger("nuru.v2.pipeline")

# ── Imports V1 rétrocompatibles ──
try:
    from complexity_classifier import get_complexity_classifier, Level, Intent, Complexity
except ImportError:
    logger.warning("complexity_classifier non disponible")
    Complexity = None
    Level = None
    Intent = None

try:
    from memory import SessionMemory
except ImportError:
    SessionMemory = None

try:
    from structured_memory import StructuredMemory
except ImportError:
    StructuredMemory = None

# ── Imports V2 ──
from model_pool_v2 import ModelPoolV2, get_model_pool
from memory_v2 import HierarchicalMemory
from reranker_v2 import FlashReranker, get_reranker
from resource_manager_v2 import ResourceManagerV2, get_resource_manager

# ── Imports optionnels ──
try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False

try:
    from rag import VectorStore, get_embedder
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from monitor import get_monitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False


class NuruPipelineV2:
    """
    Orchestrateur V2 du pipeline NURU.

    Parallélise les étapes suivantes :
      1. Classification de complexité (détermine l'intention et le niveau)
      2. RAG search + Memory retrieval (en parallèle via asyncio.gather)
      3. Re-ranking optionnel
      4. Sélection du modèle via ModelPoolV2
      5. Streaming de la génération

    Compatible stream_route() de router.py (mêmes yield types).
    """

    def __init__(
        self,
        model_pool: Optional[ModelPoolV2] = None,
        memory: Optional[HierarchicalMemory] = None,
        reranker: Optional[FlashReranker] = None,
        resource_manager: Optional[ResourceManagerV2] = None,
        vector_store: Optional[object] = None,
        enable_rag: bool = True,
        enable_rerank: bool = True,
        enable_memory: bool = True,
        enable_resource_monitor: bool = False,
    ):
        self.model_pool = model_pool or get_model_pool()
        self.memory = memory or HierarchicalMemory()
        self.reranker = reranker or get_reranker(alpha=0.7)
        self.resource_manager = resource_manager or get_resource_manager()

        self._vector_store = vector_store

        # ── Interruption intelligente ──
        self._cancel_event = threading.Event()
        self._cancelled = False

        # Flags de configuration
        self._enable_rag = enable_rag
        self._enable_rerank = enable_rerank
        self._enable_memory = enable_memory
        self._enable_resource_monitor = enable_resource_monitor

        # RAG lazy init
        self._rag_store = None
        self._classifier = None

        # Monitoring
        self._monitor = None
        if MONITOR_AVAILABLE:
            self._monitor = get_monitor()

    def cancel(self) -> None:
        """Annule la génération en cours proprement."""
        self._cancelled = True
        self._cancel_event.set()
        logger.info("Génération annulée par l'utilisateur")

    def reset_cancel(self) -> None:
        """Réinitialise l'état d'annulation pour une nouvelle génération."""
        self._cancelled = False
        self._cancel_event.clear()
        logger.info("NuruPipelineV2 initialisé")

    def _check_cancel(self) -> bool:
        """Vérifie si la génération a été annulée. Nettoie si nécessaire."""
        if self._cancelled:
            logger.info("Génération interrompue proprement")
            self.reset_cancel()
            return True
        return False

    # ── Lazy loading ──

    def _get_classifier(self):
        if self._classifier is None:
            try:
                from complexity_classifier import get_complexity_classifier
                self._classifier = get_complexity_classifier()
            except Exception as e:
                logger.warning("Classifieur non disponible : %s", e)
                self._classifier = False
        return self._classifier if self._classifier is not False else None

    def _get_rag_store(self):
        if self._rag_store is None:
            if self._vector_store is not None:
                self._rag_store = self._vector_store
            elif RAG_AVAILABLE:
                try:
                    self._rag_store = VectorStore()
                    logger.info("VectorStore ChromaDB connecté (%d documents)",
                                self._rag_store.count_documents())
                except Exception as e:
                    logger.warning("VectorStore non disponible : %s", e)
                    self._rag_store = False
        return self._rag_store if self._rag_store is not False else None

    def _map_complexity(self, level: Level) -> int:
        """Mappe Level (V1) → Complexity (V2)."""
        if level is None or Complexity is None:
            return 2  # MEDIUM par défaut
        if level.value == 1:  # LOCAL
            return 1  # SIMPLE (RAG simple)
        elif level.value == 2:  # GENERAL
            return 2  # MEDIUM
        elif level.value == 3:  # CLOUD
            return 3  # COMPLEX
        return 2

    # ── Sentence boundary detection ──

    @staticmethod
    def _has_sentence_boundary(text: str) -> bool:
        """Vérifie si le texte contient une fin de phrase."""
        return bool(re.search(r'[.!?]\s*$', text))

    # ── Orchestration synchrone (compatible V1) ──

    def handle_sync(self, query: str, **kwargs) -> Generator[dict, None, None]:
        """
        Version synchrone de handle() — compatible V1.

        Args:
            query: Requête utilisateur
            **kwargs: Paramètres additionnels (temperature, max_tokens, etc.)

        Yields:
            dict: Chunks de réponse (compatibles router.py)
        """
        # Démarrer le resource monitor si demandé
        if self._enable_resource_monitor:
            self.resource_manager.start()

        if self._monitor:
            self._monitor.start_timer("total")
            self._monitor.record_ram()

        start = time.time()

        try:
            yield from self._execute_pipeline(query, **kwargs)
        except Exception as e:
            logger.error("Erreur pipeline : %s", e, exc_info=True)
            yield {"type": "status", "msg": f"Erreur : {e}"}
            yield {"type": "token", "token": f"Une erreur s'est produite : {e}", "final": True}

        elapsed = (time.time() - start) * 1000

        if self._monitor:
            self._monitor.stop_timer("total")

        if self._enable_resource_monitor:
            self.resource_manager.stop()

    # ── Pipeline principal (synchrone) ──

    def _execute_pipeline(self, query: str, **kwargs) -> Generator[dict, None, None]:
        """
        Exécute le pipeline complet de manière synchrone.

        Étapes :
          1. Classification
          2. RAG + Memory (séquentiel car synchrone)
          3. Re-ranking
          4. Génération
        """
        # ── Étape 1 : Classification ──
        classifier = self._get_classifier()
        intent = None
        level = None

        if classifier:
            result = classifier.classify(query)
            intent = result.intent
            level = result.level
            complexity = self._map_complexity(level)

            logger.debug("Classification : intent=%s, level=%s, score=%.3f",
                         intent, level, result.score)

            # Cas spécial : identité / utilisateur
            if intent in [Intent.IDENTITY, Intent.USER]:
                desc = self.memory.describe_user()
                if desc:
                    yield {"type": "start", "level": 1, "level_name": "Mémoire", "reason": result.reason}
                    yield {"type": "token", "token": desc, "final": True}
                    yield {"type": "end", "model_used": "structured_memory", "latency_ms": 1.0}
                    return

            yield {
                "type": "start",
                "level": level.value if level else 2,
                "level_name": str(level.name) if level else "GENERAL",
                "reason": result.reason,
            }
        else:
            complexity = 2  # MEDIUM par défaut
            yield {"type": "start", "level": 2, "level_name": "GENERAL", "reason": "Fallback (classifieur indisponible)"}

        # ── Étape 2 : RAG (si activé) ──
        rag_context = ""
        sources = []

        if self._enable_rag:
            store = self._get_rag_store()
            if store is not None:
                try:
                    yield {"type": "status", "msg": "Recherche dans les documents..."}
                    rag_start = time.time()

                    if hasattr(store, 'count_documents') and store.count_documents() > 0:
                        rag_results = store.search(query, k=5)

                        if rag_results:
                            # Re-ranking optionnel
                            if self._enable_rerank and self.reranker and len(rag_results) > 1:
                                try:
                                    rag_results = self.reranker.rerank(
                                        query, rag_results, top_k=5
                                    )
                                except Exception as e:
                                    logger.debug("Re-ranking ignoré : %s", e)

                            rag_context = store.format_context(rag_results)
                            sources = [
                                r["metadata"].get("filename", "Doc")
                                for r in rag_results if r.get("metadata")
                            ]
                            rag_elapsed = (time.time() - rag_start) * 1000
                            yield {
                                "type": "rag_hits",
                                "sources": sources,
                                "latency_ms": round(rag_elapsed, 1),
                            }
                except Exception as e:
                    logger.warning("Erreur RAG : %s", e)

        # ── Étape 2bis : Memory retrieval (si activé) ──
        memory_context = ""
        if self._enable_memory and self.memory:
            try:
                memory_context = self.memory.get_context(query)
                logger.debug("Contexte mémoire récupéré (%d chars)", len(memory_context))
            except Exception as e:
                logger.warning("Erreur récupération mémoire : %s", e)

        # ── Étape 3 : Construction du prompt ──
        prompt_parts = ["<|im_start|>system"]
        prompt_parts.append("Tu es NURU, un assistant IA personnel. Réponds de manière précise et concise.")

        # Contexte utilisateur (mémoire sémantique)
        user_desc = self.memory.describe_user()
        if user_desc:
            prompt_parts.append(f"\n[Utilisateur]\n{user_desc}")

        # Contexte de travail (échanges récents)
        if memory_context:
            prompt_parts.append(f"\n[Contexte conversationnel]\n{memory_context}")

        # Contexte RAG
        if rag_context:
            prompt_parts.append(f"\n[Contexte documentaire]\n{rag_context}")

        prompt_parts.append("<|im_end|>")
        prompt_parts.append(f"<|im_start|>user\n{query}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant")

        full_prompt = "\n".join(prompt_parts)

        # ── Étape 4 : Génération via ModelPoolV2 ──
        max_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        repetition_penalty = kwargs.get("repetition_penalty", 1.15)

        # Température adaptative
        if rag_context:
            temperature = min(temperature, 0.3)

        yield {"type": "status", "msg": f"Génération (modèle: {['nano', 'default', 'reasoning'][complexity-1]})..."}

        if self._monitor:
            self._monitor.start_timer("generate")

        full_response = []
        for token in self.model_pool.stream(
            prompt=full_prompt,
            complexity=complexity,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            full_response.append(token)
            yield {"type": "token", "token": token}

        response_text = "".join(full_response)

        if self._monitor:
            self._monitor.stop_timer("generate", tokens=len(full_response))

        # Ne pas renvoyer toute la réponse à nouveau (déjà streamé)
        yield {"type": "token", "token": "", "final": True}

        # ── Post-génération : mémoire ──
        if self._enable_memory and self.memory:
            try:
                self.memory.add(query, response_text)
            except Exception as e:
                logger.warning("Erreur sauvegarde mémoire : %s", e)

        elapsed = round((time.time() - self._monitor._timers.get("total", (time.time(), 0))[0]) * 1000, 1) if self._monitor else 0

        # Récupérer le nom du modèle utilisé
        stats = self.model_pool.get_stats()
        model_used = stats.get("current_model", "default")

        yield {"type": "end", "model_used": model_used, "latency_ms": elapsed}

    # ── Version asynchrone (optionnelle) ──

    async def handle(self, query: str, **kwargs) -> AsyncGenerator[dict, None]:
        """
        Version asynchrone qui parallélise RAG + Memory.

        Args:
            query: Requête utilisateur
            **kwargs: Paramètres additionnels

        Yields:
            dict: Chunks de réponse
        """
        if not ASYNCIO_AVAILABLE:
            logger.warning("asyncio non disponible — fallback synchrone")
            for chunk in self.handle_sync(query, **kwargs):
                yield chunk
            return

        # Démarrer le resource monitor si demandé
        if self._enable_resource_monitor:
            self.resource_manager.start()

        if self._monitor:
            self._monitor.start_timer("total")
            self._monitor.record_ram()

        start = time.time()

        try:
            # ── Étape 1 : Classification (synchrone, < 50ms) ──
            classifier = self._get_classifier()
            intent = None
            level = None

            if classifier:
                result = classifier.classify(query)
                intent = result.intent
                level = result.level
                complexity = self._map_complexity(level)

                if intent == Intent.IDENTITY:
                    desc = self.memory.describe_user()
                    if desc:
                        yield {"type": "start", "level": 1, "level_name": "Local", "reason": result.reason}
                        yield {"type": "token", "token": desc, "final": True}
                        yield {"type": "end", "model_used": "structured_memory", "latency_ms": 1.0}
                        return

                yield {
                    "type": "start",
                    "level": level.value if level else 2,
                    "level_name": str(level.name) if level else "GENERAL",
                    "reason": result.reason,
                }
            else:
                complexity = 2
                yield {"type": "start", "level": 2, "level_name": "GENERAL", "reason": "Fallback"}

            # ── Étape 2 : RAG + Memory en parallèle ──
            rag_context = ""
            sources = []
            memory_context = ""

            async def _do_rag():
                nonlocal rag_context, sources
                if not self._enable_rag:
                    return
                store = self._get_rag_store()
                if store is None:
                    return
                try:
                    if hasattr(store, 'count_documents') and store.count_documents() > 0:
                        rag_results = store.search(query, k=5)
                        if rag_results:
                            if self._enable_rerank and self.reranker and len(rag_results) > 1:
                                rag_results = self.reranker.rerank(query, rag_results, top_k=5)
                            rag_context = store.format_context(rag_results)
                            sources = [r["metadata"].get("filename", "Doc") for r in rag_results if r.get("metadata")]
                except Exception as e:
                    logger.warning("Erreur RAG async : %s", e)

            async def _do_memory():
                nonlocal memory_context
                if not self._enable_memory or not self.memory:
                    return
                try:
                    memory_context = self.memory.get_context(query)
                except Exception as e:
                    logger.warning("Erreur mémoire async : %s", e)

            yield {"type": "status", "msg": "Recherche parallèle RAG + Mémoire..."}

            await asyncio.gather(_do_rag(), _do_memory())

            if sources:
                yield {"type": "rag_hits", "sources": sources, "latency_ms": 0}

            # ── Étape 3 : Construction du prompt (identique synchrone) ──
            prompt_parts = ["<|im_start|>system"]
            prompt_parts.append("Tu es NURU, un assistant IA personnel. Réponds de manière précise et concise.")

            user_desc = self.memory.describe_user()
            if user_desc:
                prompt_parts.append(f"\n[Utilisateur]\n{user_desc}")
            if memory_context:
                prompt_parts.append(f"\n[Contexte conversationnel]\n{memory_context}")
            if rag_context:
                prompt_parts.append(f"\n[Contexte documentaire]\n{rag_context}")

            prompt_parts.append("<|im_end|>")
            prompt_parts.append(f"<|im_start|>user\n{query}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant")
            full_prompt = "\n".join(prompt_parts)

            # ── Étape 4 : Génération ──
            max_tokens = kwargs.get("max_tokens", 2048)
            temperature = kwargs.get("temperature", 0.7)
            if rag_context:
                temperature = min(temperature, 0.3)

            yield {"type": "status", "msg": f"Génération (modèle: {['nano', 'default', 'reasoning'][complexity-1]})..."}

            if self._monitor:
                self._monitor.start_timer("generate")

            # Utiliser le générateur synchrone dans un thread pool
            full_response = []
            loop = asyncio.get_event_loop()

            def _generate():
                tokens = []
                for token in self.model_pool.stream(
                    prompt=full_prompt,
                    complexity=complexity,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    tokens.append(token)
                return "".join(tokens)

            response_text = await loop.run_in_executor(None, _generate)

            if self._monitor:
                self._monitor.stop_timer("generate", tokens=len(response_text.split()))

            # Ne pas renvoyer toute la réponse à nouveau (déjà streamé)
            yield {"type": "token", "token": "", "final": True}

            # ── Post-génération ──
            if self._enable_memory and self.memory:
                try:
                    self.memory.add(query, response_text)
                except Exception as e:
                    logger.warning("Erreur sauvegarde mémoire : %s", e)

            elapsed = round((time.time() - start) * 1000, 1)
            stats = self.model_pool.get_stats()
            model_used = stats.get("current_model", "default")

            yield {"type": "end", "model_used": model_used, "latency_ms": elapsed}

        except Exception as e:
            logger.error("Erreur pipeline async : %s", e, exc_info=True)
            yield {"type": "status", "msg": f"Erreur : {e}"}
            yield {"type": "token", "token": f"Une erreur s'est produite : {e}", "final": True}

        finally:
            if self._enable_resource_monitor:
                self.resource_manager.stop()

    # ── Utilitaires ──

    def get_stats(self) -> dict:
        """Retourne les statistiques combinées des sous-composants."""
        return {
            "model_pool": self.model_pool.get_stats(),
            "memory": self.memory.get_stats() if self.memory else {},
            "resource_manager": self.resource_manager.get_stats(),
            "rag_enabled": self._enable_rag,
            "rerank_enabled": self._enable_rerank,
            "memory_enabled": self._enable_memory,
        }

    def __repr__(self) -> str:
        return (
            f"NuruPipelineV2(rag={self._enable_rag}, "
            f"rerank={self._enable_rerank}, "
            f"memory={self._enable_memory})"
        )
