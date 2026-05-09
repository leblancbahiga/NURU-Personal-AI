#!/usr/bin/env python3
"""
model_pool_v2.py — Pool de modèles MLX à 3 niveaux (V2).

Niveaux :
  - nano    : Qwen2.5-1.5B-Instruct-4bit  (0.9 Go, ~70 tok/s)
  - default : Qwen2.5-3B-Instruct-4bit    (1.8 Go, ~42 tok/s)
  - reasoning: phi-4-mini-instruct-4bit   (2.3 Go, ~28 tok/s, optionnel)

Fonctionnalités :
  - Lazy loading avec LRU eviction (1 modèle chargé à la fois)
  - Streaming async generator
  - Thread-safe (threading.Lock, pas asyncio pour compatibilité V1)
  - get_stats(), unload_all()
  - Détection RAM via psutil

Usage :
    pool = ModelPoolV2()
    model, tokenizer = pool.get_model(Complexity.SIMPLE)
    async for chunk in pool.stream("Bonjour", Complexity.MEDIUM):
        print(chunk)
"""

import sys
import time
import gc
import threading
import logging
from enum import Enum
from typing import Optional, Generator, AsyncGenerator

logger = logging.getLogger("nuru.v2.model_pool")

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, generate, stream_generate
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ── Définition des modèles disponibles ──
MODEL_REGISTRY = {
    "nano": {
        "repo_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "ram_gb": 0.9,
        "tokens_per_sec": 70,
        "description": "Qwen 1.5B quantifié 4-bit — questions simples, RAG",
    },
    "default": {
        "repo_id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "ram_gb": 1.8,
        "tokens_per_sec": 42,
        "description": "Qwen 3B quantifié 4-bit — usage général, contexte RAG",
    },
    "reasoning": {
        "repo_id": "mlx-community/phi-4-mini-instruct-4bit",
        "ram_gb": 2.3,
        "tokens_per_sec": 28,
        "description": "Phi-4 Mini 4-bit — raisonnement complexe (optionnel)",
    },
}

# Mapping Complexity → nom du modèle
COMPLEXITY_MODEL_MAP = {
    1: "nano",     # Complexity.SIMPLE
    2: "default",  # Complexity.MEDIUM
    3: "reasoning",# Complexity.COMPLEX
}


class ModelPoolV2:
    """
    Pool thread-safe de modèles MLX.

    Stratégie LRU : un seul modèle chargé à la fois.
    Si le modèle demandé est déjà chargé → réutilisation immédiate.
    Sinon → déchargement + chargement du nouveau.
    """

    def __init__(self, max_loaded: int = 1):
        self.max_loaded = max_loaded
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._current_key: Optional[str] = None
        self._load_time: float = 0.0
        self._total_loads: int = 0
        self._total_streams: int = 0
        self._total_tokens: int = 0
        self._errors: int = 0
        self._reasoning_enabled: bool = self._check_reasoning_available()

    def _check_reasoning_available(self) -> bool:
        """Vérifie si assez de RAM pour le modèle reasoning (phi-4 mini)."""
        if not PSUTIL_AVAILABLE:
            return False
        mem = psutil.virtual_memory()
        needed = MODEL_REGISTRY["reasoning"]["ram_gb"]
        # Garder 1.5 Go de marge
        return mem.available / (1024**3) > needed + 1.5

    def _available_ram_gb(self) -> float:
        """Retourne la RAM disponible en Go."""
        if not PSUTIL_AVAILABLE:
            return 99.0  # Optimiste si psutil absent
        return psutil.virtual_memory().available / (1024**3)

    def _evict_if_needed(self) -> None:
        """Décharge le modèle courant si la RAM disponible est trop basse."""
        if not MLX_AVAILABLE:
            return
        available = self._available_ram_gb()
        if available < 1.5:
            logger.warning("RAM critique (%.1f Go libre) — éviction du modèle", available)
            self._unload_current()

    def _unload_current(self) -> None:
        """Libère le modèle et le tokenizer courant."""
        self._model = None
        self._tokenizer = None
        self._current_key = None
        gc.collect()
        if MLX_AVAILABLE:
            mx.clear_cache()
            logger.debug("Cache MLX vidé après déchargement")

    def enable_reasoning(self, enabled: bool) -> None:
        """Active ou désactive le modèle reasoning."""
        self._reasoning_enabled = enabled

    def get_model(self, complexity: int):
        """
        Charge (ou réutilise) le modèle adapté à la complexité donnée.

        Args:
            complexity: 1 (SIMPLE), 2 (MEDIUM), 3 (COMPLEX)

        Retourne:
            (model, tokenizer)

        Lève:
            RuntimeError si MLX non disponible
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX n'est pas installé — pip install mlx mlx-lm")

        model_key = COMPLEXITY_MODEL_MAP.get(complexity, "default")
        if model_key == "reasoning" and not self._reasoning_enabled:
            model_key = "default"
            logger.info("Modèle reasoning désactivé, fallback sur default")

        model_info = MODEL_REGISTRY[model_key]

        with self._lock:
            # Déjà chargé ?
            if self._current_key == model_key and self._model is not None:
                logger.debug("Réutilisation de %s (déjà chargé)", model_key)
                return self._model, self._tokenizer

            # Éviction si nécessaire
            self._unload_current()

            # Chargement
            repo_id = model_info["repo_id"]
            logger.info("Chargement de %s (%s)…", model_key, repo_id)
            t0 = time.time()
            try:
                self._model, self._tokenizer = load(repo_id)
                self._current_key = model_key
                self._load_time = time.time() - t0
                self._total_loads += 1
                logger.info("✓ %s chargé en %.1fs (RAM libre: %.1f Go)",
                            model_key, self._load_time, self._available_ram_gb())
            except Exception as e:
                self._errors += 1
                logger.error("Échec chargement %s : %s", repo_id, e)
                # Fallback : essayer le modèle default
                if model_key != "default":
                    logger.warning("Fallback sur le modèle default")
                    return self.get_model(2)  # Complexity.MEDIUM
                raise RuntimeError(f"Impossible de charger un modèle : {e}") from e

            return self._model, self._tokenizer

    def stream(
        self,
        prompt: str,
        complexity: int = 2,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> Generator[str, None, None]:
        """
        Génère une réponse en streaming synchrone.

        Args:
            prompt: Prompt formaté pour le modèle
            complexity: 1, 2 ou 3
            max_tokens: Nombre max de tokens à générer
            temperature: Température d'échantillonnage
            top_p: Nucleus sampling
            repetition_penalty: Pénalité de répétition

        Yields:
            str: Tokens de la réponse
        """
        if not MLX_AVAILABLE:
            yield "[ERREUR: MLX non disponible]"
            return

        model, tokenizer = self.get_model(complexity)
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]

        sampler = make_sampler(temp=temperature, top_p=top_p)
        logits_processors = [make_repetition_penalty(repetition_penalty)]

        token_count = 0
        try:
            for chunk in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                token = chunk.text
                if any(st in token for st in stop_tokens):
                    break
                token_count += 1
                yield token

            with self._lock:
                self._total_streams += 1
                self._total_tokens += token_count
        except Exception as e:
            self._errors += 1
            logger.error("Erreur de génération : %s", e)
            yield f"[ERREUR: {e}]"

    def unload_all(self) -> None:
        """Décharge tous les modèles et vide les caches MLX."""
        with self._lock:
            self._unload_current()
            self._total_loads = 0
            logger.info("Tous les modèles déchargés")

    def get_stats(self) -> dict:
        """Retourne les statistiques du pool."""
        with self._lock:
            return {
                "current_model": self._current_key,
                "load_time_sec": round(self._load_time, 2),
                "total_loads": self._total_loads,
                "total_streams": self._total_streams,
                "total_tokens": self._total_tokens,
                "errors": self._errors,
                "reasoning_enabled": self._reasoning_enabled,
                "ram_available_gb": round(self._available_ram_gb(), 2),
                "models_available": {
                    k: {
                        "repo_id": v["repo_id"],
                        "ram_gb": v["ram_gb"],
                        "tokens_per_sec": v["tokens_per_sec"],
                    }
                    for k, v in MODEL_REGISTRY.items()
                },
            }

    def __repr__(self) -> str:
        return f"ModelPoolV2(current={self._current_key}, loads={self._total_loads}, errors={self._errors})"


# ── Singleton partagé ──
_pool_instance: Optional[ModelPoolV2] = None
_pool_lock = threading.Lock()


def get_model_pool() -> ModelPoolV2:
    """Retourne l'instance singleton du pool de modèles V2."""
    global _pool_instance
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                _pool_instance = ModelPoolV2()
    return _pool_instance
