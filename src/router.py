#!/usr/bin/env python3
"""
router.py — Routeur cognitif à 3 niveaux pour NURU.
             Compatible V1 (par défaut) + V2 (optionnel via use_v2=True)

Niveau 1 : RAG Local    → Recherche dans l'index vectoriel (similarité > 0.75)
Niveau 2 : LLM Local    → Inférence via MLX + Qwen 2.5 (requêtes générales)
Niveau 3 : Cloud API    → Deepseek / OpenRouter (code, analyse, recherches web)

V2 active : ModelPoolV2, FlashReranker, ResourceManagerV2, HierarchicalMemory.
"""

import sys
import json
import time
import os
import gc
import signal
import threading
from pathlib import Path

# Gérer SIGPIPE pour éviter BrokenPipeError si la sortie est bridée (ex: | head)
if sys.platform != "win32":
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass

# Supprimer les warnings HF Hub (token manquant, progress bars)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*HF_HUB_DISABLE_SYMLINKS_WARNING.*")

from typing import Optional, Generator
from dataclasses import dataclass, field

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from memory import SessionMemory
from monitor import get_monitor
from semantic_cache import SemanticCache
from structured_memory import StructuredMemory

try:
    from action_engine import ActionEngine
    ACTION_ENGINE_AVAILABLE = True
except ImportError:
    ACTION_ENGINE_AVAILABLE = False
from transparency import get_transparency_logger
from complexity_classifier import get_complexity_classifier, Level, Intent

try:
    import yaml
except ImportError:
    yaml = None

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, generate, stream_generate
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from keychain_utils import get_key, load_config_service
    KEYCHAIN_AVAILABLE = True
except ImportError:
    KEYCHAIN_AVAILABLE = False

# RAG (lazy import)
_VECTOR_STORE = None

# Cache pour l'embedder de correction (évite de charger 500MB à chaque appel)
_correction_embedder = None

def get_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        try:
            from rag import VectorStore
            _VECTOR_STORE = VectorStore()
        except Exception as e:
            print(f"  ⚠ ChromaDB non disponible : {e}", file=sys.stderr)
            _VECTOR_STORE = False  # False = pas disponible, pas None (pour éviter de réessayer)
    return _VECTOR_STORE if _VECTOR_STORE is not False else None


# ── Niveaux de routage (définis dans complexity_classifier) ──
# Les alias sont conservés pour la compatibilité interne si nécessaire
class RouterLevel:
    LOCAL_RAG = Level.LOCAL.value
    LOCAL_LLM = Level.GENERAL.value
    CLOUD = Level.CLOUD.value


@dataclass
class RouteResult:
    """Résultat d'un routage."""
    level: Level
    content: str
    model_used: str
    latency_ms: float
    sources: list[str] = field(default_factory=list)
    level_name: str = ""


LEVEL_NAMES = {
    Level.LOCAL.value: "RAG Local",
    Level.GENERAL.value: "LLM Local",
    Level.CLOUD.value: "Cloud API",
}


class Router:
    """
    Routeur à 3 niveaux avec support streaming et gestion RAM.
    """

    def __init__(self, config_path: str | Path | None = None, memory: Optional[SessionMemory] = None,
                 use_semantic_cache: bool = True):
        self.config = self._load_config(config_path)
        self.memory = memory or SessionMemory()
        self.structured_memory = StructuredMemory()
        self.airplane_mode = False  # Mode Avion : désactive le cloud
        self.force_cloud = False   # Force l'utilisation du cloud
        self.web_search_mode = False  # Active la recherche web via Brave
        self._disable_rag = False     # Désactive le RAG (mode turbo)
        self.use_semantic_cache = use_semantic_cache
        self.semantic_cache = SemanticCache() if use_semantic_cache else None
        # HyDE : Hypothetical Document Embeddings — désactivé par défaut (config)
        self.use_hyde = self.config.get("rag", {}).get("hyde", False)
        # Re-ranker — désactivé par défaut (config)
        self.use_reranker = self.config.get("rag", {}).get("reranker", False)
        # Propager la config au module rag.py (flag global)
        try:
            from rag import set_reranker_enabled
            set_reranker_enabled(self.use_reranker)
        except Exception:
            pass
        
        # Gestion du modèle local
        self._local_model = None
        self._local_tokenizer = None
        self._model_id = self._get_model_id()
        self._last_model_use = 0
        self._auto_unload_delay = 300  # 5 minutes d'inactivité
        
        self.monitor = get_monitor()
        self.transparency = get_transparency_logger()
        self.complexity = get_complexity_classifier()
        self.action_engine = ActionEngine() if ACTION_ENGINE_AVAILABLE else None
        self._preload_user_facts()
        # Préchargement du modèle d'embedding RAG en arrière-plan
        # (évite le blocage de 30s au premier appel RAG)
        threading.Thread(target=self._warmup_embedder, daemon=True).start()

        # ── V2 : Modules optionnels (ModelPool, FlashReranker, ResourceManager, Memory) ──
        self.v2_enabled = False
        self.v2_model_pool = None
        self.v2_reranker = None
        self.v2_resource_manager = None
        self.v2_memory = None
        self.v2_pipeline = None

    def _warmup_embedder(self):
        """Préchauffe le modèle d'embedding RAG sans bloquer l'init."""
        try:
            from rag import get_embedder
            get_embedder()
        except Exception:
            pass

    # ── Gestion de la RAM ──

    def check_memory_maintenance(self):
        """Décharge le modèle si inactif trop longtemps."""
        if self._local_model is not None and (time.time() - self._last_model_use) > self._auto_unload_delay:
            self.unload_model()

    def unload_model(self):
        """Libère la RAM en supprimant le modèle chargé."""
        if self._local_model is not None:
            print(f"  ♻️ Déchargement de {self._model_id} (inactivité)", file=sys.stderr)
            self._local_model = None
            self._local_tokenizer = None
            gc.collect()
            if MLX_AVAILABLE:
                mx.clear_cache()

    # ── Préchargement des faits utilisateur ──

    def _preload_user_facts(self):
        """Précharge les faits connus sur l'utilisateur dans la mémoire structurée."""
        known_facts = {
            "nom": "Leblanc BAHIGA Mudarhi",
            "prenom": "Leblanc",
            "profession": "Ingénieur agronome & informaticien",
            "lieu": "RDC (Kinshasa)",
            "projet": "NURU — assistant IA personnel",
        }
        for key, value in known_facts.items():
            if not self.structured_memory.get_fact(key):
                self.structured_memory.store_fact(key, value, confidence=0.95)

    # ── Configuration ──

    def _load_config(self, config_path: str | Path | None) -> dict:
        if yaml is None or config_path is None:
            return {}
        path = Path(config_path) if isinstance(config_path, str) else config_path
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _get_model_id(self) -> str:
        try:
            return self.config["models"]["local"]["llm"]["repo_id"]
        except (KeyError, TypeError):
            return "mlx-community/Qwen2.5-3B-Instruct-4bit"

    # ── Fenêtre de contexte dynamique ──

    def _get_dynamic_config(self, score: float) -> dict:
        """
        Retourne la config dynamique (max_tokens, rag_chunks, memory_tokens)
        selon le score de complexité de la requête.
        Lit depuis config.yaml si disponible, sinon utilise les valeurs par défaut.
        """
        try:
            dc = self.config.get("rag", {}).get("dynamic_context", {})
            if not dc.get("enabled", True):
                return {"max_tokens": 2048, "rag_chunks": 5, "memory_tokens": 500}

            if score < 0.36:
                profile = dc.get("simple", {})
            elif score > 0.55:
                profile = dc.get("complex", {})
            else:
                profile = dc.get("medium", {})
        except Exception:
            profile = {}

        return {
            "max_tokens": profile.get("max_tokens", 1024),
            "rag_chunks": profile.get("rag_chunks", 3),
            "memory_tokens": profile.get("memory_tokens", 300),
        }

    # ── Routage (décision) ──

    def _get_prompt_cache(self, prompt: str) -> list[int] | None:
        """
        Cache les tokens du prompt système pour éviter la re-tokenization.
        Retourne les tokens si le prompt est identique au cache, None sinon.
        """
        if not hasattr(self, '_cached_prompt') or not hasattr(self, '_cached_tokens'):
            return None
        if self._cached_prompt == prompt:
            return self._cached_tokens
        return None

    def _set_prompt_cache(self, prompt: str, tokens: list[int]):
        """Stocke le prompt et ses tokens dans le cache."""
        self._cached_prompt = prompt
        self._cached_tokens = tokens

    def decide_level(self, query: str, rag_score: float | None = None) -> tuple[Intent, Level, str]:
        """
        Décide à quel niveau router la requête en utilisant le ComplexityClassifier.
        """
        if (self.force_cloud or self.web_search_mode) and not self.airplane_mode:
            return Intent.WEB if self.web_search_mode else Intent.CHAT, Level.CLOUD, "Mode Cloud/Web forcé par l'utilisateur"

        classifier = self.complexity
        result = classifier.classify(query, rag_score)

        return result.intent, result.level, result.reason

    # ── Exécution Streaming ──

    def stream_route(self, query: str, user_confirmed_cloud: bool = False) -> Generator[dict, None, None]:
        """
        Route et exécute une requête en mode streaming.

        Nouveau flux N1/N2/N3 basé sur le ComplexityClassifier :
        - N1 (LOCAL_RAG) : Qwen avec contexte documentaire
        - N2 (LOCAL_LLM) : Qwen connaissances générales
        - N3 (CLOUD)     : Brave Search + Deepseek v4 Flash
        """
        # Vérifier les corrections prioritaires
        correction = self._check_corrections(query)
        if correction:
            yield {"type": "status", "msg": "Correction trouvée..."}
            yield {"type": "token", "token": correction, "final": True}
            yield {"type": "end", "model_used": "correction_memoire", "latency_ms": 1.0}
            return

        self.check_memory_maintenance()
        mon = self.monitor
        mon.start_timer("total")
        mon.record_ram()

        start = time.time()
        
        # ── Étape 0 : Interception immédiate (Identité/User) ──
        # On le fait AVANT le RAG pour éviter la latence de chargement des modèles d'embedding
        intent_pre, level_pre, reason_pre = self.decide_level(query, 0.0)
        
        if intent_pre == Intent.USER:
            desc = self.structured_memory.describe_user()
            if desc:
                yield {"type": "start", "level": Level.LOCAL.value, "intent": Intent.USER.name,
                       "level_name": "Mémoire", "reason": "Identité utilisateur détectée"}
                yield {"type": "token", "token": desc, "final": True}
                yield {"type": "end", "model_used": "structured_memory", "latency_ms": 1.0}
                return
        elif intent_pre == Intent.NURU:
            nuru_presentation = (
                "Je suis NURU, un assistant IA personnel de nouvelle génération conçu et piloté par Leblanc BAHIGA Mudarhi. "
                "Je fonctionne localement sur Apple Silicon avec un modèle Qwen 2.5 optimisé, "
                "et je peux accéder au cloud Deepseek pour les tâches complexes.\n\n"
                "**Mes capacités clés :**\n"
                "- 💻 **Programmation** : Je maîtrise Python, JavaScript et l'architecture logicielle.\n"
                "- 🔍 **RAG Local** : J'analyse tes documents personnels en toute confidentialité.\n"
                "- 🌐 **Recherche Web** : Je peux consulter Internet pour des données en temps réel.\n"
                "- 🧠 **Apprentissage** : J'apprends de nos échanges pour devenir ton partenaire idéal."
            )
            yield {"type": "start", "level": Level.LOCAL.value, "intent": Intent.NURU.name,
                   "level_name": "Mémoire", "reason": "Identité assistante détectée"}
            yield {"type": "token", "token": nuru_presentation, "final": True}
            yield {"type": "end", "model_used": "structured_memory", "latency_ms": 1.0}
            return

        rag_context = ""
        corrections_context = ""
        sources = []
        rag_score = None

        # ── Étape 1 : Recherche RAG (AVANT decide_level pour le score) ──
        # ⚡ Sauter le RAG si le mode Web/Cloud est forcé (bouton 🌐)
        skip_rag = self.web_search_mode or self.force_cloud
        if skip_rag:
            print(f"  ⚡ RAG sauté : web_search_mode={self.web_search_mode}, force_cloud={self.force_cloud}", file=sys.stderr)
            pass  # Pas de RAG → va directement en N3 via decide_level
        elif not self._disable_rag:
            try:
                store = get_vector_store()
                if store is not None and store.count_documents() > 0:
                    # Réécrire la requête pour améliorer la recherche RAG
                    yield {"type": "status", "msg": "Optimisation de la requête RAG..."}
                    rewritten_query = self._rewrite_query_for_rag(query)
                    if rewritten_query != query:
                        print(f"  ℹ Requête RAG réécrite : '{query[:50]}…' → '{rewritten_query[:60]}…'", file=sys.stderr)

                    # HyDE : Générer une réponse hypothétique pour enrichir la recherche
                    hyde_doc = None
                    if self.use_hyde:
                        yield {"type": "status", "msg": "🧠 Génération HyDE (réponse hypothétique)..."}
                        hyde_doc = self._generate_hyde_doc(rewritten_query)
                        if hyde_doc:
                            print(f"  ℹ HyDE généré ({len(hyde_doc)} chars)", file=sys.stderr)

                    yield {"type": "status", "msg": "Recherche dans les documents..."}
                    rag_start = time.time()
                    rag_results = store.search(rewritten_query, hyde_doc=hyde_doc, k=5)
                    rag_elapsed = (time.time() - rag_start) * 1000
                    if rag_results:
                        # Séparer les corrections (few-shot) des documents normaux
                        corrections_list = [r for r in rag_results if r.get("source") == "corrections"]
                        docs_list = [r for r in rag_results if r.get("source") != "corrections"]
                        
                        rag_context = store.format_context(docs_list)
                        corrections_context = "\n".join([f"- {r['text']}" for r in corrections_list])
                        
                        sources = [r["metadata"].get("filename", "Doc") for r in docs_list if r.get("metadata")]
                        # Calculer le meilleur score RAG (tout type confondu)
                        best_score = max(r.get("score", 0) for r in rag_results)
                        rag_score = best_score
                        yield {
                            "type": "rag_hits",
                            "sources": sources,
                            "latency_ms": round(rag_elapsed, 1)
                        }
            except Exception as e:
                print(f"  ⚠ Erreur RAG : {e}", file=sys.stderr)

        # ── Étape 2 : Décider de l'intention et du niveau via le nouveau classifieur ──
        intent, level, reason = self.decide_level(query, rag_score)
        
        # Config dynamique : max_tokens + RAG chunks selon score
        # Note : on utilise le score du classify déjà fait dans decide_level
        classify_result = self.complexity.classify(query, rag_score)
        dynamic = self._get_dynamic_config(classify_result.score)
        print(f"  ℹ Dynamic context: max_tokens={dynamic['max_tokens']}, rag_chunks={dynamic['rag_chunks']}", file=sys.stderr)
        
        # Limiter les chunks RAG selon la config dynamique (protégé : docs_list existe seulement si RAG a retourné des résultats)
        if rag_context and 'docs_list' in dir() and dynamic["rag_chunks"] < len(docs_list):
            docs_list = docs_list[:dynamic["rag_chunks"]]
            rag_context = store.format_context(docs_list)
        
        mon.set_level(level.value)

        # (Anciennes interceptions d'identité déplacées à l'étape 0)

        yield {
            "type": "start",
            "level": level.value,
            "intent": intent.name,
            "level_name": LEVEL_NAMES.get(level.value, "Inconnu"),
            "reason": reason
        }

        # Pas de confirmation requise — le classifieur a déjà décidé

        # Vérifier le cache sémantique
        if self.use_semantic_cache and self.semantic_cache is not None:
            cached_response = self.semantic_cache.get(query)
            if cached_response is not None:
                yield {"type": "token", "token": cached_response, "final": True}
                yield {"type": "end", "model_used": "semantic_cache", "latency_ms": 1.0}
                return

        # ── Étape 3 : Exécution selon le niveau (N1/N2/N3) ──

        if level == Level.CLOUD:
            # ── N3 : Cloud avec Brave Search ──
            yield {"type": "status", "msg": "Recherche web via Brave Search..."}

            web_context = self._do_web_search(query)
            if web_context:
                yield {"type": "status", "msg": "Résultats web récupérés"}

            # Construire le prompt avec contexte
            forced_instruction = "Tu es NURU. Réponds à la question en utilisant le contexte fourni (Documents et Recherche Web) et tes connaissances.\n\n"

            prompt_cloud = f"{forced_instruction}"
            # V2 : utiliser prompts_v2 pour le prompt cloud
            if self.v2_enabled:
                try:
                    from prompts_v2 import build_prompt
                    prompt_cloud = build_prompt(
                        route="N3",
                        query=query,
                        memory_ctx=self.structured_memory.get_context(),
                        web_results=web_context,
                        rag_chunks=rag_results if rag_context else None,
                        rag_score=rag_score or 0.0,
                    )
                except Exception:
                    pass  # fallback V1
            if not self.v2_enabled or prompt_cloud == forced_instruction:
                if rag_context:
                    prompt_cloud += f"--- DOCUMENTS ---\n{rag_context}\n--- FIN DOCUMENTS ---\n\n"
                if web_context:
                    prompt_cloud += f"--- RECHERCHE WEB ---\n{web_context}\n--- FIN WEB ---\n\n"
                prompt_cloud += f"Question : {query}"

            yield {"type": "status", "msg": "Interrogation du Cloud (Deepseek v4 Flash)..."}
            content, model_used = self._execute_cloud(prompt_cloud)
            full_response = content
            yield {"type": "token", "token": content, "final": True}

        elif level == Level.LOCAL:
            # ── N1 : Local avec/sans RAG ──
            if rag_context:
                yield {"type": "status", "msg": f"Génération locale avec {len(sources)} documents..."}
                print(f"  ℹ RAG actif : {len(sources)} sources, {len(rag_context)} chars de contexte", file=sys.stderr)
            else:
                yield {"type": "status", "msg": "Génération locale (sans documents)..."}
                print(f"  ℹ RAG inactif : aucun document pertinent trouvé malgré niveau N1", file=sys.stderr)

            prompt = self._format_local_prompt(query, rag_context, corrections_context)
            # V2 : utiliser le nouveau système de prompts si activé
            if self.v2_enabled:
                try:
                    from prompts_v2 import build_prompt
                    route = "N1" if rag_context else "N2"
                    prompt = build_prompt(
                        route=route,
                        query=query,
                        memory_ctx=self.structured_memory.get_context(),
                        rag_chunks=rag_results if rag_context else None,
                        rag_score=rag_score or 0.0,
                        use_nano=(dynamic["max_tokens"] <= 256),
                    )
                except Exception:
                    pass  # fallback silencieux vers le prompt V1
            model, tokenizer = self._load_local_model()
            model_used = self._model_id
            self._last_model_use = time.time()
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
            temp = 0.3 if rag_context else 0.7

            tokens = []
            for response_chunk in stream_generate(
                model=model, tokenizer=tokenizer, prompt=prompt,
                max_tokens=dynamic["max_tokens"], sampler=make_sampler(temp=temp, top_p=0.9),
                logits_processors=[make_repetition_penalty(1.15)],
            ):
                token = response_chunk.text
                if any(st in token for st in stop_tokens):
                    break
                tokens.append(token)

            full_response = "".join(tokens)

            # Vérifier les signaux d'escalade (via regex, résistant au bruit)
            escalation_action = self._detect_escalade(full_response)
            if escalation_action == "[[ESCALADE:NIVEAU3]]":
                yield {"type": "status", "msg": "🌐 Recherche web..."}
                web_context_esc = self._do_web_search(query)
                yield {"type": "status", "msg": "⏫ Escalade vers le Cloud..."}
                prompt_cloud = query
                if web_context_esc:
                    prompt_cloud = web_context_esc + f"\n\nQuestion : {query}"
                content, cloud_model = self._execute_cloud(prompt_cloud)
                full_response = content
                model_used = cloud_model
                level = Level.CLOUD
            elif escalation_action == "[[ESCALADE:INTERNET]]":
                yield {"type": "status", "msg": "🌐 Recherche web..."}
                web_context_esc = self._do_web_search(query)
                if web_context_esc:
                    prompt_cloud = web_context_esc + f"\n\nQuestion : {query}"
                    content, cloud_model = self._execute_cloud(prompt_cloud)
                    full_response = content
                    model_used = cloud_model
                    level = Level.CLOUD
                else:
                    full_response = "La recherche web n'a retourné aucun résultat."
            elif escalation_action == "[[ESCALADE:INCONNU]]":
                full_response = "Je ne dispose pas d'informations suffisantes pour répondre à cette question de façon fiable."

            yield {"type": "token", "token": full_response, "final": True}

        else:  # Level.GENERAL
            # ── N2 : Local sans RAG (connaissances générales) ──
            yield {"type": "status", "msg": "Génération locale (connaissances générales)..."}
            print(f"  ℹ N2 : Génération sans RAG — connaissances générales", file=sys.stderr)

            prompt = self._format_local_prompt(query, "", corrections_context)
            # V2 : nouveaux prompts si activé
            if self.v2_enabled:
                try:
                    from prompts_v2 import build_prompt
                    prompt = build_prompt(
                        route="N2",
                        query=query,
                        memory_ctx=self.structured_memory.get_context(),
                        use_nano=(dynamic["max_tokens"] <= 256),
                    )
                except Exception:
                    pass  # fallback V1
            model, tokenizer = self._load_local_model()
            model_used = self._model_id
            self._last_model_use = time.time()
            stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]

            tokens = []
            for response_chunk in stream_generate(
                model=model, tokenizer=tokenizer, prompt=prompt,
                max_tokens=dynamic["max_tokens"], sampler=make_sampler(temp=0.7, top_p=0.9),
                logits_processors=[make_repetition_penalty(1.15)],
            ):
                token = response_chunk.text
                if any(st in token for st in stop_tokens):
                    break
                tokens.append(token)

            full_response = "".join(tokens)

            # Vérifier les signaux d'escalade (via regex, résistant au bruit)
            escalation_action = self._detect_escalade(full_response)
            if escalation_action == "[[ESCALADE:NIVEAU3]]":
                yield {"type": "status", "msg": "🌐 Recherche web..."}
                web_context_esc = self._do_web_search(query)
                yield {"type": "status", "msg": "⏫ Escalade vers le Cloud..."}
                prompt_cloud = query
                if web_context_esc:
                    prompt_cloud = web_context_esc + f"\n\nQuestion : {query}"
                content, cloud_model = self._execute_cloud(prompt_cloud)
                full_response = content
                model_used = cloud_model
                level = Level.CLOUD
            elif escalation_action == "[[ESCALADE:INTERNET]]":
                yield {"type": "status", "msg": "🌐 Recherche web..."}
                web_context_esc = self._do_web_search(query)
                if web_context_esc:
                    prompt_cloud = web_context_esc + f"\n\nQuestion : {query}"
                    content, cloud_model = self._execute_cloud(prompt_cloud)
                    full_response = content
                    model_used = cloud_model
                    level = Level.CLOUD
                else:
                    full_response = "La recherche web n'a retourné aucun résultat."
            elif escalation_action == "[[ESCALADE:INCONNU]]":
                full_response = "Je ne dispose pas d'informations suffisantes pour répondre à cette question de façon fiable."

            yield {"type": "token", "token": full_response, "final": True}

        # ── Post-génération ──
        elapsed = (time.time() - start) * 1000
        if self.use_semantic_cache and self.semantic_cache is not None:
            # Mapper le niveau numérique (1/2/3) vers N1/N2/N3 pour le TTL du cache
            niveau_str = {Level.LOCAL: "N1", Level.GENERAL: "N2", Level.CLOUD: "N3"}.get(level, "N2")
            self.semantic_cache.put(query, full_response, niveau=niveau_str)

        self.structured_memory.extract_and_store(query + " " + full_response)

        action_result = self._check_and_execute_action(full_response)
        if action_result != full_response:
            extra = action_result[len(full_response):]
            yield {"type": "token", "token": extra}
            full_response = action_result

        result = RouteResult(
            level=level,
            content=full_response,
            model_used=model_used,
            latency_ms=round(elapsed, 1),
            sources=sources,
            level_name=LEVEL_NAMES.get(level, "Inconnu")
        )
        self._log_decision(query, result, reason)

        yield {
            "type": "end",
            "model_used": model_used,
            "latency_ms": round(elapsed, 1),
            "sources": sources
        }

    # ── Exécution (legacy compat) ──

    def route(self, query: str, user_confirmed_cloud: bool = False) -> RouteResult:
        """Version synchrone de route."""
        full_content = ""
        last_res = None
        start_meta = {}
        
        for chunk in self.stream_route(query, user_confirmed_cloud):
            if chunk["type"] == "start":
                start_meta = chunk
            elif chunk["type"] == "token":
                full_content += chunk["token"]
            elif chunk["type"] == "end":
                last_res = chunk
            elif chunk["type"] == "confirm":
                return RouteResult(
                    level=Level.CLOUD,
                    content=f"CLOUD_NEEDS_CONFIRM:{chunk['reason']}",
                    model_used="",
                    latency_ms=0,
                    level_name=LEVEL_NAMES[Level.CLOUD]
                )
        
        return RouteResult(
            level=start_meta.get("level", Level.GENERAL),
            content=full_content,
            model_used=last_res.get("model_used", "inconnu") if last_res else "inconnu",
            latency_ms=last_res.get("latency_ms", 0) if last_res else 0,
            sources=last_res.get("sources", []) if last_res else [],
            level_name=start_meta.get("level_name", "")
        )

    # ── Moteurs ──

    def _load_local_model(self):
        """Charge le modèle local avec fallback OOM."""
        if self._local_model is not None:
            self._last_model_use = time.time()
            return self._local_model, self._local_tokenizer

        if not MLX_AVAILABLE:
            raise RuntimeError("MLX non installé")

        # Tenter de libérer de la RAM avant chargement
        gc.collect()
        if MLX_AVAILABLE:
            mx.clear_cache()

        original_id = self._model_id
        # Fallback de 3B vers 1.5B ou 0.5B si besoin (Qwen uniquement)
        fallbacks = [original_id, "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"]
        
        for attempt, model_id in enumerate(fallbacks):
            try:
                self.monitor.start_timer("load")
                print(f"Chargement de {model_id}...", file=sys.stderr, end=" ", flush=True)
                t0 = time.time()
                self._local_model, self._local_tokenizer = load(model_id)
                self._model_id = model_id
                self.monitor.stop_timer("load")
                print(f"✓ ({time.time() - t0:.1f}s)", file=sys.stderr)
                self._last_model_use = time.time()
                return self._local_model, self._local_tokenizer
            except (MemoryError, RuntimeError) as e:
                print(f"⚠ Échec chargement {model_id}: {e}", file=sys.stderr)
                self._local_model = None
                self._local_tokenizer = None
                gc.collect()
                if MLX_AVAILABLE:
                    mx.clear_cache()
                if attempt < len(fallbacks) - 1:
                    from rag import unload_embedding, unload_reranker
                    try:
                        unload_embedding()
                        unload_reranker()
                    except Exception:
                        pass
                    print(f"  ♻️ RAM libérée — tentative fallback...", file=sys.stderr)
                else:
                    raise RuntimeError("Impossible de charger un modèle local (OOM)")

        raise RuntimeError("Impossible de charger un modèle local (OOM)")

    def _format_local_prompt(self, user_msg: str, rag_context: str = "", corrections: str = "") -> str:
        """
        Prompt local avec trois variantes (V1).
        V2 utilise prompts_v2.build_prompt() à la place.
        """
        # Faits utilisateur (communs aux deux variantes)
        user_facts = ""
        structured_context = self.structured_memory.get_context()
        if structured_context:
            facts = [line for line in structured_context.split("\n") if line.strip().startswith("- ")]
            if facts:
                user_facts = "\n".join(facts) + "\n"

        # Section corrections (Few-shot dynamique)
        correction_section = ""
        if corrections:
            correction_section = (
                "\n## --- ERREURS PASSÉES À ÉVITER (FEW-SHOT) ---\n"
                "Voici des points sur lesquels tu as été corrigé par le passé. "
                "Assure-tu de respecter ces corrections dans ta réponse actuelle :\n"
                f"{corrections}\n"
            )

        if rag_context:
            # ── Variante RAG local ──
            system = (
                "## Identité\n"
                "Tu es NURU, un assistant IA avancé et polyvalent.\n"
                "Tu possèdes des compétences expertes en programmation (Python, JS), analyse et rédaction.\n"
                "\n"
                "## Signaux d'action (priorité absolue)\n"
                "Évalue d'abord si ta réponse nécessite :\n"
                "  - Des données récentes (événements après 2023) → [[ESCALADE:INTERNET]]\n"
                "  - Un raisonnement très complexe ou long code → [[ESCALADE:NIVEAU3]]\n"
                "\n"
                "Si l'un de ces cas s'applique, réponds UNIQUEMENT avec le signal correspondant.\n"
                "\n"
                "## Règles de réponse\n"
                "- Tu réponds avec assurance en utilisant tes connaissances internes.\n"
                "- Si tu génères du code, assure-toi qu'il soit propre et commenté.\n"
                "- Tu n'utilises PAS de phrase de type 'D'après mes connaissances générales' sauf si tu es réellement incertain sur un fait historique ou factuel précis.\n"
                "- Pour le code ou la logique, tu es un expert direct.\n"
            )
            if user_facts:
                system += f"\n{user_facts}"
            if correction_section:
                system += f"\n{correction_section}"
            
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
            
            history = self.memory.get_context()
            if history:
                prompt += history
            
            prompt += (
                f"<|im_start|>user\n"
                f"## Documents disponibles\n"
                f"{rag_context}\n"
                f"\n"
                f"## Question\n"
                f"{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            # ── Variante connaissances générales ──
            system = (
                "## Identité\n"
                "Tu es NURU, un assistant fiable opérant sur tes connaissances générales.\n"
                "Aucun document n'est disponible pour cette requête.\n"
                "\n"
                "## Signaux d'action (priorité absolue)\n"
                "Nous sommes en MAI 2026. Évalue d'abord si ta réponse nécessite :\n"
                "  - Des données récentes (événements après 2023, élections, noms de dirigeants actuels) → [[ESCALADE:INTERNET]]\n"
                "  - Un raisonnement complexe dépassant tes capacités → [[ESCALADE:NIVEAU3]]\n"
                "\n"
                "Si l'un de ces cas s'applique, réponds UNIQUEMENT avec le signal correspondant,\n"
                "sans aucun autre texte.\n"
                "\n"
                "## Règles de réponse (si aucun signal d'action n'est nécessaire)\n"
                "- Tu réponds avec tes connaissances générales en signalant clairement leur nature :\n"
                '  "D\'après mes connaissances générales (non vérifiées par un document)..."\n'
                "- Tu n'inventes jamais de chiffres précis, dates ou noms si tu n'en es pas certain.\n"
                "- En cas de doute sur un fait précis, tu dis : 'Je ne suis pas certain de ce point,\n"
                "  une vérification est recommandée.'\n"
                "- Tu restes dans le périmètre de la question posée."
            )
            if user_facts:
                system += f"\n{user_facts}"
            if correction_section:
                system += f"\n{correction_section}"
            
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
            
            history = self.memory.get_context()
            if history:
                prompt += history
            
            prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt

    def _execute_local(self, query: str, rag_context: str = "") -> tuple[str, str]:
        """Exécution synchrone avec détection d'escalade."""
        self.monitor.start_timer("generate")
        model, tokenizer = self._load_local_model()
        prompt = self._format_local_prompt(query, rag_context)
        response = generate(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=1024, verbose=False)
        response = response.strip()
        ntokens = len(response.split())
        self.monitor.stop_timer("generate", tokens=ntokens)
        self._last_model_use = time.time()

        # Détection d'escalade : si le modèle local demande de l'aide,
        # on fait la recherche web + cloud call automatiquement
        escalation_action = self._detect_escalade(response)
        if escalation_action in ("[[ESCALADE:INTERNET]]", "[[ESCALADE:NIVEAU3]]"):
            print("  ⬆ Escalade détectée — recherche web + cloud...", file=sys.stderr)
            web_context = self._do_web_search(query)
            if web_context:
                prompt_cloud = web_context + f"\n\nQuestion : {query}"
                content, cloud_model = self._execute_cloud(prompt_cloud)
                return content, cloud_model
            else:
                return "La recherche web n'a retourné aucun résultat.", self._model_id
        elif escalation_action == "[[ESCALADE:INCONNU]]":
            return "Je ne dispose pas d'informations suffisantes pour répondre à cette question de façon fiable.", self._model_id

        return response, self._model_id

    def _execute_cloud(self, query: str) -> tuple[str, str]:
        """Exécution via API Cloud — Deepseek d'abord, OpenRouter en fallback."""
        self.monitor.start_timer("cloud")
        if not KEYCHAIN_AVAILABLE:
            return self._execute_local(query)
        service = self.config.get("system", {}).get("keychain_service", "com.nuru.assistant")
        provider = self.config.get("models", {}).get("cloud", {}).get("provider", "deepseek")

        # ── Choix du provider primaire ──
        if provider == "deepseek":
            result = self._call_deepseek(query, service)
            # Fallback automatique vers OpenRouter si Deepseek échoue
            content, model_used = result
            _err_kw = ["Erreur", "Clé manquante", "error", "Error", "API key"]
            if len(content) < 50 and any(kw in content for kw in _err_kw):
                print("  ⚠ Deepseek indisponible → fallback OpenRouter", file=sys.stderr)
                fallback = self._call_openrouter(query, service)
                if not (len(fallback[0]) < 50 and any(kw in fallback[0] for kw in _err_kw)):
                    result = fallback
        elif provider == "openrouter":
            result = self._call_openrouter(query, service)
        else:
            result = self._execute_local(query)

        # ── Détection d'erreur finale ──
        content, model_used = result
        _ERROR_KEYWORDS = ["Erreur", "Clé manquante", "error", "Error", "API key"]
        if len(content) < 50 and any(kw in content for kw in _ERROR_KEYWORDS):
            raise RuntimeError(f"Cloud API error: {content}")
        self.monitor.stop_timer("cloud", tokens=len(content.split()) if content else 0)
        return result

    def _call_openrouter(self, query: str, service: str) -> tuple[str, str]:
        """Appel API OpenRouter (OpenAI-compatible, GPT/Claude/Llama via OpenRouter)."""
        api_key = get_key(service, "openrouter")
        if not api_key:
            return ("Clé manquante pour OpenRouter. Configurez via : security add-generic-password -s 'com.nuru.assistant' -a 'openrouter' -w 'votre_clé'", "openrouter")
        try:
            import openai
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=60,
            )
            model_name = self.config.get("models", {}).get("cloud", {}).get("openrouter_model", "google/gemma-4-31b-it:free")

            # Même system prompt que Deepseek (signaux de confiance + escalade)
            system_content = (
                "## Identité\n"
                "Tu es NURU, un assistant fiable opérant en mode cloud pour les requêtes complexes.\n"
                "Tu peux utiliser tes connaissances avancées et, si des résultats de recherche\n"
                "web sont fournis, tu les intègres à ta réponse.\n"
                "\n"
                "## Signaux de confiance\n"
                "✅ [CONFIRMÉ]      — Fait établi, source fiable ou résultat de recherche web\n"
                "🔶 [DÉDUIT]        — Raisonnement ou inférence logique\n"
                "⚠️  [INCERTAIN]    — Information incomplète, débattue ou non vérifiable\n"
                "❌ [HORS CONTEXTE] — Question hors de portée, même pour ce niveau\n"
                "\n"
                "## Signal d'action\n"
                "Si même ce niveau ne peut pas répondre de façon fiable :\n"
                "  [[ESCALADE:INCONNU]]\n"
                "\n"
                "## Règles fondamentales\n"
                "- Tu indiques systématiquement le signal de confiance par bloc d'information.\n"
                "- Si des résultats de recherche web sont fournis dans le contexte, tu les utilises\n"
                "  en priorité et tu les cites comme source.\n"
                "- Tu n'inventes jamais de données précises sans les signaler avec ⚠️ [INCERTAIN].\n"
                "- Tu restes factuel, structuré, et précis.\n"
            )

            structured_context = self.structured_memory.get_context()
            if structured_context:
                facts_lines = [line for line in structured_context.split("\n") if line.startswith("- ")]
                if facts_lines:
                    system_content += "\nInformations sur l'utilisateur :\n" + "\n".join(facts_lines)

            messages = [{"role": "system", "content": system_content}]
            for ex in self.memory.get_exchanges()[-5:]:
                if "qui suis" in ex.user.lower() and "nuru" in ex.assistant.lower():
                    continue
                messages.append({"role": "user", "content": ex.user})
                messages.append({"role": "assistant", "content": ex.assistant})
            messages.append({"role": "user", "content": query})

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                extra_headers={
                    "HTTP-Referer": "https://github.com/leblancbahiga/nuru",
                    "X-Title": "NURU Assistant",
                },
            )
            return response.choices[0].message.content, f"openrouter/{model_name}"
        except Exception as e:
            return (f"Erreur OpenRouter : {e}", "openrouter")

    def _call_deepseek(self, query: str, service: str) -> tuple[str, str]:
        api_key = get_key(service, "deepseek")
        if not api_key: return ("Clé manquante", "deepseek")
        import time as _time
        last_err = None
        for attempt in range(3):
            try:
                import openai
                client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=60)
                model_name = self.config.get("models", {}).get("cloud", {}).get("deepseek_model", "deepseek-v4-flash")

                # System prompt Cloud — signaux de confiance + escalade
                system_content = (
                    "## Identité\n"
                    "Tu es NURU, un assistant fiable opérant en mode cloud pour les requêtes complexes.\n"
                    "Tu peux utiliser tes connaissances avancées et, si des résultats de recherche\n"
                    "web sont fournis, tu les intègres à ta réponse.\n"
                    "\n"
                    "## Signaux de confiance\n"
                    "✅ [CONFIRMÉ]      — Fait établi, source fiable ou résultat de recherche web\n"
                    "🔶 [DÉDUIT]        — Raisonnement ou inférence logique\n"
                    "⚠️  [INCERTAIN]    — Information incomplète, débattue ou non vérifiable\n"
                    "❌ [HORS CONTEXTE] — Question hors de portée, même pour ce niveau\n"
                    "\n"
                    "## Signal d'action\n"
                    "Si même ce niveau ne peut pas répondre de façon fiable :\n"
                    "  [[ESCALADE:INCONNU]]\n"
                    "\n"
                    "## Règles fondamentales\n"
                    "- Tu indiques systématiquement le signal de confiance par bloc d'information.\n"
                    "- Si des résultats de recherche web sont fournis dans le contexte, tu les utilises\n"
                    "  en priorité et tu les cites comme source.\n"
                    "- Tu n'inventes jamais de données précises sans les signaler avec ⚠️ [INCERTAIN].\n"
                    "- Tu restes factuel, structuré, et précis.\n"
                )

                structured_context = self.structured_memory.get_context()
                if structured_context:
                    facts_lines = [line for line in structured_context.split("\n") if line.startswith("- ")]
                    if facts_lines:
                        system_content += "\nInformations sur l'utilisateur :\n" + "\n".join(facts_lines)

                messages = [{"role": "system", "content": system_content}]
                # Ajouter l'historique récent en filtrant les réponses erronées
                for ex in self.memory.get_exchanges()[-5:]:
                    if "qui suis" in ex.user.lower() and "nuru" in ex.assistant.lower():
                        continue
                    messages.append({"role": "user", "content": ex.user})
                    messages.append({"role": "assistant", "content": ex.assistant})
                messages.append({"role": "user", "content": query})

                response = client.chat.completions.create(
                    model=model_name, messages=messages, temperature=0.7, max_tokens=4096
                )
                return response.choices[0].message.content, f"deepseek/{model_name}"
            except (BrokenPipeError, ConnectionError, TimeoutError) as e:
                last_err = e
                if attempt < 2:
                    _time.sleep(2 ** attempt)
                    continue
                return (f"Erreur après 3 tentatives : {last_err}", "deepseek")
            except Exception as e:
                return (f"Erreur : {e}", "deepseek")

    def _brave_web_search(self, query: str) -> list[dict]:
        import urllib.request, urllib.parse, ssl
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        api_key = None
        if KEYCHAIN_AVAILABLE:
            try:
                service = load_config_service()
                api_key = get_key(service, "brave")
            except: pass
        if not api_key: return []
        try:
            url = f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}&count=5"
            req = urllib.request.Request(url, headers={"X-Subscription-Token": api_key})
            with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as resp:
                data = json.loads(resp.read())
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("description", "")})
            return results
        except: return []

    def _do_web_search(self, q: str) -> str:
        """Recherche web : Tavily (primaire) → Brave (backup).

        L'ordre des moteurs est défini dans config.yaml (search.primary / search.backup).
        Les clés API sont stockées dans le trousseau macOS.
        """
        import json

        ctx = ""
        search_cfg = self.config.get("search", {})
        primary = search_cfg.get("primary", "tavily")
        backup = search_cfg.get("backup", "brave")
        service = load_config_service() if KEYCHAIN_AVAILABLE else None

        def _try_tavily(q: str) -> str:
            """Tente une recherche via Tavily Search API."""
            if not KEYCHAIN_AVAILABLE:
                return ""
            try:
                api_key = get_key(service, "tavily")
                if not api_key:
                    print("  ⚠ Tavily : clé API manquante dans le trousseau", file=__import__('sys').stderr)
                    return ""
                from tavily_search import TavilySearchClient
                tavily_cfg = search_cfg.get("tavily", {})
                client = TavilySearchClient(
                    api_key=api_key,
                    max_results=tavily_cfg.get("max_results", 5),
                    search_depth=tavily_cfg.get("depth", "basic"),
                    include_answer=tavily_cfg.get("include_answer", True),
                )
                resp = client.search(q)
                if resp.success:
                    return resp.raw_context
                else:
                    print(f"  ⚠ Tavily : {resp.error}", file=__import__('sys').stderr)
                    return ""
            except Exception as e:
                print(f"  ⚠ Tavily : {e}", file=__import__('sys').stderr)
                return ""

        def _try_brave(q: str) -> str:
            """Tente une recherche via Brave Search Client (httpx + retry)."""
            if not KEYCHAIN_AVAILABLE:
                return ""
            try:
                api_key = get_key(service, "brave")
                if not api_key:
                    return ""
                from brave_search import BraveSearchClient
                brave_cfg = search_cfg.get("brave", {})
                client = BraveSearchClient(
                    api_key=api_key,
                    max_results=brave_cfg.get("max_results", 5),
                    country=brave_cfg.get("country", "fr"),
                    language=brave_cfg.get("language", "fr"),
                )
                resp = client.search(q)
                if resp.success:
                    return resp.raw_context
                return ""
            except Exception:
                return ""

        def _try_brave_fallback(q: str) -> str:
            """Dernier recours : Brave en urllib (sans httpx)."""
            results = self._brave_web_search(q)
            if results:
                web_lines = [f"{i}. {r['title']} — {r['snippet']} ({r['url']})"
                            for i, r in enumerate(results, 1)]
                return "Résultats de recherche web :\n" + "\n".join(web_lines)
            return ""

        # ── 1. Primaire ──
        if primary == "tavily":
            print("  🌐 Recherche Tavily (primaire)...", file=__import__('sys').stderr)
            ctx = _try_tavily(q)

        # ── 2. Backup si primaire échoue ──
        if not ctx:
            print(f"  🔄 Backup {backup}...", file=__import__('sys').stderr)
            if backup == "tavily":
                ctx = _try_tavily(q)
            elif backup == "brave":
                ctx = _try_brave(q)

        # ── 3. Dernier recours : Brave urllib ──
        if not ctx:
            ctx = _try_brave_fallback(q)

        return ctx

    def _check_corrections(self, query: str) -> Optional[str]:
        global _correction_embedder
        store = get_vector_store()
        if store is None: return None
        try:
            if _correction_embedder is None:
                from sentence_transformers import SentenceTransformer
                _correction_embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            q_emb = _correction_embedder.encode(query).tolist()
            res = store.col_corrections.query(query_embeddings=[q_emb], n_results=1)
            if res and res["ids"][0]:
                dist = res["distances"][0][0]
                doc = res["documents"][0][0]
                print(f"  DEBUG: Correction check for '{query}' -> Match: '{doc}' (Dist: {dist:.4f})", file=sys.stderr)
                if (1.0 - dist) >= 0.85:
                    return doc
        except: pass
        return None

    def toggle_airplane_mode(self) -> bool:
        self.airplane_mode = not self.airplane_mode
        return self.airplane_mode

    def set_airplane_mode(self, enabled: bool): self.airplane_mode = enabled
    def set_force_cloud(self, enabled: bool): self.force_cloud = enabled
    def set_web_search(self, enabled: bool): self.web_search_mode = enabled

    # ── V2 : Activer les modules optionnels ──

    def enable_v2(self, use_reranker: bool = True, use_resource_manager: bool = True) -> bool:
        """
        Active les modules V2 (ModelPool, FlashReranker, ResourceManager, Hiérarchie mémoire).
        Ne remplace PAS stream_route() — V1 continue de fonctionner.
        Les modules V2 sont disponibles comme services additionnels.

        Retourne True si activé, False si modules manquants.
        """
        try:
            from model_pool_v2 import ModelPoolV2
            from reranker_v2 import FlashReranker
            from resource_manager_v2 import ResourceManagerV2
            from memory_v2 import HierarchicalMemory
            from pipeline_v2 import NuruPipelineV2

            self.v2_model_pool = ModelPoolV2(max_loaded=1)
            self.v2_reranker = FlashReranker(alpha=0.7) if use_reranker else None
            self.v2_resource_manager = ResourceManagerV2(
                on_ram_critical=lambda ram_gb: self.v2_model_pool.unload_all()
            ) if use_resource_manager else None
            self.v2_memory = HierarchicalMemory()

            self.v2_pipeline = NuruPipelineV2(
                model_pool=self.v2_model_pool,
                memory=self.v2_memory,
                reranker=self.v2_reranker,
                resource_manager=self.v2_resource_manager,
            )

            if self.v2_resource_manager:
                self.v2_resource_manager.start()

            self.v2_enabled = True
            print("  ✅ Modules V2 activés (ModelPool + FlashReranker + ResourceManager + Hiérarchie mémoire)", file=sys.stderr)
            return True
        except ImportError as e:
            print(f"  ⚠ Modules V2 non disponibles : {e}", file=sys.stderr)
            self.v2_enabled = False
            return False
        except Exception as e:
            print(f"  ⚠ Erreur activation V2 : {e}", file=sys.stderr)
            self.v2_enabled = False
            return False

    def disable_v2(self) -> None:
        """Désactive et nettoie les modules V2."""
        if self.v2_resource_manager:
            self.v2_resource_manager.stop()
        if self.v2_model_pool:
            self.v2_model_pool.unload_all()
        self.v2_enabled = False
        self.v2_model_pool = None
        self.v2_reranker = None
        self.v2_resource_manager = None
        self.v2_memory = None
        self.v2_pipeline = None
        print("  ♻️ Modules V2 désactivés", file=sys.stderr)

    # ── V2 : Route via pipeline V2 (alternative à stream_route) ──

    def stream_v2(self, query: str) -> Generator[dict, None, None]:
        """
        Version V2 de stream_route() utilisant NuruPipelineV2.
        Compatible avec le même format de yield que stream_route().
        Si V2 n'est pas activé, fallback vers stream_route() standard.
        """
        if not self.v2_enabled or self.v2_pipeline is None:
            yield from self.stream_route(query)
            return

        # Utiliser le pipeline V2
        yield from self.v2_pipeline.handle_sync(
            query,
            force_n3=self.force_cloud or self.web_search_mode,
            airplane=self.airplane_mode,
        )

    def _rewrite_query_for_rag(self, query: str) -> str:
        """
        Réécrit la requête utilisateur pour améliorer la recherche RAG.

        Stratégie :
        1. Si le modèle local est déjà chargé (warm), utilise un prompt court pour reformuler.
        2. Sinon, applique des règles légères d'expansion contextuelle.

        Args:
            query: Requête brute de l'utilisateur

        Returns:
            Requête optimisée pour la recherche vectorielle
        """
        # ── Règles légères (toujours appliquées) ──
        rewritten = query.strip()

        # Expansion des noms de mois/dates pour la similarité sémantique
        import datetime
        current_year = datetime.datetime.now().year
        
        # Si la requête contient des mots temporels implicites, ajouter l'année
        temporal_hints = ["actuel", "actuelle", "récent", "récente", "aujourd'hui",
                          "prix", "cours", "actualité", "nouveau", "nouvelle", "dernier"]
        has_temporal = any(hint in rewritten.lower() for hint in temporal_hints)
        has_year = str(current_year) in rewritten or str(current_year - 1) in rewritten
        if has_temporal and not has_year and len(rewritten) > 10:
            rewritten += f" {current_year}"

        # Expansion d'abréviations agricoles courantes
        agri_abbrev = {
            "rdc": "République Démocratique du Congo",
            "ins": "Institut National de la Statistique",
            "sa": "Société Anonyme",
            "pib": "Produit Intérieur Brut",
        }
        words = rewritten.split()
        expanded = []
        for w in words:
            lower = w.lower().strip(".,;:!?")
            if lower in agri_abbrev:
                expanded.append(agri_abbrev[lower])
            else:
                expanded.append(w)
        if expanded != words:
            rewritten = " ".join(expanded)

        # ── Si le modèle local est déjà chaud, affiner avec un prompt court ──
        if self._local_model is not None and MLX_AVAILABLE and len(query) < 200:
            try:
                import datetime
                now = datetime.datetime.now()
                date_str = now.strftime("%d %B %Y")
                rewrite_prompt = (
                    f"<|im_start|>system\n"
                    f"Nous sommes aujourd'hui le {date_str}. "
                    f"Ton but est d'optimiser cette requête pour une recherche dans une base de documents.\n"
                    f"1. Si la requête est claire et précise, renvoie-la TELLE QUELLE.\n"
                    f"2. Si elle est trop courte ou contient des pronoms ambigus, explicite-la.\n"
                    f"3. Si c'est une instruction ou une phrase méta sur ton comportement, NE LA CHANGE PAS.\n"
                    f"4. Ne change jamais le sens original. Réponds UNIQUEMENT avec la requête optimisée.<|im_end|>\n"
                    f"<|im_start|>user\n"
                    f"{query}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                from mlx_lm import generate
                refined = generate(
                    model=self._local_model,
                    tokenizer=self._local_tokenizer,
                    prompt=rewrite_prompt,
                    max_tokens=64,
                    verbose=False,
                ).strip()
                if refined and len(refined) > len(query) * 0.5 and len(refined) < len(query) * 3:
                    rewritten = refined
            except Exception:
                pass  # Fallback silencieux à la version rules-based

        return rewritten

    def _generate_hyde_doc(self, query: str) -> str:
        """
        Génère une réponse hypothétique pour HyDE (Hypothetical Document Embeddings).
        """
        if not MLX_AVAILABLE:
            return ""

        try:
            # S'assurer que le modèle est chargé
            model, tokenizer = self._load_local_model()
            if not model:
                return ""

            import datetime
            now = datetime.datetime.now()
            date_str = now.strftime("%d %B %Y")
            hyde_prompt = (
                f"<|im_start|>system\n"
                f"Nous sommes aujourd'hui le {date_str}. "
                f"Tu es un expert qui génère une réponse hypothétique courte et factuelle pour aider à la recherche documentaire.\n"
                f"Génère une réponse possible à la question suivante en respectant la date d'aujourd'hui.\n"
                f"Si la requête ressemble à une instruction ou une question méta, réponds de façon neutre et courte.\n"
                f"Réponds directement avec le contenu factuel sans phrase d'introduction.<|im_end|>\n"
                f"<|im_start|>user\n{query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            from mlx_lm import generate
            hyde_doc = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=hyde_prompt,
                max_tokens=150,
                verbose=False,
            ).strip()
            return hyde_doc
        except Exception as e:
            print(f"  ⚠ Échec HyDE : {e}", file=sys.stderr)
            return ""

    def _log_decision(self, query: str, result: RouteResult, reason: str = ""):
        # Convertir Level enum en sa valeur entière pour la sérialisation JSON
        level_value = result.level.value if hasattr(result.level, 'value') else result.level
        self.transparency.log_decision(query=query, level=level_value, level_name=result.level_name, reason=reason, latency_ms=result.latency_ms, model_used=result.model_used)
        try:
            self.complexity.log_result(query=query, level_chosen=level_value, latency_ms=result.latency_ms, tokens_generated=len(result.content.split()))
        except AttributeError:
            pass  # Nouveau ComplexityClassifier n'a pas log_result

    def _check_and_execute_action(self, content: str) -> str:
        if self.action_engine is None: return content
        try:
            res = self.action_engine.parse_and_execute(content)
            if res:
                content += f"\n\nAction : {res.get('message', '')}"
                if res.get("stdout"): content += f"\nSortie :\n{res['stdout']}"
        except: pass
        return content

    def _detect_escalade(self, response: str) -> str | None:
        """Détecte un signal d'escalade [[ESCALADE:...]] par regex (1 ou 2 crochets)."""
        import re as _re
        # Accepte [[...]] ou [...] — le Qwen 3B oublie parfois un crochet
        m = _re.search(r'\[{0,2}ESCALADE:(INTERNET|NIVEAU3|INCONNU)\]{0,2}', response, _re.IGNORECASE)
        if m:
            return f"[[ESCALADE:{m.group(1).upper()}]]"
        return None
