import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

# Détection de langue (importé une fois au module, pas à chaque appel)
try:
    from langdetect import detect as _detect_lang
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    def _detect_lang(text): return 'fr'

# ─────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────

class Level(Enum):
    LOCAL = 1     # N1 : RAG local
    GENERAL = 2   # N2 : LLM local sans docs
    CLOUD = 3     # N3 : Recherche Web + Cloud

class Intent(Enum):
    CHAT = "chat"
    RAG = "rag"
    WEB = "web"
    CODE = "code"
    ACTION = "action"
    IDENTITY = "identity"    # conservé pour rétrocompatibilité
    USER = "user"            # "qui suis-je" → décrit l'utilisateur
    NURU = "nuru"            # "qui es-tu" → NURU se présente

@dataclass
class ClassificationResult:
    intent: Intent
    level: Level
    score: float
    breakdown: dict[str, float]
    reason: str


# ─────────────────────────────────────────────
# Mots-clés par catégorie — Multilingue
# ─────────────────────────────────────────────

# ── Français ──
KEYWORDS_COMPLEX_FR = [
    # Raisonnement avancé
    "explique pourquoi", "analyse", "compare", "évalue", "critique",
    "synthétise", "argumente", "démontre", "prouve", "réfute",
    "quelles sont les implications", "quel est l'impact",
    # Temporalité récente
    "aujourd'hui", "récemment", "actuellement", "en ce moment",
    "cette semaine", "ce mois", "2024", "2025",
    "dernières nouvelles", "actualité",
    # Multi-étapes
    "étape par étape", "plan détaillé", "stratégie complète",
    "comment mettre en place", "comment concevoir",
    # Créativité / génération longue
    "rédige", "écris un rapport", "crée un plan", "génère",
]

KEYWORDS_SIMPLE_FR = [
    "qu'est-ce que", "c'est quoi", "définis", "définition",
    "quand", "où", "qui", "combien", "quel est le nom",
    "liste", "énumère", "cite",
]

KEYWORDS_WEB_FR = [
    "prix actuel", "taux actuel", "cours de", "météo", "dernières nouvelles",
    "actualité", "aujourd'hui", "en direct", "live", "dollar", "euro",
    "récemment", "récent", "récents", "derniers", "dernières", "infos sur",
    "résultats sur", "trouve des", "cherche des", "recherche des", "actu",
    "vient de", "annoncé", "publié", "bourse", "crypto",
    "taux de change", "taux du", "cotation", "marché",
    "bourse", "indice", "flation", "taux d'intérêt",
    "taux", "dollar", "euro", "bourse", "valeur",
    "actuel", "actuelle", "président", "élection", "gouvernement", "ministre"
]

# ── Anglais ──
KEYWORDS_COMPLEX_EN = [
    "explain why", "analyze", "compare", "evaluate", "critique",
    "synthesize", "argue", "demonstrate", "prove", "refute",
    "what are the implications", "what is the impact",
    "today", "recently", "currently", "right now",
    "this week", "this month", "2024", "2025", "2026",
    "latest news", "breaking news",
    "step by step", "detailed plan", "complete strategy",
    "how to implement", "how to design",
    "write", "write a report", "create a plan", "generate",
]

KEYWORDS_SIMPLE_EN = [
    "what is", "define", "definition",
    "when", "where", "who", "how many", "what is the name",
    "list", "enumerate", "cite", "tell me",
]

KEYWORDS_WEB_EN = [
    "current price", "price of", "weather", "latest news",
    "breaking news", "today", "live", "live update",
    "recently", "just announced", "just published",
    "exchange rate", "exchange rates", "stock market",
    "market index", "inflation", "interest rate",
    "current", "president", "election", "government", "minister"
]

# ── Swahili ──
KEYWORDS_COMPLEX_SW = [
    "elezea kwa nini", "chambua", "linganisha", "tathmini", "kosoa",
    "sintetisha", "jadili", "onyesha", "thibitisha", "kanusha",
    "ni nini athari", "ni nini matokeo",
    "leo", "hivi karibuni", "sasa hivi", "wakati huu",
    "wiki hii", "mwezi huu",
    "hatua kwa hatua", "mpango kamili", "mkakati kamili",
    "jinsi ya kutekeleza", "jinsi ya kubuni",
    "andika", "andika ripoti", "unda mpango",
]

KEYWORDS_SIMPLE_SW = [
    "nini", "fafanua", "ufafanuzi",
    "lini", "wapi", "nani", "ngapi", "jina lake ni nini",
    "orodhesha", "taja",
]

KEYWORDS_WEB_SW = [
    "bei ya sasa", "bei ya", "hali ya hewa", "habari mpya",
    "habari za mwisho", "leo", "moja kwa moja",
    "hivi karibuni", "ilitangazwa", "ilichapishwa",
]

# ── Code (Programmation) ──
KEYWORDS_CODE = [
    "code", "python", "javascript", "fonction", "boucle", "algorithme", 
    "déboguer", "script", "développe", "implémente", "write a function",
    "how to code", "bug", "syntaxe", "compilation", "repository", "github"
]

# ── Actions (Système) ──
KEYWORDS_ACTION = [
    "crée un dossier", "ouvre", "lance", "exécute", "efface", "supprime",
    "déplace", "copie", "alerte", "notifie", "create folder", "open",
    "launch", "execute", "delete", "remove", "move", "copy", "notify",
    "système", "processus", "terminal", "commande"
]

# ── Identité (NURU / Leblanc) ──
# Qui suis-je / who am I → à propos de l'utilisateur
KEYWORDS_USER_IDENTITY = [
    "qui suis-je", "who am i", "leblanc bahiga", "who is leblanc",
    "qui est leblanc", "c'est qui leblanc",
    "parcours professionnel", "expérience professionnelle", "mon travail", "mon cv", "quel est mon poste", "quelle est ma profession",
]
# Qui es-tu / who are you → à propos de NURU
KEYWORDS_NURU_IDENTITY = [
    "qui es-tu", "c'est quoi nuru", "qui est ton créateur",
    "ton nom", "qui t'a créé", "who are you", "what is nuru",
    "tu es qui", "c'est qui nuru", "présente-toi",
]
KEYWORDS_IDENTITY = KEYWORDS_USER_IDENTITY + KEYWORDS_NURU_IDENTITY

# Agrégation multilingue pour le scoring
KEYWORDS_COMPLEX = KEYWORDS_COMPLEX_FR + KEYWORDS_COMPLEX_EN + KEYWORDS_COMPLEX_SW
KEYWORDS_SIMPLE  = KEYWORDS_SIMPLE_FR  + KEYWORDS_SIMPLE_EN  + KEYWORDS_SIMPLE_SW
KEYWORDS_WEB     = KEYWORDS_WEB_FR     + KEYWORDS_WEB_EN     + KEYWORDS_WEB_SW


# ─────────────────────────────────────────────
# Critères individuels (chacun retourne 0.0–1.0)
# ─────────────────────────────────────────────

def score_longueur(query: str) -> float:
    n = len(query.split())
    if n <= 8:
        return 0.0
    elif n >= 30:
        return 1.0
    else:
        return (n - 8) / (30 - 8)


def score_mots_cles(query: str) -> tuple[float, bool]:
    q = query.lower()
    needs_web = any(kw in q for kw in KEYWORDS_WEB)
    complex_hits = sum(1 for kw in KEYWORDS_COMPLEX if kw in q)
    simple_hits  = sum(1 for kw in KEYWORDS_SIMPLE  if kw in q)
    raw = (complex_hits - simple_hits)
    score = max(0.0, min(1.0, (raw + 3) / 6))
    return score, needs_web


def score_ponctuation_structure(query: str) -> float:
    connecteurs = ["parce que", "donc", "ainsi", "cependant", "néanmoins",
                   "en revanche", "d'une part", "d'autre part", "or", "mais"]
    q = query.lower()
    n_connecteurs = sum(1 for c in connecteurs if c in q)
    n_phrases = len(re.findall(r'[.!?]+', query))
    n_questions = query.count('?')
    raw = n_connecteurs * 0.4 + n_phrases * 0.2 + (n_questions - 1) * 0.3
    return min(1.0, raw)


def score_rag(rag_score: Optional[float], rag_threshold: float = 0.50) -> float:
    if rag_score is None:
        return 0.8
    if rag_score >= rag_threshold:
        return max(0.0, 1.0 - rag_score)
    else:
        return 0.7


def score_langue(query: str) -> tuple[float, str]:
    """
    Détecte la langue de la requête (5ᵉ critère).
    Retourne (bonus_score, lang_code).
    
    - 'fr' : bonus 0.0 (langue principale de NURU)
    - 'en' : bonus 0.1 (langue secondaire, mots-clés EN actifs)
    - 'sw' : bonus 0.05 (swahili, mots-clés SW actifs)
    - autre : bonus 0.15 (langue inconnue → escalade plus probable)
    """
    word_count = len(query.split())
    try:
        lang = _detect_lang(query)
    except Exception:
        lang = 'fr'

    # langdetect est peu fiable sur les requêtes très courtes (< 4 mots)
    # Si le résultat est surprenant (ni fr, ni en, ni sw), on default à 'fr'
    if word_count < 4 and lang not in ('fr', 'en', 'sw'):
        lang = 'fr'

    bonus = {'fr': 0.0, 'en': 0.1, 'sw': 0.05}.get(lang, 0.15)
    return bonus, lang


# ─────────────────────────────────────────────
# Classifier principal
# ─────────────────────────────────────────────

class ComplexityClassifier:
    def __init__(
        self,
        w_rag: float = 0.25,
        w_keywords: float = 0.40,
        w_length: float = 0.20,
        w_structure: float = 0.15,
        threshold_n1: float = 0.36,
        threshold_n3: float = 0.55,
        rag_threshold: float = 0.30,
    ):
        total = w_rag + w_keywords + w_length + w_structure
        assert abs(total - 1.0) < 1e-6, f"Les poids doivent sommer à 1.0 (total={total})"

        self.w_rag        = w_rag
        self.w_keywords   = w_keywords
        self.w_length     = w_length
        self.w_structure  = w_structure
        self.threshold_n1 = threshold_n1
        self.threshold_n3 = threshold_n3
        self.rag_threshold = rag_threshold

    def classify(
        self,
        query: str,
        rag_score: Optional[float] = None,
    ) -> ClassificationResult:
        q = query.lower()
        s_length, (s_keywords, needs_web) = score_longueur(query), score_mots_cles(query)
        s_structure = score_ponctuation_structure(query)
        s_rag       = score_rag(rag_score, self.rag_threshold)
        s_lang, lang_code = score_langue(query)

        # Ajuster les mots-clés selon la langue détectée
        if lang_code not in ('fr', 'en', 'sw'):
            # Langue non supportée → mots-clés désactivés, se fier aux autres critères
            s_keywords = 0.0

        final = (
            self.w_rag       * s_rag       +
            self.w_keywords  * s_keywords  +
            self.w_length    * s_length    +
            self.w_structure * s_structure
        )
        # Bonus langue (0.0 pour FR, 0.1 pour EN, 0.15 pour autre)
        final = min(1.0, final + s_lang)

        # ── Détection d'Intention (Pre-classifier sub-50ms) ──
        intent = Intent.CHAT
        if any(kw in q for kw in KEYWORDS_USER_IDENTITY):
            intent = Intent.USER
            level = Level.LOCAL
            reason = "Question sur l'identité de l'utilisateur → Décrit Leblanc"
        elif any(kw in q for kw in KEYWORDS_NURU_IDENTITY):
            intent = Intent.NURU
            level = Level.LOCAL
            reason = "Question sur l'identité de NURU → Se présente"
        elif needs_web:
            intent = Intent.WEB
            level = Level.CLOUD
            reason = "Données récentes ou internet requis → Niveau 3"
        elif any(kw in q for kw in KEYWORDS_CODE):
            intent = Intent.CODE
            level = Level.CLOUD if final > 0.4 else Level.GENERAL
            reason = f"Tâche de programmation détectée (Score complexité: {final:.2f})"
        elif any(kw in q for kw in KEYWORDS_ACTION):
            intent = Intent.ACTION
            level = Level.GENERAL
            reason = "Action système détectée → Niveau 2 (Action Engine)"
        elif rag_score is not None and rag_score >= self.rag_threshold:
            intent = Intent.RAG
            level = Level.LOCAL
            reason = f"Documents pertinents trouvés (RAG Score: {rag_score:.2f})"
        else:
            # Choix par score de complexité pur
            if final < self.threshold_n1:
                level  = Level.LOCAL
                reason = f"Score {final:.2f} < {self.threshold_n1} → Niveau 1 (Local)"
            elif final > self.threshold_n3:
                level  = Level.CLOUD
                reason = f"Score {final:.2f} > {self.threshold_n3} → Niveau 3 (Cloud)"
            else:
                level  = Level.GENERAL
                reason = f"Score {final:.2f} zone médiane → Niveau 2 (Général)"

        # ── Routage Sémantique (Ajustement) ──
        # Si le score est proche d'un seuil, l'analyse sémantique peut faire basculer
        semantic_bonus = self._get_semantic_adjustment(query)
        if semantic_bonus > 0 and level != Level.CLOUD:
            final = min(1.0, final + semantic_bonus)
            if final > self.threshold_n3:
                level = Level.CLOUD
                reason = f"Analyse sémantique (bonus +{semantic_bonus}) → Escalade Niveau 3"
            elif final > self.threshold_n1 and level == Level.LOCAL:
                level = Level.GENERAL
                reason = f"Analyse sémantique (bonus +{semantic_bonus}) → Passage Niveau 2"

        return ClassificationResult(
            intent=intent,
            level=level,
            score=round(final, 3),
            breakdown={
                "rag": round(s_rag, 3),
                "keywords": round(s_keywords, 3),
                "length": round(s_length, 3),
                "structure": round(s_structure, 3),
                "semantic_bonus": round(semantic_bonus, 3),
                "lang": lang_code,
                "lang_bonus": round(s_lang, 3),
            },
            reason=reason,
        )

    def _get_semantic_adjustment(self, query: str) -> float:
        """
        Calcule un bonus de complexité basé sur la similarité sémantique 
        avec des intentions de 'haut niveau' (N3).
        """
        try:
            # Import lazy pour éviter les dépendances circulaires
            from rag import get_embedder
            embedder = get_embedder()
            if embedder is None:
                return 0.0

            # Intentions types nécessitant le Cloud (N3)
            complex_anchors = [
                "Analyse approfondie et critique de la situation",
                "Comparaison détaillée des avantages et inconvénients",
                "Rédaction d'un rapport professionnel structuré",
                "Explication d'un mécanisme technique complexe",
                "Synthèse de multiples points de vue divergents",
                "Planification stratégique à long terme"
            ]

            # Embedding de la requête
            q_emb = embedder.encode(query)
            
            # Pour la performance, on pourrait pré-calculer ces embeddings, 
            # mais get_embedder est déjà mis en cache.
            anchor_embs = embedder.encode(complex_anchors)

            # Calcul de la similarité cosinus maximale
            # sim = (A . B) / (|A| * |B|)
            similarities = []
            for a_emb in anchor_embs:
                norm_q = np.linalg.norm(q_emb)
                norm_a = np.linalg.norm(a_emb)
                if norm_q > 0 and norm_a > 0:
                    sim = np.dot(q_emb, a_emb) / (norm_q * norm_a)
                    similarities.append(sim)
            
            max_sim = max(similarities) if similarities else 0
            
            # Seuil de déclenchement du bonus
            if max_sim > 0.85:
                return 0.25 # Très forte intention complexe
            if max_sim > 0.75:
                return 0.15 # Intention complexe probable
            
            return 0.0
        except Exception:
            return 0.0


# ── Complexity Enum V2 (rétrocompatible) ──
class Complexity(Enum):
    """Niveau de complexité pour la sélection de modèle V2."""
    SIMPLE = 1
    MEDIUM = 2
    COMPLEX = 3


# ── Singleton ──
_classifier_instance: Optional[ComplexityClassifier] = None

def get_complexity_classifier() -> ComplexityClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ComplexityClassifier()
    return _classifier_instance
