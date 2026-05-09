#!/usr/bin/env python3
"""
prompts_v2.py — Prompts système optimisés pour NURU V2.

Principes :
  1. Préfixe identité unique cacheable (identique N1/N2/N3 → KV cache réutilisable)
  2. Pas de signaux [[ESCALADE:...]] — le pre-classifier gère le routage
  3. [[INSUFFISANT]] conservé comme safety valve (faux négatifs du classifier)
  4. Prompt NANO ultra-compact pour Qwen 1.5B
  5. Injection mémoire conditionnelle (zéro token si vide)
  6. Date système dynamique
  7. Mode d'emploi : build_prompt(route, query, ...) → str
"""

from datetime import datetime
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  PRÉFIXE COMMUN — Identique dans N1, N2, N3 → cacheable
#  ~75 tokens stables, zéro instruction de routing
# ═══════════════════════════════════════════════════════════════
CACHED_PREFIX = (
    "Tu es NURU, assistant de Leblanc BAHIGA Mudarhi.\n"
    "Compétences : agronomie · Python/JS · analyse.\n"
    "Réponds dans la langue de la question. Sois direct.\n"
)

# ═══════════════════════════════════════════════════════════════
#  N1 — RAG LOCAL
#  Extraction + synthèse à partir des documents fournis.
# ═══════════════════════════════════════════════════════════════
PROMPT_N1 = CACHED_PREFIX + """\
Mode : RAG LOCAL — {rag_count} extrait(s) [score {rag_score:.2f}]
Date système : {date}

Règles :
- Appuie-toi sur les documents comme source primaire.
- Complète avec tes connaissances si les extraits sont partiels — signale-le.
- Si les docs ne couvrent pas la question, mentionne-le.
- Si impossible de répondre avec les docs disponibles : [[INSUFFISANT]]

{memory_block}
## Documents
{rag_context}

## Question
{query}"""


# ═══════════════════════════════════════════════════════════════
#  N2 — GÉNÉRAL LOCAL
#  Connaissances générales, code, définitions.
# ═══════════════════════════════════════════════════════════════
PROMPT_N2 = CACHED_PREFIX + """\
Mode : GÉNÉRAL
Date système : {date}

Règles :
- Réponds avec assurance sur ce que tu maîtrises.
- Pour les faits récents (post-2023) ou incertains : note-le brièvement.
- Pas de préambule. Va droit au but.
- Si tu ne peux pas répondre du tout : [[INSUFFISANT]]

{memory_block}
## Question
{query}"""


# ═══════════════════════════════════════════════════════════════
#  N3 — CLOUD + RECHERCHE WEB
#  Synthèse des résultats web + raisonnement expert.
# ═══════════════════════════════════════════════════════════════
PROMPT_N3 = CACHED_PREFIX + """\
Mode : CLOUD
Date système : {date}
Signaux : ✅ confirmé | 🔶 déduit | ⚠️ incertain

Règles :
- Synthétise les résultats web, ne les liste pas brut.
- Cite la source pour les faits clés.
- Croise avec les documents RAG si disponibles.
- Si les résultats sont insuffisants : [[INSUFFISANT]]

{memory_block}
{web_block}
{rag_block}
## Question
{query}"""


# ═══════════════════════════════════════════════════════════════
#  NANO — Ultra-compact pour Qwen 1.5B
#  < 10 tokens d'instructions, pas de règles complexes.
# ═══════════════════════════════════════════════════════════════
PROMPT_NANO = """\
Tu es NURU, assistant de Leblanc.
Date : {date}
{memory_block}{rag_block}
## Question
{query}"""


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _memory_block(memory_ctx: str) -> str:
    """Bloc mémoire conditionnel — vide si pas de contexte."""
    if not memory_ctx or not memory_ctx.strip():
        return ""
    return f"## Contexte\n{memory_ctx.strip()}\n\n"


def _rag_block(rag_chunks: list) -> str:
    """Bloc RAG formaté avec score pour chaque extrait."""
    if not rag_chunks:
        return ""
    texts = "\n---\n".join(
        f"[{i+1}] (score {c.get('score', 0):.2f})\n{c['text'][:500]}"
        for i, c in enumerate(rag_chunks)
    )
    return f"## Documents\n{texts}\n\n"


def _web_block(web_results: str) -> str:
    """Bloc résultats web conditionnel."""
    if not web_results or not web_results.strip():
        return ""
    return f"## Résultats web\n{web_results.strip()}\n\n"


def build_prompt(
    route: str,
    query: str,
    memory_ctx: str = "",
    rag_chunks: Optional[list] = None,
    web_results: str = "",
    rag_score: float = 0.0,
    use_nano: bool = False,
) -> str:
    """
    Construit le prompt complet pour NURU V2.

    Args:
        route: "N1", "N2", "N3"
        query: Question utilisateur
        memory_ctx: Contexte mémoire formaté (ou chaîne vide)
        rag_chunks: Liste de dicts avec 'text', 'score', 'source'
        web_results: Résultats web bruts formatés
        rag_score: Meilleur score RAG
        use_nano: Si True, utilise PROMPT_NANO (pour Qwen 1.5B)

    Retourne:
        Prompt formaté prêt pour le LLM
    """
    date = datetime.now().strftime("%d %B %Y")
    rag_chunks = rag_chunks or []
    mb = _memory_block(memory_ctx)

    if use_nano:
        rb = _rag_block(rag_chunks) if rag_chunks else ""
        return PROMPT_NANO.format(
            date=date,
            memory_block=mb,
            rag_block=rb,
            query=query,
        )

    if route == "N1":
        rag_context_lines = []
        for i, c in enumerate(rag_chunks):
            source = c.get("source", "doc")
            score = c.get("score", 0)
            text = c.get("text", "")[:600]
            rag_context_lines.append(f"[{i+1}] {source} (score {score:.2f})\n{text}")
        rag_context = "\n---\n".join(rag_context_lines) if rag_context_lines else "(aucun extrait)"
        return PROMPT_N1.format(
            date=date,
            rag_count=len(rag_chunks),
            rag_score=rag_score,
            memory_block=mb,
            rag_context=rag_context,
            query=query,
        )

    if route == "N2":
        return PROMPT_N2.format(
            date=date,
            memory_block=mb,
            query=query,
        )

    if route == "N3":
        return PROMPT_N3.format(
            date=date,
            memory_block=mb,
            web_block=_web_block(web_results),
            rag_block=_rag_block(rag_chunks) if rag_chunks else "",
            query=query,
        )

    # Fallback : N2
    return PROMPT_N2.format(
        date=date,
        memory_block=mb,
        query=query,
    )


def classify_insufficient(response: str) -> bool:
    """Détecte le signal [[INSUFFISANT]] dans la réponse du LLM."""
    return "[[INSUFFISANT]]" in response
