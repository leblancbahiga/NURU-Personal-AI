#!/usr/bin/env python3
"""
context_compressor.py — Compression de chunks RAG par règles textuelles pour NURU V2.

Compresse les chunks RAG en appliquant des règles textuelles locales :
  - Normalisation du texte (espaces, sauts de ligne)
  - Suppression de contenus répétés (phrases dupliquées)
  - Élimination de boilerplate (en-têtes, pieds de page, copyright)
  - Extraction de phrases clés par densité de mots significatifs
  - Troncature intelligente aux limites sémantiques

Aucun modèle ML chargé — 100% règles symboliques, 0 VRAM, exécution instantanée.

Usage :
    compressor = ContextCompressor()
    result = compressor.compress("Long texte de chunk RAG...", max_tokens=256)
    compressed = result["compressed_text"]
    stats = result["stats"]
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("nuru.v2.context_compressor")


# ── Constantes de configuration par défaut ──
DEFAULT_MAX_TOKENS = 512
DEFAULT_MIN_TOKENS = 32
ESTIMATED_CHARS_PER_TOKEN = 4  # ~4 caractères par token pour le français


def _estimate_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens."""
    return len(text) // ESTIMATED_CHARS_PER_TOKEN


# ── Règles de compression ──

def _normalize_whitespace(text: str) -> str:
    """
    Normalise les espaces.

    - Remplace les séquences de 3+ newlines par 2 newlines max
    - Supprime les espaces en début/fin de chaque ligne
    - Supprime les lignes vides consécutives au-delà de 1
    """
    # Remplacer 3+ newlines par 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Supprimer les espaces en début/fin de ligne
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    # Supprimer les lignes composées uniquement d'espaces
    lines = [l.strip() for l in text.split('\n')]
    # Éliminer les lignes vides consécutives (en garder max 1)
    cleaned = []
    prev_empty = False
    for line in lines:
        if not line:
            if prev_empty:
                continue
            prev_empty = True
        else:
            prev_empty = False
        cleaned.append(line)
    return '\n'.join(cleaned).strip()


def _remove_repeated_content(text: str) -> str:
    """
    Supprime les phrases ou fragments dupliqués consécutifs.

    Détecte :
      - Phrases entières dupliquées à la suite
      - Lignes répétées (titres, séparateurs)
    """
    lines = text.split('\n')
    deduped = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            deduped.append(line)
            continue
        # Vérifier si la ligne est une répétition exacte de la précédente
        if deduped and cleaned == deduped[-1].strip():
            continue
        deduped.append(line)

    # Traiter les phrases dupliquées à l'intérieur d'un même paragraphe
    paragraphs = '\n'.join(deduped).split('\n\n')
    cleaned_paragraphs = []
    for para in paragraphs:
        # Découper en phrases (délimiteurs . ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        seen = set()
        unique_sentences = []
        for sent in sentences:
            norm = sent.strip().lower()
            if norm and len(norm) > 10 and norm in seen:
                continue
            if norm:
                seen.add(norm)
            unique_sentences.append(sent)
        cleaned_paragraphs.append(' '.join(unique_sentences))

    return '\n\n'.join(cleaned_paragraphs)


def _remove_boilerplate(text: str) -> str:
    """
    Supprime les lignes de boilerplate courantes.

    Patterns détectés (insensibles à la casse) :
      - Copyright, tous droits réservés
      - En-têtes de navigation (Page X of Y)
      - URLs génériques de navigation
      - Mentions de confidentialité et CGU
      - Filigranes numériques
    """
    patterns = [
        r'(?i)^copyright\s+©?\s*\d{4}.*$',
        r'(?i)^tous droits réservés.*$',
        r'(?i)^all rights reserved.*$',
        r'(?i)^page\s+\d+\s+(of|sur|de|/)\s+\d+.*$',
        r'(?i)^https?://\S+$',
        r'(?i)^www\.\S+\.\w{2,}$',
        r'(?i)^confidentialité.*$',
        r'(?i)^privacy\s+policy.*$',
        r'(?i)^conditions?\s+générales?.*$',
        r'(?i)^terms?\s+of\s+service.*$',
        r'(?i)^généré\s+(par|le).*$',
        r'(?i)^generated\s+(by|on).*$',
        r'(?i)^document\s+propriétaire.*$',
        r'(?i)^confidential\s+document.*$',
    ]
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append(line)
            continue
        is_boilerplate = False
        for pat in patterns:
            if re.match(pat, stripped):
                is_boilerplate = True
                break
        if not is_boilerplate:
            filtered.append(line)

    return '\n'.join(filtered)


def _extract_key_sentences(text: str, ratio: float = 0.6, min_sentences: int = 2) -> str:
    """
    Extrait les phrases les plus informatives par densité de mots-clés.

    Stratégie :
      1. Découper le texte en phrases.
      2. Compter la fréquence des mots pleins (>= 3 lettres, non-stopwords).
      3. Scorer chaque phrase par somme des fréquences des mots qu'elle contient,
         normalisée par la longueur de la phrase.
      4. Garder les meilleures phrases jusqu'à `ratio` du texte original.

    Args:
        text: Texte à filtrer.
        ratio: Proportion du texte à conserver (entre 0.0 et 1.0).
        min_sentences: Nombre minimum de phrases à conserver.

    Retourne:
        Texte composé des phrases les plus pertinentes.
    """
    # Stopwords français + anglais basiques
    stopwords = {
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 'sont',
        'dans', 'pour', 'sur', 'avec', 'par', 'pas', 'pas', 'plus', 'très',
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for', 'and', 'or',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'shall', 'can', 'need', 'dare', 'ought', 'used',
        'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
        'mes', 'tes', 'ses', 'nos', 'vos', 'leurs', 'notre', 'votre', 'leur',
        'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
        'me', 'te', 'se', 'nous', 'vous', 'se', 'moi', 'toi', 'soi',
        'lui', 'elle', 'eux', 'elles', 'que', 'qui', 'dont', 'où',
        'quoi', 'lequel', 'laquelle', 'lesquels', 'lesquelles',
        'au', 'aux', 'à', 'aux', 'ne', 'ni', 'ou', 'mais', 'car', 'donc',
        'or', 'si', 'comme', 'quand', 'lorsque', 'depuis', 'pendant',
        'avant', 'après', 'entre', 'sans', 'sous', 'dans', 'par',
        'ceci', 'cela', 'ça', 'celui', 'celle', 'ceux', 'celles',
        'c\'est', 's\'est', 'n\'est', 'l\'on', 'c\'', 's\'', 'n\'', 'd\'',
    }

    # Découpage en phrases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= min_sentences:
        return text

    # Compter les fréquences des mots pleins
    word_freq = {}
    for sent in sentences:
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', sent.lower())
        for w in words:
            if w not in stopwords:
                word_freq[w] = word_freq.get(w, 0) + 1

    if not word_freq:
        # Fallback: garder les premières phrases
        target_chars = int(len(text) * ratio)
        result = ''
        for sent in sentences:
            if len(result) + len(sent) > target_chars:
                break
            result += sent + ' '
        return result.strip()

    # Scorer chaque phrase
    scored = []
    for sent in sentences:
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', sent.lower())
        if not words:
            continue
        # Score = somme des fréquences / nombre de mots
        score = sum(word_freq.get(w, 0) for w in words) / len(words)
        scored.append((sent, score))

    # Trier par score descendant
    scored.sort(key=lambda x: x[1], reverse=True)

    # Garder les phrases jusqu'à atteindre le ratio
    total_chars = len(text)
    target_chars = int(total_chars * ratio)
    min_chars = min(target_chars, sum(len(s[0]) for s in scored[:min_sentences]))

    selected = []
    seen_texts = set()
    current_chars = 0

    for sent, _ in scored:
        sent_text = sent.strip()
        if sent_text.lower() in seen_texts:
            continue
        if current_chars + len(sent_text) > target_chars and current_chars >= min_chars:
            break
        selected.append(sent_text)
        seen_texts.add(sent_text.lower())
        current_chars += len(sent_text)

    # Réordonner selon l'ordre original pour garder la cohérence
    ordered = []
    seen_ordered = set()
    for sent in sentences:
        stripped = sent.strip()
        norm = stripped.lower()
        if norm in seen_texts and norm not in seen_ordered:
            ordered.append(stripped)
            seen_ordered.add(norm)

    return ' '.join(ordered) if ordered else text


def _intelligent_truncate(text: str, max_chars: int) -> str:
    """
    Troncature intelligente aux limites sémantiques.

    Préfère couper :
      1. À une limite de paragraphe (\n\n)
      2. À une fin de phrase (. ! ?)
      3. À un saut de ligne simple
      4. Sinon, à la limite exacte
    """
    if len(text) <= max_chars:
        return text

    # Essayer limite de paragraphe
    truncated = text[:max_chars]
    para_cut = truncated.rfind('\n\n')
    if para_cut > max_chars * 0.5:  # Au moins 50% de la cible
        return text[:para_cut].strip()

    # Essayer limite de phrase
    sent_cut = max(
        truncated.rfind('. '),
        truncated.rfind('! '),
        truncated.rfind('? '),
        truncated.rfind('.\n'),
        truncated.rfind('!\n'),
        truncated.rfind('?\n'),
    )
    if sent_cut > max_chars * 0.5:
        return text[:sent_cut + 1].strip()

    # Essayer limite de ligne
    line_cut = truncated.rfind('\n')
    if line_cut > max_chars * 0.5:
        return text[:line_cut].strip()

    # Fallback: couper à max_chars
    return text[:max_chars].strip() + '…'


# ── Compressor principal ──

class ContextCompressor:
    """
    Compresseur de chunks RAG par règles textuelles.

    Applique une chaîne de transformations configurables :
      1. Normalisation (espaces, sauts de ligne)
      2. Déduplication (phrases répétées)
      3. Suppression de boilerplate
      4. Extraction de phrases clés (optionnelle)
      5. Troncature intelligente

    Args:
        enable_normalization: Activer la normalisation du texte.
        enable_dedup: Activer la déduplication des phrases.
        enable_boilerplate: Activer la suppression de boilerplate.
        enable_key_extraction: Activer l'extraction de phrases clés.
        key_sentence_ratio: Ratio de conservation pour l'extraction (défaut: 0.6).
        min_tokens: Nombre minimum de tokens à conserver après compression.
    """

    def __init__(
        self,
        enable_normalization: bool = True,
        enable_dedup: bool = True,
        enable_boilerplate: bool = True,
        enable_key_extraction: bool = True,
        key_sentence_ratio: float = 0.6,
        min_tokens: int = DEFAULT_MIN_TOKENS,
    ):
        self.enable_normalization = enable_normalization
        self.enable_dedup = enable_dedup
        self.enable_boilerplate = enable_boilerplate
        self.enable_key_extraction = enable_key_extraction
        self.key_sentence_ratio = max(0.1, min(1.0, key_sentence_ratio))
        self.min_tokens = max(16, min_tokens)

        logger.debug(
            "ContextCompressor: norm=%s dedup=%s boilerplate=%s extract=%s ratio=%.2f min_tok=%d",
            enable_normalization, enable_dedup, enable_boilerplate,
            enable_key_extraction, self.key_sentence_ratio, self.min_tokens,
        )

    def compress(
        self,
        text: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict:
        """
        Compresse un chunk RAG par règles textuelles.

        Args:
            text: Texte du chunk à compresser.
            max_tokens: Nombre maximum de tokens cible après compression.

        Retourne:
            Dict avec :
              - compressed_text : texte compressé
              - stats : dict de statistiques de compression
                - original_tokens : tokens estimés avant compression
                - compressed_tokens : tokens estimés après compression
                - compression_ratio : ratio de compression (0.0 à 1.0)
                - stages_applied : liste des étapes appliquées
                - truncated : booléen indiquant si une troncature a eu lieu
        """
        if not text or not text.strip():
            return {
                "compressed_text": "",
                "stats": {
                    "original_tokens": 0,
                    "compressed_tokens": 0,
                    "compression_ratio": 1.0,
                    "stages_applied": [],
                    "truncated": False,
                },
            }

        original_text = text
        original_tokens = _estimate_tokens(original_text)
        stages_applied = []
        truncated = False
        min_chars = self.min_tokens * ESTIMATED_CHARS_PER_TOKEN

        # 1. Normalisation
        if self.enable_normalization:
            text = _normalize_whitespace(text)
            stages_applied.append("normalization")

        # 2. Déduplication
        if self.enable_dedup:
            text = _remove_repeated_content(text)
            stages_applied.append("dedup")

        # 3. Suppression de boilerplate
        if self.enable_boilerplate:
            text = _remove_boilerplate(text)
            stages_applied.append("boilerplate_removal")

        # Vérifier que le texte n'est pas vide après nettoyage
        if not text.strip():
            # Fallback: garder l'original normalisé
            text = original_text.strip()

        # 4. Extraction de phrases clés
        if self.enable_key_extraction:
            # Appliquer seulement si le texte dépasse le seuil minimum
            text_tokens = _estimate_tokens(text)
            if text_tokens > self.min_tokens * 2:
                ratio = min(self.key_sentence_ratio, max_tokens / max(text_tokens, 1))
                text = _extract_key_sentences(text, ratio=ratio)
                stages_applied.append("key_extraction")

        # 5. Troncature intelligente
        max_chars = max_tokens * ESTIMATED_CHARS_PER_TOKEN
        if len(text) > max_chars and len(text) > min_chars:
            # Calculer une troncature à la cible
            target = max(max_chars, min_chars)
            if len(text) > target:
                text = _intelligent_truncate(text, target)
                truncated = True
                stages_applied.append("truncation")

        # Vérifier que le résultat a au moins min_tokens
        if _estimate_tokens(text) < self.min_tokens and _estimate_tokens(original_text) > self.min_tokens:
            # Si trop compressé, garder l'original tronqué
            text = _intelligent_truncate(original_text, max_chars)
            stages_applied.append("fallback_truncation")

        compressed_tokens = _estimate_tokens(text)
        original_nonzero = max(original_tokens, 1)
        compression_ratio = round(compressed_tokens / original_nonzero, 4)

        stats = {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": min(compression_ratio, 1.0),
            "stages_applied": stages_applied,
            "truncated": truncated,
        }

        logger.debug(
            "Compression: %d → %d tok (ratio=%.2f) stages=%s",
            original_tokens, compressed_tokens, compression_ratio, stages_applied,
        )

        return {
            "compressed_text": text,
            "stats": stats,
        }

    def compress_batch(
        self,
        chunks: list[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        key: str = "text",
    ) -> list[dict]:
        """
        Compresse une liste de chunks RAG.

        Args:
            chunks: Liste de dicts, chacun devant contenir une clé 'text' (ou key spécifié).
            max_tokens: Nombre maximum de tokens par chunk compressé.
            key: Nom de la clé contenant le texte dans chaque dict.

        Retourne:
            La liste des chunks modifiés avec :
              - Texte compressé dans la clé d'origine
              - 'compression_stats' ajouté à chaque chunk
        """
        compressed_chunks = []
        for chunk in chunks:
            text = chunk.get(key, "")
            result = self.compress(text, max_tokens=max_tokens)
            new_chunk = dict(chunk)
            new_chunk[key] = result["compressed_text"]
            new_chunk["compression_stats"] = result["stats"]
            compressed_chunks.append(new_chunk)
        return compressed_chunks

    def get_stats(self) -> dict:
        """Retourne la configuration actuelle du compresseur."""
        return {
            "enable_normalization": self.enable_normalization,
            "enable_dedup": self.enable_dedup,
            "enable_boilerplate": self.enable_boilerplate,
            "enable_key_extraction": self.enable_key_extraction,
            "key_sentence_ratio": self.key_sentence_ratio,
            "min_tokens": self.min_tokens,
        }


# ── Fonctions utilitaires ──

def compress_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_normalization: bool = True,
    enable_dedup: bool = True,
    enable_boilerplate: bool = True,
    enable_key_extraction: bool = True,
) -> str:
    """
    Fonction raccourci pour compresser un texte rapidement.

    Usage :
        compressed = compress_text("Long texte...", max_tokens=256)
    """
    compressor = ContextCompressor(
        enable_normalization=enable_normalization,
        enable_dedup=enable_dedup,
        enable_boilerplate=enable_boilerplate,
        enable_key_extraction=enable_key_extraction,
    )
    result = compressor.compress(text, max_tokens=max_tokens)
    return result["compressed_text"]


def compress_chunks(
    chunks: list[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    key: str = "text",
) -> list[dict]:
    """
    Fonction raccourci pour compresser une liste de chunks rapidement.

    Usage :
        compressed = compress_chunks(rag_results, max_tokens=256)
    """
    compressor = ContextCompressor()
    return compressor.compress_batch(chunks, max_tokens=max_tokens, key=key)


# ── Singleton ──
_compressor_instance: Optional[ContextCompressor] = None


def get_compressor(
    enable_normalization: bool = True,
    enable_dedup: bool = True,
    enable_boilerplate: bool = True,
    enable_key_extraction: bool = True,
    key_sentence_ratio: float = 0.6,
) -> ContextCompressor:
    """
    Retourne l'instance singleton du ContextCompressor.

    Usage :
        compressor = get_compressor()
        result = compressor.compress(chunk_text)
    """
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = ContextCompressor(
            enable_normalization=enable_normalization,
            enable_dedup=enable_dedup,
            enable_boilerplate=enable_boilerplate,
            enable_key_extraction=enable_key_extraction,
            key_sentence_ratio=key_sentence_ratio,
        )
    return _compressor_instance
