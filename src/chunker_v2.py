#!/usr/bin/env python3
"""
chunker_v2.py — Small-to-Big chunker pour NURU V2.

Indexe des petits chunks (128 tokens) pour la précision de la recherche,
mais retourne le grand chunk parent (512 tokens) comme contexte.

Stratégie :
  - Petits chunks (128 tok) → indexés dans la vector DB pour la recherche
  - Grands chunks parents (512 tok) → retournés comme contexte de réponse
  - Découpage sémantique aux frontières naturelles (paragraphes, titres)

Usage :
    chunker = SmallToBigChunker(small_size=128, parent_size=512)
    small_chunks, parent_chunks = chunker.chunk_document(text, source="doc.pdf")
"""

import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("nuru.v2.chunker")


@dataclass
class SmallChunk:
    """Petit chunk indexé pour la recherche (128 tokens)."""
    id: str
    text: str
    parent_id: str       # Référence au parent chunk
    source: str
    chunk_index: int


@dataclass
class ParentChunk:
    """Grand chunk retourné comme contexte (512 tokens)."""
    id: str
    text: str
    source: str
    small_chunk_ids: List[str] = field(default_factory=list)


def _estimate_tokens(text: str) -> int:
    """Estimation rapide : ~4 caractères par token en français."""
    return len(text) // 4


class SmallToBigChunker:
    """
    Chunker Small-to-Big avec découpage sémantique.

    Args:
        small_size: Taille des petits chunks en tokens (défaut: 128)
        parent_size: Taille des chunks parents en tokens (défaut: 512)
        overlap: Chevauchement entre chunks consécutifs en tokens (défaut: 16)
    """

    def __init__(
        self,
        small_size: int = 128,
        parent_size: int = 512,
        overlap: int = 16,
    ):
        self.small_size = small_size
        self.parent_size = parent_size
        self.overlap = overlap
        logger.debug(
            "SmallToBigChunker: small=%d, parent=%d, overlap=%d",
            small_size, parent_size, overlap,
        )

    def chunk_document(
        self,
        text: str,
        source: str,
    ) -> tuple[List[SmallChunk], List[ParentChunk]]:
        """
        Découpe un document en petits chunks + chunks parents.

        Args:
            text: Texte complet du document.
            source: Identifiant de la source (nom de fichier).

        Retourne:
            (small_chunks, parent_chunks)
        """
        if not text or not text.strip():
            return [], []

        # 1. Découpage sémantique aux frontières naturelles
        paragraphs = self._split_by_semantics(text)
        if not paragraphs:
            return [], []

        small_chunks: List[SmallChunk] = []
        parent_chunks: List[ParentChunk] = []
        parent_idx = 0
        small_idx = 0

        current_parent_text = ""
        current_parent_small_ids: List[str] = []

        for para in paragraphs:
            if not para.strip():
                continue

            words = para.split()
            # Découper en petits chunks de `small_size` tokens
            step = max(1, self.small_size - self.overlap)
            for i in range(0, len(words), step):
                small_text = " ".join(words[i:i + self.small_size])
                if not small_text.strip():
                    continue

                small_id = f"{source}:s:{small_idx}"
                small_idx += 1

                # Accumuler dans le parent
                current_parent_text += " " + small_text
                current_parent_small_ids.append(small_id)

                parent_words = current_parent_text.split()
                if len(parent_words) >= self.parent_size:
                    parent_id = f"{source}:p:{parent_idx}"
                    parent_chunks.append(ParentChunk(
                        id=parent_id,
                        text=current_parent_text.strip(),
                        source=source,
                        small_chunk_ids=current_parent_small_ids.copy(),
                    ))
                    parent_idx += 1

                    # Reset avec overlap
                    overflow = " ".join(parent_words[-self.overlap:]) if self.overlap > 0 else ""
                    current_parent_text = overflow
                    current_parent_small_ids = []

                small_chunks.append(SmallChunk(
                    id=small_id,
                    text=small_text,
                    parent_id=f"{source}:p:{parent_idx}",
                    source=source,
                    chunk_index=small_idx - 1,
                ))

        # Dernier parent chunk s'il reste du texte
        if current_parent_text.strip():
            parent_id = f"{source}:p:{parent_idx}"
            parent_chunks.append(ParentChunk(
                id=parent_id,
                text=current_parent_text.strip(),
                source=source,
                small_chunk_ids=current_parent_small_ids,
            ))

        logger.debug(
            "Document '%s': %d small chunks, %d parent chunks",
            source, len(small_chunks), len(parent_chunks),
        )
        return small_chunks, parent_chunks

    def _split_by_semantics(self, text: str) -> List[str]:
        """
        Découpe le texte aux frontières sémantiques naturelles.

        Méthodes de découpage (par ordre de priorité) :
          1. Titres Markdown (# ## ### etc.)
          2. Doubles sauts de ligne (paragraphes)
          3. Listes numérotées ou à puces
        """
        if not text:
            return []

        # Pattern combiné : titres markdown OU doubles newlines OU listes
        pattern = r'(?:^|\n)(#{1,6}\s.*|\n\n+|(?=\d+\.\s)|(?=[-*•]\s))'
        parts = re.split(pattern, text)

        # Nettoyage et filtrage
        result = []
        for p in parts:
            cleaned = p.strip()
            if cleaned and len(cleaned) > 10:  # Ignorer les fragments trop courts
                result.append(cleaned)

        return result if result else [text.strip()]

    def get_stats(self) -> dict:
        """Statistiques du chunker."""
        return {
            "small_size": self.small_size,
            "parent_size": self.parent_size,
            "overlap": self.overlap,
        }


# ── Singleton ──
_chunker_instance: Optional[SmallToBigChunker] = None


def get_chunker(
    small_size: int = 128,
    parent_size: int = 512,
    overlap: int = 16,
) -> SmallToBigChunker:
    """Retourne l'instance singleton du chunker."""
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = SmallToBigChunker(small_size, parent_size, overlap)
    return _chunker_instance
