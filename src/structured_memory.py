#!/usr/bin/env python3
"""
structured_memory.py — Mémoire structurée clé-valeur pour NURU.

Extrait des faits des conversations via regex et les stocke dans un fichier JSON.
Injecte les faits mémorisés dans le prompt système pour personnaliser les réponses.

Usage :
    mem = StructuredMemory()
    facts = mem.extract_facts("Je m'appelle Jean et mon projet est NURU.")
    mem.store_fact("nom", "Jean", confidence=0.95)
    context = mem.get_context()
    mem.forget("projet")
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional


# ── Patterns regex pour l'extraction de faits ──
# Chaque pattern capture la valeur dans un groupe nommé 'value'.
# L'ordre des patterns détermine la priorité si plusieurs matchs.

FACT_PATTERNS: list[tuple[str, str, str]] = [
    # (key, label, regex_pattern)
    ("nom", "nom", r"(?:je\s+m['']appelle)\s+(?P<value>[A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+)|[.,!?;]|$)"),
    ("projet", "projet", r"(?:mon\s+projet\s+(?:est|s['']appelle|appelle))\s+(?P<value>[A-Za-z0-9À-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+)|[.,!?;]|$)"),
    ("langue", "langue", r"(?:je\s+parle)\s+(?P<value>[A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+|couramment|un\s+peu)|[.,!?;]|$)"),
    ("email", "email", r"(?:mon\s+email\s+(?:est|:))\s*(?P<value>[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"),
    ("numero", "numéro", r"(?:mon\s+num[eé]ro\s+(?:est|:))\s*(?P<value>[\+\d\s\-\.\(\)]{6,20})"),
    ("habitation", "habitation", r"(?:j['']habite\s+(?:à|en|dans))\s+(?P<value>[A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+|depuis\s+)|[.,!?;]|$)"),
    ("travail", "travail", r"(?:je\s+travaille\s+(?:chez|sur|dans|pour))\s+(?P<value>[A-Za-z0-9À-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+|depuis\s+)|[.,!?;]|$)"),
    ("aime", "aime", r"(?:j['']aime)\s+(?P<value>[A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+|beaucoup|particulièrement)|[.,!?;]|$)"),
    ("prefere", "préfère", r"(?:je\s+pr[eé]f[eè]re)\s+(?P<value>[A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+|à\s+)|[.,!?;]|$)"),
]


class StructuredMemory:
    """
    Mémoire structurée clé-valeur persistée dans un fichier JSON.

    Chaque fait a :
      - key : identifiant unique (ex: "nom", "email")
      - value : valeur extraite
      - confidence : score de confiance (0.0 - 1.0)
      - timestamp : horodatage Unix
      - status : "active" ou "replaced" (si un nouveau fait a remplacé celui-ci)
      - source : texte d'origine (tronqué)
    """

    def __init__(self, storage_path: str | Path | None = None):
        self._storage_path = Path(storage_path) if storage_path else Path(__file__).parent.parent / "data" / "structured_memory.json"
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._facts: dict[str, dict] = {}
        self._load()

    # ── Persistance ──

    def _load(self) -> None:
        """Charge les faits depuis le fichier JSON."""
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    self._facts = json.load(f)
            except (json.JSONDecodeError, PermissionError) as e:
                print(f"  ⚠ Erreur chargement mémoire structurée : {e}", file=__import__('sys').stderr)
                self._facts = {}

    def _save(self) -> None:
        """Sauvegarde les faits dans le fichier JSON."""
        try:
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(self._facts, f, ensure_ascii=False, indent=2)
        except PermissionError as e:
            print(f"  ⚠ Erreur sauvegarde mémoire structurée : {e}", file=__import__('sys').stderr)

    # ── Extraction de faits ──

    def extract_facts(self, text: str) -> dict[str, str]:
        """
        Extrait des faits d'un texte en utilisant des regex.

        Args:
            text: Texte à analyser (requête utilisateur + réponse).

        Retourne:
            dict {key: value} des faits extraits (un seul par catégorie).
        """
        if not text or not isinstance(text, str):
            return {}

        facts: dict[str, str] = {}

        for key, label, pattern in FACT_PATTERNS:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group("value").strip()
                    # Nettoyage : enlever la ponctuation finale résiduelle
                    value = re.sub(r"[.,!?;]+$", "", value).strip()
                    if value and len(value) > 1:  # Ignorer les valeurs trop courtes
                        facts[key] = value
            except re.error:
                continue

        return facts

    def extract_and_store(self, text: str) -> dict[str, str]:
        """
        Extrait des faits d'un texte et les stocke immédiatement.

        Args:
            text: Texte à analyser.

        Retourne:
            dict des faits nouvellement extraits.
        """
        new_facts = self.extract_facts(text)
        for key, value in new_facts.items():
            # Si le fait existe déjà, on ne met à jour que si la nouvelle valeur est différente
            if key not in self._facts or self._facts[key]["value"] != value:
                self.store_fact(key, value, confidence=0.85)
        return new_facts

    # ── Gestion des faits ──

    def store_fact(self, key: str, value: str, confidence: float = 0.8) -> dict:
        """
        Stocke un fait.

        Détection de conflit : si un fait avec la même clé existe déjà
        avec une VALEUR DIFFÉRENTE, l'ancien est marqué status='replaced'
        et le nouveau est créé avec status='active'.

        Args:
            key: Identifiant du fait (ex: "nom", "langue").
            value: Valeur du fait.
            confidence: Score de confiance (défaut: 0.8).

        Retourne:
            Le dictionnaire du fait stocké.
        """
        entry = {
            "value": value,
            "confidence": min(max(confidence, 0.0), 1.0),
            "timestamp": time.time(),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active",
        }

        if key in self._facts:
            old = self._facts[key]
            if old["value"] == value:
                # Même valeur → augmentation de confiance, garde actif
                entry["confidence"] = min(1.0, old["confidence"] + 0.1)
            else:
                # Valeur différente → conflit : marque l'ancien comme remplacé
                old["status"] = "replaced"

        self._facts[key] = entry
        self._save()
        return entry

    def get_fact(self, key: str) -> Optional[dict]:
        """Récupère un fait par sa clé."""
        return self._facts.get(key)

    def get_all_facts(self) -> dict[str, dict]:
        """Retourne tous les faits stockés."""
        return dict(self._facts)

    def forget(self, key: str) -> bool:
        """
        Supprime un fait de la mémoire.

        Args:
            key: Clé du fait à oublier.

        Retourne:
            True si le fait existait et a été supprimé, False sinon.
        """
        if key in self._facts:
            del self._facts[key]
            self._save()
            return True
        return False

    def forget_all(self) -> None:
        """Supprime tous les faits stockés."""
        self._facts.clear()
        self._save()

    # ── Description utilisateur ──

    def describe_user(self) -> str | None:
        """
        Génère une description en français de l'utilisateur à partir des faits stockés.
        Retourne None si aucun fait pertinent n'est trouvé.

        Exemple :
            "Tu es Leblanc BAHIGA Mudarhi, un ingénieur agronome & informaticien basé en RDC (Kinshasa).
             Tu travailles sur le projet NURU — assistant IA personnel."
        """
        facts = self._facts
        active = {k: v for k, v in facts.items() if v.get("status", "active") == "active"}
        if not active:
            return None

        parts = []
        nom = active.get("nom", {}).get("value", "")
        prenom = active.get("prenom", {}).get("value", "")
        profession = active.get("profession", {}).get("value", "")
        lieu = active.get("lieu", {}).get("value", "")
        projet = active.get("projet", {}).get("value", "")

        # Construction de la phrase d'introduction
        if nom:
            intro = f"Tu es {nom}"
            parts.append(intro)
        elif prenom:
            parts.append(f"Tu es {prenom}")

        # Ajouter la profession
        if profession:
            if parts:
                parts[-1] += f", un {profession}"
            else:
                parts.append(f"Tu es {profession}")

        # Ajouter le lieu
        if lieu:
            parts[-1] += f" basé en {lieu}" if "en " not in lieu[:4].lower() else f" basé {lieu}"

        # Ajouter le projet
        if projet:
            parts.append(f"Tu travailles sur le projet {projet}")

        return ". ".join(parts) + "."

    # ── Contexte pour le prompt ──

    def get_context(self) -> str:
        """
        Génère un bloc de texte des faits structurés pour injection dans le prompt système.

        Format :
            <|im_start|>system
            [Mémoire structurée — faits connus sur l'utilisateur]
            - nom : Jean
            - langue : français
            ...
            <|im_end|>

        Retourne:
            Chaîne vide si aucun fait, sinon le bloc formaté.
        """
        if not self._facts:
            return ""

        LABELS = {
            "nom": "Nom",
            "prenom": "Prénom",
            "profession": "Profession",
            "lieu": "Lieu",
            "projet": "Projet",
            "langue": "Langue(s)",
            "email": "Email",
            "numero": "Numéro",
            "habitation": "Habitation",
            "travail": "Travail",
            "aime": "Aime",
            "prefere": "Préfère",
        }

        lines = ["<|im_start|>system",
                 "[Mémoire structurée — informations connues sur l'utilisateur]"]

        for key, entry in self._facts.items():
            if entry.get("status", "active") != "active":
                continue
            label = LABELS.get(key, key.capitalize())
            value = entry["value"]
            lines.append(f"- {label} : {value}")

        lines.append("Utilise ces informations pour personnaliser tes réponses.")
        lines.append("<|im_end|>")

        return "\n".join(lines)

    # ── Statistiques ──

    def __len__(self) -> int:
        return len(self._facts)

    def __repr__(self) -> str:
        return f"StructuredMemory(faits={len(self._facts)}, path={self._storage_path})"

    def get_stats(self) -> dict:
        """Statistiques de la mémoire structurée."""
        return {
            "total_facts": len(self._facts),
            "keys": list(self._facts.keys()),
            "storage_path": str(self._storage_path),
            "file_exists": self._storage_path.exists(),
        }
