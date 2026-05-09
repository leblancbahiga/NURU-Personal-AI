#!/usr/bin/env python3
"""
feedback.py — Système de correction et d'apprentissage continu pour NURU.

Permet à l'utilisateur de :
  - Corriger vocalement : "Corrige : [nouvelle réponse]"
  - Corriger par texte : idem via CLI
  - Afficher l'historique des corrections
  - Supprimer une correction
  - Désactiver temporairement une correction

Les corrections sont stockées dans la collection 'corrections_prioritaires'
de ChromaDB. Lors des futures requêtes similaires, la correction est
injectée en priorité dans le contexte.

Usage :
    python3 src/feedback.py "Corrige : la réponse est X"
    python3 src/feedback.py --list
    python3 src/feedback.py --delete corr_12345
    python3 src/feedback.py --disable corr_12345
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

try:
    from rag import VectorStore
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None


CORRECTIONS_FILE = Path(__file__).parent.parent / "data" / "corrections_history.json"


@dataclass
class Correction:
    """Une correction apprise avec suivi avancé."""
    id: str
    original_query: str
    corrected_response: str
    created_at: float = field(default_factory=time.time)
    frequency: int = 1
    success_count: int = 0
    disabled: bool = False
    category: str = "general"
    confidence_score: float = 0.5        # Score de confiance (0.0 → 1.0)
    last_used_at: float = field(default_factory=time.time)
    usage_count: int = 0                 # Nombre d'utilisations réelles
    confirmation_count: int = 0          # Nombre de confirmations/renforcements


class FeedbackManager:
    """
    Gère les corrections utilisateur.
    Interface entre l'utilisateur et ChromaDB (corrections_prioritaires).
    """

    def __init__(self):
        self.store: Optional[VectorStore] = None
        if RAG_AVAILABLE:
            try:
                self.store = VectorStore()
            except Exception:
                pass
        self._history = self._load_history()

    def _load_history(self) -> list[dict]:
        """Charge l'historique des corrections depuis le fichier JSON."""
        if CORRECTIONS_FILE.exists():
            try:
                with open(CORRECTIONS_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_history(self):
        """Sauvegarde l'historique des corrections."""
        CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CORRECTIONS_FILE, "w") as f:
            json.dump(self._history, f, indent=2, default=str)

    def add_correction(self, query: str, correction: str) -> bool:
        """
        Ajoute une correction.

        Args:
            query: La question/règle déclencheuse
            correction: La réponse correcte

        Retourne:
            True si ajouté avec succès
        """
        corr_id = f"corr_{int(time.time())}_{hash(query) % 100000}"

        # Stocker dans ChromaDB (collection corrections_prioritaires)
        if self.store is not None:
            try:
                self.store.add_correction(query, correction)
            except Exception as e:
                print(f"  ⚠ Erreur ChromaDB : {e}")

        # Stocker dans l'historique JSON
        entry = {
            "id": corr_id,
            "query": query,
            "correction": correction,
            "created_at": time.time(),
            "frequency": 1,
            "disabled": False,
            "confidence_score": 0.5,
            "last_used_at": time.time(),
            "usage_count": 0,
            "confirmation_count": 0,
        }
        self._history.append(entry)
        self._save_history()

        print(f"  ✅ Correction enregistrée : {corr_id}")
        print(f"    Requête : {query[:60]}...")
        print(f"    Réponse : {correction[:60]}...")
        return True

    def get_corrections(self, include_disabled: bool = False) -> list[dict]:
        """Liste toutes les corrections."""
        if include_disabled:
            return list(self._history)
        return [c for c in self._history if not c.get("disabled", False)]

    def delete_correction(self, corr_id: str) -> bool:
        """Supprime une correction."""
        for i, c in enumerate(self._history):
            if c["id"] == corr_id:
                self._history.pop(i)
                self._save_history()
                print(f"  🗑️ Correction supprimée : {corr_id}")
                return True
        print(f"  ✗ Correction introuvable : {corr_id}")
        return False

    def toggle_correction(self, corr_id: str) -> bool:
        """Active/désactive une correction."""
        for c in self._history:
            if c["id"] == corr_id:
                c["disabled"] = not c.get("disabled", False)
                self._save_history()
                status = "désactivée" if c["disabled"] else "réactivée"
                print(f"  🔄 Correction {status} : {corr_id}")
                return True
        print(f"  ✗ Correction introuvable : {corr_id}")
        return False

    def increment_frequency(self, query: str) -> bool:
        """Incrémente le compteur d'utilisation d'une correction."""
        for c in self._history:
            if c["query"].lower() == query.lower() and not c.get("disabled", False):
                c["frequency"] = c.get("frequency", 1) + 1
                c["last_used_at"] = time.time()
                c["usage_count"] = c.get("usage_count", 0) + 1
                self._save_history()
                return True
        return False

    def confirm_correction(self, corr_id: str) -> bool:
        """
        Confirme/renforce une correction, augmentant sa confiance.
        """
        for c in self._history:
            if c["id"] == corr_id:
                c["confirmation_count"] = c.get("confirmation_count", 0) + 1
                c["confidence_score"] = self._compute_confidence_score(c)
                self._save_history()
                print(f"  ✅ Correction renforcée : {corr_id} (confiance: {c['confidence_score']:.2f})")
                return True
        print(f"  ✗ Correction introuvable : {corr_id}")
        return False

    @staticmethod
    def _compute_confidence_score(correction: dict) -> float:
        """
        Calcule le score de confiance à partir des confirmations.

        Formule : 0.5 + min(0.5, confirmations * 0.1)
        Soit : 0.5 (initiale) → 0.6 (1 confirmation) → ... → 1.0 (5+ confirmations)
        """
        base = 0.5
        confirmations = correction.get("confirmation_count", 0)
        boost = min(0.5, confirmations * 0.1)
        return round(min(1.0, base + boost), 4)

    def apply_expiration(self) -> int:
        """
        Applique la règle d'expiration :
        Une correction non utilisée depuis 30 jours voit son score réduit de 0.1 par semaine.

        Retourne le nombre de corrections affectées.
        """
        now = time.time()
        thirty_days = 30 * 24 * 3600
        one_week = 7 * 24 * 3600
        affected = 0

        for c in self._history:
            if c.get("disabled", False):
                continue

            last_used = c.get("last_used_at", c.get("created_at", now))
            age_since_use = now - last_used

            if age_since_use > thirty_days:
                # Calcul du nombre de semaines de dépassement
                weeks_overdue = (age_since_use - thirty_days) / one_week
                reduction = weeks_overdue * 0.1
                current_conf = c.get("confidence_score", 0.5)
                c["confidence_score"] = round(max(0.0, current_conf - reduction), 4)
                affected += 1

        if affected > 0:
            self._save_history()

        return affected

    def get_stats(self) -> dict:
        """Statistiques détaillées des corrections."""
        active = [c for c in self._history if not c.get("disabled", False)]
        avg_confidence = sum(c.get("confidence_score", 0.5) for c in self._history) / max(len(self._history), 1)

        # Appliquer l'expiration pour des stats à jour
        expired_count = self.apply_expiration()

        by_category = {}
        for c in self._history:
            cat = c.get("category", "general")
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total": len(self._history),
            "active": len(active),
            "disabled": len(self._history) - len(active),
            "expired_reduced": expired_count,
            "avg_confidence": round(avg_confidence, 3),
            "most_frequent": max(self._history, key=lambda c: c.get("frequency", 0))
                if self._history else None,
            "highest_confidence": max(self._history, key=lambda c: c.get("confidence_score", 0))
                if self._history else None,
            "by_category": by_category,
        }

    def parse_feedback(self, text: str) -> Optional[tuple[str, str]]:
        """
        Analyse une commande de feedback.

        "Corrige : [réponse]" → enregistre une correction
        "Non, la bonne réponse est [réponse]" → enregistre une correction

        Retourne:
            (dernière_question, correction) ou None si pas de feedback
        """
        text_lower = text.strip().lower()
        correction = None

        # Patterns de correction
        patterns = [
            (r"corrige\s*:\s*(.+)", 1),
            (r"non,\s*(?:la bonne réponse est|la réponse est|c'est)\s*(.+)", 1),
            (r"rectifie\s*:\s*(.+)", 1),
            (r"correction\s*:\s*(.+)", 1),
        ]

        import re
        for pattern, group in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                correction = match.group(group).strip()
                break

        if correction:
            # Retourner la dernière question de la session comme déclencheur
            return (None, correction)  # None = on utilisera le dernier contexte

        return None

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"FeedbackManager(corrections={stats['active']}/{stats['total']})"


def main():
    parser = argparse.ArgumentParser(description="NURU — Gestion des corrections")
    parser.add_argument("correction", nargs="?", help="Correction : 'Corrige : [réponse]'")
    parser.add_argument("--list", action="store_true", help="Lister les corrections")
    parser.add_argument("--delete", type=str, metavar="ID", help="Supprimer une correction")
    parser.add_argument("--toggle", type=str, metavar="ID", help="Activer/désactiver une correction")
    parser.add_argument("--confirm", type=str, metavar="ID", help="Confirmer/renforcer une correction")
    parser.add_argument("--stats", action="store_true", help="Statistiques des corrections")
    args = parser.parse_args()

    fb = FeedbackManager()

    if args.list:
        corrections = fb.get_corrections()
        if not corrections:
            print("📭 Aucune correction enregistrée.")
        else:
            print(f"Corrections ({len(corrections)}) :")
            for c in corrections:
                marker = "🔇" if c.get("disabled") else "✅"
                freq = c.get("frequency", 1)
                conf = c.get("confidence_score", 0.5)
                usage = c.get("usage_count", 0)
                print(f"  {marker} {c['id']} (x{freq} | conf:{conf:.2f} | used:{usage}x)")
                print(f"    Requête : {c['query'][:60]}...")
                print(f"    Réponse : {c['correction'][:60]}...")
                print()

    elif args.delete:
        fb.delete_correction(args.delete)

    elif args.toggle:
        fb.toggle_correction(args.toggle)

    elif args.confirm:
        fb.confirm_correction(args.confirm)

    elif args.stats:
        stats = fb.get_stats()
        print(f"📊 Stats corrections :")
        print(f"  Total         : {stats['total']}")
        print(f"  Actives       : {stats['active']}")
        print(f"  Désactivées   : {stats['disabled']}")
        print(f"  Expirées      : {stats['expired_reduced']}")
        print(f"  Conf. moyenne : {stats['avg_confidence']:.3f}")
        print(f"  Par catégorie : {stats['by_category']}")
        if stats['most_frequent']:
            mf = stats['most_frequent']
            print(f"  Plus fréquente : {mf['id']} (x{mf.get('frequency', 1)} | conf:{mf.get('confidence_score', 0.5):.2f})")
        if stats['highest_confidence']:
            hc = stats['highest_confidence']
            print(f"  Plus fiable    : {hc['id']} (conf:{hc.get('confidence_score', 0.5):.2f})")

    elif args.correction:
        result = fb.parse_feedback(args.correction)
        if result:
            # Mode direct : on connaît la correction
            fb.add_correction(args.correction[:60], result[1])
        else:
            # Pas un pattern de correction → ajout direct
            # Dans le daemon, le contexte serait la dernière question
            fb.add_correction(args.correction[:60], args.correction)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
