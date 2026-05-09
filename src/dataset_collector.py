#!/usr/bin/env python3
"""
dataset_collector.py — Collecte les conversations NURU pour fine-tuning futur.

Sauvegarde chaque échange (query + response + métadonnées) dans un fichier JSON
à mesure qu'ils se produisent. Prêt à alimenter un pipeline MLX LoRA.

Format final :
  [
    {
      "id": "uuid",
      "timestamp": "2026-05-04T12:34:56",
      "user": "question de l'utilisateur",
      "assistant": "réponse de NURU",
      "model": "deepseek/deepseek-v4-flash",
      "level": "cloud",
      "latency_ms": 4560,
      "feedback": null  # ou "good" / "bad"
    },
    ...
  ]

Usage :
    from dataset_collector import DatasetCollector
    dc = DatasetCollector()
    dc.add("Qui suis-je?", "Tu es Leblanc...", "deepseek/...", "cloud", 4560)
    dc.feedback("uuid", "good")
"""

import json
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone

# ── Fichier de données ──
DATASET_DIR = Path.home() / ".nuru" / "dataset"
DATASET_FILE = DATASET_DIR / "conversations.jsonl"  # JSON Lines (1 objet par ligne)


class DatasetCollector:
    """Collecte les échanges NURU pour le fine-tuning futur."""

    def __init__(self):
        self._last_id = None
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        if not DATASET_FILE.exists():
            DATASET_FILE.write_text("")

    def add(self, user_query: str, assistant_response: str,
            model_used: str = "", level: str = "",
            latency_ms: float = 0.0) -> str:
        """Ajoute un échange au dataset.

        Retourne l'ID de l'échange (pour feedback ultérieur).
        """
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_query,
            "assistant": assistant_response,
            "model": model_used,
            "level": level,
            "latency_ms": round(latency_ms, 1),
            "feedback": None,  # "good" | "bad" | None
        }
        self._last_id = entry["id"]

        with open(DATASET_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry["id"]

    def feedback(self, exchange_id: str, rating: str):
        """Ajoute un feedback à un échange existant.

        rating : "good" | "bad"
        """
        if rating not in ("good", "bad"):
            return

        lines = DATASET_FILE.read_text().splitlines()
        updated = []
        for line in lines:
            entry = json.loads(line)
            if entry["id"] == exchange_id:
                entry["feedback"] = rating
            updated.append(json.dumps(entry, ensure_ascii=False))
        DATASET_FILE.write_text("\n".join(updated) + "\n")

    def count(self) -> int:
        """Nombre d'échanges collectés."""
        if not DATASET_FILE.exists():
            return 0
        return len(DATASET_FILE.read_text().splitlines())

    def export_json(self, path: str | None = None) -> str:
        """Exporte tout le dataset au format JSON (array).

        Utile pour alimenter un script de fine-tuning.
        """
        if not DATASET_FILE.exists():
            return "[]"
        entries = [json.loads(line) for line in DATASET_FILE.read_text().splitlines()]
        out_path = Path(path or (DATASET_DIR / "export.json"))
        out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
        return str(out_path)

    def stats(self) -> dict:
        """Statistiques du dataset."""
        total = self.count()
        good = 0
        bad = 0
        by_model = {}
        if total > 0:
            for line in DATASET_FILE.read_text().splitlines():
                entry = json.loads(line)
                fb = entry.get("feedback")
                if fb == "good":
                    good += 1
                elif fb == "bad":
                    bad += 1
                model = entry.get("model", "?")
                by_model[model] = by_model.get(model, 0) + 1
        return {
            "total": total,
            "good": good,
            "bad": bad,
            "by_model": by_model,
            "file": str(DATASET_FILE),
        }


if __name__ == "__main__":
    # Test rapide
    dc = DatasetCollector()
    eid = dc.add("Bonjour", "Bonjour !", "test", "local")
    print(f"Ajouté : {eid}")
    print(f"Stats : {dc.stats()}")
    print(f"Export : {dc.export_json()}")
