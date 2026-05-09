#!/usr/bin/env python3
"""
transparency.py — Mode Transparence pour NURU.

Enregistre chaque décision du routeur avec timestamp, requête, niveau choisi,
raison (keywords détectés, estimation tokens, scores), latence, modèle utilisé.

Stockage :
  - Mémoire : deque (max 100 entrées)
  - Fichier  : data/transparency_log.json (accumulatif, append)

Usage :
    from transparency import get_transparency_logger
    tl = get_transparency_logger()
    tl.log_decision(query, level, reason, latency_ms, model_used, extra_scores=None)
    for entry in tl.get_timeline(count=5):
        print(entry)
    html = tl.get_timeline_html()
"""

import json
import time
import sys
from pathlib import Path
from collections import deque
from typing import Optional


class TransparencyLogger:
    """
    Journalise chaque décision de routage pour l'audabilité / mode transparence.
    """

    def __init__(self, maxlen: int = 100):
        self.maxlen = maxlen
        self._buffer: deque[dict] = deque(maxlen=maxlen)
        self._log_path = self._resolve_log_path()

        # Charger les entrées existantes depuis le fichier
        self._load_from_disk()

    # ── Chemins ──

    @staticmethod
    def _resolve_log_path() -> Path:
        """Retourne le chemin absolu du fichier de log."""
        return Path(__file__).parent.parent / "data" / "transparency_log.json"

    # ── Persistance ──

    def _load_from_disk(self):
        """Recharge les dernières entrées depuis le fichier JSON."""
        path = self._log_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ne charger que les maxlen dernières entrées
            if isinstance(data, list):
                for entry in data[-self.maxlen:]:
                    self._buffer.append(entry)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠ Transparency: erreur lecture {path} : {e}", file=sys.stderr)

    def _append_to_disk(self, entry: dict):
        """Ajoute une entrée au fichier JSON (accumulatif)."""
        path = self._log_path
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            else:
                data = []
        except (json.JSONDecodeError, IOError):
            data = []

        data.append(entry)

        # Limiter la taille du fichier (garder les 1000 dernières)
        if len(data) > 1000:
            data = data[-1000:]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Journalisation ──

    def log_decision(
        self,
        query: str,
        level: int,
        level_name: str,
        reason: str,
        latency_ms: float,
        model_used: str,
        extra_scores: Optional[dict] = None,
    ):
        """
        Enregistre une décision de routage.

        Args:
            query: Requête utilisateur originale
            level: Niveau choisi (1, 2, ou 3)
            level_name: Nom lisible du niveau (ex: "RAG Local")
            reason: Raison textuelle courte
            latency_ms: Latence de la requête en ms
            model_used: Identifiant du modèle utilisé
            extra_scores: Dict optionnel avec détails supplémentaires
                          (keywords, estimated_tokens, scores, etc.)
        """
        entry = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query[:500],  # Tronquer les très longues requêtes
            "query_length": len(query),
            "level": level,
            "level_name": level_name,
            "reason": reason,
            "latency_ms": round(latency_ms, 1),
            "model_used": model_used,
            "extra": extra_scores or {},
        }

        self._buffer.append(entry)
        self._append_to_disk(entry)

    # ── Affichage console ──

    def get_timeline(self, count: int = 10) -> list[str]:
        """
        Retourne une liste de lignes formatées pour affichage console.

        Args:
            count: Nombre d'entrées à retourner (les plus récentes)

        Returns:
            Liste de chaînes prêtes à être imprimées
        """
        if not self._buffer:
            return ["(Aucune décision enregistrée)"]

        lines = []
        entries = list(self._buffer)[-count:]

        for entry in reversed(entries):
            ts = entry.get("timestamp_iso", "?")
            lvl = entry.get("level", "?")
            lvl_name = entry.get("level_name", "")
            query = entry.get("query", "")[:60]
            reason = entry.get("reason", "")
            lat = entry.get("latency_ms", 0)
            model = entry.get("model_used", "")

            # Niveau coloré (codes ANSI)
            level_colors = {1: "\033[92m", 2: "\033[96m", 3: "\033[93m"}
            lvl_color = level_colors.get(lvl, "\033[90m")
            reset = "\033[0m"
            dim = "\033[2m"

            line = (
                f"{dim}[{ts}]{reset} "
                f"{lvl_color}N{lvl}{reset} "
                f"({lvl_name}) "
                f"{dim}|{reset} "
                f"\"{query}{'...' if len(entry.get('query', '')) > 60 else ''}\" "
                f"{dim}|{reset} "
                f"{lat}ms "
                f"{dim}|{model}{reset}"
            )
            lines.append(line)

        return lines

    # ── Affichage HTML (interface web) ──

    def get_timeline_html(self, count: int = 20) -> str:
        """
        Retourne un bloc HTML <table> pour intégration dans l'interface web.

        Args:
            count: Nombre d'entrées à afficher (les plus récentes)

        Returns:
            Chaîne HTML avec une table stylisée
        """
        if not self._buffer:
            return '<p style="color:rgba(0,180,255,0.3);">Aucune décision enregistrée.</p>'

        rows = []
        entries = list(self._buffer)[-count:]

        level_badges = {
            1: '<span class="badge ok">RAG Local</span>',
            2: '<span class="badge" style="background:rgba(0,180,255,0.1);color:#00b4ff;">LLM Local</span>',
            3: '<span class="badge warn">Cloud API</span>',
        }

        for entry in reversed(entries):
            ts = entry.get("timestamp_iso", "?")
            lvl = entry.get("level", "?")
            badge = level_badges.get(lvl, f'<span class="badge">N{lvl}</span>')
            query = entry.get("query", "")[:80]
            reason = entry.get("reason", "")[:60]
            lat = entry.get("latency_ms", 0)
            model = entry.get("model_used", "")

            # Extraire les détails supplémentaires
            extra = entry.get("extra", {})
            details_parts = []
            if extra.get("estimated_tokens"):
                details_parts.append(f"~{extra['estimated_tokens']} tok")
            if extra.get("keywords_detected"):
                kw = extra["keywords_detected"]
                if isinstance(kw, list):
                    details_parts.append(f"KW: {', '.join(kw[:3])}")
            scores = extra.get("scores", {})
            if scores:
                score_str = " ".join(f"{k}={v}" for k, v in scores.items())
                details_parts.append(score_str)
            details = " | ".join(details_parts) if details_parts else "—"

            rows.append(f"""<tr>
                <td style="font-family:monospace;font-size:11px;white-space:nowrap;">{ts}</td>
                <td>{badge}</td>
                <td style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{query}</td>
                <td style="font-size:11px;color:rgba(0,180,255,0.4);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{reason}</td>
                <td style="font-size:11px;color:rgba(0,212,170,0.6);">{lat} ms</td>
                <td style="font-size:11px;color:rgba(0,180,255,0.4);">{model}</td>
                <td style="font-size:10px;color:rgba(0,180,255,0.25);">{details}</td>
            </tr>""")

        return f"""<table>
            <thead><tr>
                <th>Timestamp</th>
                <th>Niveau</th>
                <th>Requête</th>
                <th>Raison</th>
                <th>Latence</th>
                <th>Modèle</th>
                <th>Détails</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>"""

    # ── Utilitaires ──

    @property
    def count(self) -> int:
        """Nombre d'entrées en mémoire tampon."""
        return len(self._buffer)

    def clear(self):
        """Vide le tampon mémoire (ne touche pas au fichier)."""
        self._buffer.clear()

    def get_entries(self, count: int = 10) -> list[dict]:
        """Retourne les N dernières entrées brutes."""
        return list(self._buffer)[-count:]


# ── Singleton partagé ──
_transparency_instance: Optional[TransparencyLogger] = None


def get_transparency_logger(maxlen: int = 100) -> TransparencyLogger:
    """Retourne l'instance singleton du logger de transparence."""
    global _transparency_instance
    if _transparency_instance is None:
        _transparency_instance = TransparencyLogger(maxlen=maxlen)
    return _transparency_instance
