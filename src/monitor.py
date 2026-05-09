#!/usr/bin/env python3
"""
monitor.py — Module de monitoring temps réel pour NURU.

Tracke en continu :
  - RAM utilisée (psutil)
  - Latence de chaque étape du routeur (transcription, embedding,
    LLM local, cloud, TTS)
  - Nombre de tokens
  - Niveau de routage

Usage :
    from monitor import Monitor
    mon = Monitor(window=100)
    mon.start_timer("transcription")
    ...
    mon.stop_timer("transcription", tokens=42)
    print(mon.report())
    mon.save_log("data/logs/perf.json")
"""

import os
import json
import time
import sys
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ── Couleurs pour le terminal ──
class _C:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# ── Noms d'étapes standard ──
STAGE_NAMES = {
    "transcription": "Transcription",
    "embedding": "Embedding",
    "load": "Chargement modèle",
    "generate": "Génération LLM",
    "cloud": "Appel Cloud",
    "tts": "TTS",
    "total": "Total requête",
}

STAGE_COLORS = {
    "transcription": _C.MAGENTA,
    "embedding": _C.CYAN,
    "load": _C.YELLOW,
    "generate": _C.GREEN,
    "cloud": _C.RED,
    "tts": _C.MAGENTA,
    "total": _C.BOLD,
}


@dataclass
class MetricSample:
    """Une mesure unitaire de monitoring."""
    stage: str
    latency_ms: float
    tokens: int = 0
    level: int = 0
    timestamp: float = field(default_factory=time.time)


class Monitor:
    """
    Moniteur temps réel des performances de NURU.

    Utilise une deque pour garder les N dernières mesures de chaque étape.
    Propose chronométrage par start_timer / stop_timer / lap.
    """

    def __init__(self, window: int = 100):
        """
        Args:
            window: Nombre maximum de mesures conservées par étape.
        """
        self.window = window
        # stockage par étape : {stage_name: deque[MetricSample, ...]}
        self._samples: dict[str, deque[MetricSample]] = {}
        # chronomètres actifs : {stage_name: (start_time, lap_count)}
        self._timers: dict[str, tuple[float, int]] = {}
        # accumulateur pour le niveau de routage courant
        self._current_level: int = 0
        self._current_tokens: int = 0

        # Créer les deques pour les étapes connues
        for name in STAGE_NAMES:
            self._samples[name] = deque(maxlen=window)

    # ── Chronométrage ──

    def start_timer(self, stage: str):
        """Démarre le chronomètre pour une étape."""
        if stage not in self._samples:
            self._samples[stage] = deque(maxlen=self.window)
        self._timers[stage] = (time.time(), 0)

    def lap(self, stage: str, tokens: int = 0) -> float:
        """
        Enregistre un tour intermédiaire sans arrêter le chrono.
        Retourne la latence de ce segment.
        """
        now = time.time()
        if stage not in self._timers:
            self.start_timer(stage)
            return 0.0

        start, lap_count = self._timers[stage]
        elapsed_ms = round((now - start) * 1000, 1)

        sample = MetricSample(
            stage=stage,
            latency_ms=elapsed_ms,
            tokens=tokens,
            level=self._current_level,
            timestamp=now,
        )
        self._samples[stage].append(sample)

        # Redémarrer le timer pour le prochain segment
        self._timers[stage] = (now, lap_count + 1)
        return elapsed_ms

    def stop_timer(self, stage: str, tokens: int = 0) -> float:
        """
        Arrête le chronomètre et enregistre la mesure.
        Retourne la latence en ms.
        """
        now = time.time()
        if stage not in self._timers:
            self.start_timer(stage)
            self._timers[stage] = (now - 0.001, 0)  # évite division par zéro
            now = self._timers[stage][0] + 0.001

        start, _ = self._timers[stage]
        elapsed_ms = round((now - start) * 1000, 1)
        del self._timers[stage]

        sample = MetricSample(
            stage=stage,
            latency_ms=elapsed_ms,
            tokens=tokens or self._current_tokens,
            level=self._current_level,
            timestamp=now,
        )
        if stage not in self._samples:
            self._samples[stage] = deque(maxlen=self.window)
        self._samples[stage].append(sample)

        return elapsed_ms

    # ── API de monitoring ──

    def set_level(self, level: int):
        """Définit le niveau de routage courant."""
        self._current_level = level

    def set_tokens(self, tokens: int):
        """Définit le nombre de tokens courant."""
        self._current_tokens = tokens

    def record_ram(self):
        """Enregistre un snapshot RAM dans l'étape 'ram'."""
        if not PSUTIL_AVAILABLE:
            return
        mem = psutil.virtual_memory()
        sample = MetricSample(
            stage="ram",
            latency_ms=round(mem.percent, 1),
            tokens=round(mem.used / (1024**3), 2),  # Go
            level=self._current_level,
        )
        if "ram" not in self._samples:
            self._samples["ram"] = deque(maxlen=self.window)
        self._samples["ram"].append(sample)

    def current_ram_info(self) -> dict:
        """Retourne les infos RAM actuelles."""
        if not PSUTIL_AVAILABLE:
            return {"available": False, "percent": 0, "used_gb": 0, "total_gb": 0}
        mem = psutil.virtual_memory()
        return {
            "available": True,
            "percent": mem.percent,
            "used_gb": round(mem.used / (1024**3), 2),
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
        }

    # ── Rapports ──

    def _stats_for_stage(self, stage: str) -> dict:
        """Calcule les stats (min, max, avg, count, last) pour une étape."""
        samples = self._samples.get(stage)
        if not samples:
            return {"count": 0}
        values = [s.latency_ms for s in samples]
        tokens = [s.tokens for s in samples if s.tokens > 0]
        return {
            "count": len(values),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
            "avg": round(sum(values) / len(values), 1),
            "last": round(values[-1], 1) if values else 0,
            "tokens_avg": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        }

    def report(self) -> str:
        """
        Génère un tableau console coloré avec les stats de monitoring.
        """
        lines = []
        header = (
            f"{_C.CYAN}{_C.BOLD}"
            f"┌───────────────────────────── PERFORMANCE ─────────────────────────────┐"
            f"{_C.RESET}"
        )
        lines.append("")
        lines.append(header)

        # ── RAM ──
        ram = self.current_ram_info()
        if ram["available"]:
            pct = ram["percent"]
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            color = _C.GREEN if pct < 50 else (_C.YELLOW if pct < 80 else _C.RED)
            lines.append(
                f"  {_C.BOLD}RAM{_C.RESET}      "
                f"{bar} {color}{pct:.0f}%{_C.RESET} "
                f"({ram['used_gb']:.1f}/{ram['total_gb']:.1f} Go)"
            )

        # ── En-tête tableau ──
        lines.append(
            f"{_C.DIM}"
            f"  {'Étape':<20} {'Moy(ms)':<10} {'Min(ms)':<10} "
            f"{'Max(ms)':<10} {'Dernier(ms)':<12} {'Tok/moy':<8}"
            f"{_C.RESET}"
        )
        lines.append(f"  {'─'*70}")

        # Lignes par étape (dans l'ordre de STAGE_NAMES)
        for stage_key, stage_label in STAGE_NAMES.items():
            s = self._stats_for_stage(stage_key)
            if s["count"] == 0:
                continue
            color = STAGE_COLORS.get(stage_key, _C.DIM)
            tok_str = f"{s['tokens_avg']:.0f}" if s["tokens_avg"] else "—"
            lines.append(
                f"  {color}{stage_label:<20}{_C.RESET}"
                f" {s['avg']:<10.1f}"
                f" {s['min']:<10.1f}"
                f" {s['max']:<10.1f}"
                f" {s['last']:<12.1f}"
                f" {tok_str:<8}"
            )

        # Niveaux de routage
        level_dist = self._level_distribution()
        if level_dist:
            lstr = ", ".join(
                f"N{lvl}={cnt}" for lvl, cnt in sorted(level_dist.items())
            )
            lines.append(f"  {_C.DIM}Routage : {lstr}{_C.RESET}")

        # Nombre total d'échantillons
        total_samples = sum(len(dq) for dq in self._samples.values())
        lines.append(
            f"  {_C.DIM}Échantillons : {total_samples} "
            f"(fenêtre max={self.window}/étape){_C.RESET}"
        )

        lines.append(
            f"{_C.CYAN}{_C.BOLD}"
            f"└────────────────────────────────────────────────────────────────────────┘"
            f"{_C.RESET}"
        )
        lines.append("")
        return "\n".join(lines)

    def _level_distribution(self) -> dict[int, int]:
        """Compte les occurrences de chaque niveau de routage."""
        dist: dict[int, int] = {}
        for dq in self._samples.values():
            for s in dq:
                if s.level > 0:
                    dist[s.level] = dist.get(s.level, 0) + 1
        return dist

    # ── Sauvegarde ──

    def save_log(self, path: str | Path | None = None) -> str:
        """
        Sauvegarde les métriques dans un fichier JSON.

        Args:
            path: Chemin du fichier. Par défaut data/logs/perf_YYYYMMDD_HHMMSS.json

        Returns:
            Le chemin absolu du fichier créé.
        """
        if path is None:
            logs_dir = Path(__file__).parent.parent / "data" / "logs"
        else:
            logs_dir = Path(path).parent
            if not logs_dir.exists():
                logs_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = logs_dir / f"perf_{ts}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ramasser toutes les mesures
        all_samples = []
        for stage, dq in self._samples.items():
            for s in dq:
                all_samples.append(asdict(s))

        data = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ram": self.current_ram_info(),
            "window": self.window,
            "samples": all_samples,
            "summary": {
                stage: self._stats_for_stage(stage)
                for stage in list(STAGE_NAMES.keys()) + ["ram"]
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(path.resolve())

    # ── API temps réel ──

    def get_realtime_stats(self) -> dict:
        """
        Retourne un dict des stats actuelles pour l'API web.

        Utilisé par l'endpoint /api/perf dans webui.py.
        """
        ram = self.current_ram_info()
        stages = {}
        for stage_key in STAGE_NAMES:
            s = self._stats_for_stage(stage_key)
            if s["count"] > 0:
                stages[stage_key] = s

        return {
            "ram": ram,
            "stages": stages,
            "routing_levels": self._level_distribution(),
            "total_samples": sum(len(dq) for dq in self._samples.values()),
            "window": self.window,
            "timestamp": time.time(),
        }


# ── Singleton partagé ──
_monitor_instance: Optional[Monitor] = None


def get_monitor(window: int = 100) -> Monitor:
    """Retourne l'instance singleton du moniteur."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = Monitor(window=window)
    return _monitor_instance
