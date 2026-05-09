#!/usr/bin/env python3
"""
resource_manager_v2.py — Gestionnaire RAM + thermique pour NURU V2.

Fonctionnalités :
  - Monitoring RAM (psutil, toutes les 5s)
  - Éviction agressive si RAM < 1.5 Go libre
  - Détection power mode (AC/battery via pmset)
  - Configuration MLX threads adaptative
  - Thread dédié avec start()/stop()
  - Callback optionnel on_ram_critical
  - get_stats() → dict
  - force_clear() statique : gc.collect() + mx.clear_cache()

Usage :
    mgr = ResourceManagerV2()
    mgr.start()
    # ... utilisation normale ...
    print(mgr.get_stats())
    mgr.stop()
"""

import os
import sys
import gc
import time
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger("nuru.v2.resource_manager")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import subprocess
    PMSET_AVAILABLE = True
except ImportError:
    PMSET_AVAILABLE = False


# ── Constantes ──
RAM_CRITICAL_THRESHOLD = 0.8  # Go — seuil d'éviction agressive (abaissé de 1.5)
RAM_WARNING_THRESHOLD = 1.5   # Go — seuil d'avertissement (abaissé de 2.5)

# Niveaux de pression RAM
PRESSURE_NONE = 0
PRESSURE_LOW = 1
PRESSURE_MEDIUM = 2
PRESSURE_HIGH = 3
PRESSURE_CRITICAL = 4

# Seuils pour les niveaux de pression (Go libres)
PRESSURE_NONE_THRESHOLD = 3.0         # > 3 Go libre → aucune pression
PRESSURE_LOW_THRESHOLD = RAM_WARNING_THRESHOLD    # ≤ 3 Go, > 1.5 Go → faible
PRESSURE_MEDIUM_THRESHOLD = RAM_CRITICAL_THRESHOLD # ≤ 1.5 Go, > 0.8 Go → moyenne
PRESSURE_HIGH_THRESHOLD = 0.4         # ≤ 0.8 Go, > 0.4 Go → élevée
                                       # ≤ 0.4 Go → critique

PRESSURE_LABELS = {
    PRESSURE_NONE: "none",
    PRESSURE_LOW: "low",
    PRESSURE_MEDIUM: "medium",
    PRESSURE_HIGH: "high",
    PRESSURE_CRITICAL: "critical",
}

MONITOR_INTERVAL = 10.0       # secondes entre chaque check (augmenté de 5s)
MLX_THREADS_AC = 6            # threads MLX sur secteur
MLX_THREADS_BATTERY = 4       # threads MLX sur batterie
MLX_THREADS_LOW_BATTERY = 2   # threads MLX si batterie < 20%
EVICTION_COOLDOWN = 60.0      # secondes minimum entre deux évictions


class ResourceManagerV2:
    """
    Gestionnaire de ressources système pour NURU.

    Lance un thread dédié qui :
      1. Surveille la RAM libre toutes les 5s
      2. Évince agressivement si < 1.5 Go libre
      3. Détecte le mode d'alimentation (AC/battery)
      4. Ajuste les threads MLX dynamiquement
    """

    def __init__(
        self,
        ram_critical_threshold: float = RAM_CRITICAL_THRESHOLD,
        check_interval: float = MONITOR_INTERVAL,
        on_ram_critical: Optional[Callable] = None,
    ):
        self._ram_critical_threshold = ram_critical_threshold
        self._check_interval = check_interval
        self._on_ram_critical = on_ram_critical

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Statistiques
        self._check_count: int = 0
        self._evictions = 0
        self._thread_adjustments = 0
        self._last_eviction_time = 0.0
        self._check_count = 0
        self._power_mode: str = "unknown"
        self._mlx_threads_current: int = self._get_default_threads()
        self._last_ram_gb: float = 0.0
        self._last_ram_percent: float = 0.0
        self._battery_percent: Optional[float] = None
        self._running: bool = False

        # Appliquer la configuration initiale des threads
        self._apply_mlx_threads()

    # ── Détection du mode d'alimentation ──

    @staticmethod
    def _detect_power_mode() -> str:
        """
        Détecte le mode d'alimentation via pmset (macOS).

        Retourne :
            "ac" : sur secteur
            "battery" : sur batterie
            "unknown" : impossible de détecter
        """
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            output = result.stdout + result.stderr
            if "AC Power" in output or "charg" in output.lower():
                return "ac"
            elif "Battery Power" in output or "battery" in output.lower():
                return "battery"
            return "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return "unknown"

    @staticmethod
    def _get_battery_percent() -> Optional[float]:
        """Retourne le pourcentage de batterie, ou None si sur secteur."""
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            import re
            match = re.search(r'(\d+)%', result.stdout)
            if match:
                return float(match.group(1))
            return None
        except Exception:
            return None

    @staticmethod
    def _get_default_threads() -> int:
        """Retourne le nombre par défaut de threads MLX."""
        try:
            import os
            return int(os.environ.get("MLX_NUM_THREADS", str(os.cpu_count() or 4)))
        except Exception:
            return 4

    # ── Gestion des threads MLX ──

    @staticmethod
    def _apply_mlx_threads(n_threads: Optional[int] = None) -> None:
        """
        Configure le nombre de threads pour MLX.

        MLX utilise OMP_NUM_THREADS pour le parallélisme CPU.
        """
        if not MLX_AVAILABLE:
            return
        try:
            threads = n_threads if n_threads is not None else MLX_THREADS_AC
            os.environ["MLX_NUM_THREADS"] = str(threads)
            # Certaines opérations MLX utilisent OMP
            os.environ["OMP_NUM_THREADS"] = str(threads)
            logger.debug("Threads MLX configurés : %d", threads)
        except Exception as e:
            logger.warning("Impossible de configurer les threads MLX : %s", e)

    # ── Nettoyage forcé ──

    @staticmethod
    def force_clear() -> None:
        """Libère un maximum de mémoire : GC + caches MLX."""
        gc.collect()
        if MLX_AVAILABLE:
            try:
                mx.clear_cache()
                logger.debug("Cache MLX vidé")
            except Exception:
                pass
        logger.info("Nettoyage mémoire forcé effectué")

    # ── Éviction ──

    def _evict_if_critical(self, available_gb: float) -> bool:
        """
        Évince les ressources si la RAM disponible est critique.
        Ajoute un délai de 'cooloff' pour éviter le thrashing.
        """
        now = time.time()
        if available_gb < self._ram_critical_threshold:
            if now - self._last_eviction_time < EVICTION_COOLDOWN:
                logger.debug("RAM critique mais cooloff d'éviction actif (%.1fs restants)", 
                             EVICTION_COOLDOWN - (now - self._last_eviction_time))
                return False

            logger.warning(
                "RAM critique : %.2f Go libre (seuil: %.1f Go) — éviction agressive",
                available_gb, self._ram_critical_threshold,
            )
            self.force_clear()
            with self._lock:
                self._evictions += 1
                self._last_eviction_time = now
            
            if self._on_ram_critical:
                try:
                    self._on_ram_critical(available_gb)
                except Exception as e:
                    logger.error("Erreur dans le callback on_ram_critical : %s", e)
            return True
        return False

    # ── Boucle de monitoring ──

    def _monitor_loop(self) -> None:
        """Boucle principale du thread de monitoring."""
        logger.info("Monitoring des ressources démarré (intervalle: %.1fs)", self._check_interval)

        while not self._stop_event.is_set():
            try:
                self._check_resources()
            except Exception as e:
                logger.error("Erreur dans le monitoring : %s", e)

            self._stop_event.wait(self._check_interval)

        logger.info("Monitoring des ressources arrêté")

    def _check_resources(self) -> None:
        """Vérifie l'état des ressources et prend les actions nécessaires."""
        with self._lock:
            self._check_count += 1

        # 1. RAM
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            percent = mem.percent

            with self._lock:
                self._last_ram_gb = round(available_gb, 2)
                self._last_ram_percent = round(percent, 1)

            # Alerte RAM si < seuil d'avertissement
            if available_gb < RAM_WARNING_THRESHOLD:
                logger.info("RAM : %.2f Go libre (%.1f%%)", available_gb, percent)

            # Éviction critique
            self._evict_if_critical(available_gb)

        # 2. Mode d'alimentation et threads MLX
        power_mode = self._detect_power_mode()
        battery_pct = self._get_battery_percent()

        with self._lock:
            self._power_mode = power_mode
            self._battery_percent = battery_pct

        # 3. Ajustement des threads MLX
        if MLX_AVAILABLE:
            if power_mode == "ac":
                target_threads = MLX_THREADS_AC
            elif battery_pct is not None and battery_pct < 20:
                target_threads = MLX_THREADS_LOW_BATTERY
            else:
                target_threads = MLX_THREADS_BATTERY

            with self._lock:
                if target_threads != self._mlx_threads_current:
                    self._mlx_threads_current = target_threads
                    self._thread_adjustments += 1
                    logger.info(
                        "Ajustement threads MLX : %d → %d (mode: %s, batterie: %s%%)",
                        self._mlx_threads_current, target_threads, power_mode,
                        f"{battery_pct:.0f}" if battery_pct is not None else "N/A",
                    )
                    self._apply_mlx_threads(target_threads)

    # ── API publique ──

    def start(self) -> None:
        """Démarre le thread de monitoring."""
        if self._running:
            logger.warning("ResourceManager déjà en cours d'exécution")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="nuru-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info("ResourceManagerV2 démarré")

    def stop(self) -> None:
        """Arrête proprement le thread de monitoring."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._running = False
        logger.info("ResourceManagerV2 arrêté")

    def get_stats(self) -> dict:
        """Retourne les statistiques et l'état actuel."""
        with self._lock:
            return {
                "running": self._running,
                "power_mode": self._power_mode,
                "battery_percent": self._battery_percent,
                "ram_available_gb": self._last_ram_gb if PSUTIL_AVAILABLE else None,
                "ram_percent": self._last_ram_percent if PSUTIL_AVAILABLE else None,
                "mlx_threads": self._mlx_threads_current,
                "check_count": self._check_count,
                "evictions": self._evictions,
                "thread_adjustments": self._thread_adjustments,
                "ram_critical_threshold_gb": self._ram_critical_threshold,
                "psutil_available": PSUTIL_AVAILABLE,
                "mlx_available": MLX_AVAILABLE,
                "pressure_level": self.get_pressure_level(),
            }

    def get_pressure_level(self) -> dict:
        """
        Calcule et retourne le niveau de pression RAM actuel.

        Utilise psutil si disponible, sinon se base sur la dernière mesure
        enregistrée par le thread de monitoring.

        Niveaux :
            - none     (0) : RAM libre > PRESSURE_NONE_THRESHOLD (4 Go)
            - low      (1) : PRESSURE_LOW_THRESHOLD < RAM ≤ 4 Go
            - medium   (2) : PRESSURE_MEDIUM_THRESHOLD < RAM ≤ 2.5 Go
            - high     (3) : PRESSURE_HIGH_THRESHOLD < RAM ≤ 1.5 Go
            - critical (4) : RAM ≤ 0.5 Go

        Retourne un dict :
            level: int (0-4)
            label: str ("none", "low", "medium", "high", "critical")
            ram_available_gb: float ou None
            ram_percent: float ou None
        """
        # Essayer une mesure live si psutil est dispo
        ram_gb = None
        ram_pct = None
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                ram_gb = mem.available / (1024**3)
                ram_pct = mem.percent
            except Exception:
                pass

        # Fallback sur les dernières valeurs du thread
        if ram_gb is None:
            ram_gb = self._last_ram_gb
            ram_pct = self._last_ram_percent

        # Déterminer le niveau de pression
        if ram_gb is None:
            level = PRESSURE_NONE
        elif ram_gb <= PRESSURE_HIGH_THRESHOLD:
            level = PRESSURE_CRITICAL
        elif ram_gb <= self._ram_critical_threshold:
            level = PRESSURE_HIGH
        elif ram_gb <= RAM_WARNING_THRESHOLD:
            level = PRESSURE_MEDIUM
        elif ram_gb <= PRESSURE_NONE_THRESHOLD:
            level = PRESSURE_LOW
        else:
            level = PRESSURE_NONE

        return {
            "level": level,
            "label": PRESSURE_LABELS.get(level, "unknown"),
            "ram_available_gb": round(ram_gb, 2) if ram_gb is not None else None,
            "ram_percent": round(ram_pct, 1) if ram_pct is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"ResourceManagerV2(running={self._running}, "
            f"RAM={self._last_ram_gb:.1f} Go libre, "
            f"mode={self._power_mode}, "
            f"threads={self._mlx_threads_current})"
        )


# ── Singleton partagé ──
_resource_manager_instance: Optional[ResourceManagerV2] = None
_resource_manager_lock = threading.Lock()


def get_resource_manager() -> ResourceManagerV2:
    """Retourne l'instance singleton du gestionnaire de ressources."""
    global _resource_manager_instance
    if _resource_manager_instance is None:
        with _resource_manager_lock:
            if _resource_manager_instance is None:
                _resource_manager_instance = ResourceManagerV2()
    return _resource_manager_instance
