#!/usr/bin/env python3
"""
indexer_daemon.py — Service d'indexation automatique pour NURU.

Scanne les dossiers configurés, détecte les fichiers nouveaux/modifiés,
les parse, les chunke, les vectorise et les stocke dans ChromaDB.

Fonctionne en arrière-plan avec un intervalle configurable.
Respecte les ressources système (batterie, mémoire).

Usage :
    python3 src/indexer_daemon.py            # Scan unique + démarrage daemon
    python3 src/indexer_daemon.py --once     # Scan unique puis arrêt
    python3 src/indexer_daemon.py --status   # Afficher l'état
"""

import hashlib
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Set, List

# Ajouter src/ au path pour les imports du projet
_src_dir = str(Path(__file__).parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

logger = logging.getLogger("nuru.indexer")


class IndexerDaemon:
    """
    Service d'indexation automatique de fichiers.

    Parcourt les dossiers configurés, détecte les fichiers nouveaux ou modifiés
    (via MD5), les parse, les chunke, les vectorise et les stocke dans ChromaDB.

    Attributes:
        config: Section 'indexing' de la config NURU
        rag_config: Section 'rag' de la config (héritage des extensions/chunk size)
        store: Instance de rag.VectorStore (initialisée à la demande)
    """

    def __init__(self, config: dict, vector_store=None):
        """
        Initialise l'indexeur.

        Args:
            config: Configuration NURU complète (dict)
            vector_store: Instance existante de VectorStore (optionnelle)
        """
        self.config = config.get("indexing", {})
        self.rag_config = config.get("rag", {})
        self._store = vector_store
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stats = {
            "last_scan": 0,
            "total_scanned": 0,
            "total_indexed": 0,
            "total_skipped": 0,
            "total_errors": 0,
            "running": False,
        }
        self._lock = threading.Lock()
        self._project_root = Path(__file__).parent.parent
        logger.info(
            "IndexerDaemon initialisé (intervalle=%d min, racines=%s)",
            self.config.get("interval_minutes", 30),
            self.config.get("scan_roots", ["~"]),
        )

    # ── Cycle de vie ──

    def start(self):
        """Démarre le thread d'indexation en arrière-plan."""
        if self._thread and self._thread.is_alive():
            logger.warning("Indexer déjà en cours d'exécution")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="nuru-indexer")
        self._thread.start()
        logger.info("Indexer démarré")

    def stop(self, wait: bool = True):
        """Arrête le thread d'indexation."""
        self._stop_event.set()
        if wait and self._thread:
            self._thread.join(timeout=10)
        logger.info("Indexer arrêté")

    def _run_loop(self):
        """
        Boucle principale.

        Exécute un premier scan immédiat, puis boucle sur l'intervalle
        configuré. L'arrêt est détecté via _stop_event.
        """
        interval = self.config.get("interval_minutes", 30) * 60
        with self._lock:
            self._stats["running"] = True

        # Premier scan immédiat
        self._scan_pass()

        while not self._stop_event.is_set():
            # Attendre l'intervalle (par pas de 1s pour réactivité)
            for _ in range(max(1, int(interval))):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

            if not self._stop_event.is_set():
                self._scan_pass()

        with self._lock:
            self._stats["running"] = False

    # ── Scan ──

    def scan_once(self) -> dict:
        """Déclenche un scan unique immédiat (thread-safe). Retourne les stats."""
        self._scan_pass()
        return self.get_stats()

    def _ensure_store(self) -> bool:
        """Initialise VectorStore si nécessaire. Retourne True si OK."""
        if self._store is not None:
            return True

        try:
            from rag import VectorStore, CHROMA_AVAILABLE

            if not CHROMA_AVAILABLE:
                logger.error("ChromaDB non installé — impossible d'indexer")
                return False

            chroma_path = self.rag_config.get("chroma_db_path", "data/chroma_db")
            if not os.path.isabs(chroma_path):
                chroma_path = str(self._project_root / chroma_path)

            self._store = VectorStore(persist_dir=chroma_path)
            return True
        except Exception as e:
            logger.error("Impossible d'initialiser VectorStore : %s", e)
            return False

    def _scan_pass(self):
        """Une passe de scan complète et silencieuse."""
        if not self._ensure_store():
            return

        extensions = self._get_extensions()
        scan_roots = self._get_scan_roots()
        exclude_set = self._get_exclude_set()
        exclude_hidden = self.config.get("exclude_hidden", True)
        max_bytes = self.config.get("max_file_size_mb", 50) * 1024 * 1024
        override_battery = self.config.get("override_battery_check", False)
        retry_on_low_battery = self.config.get("retry_on_low_battery", True)

        scanned = 0
        indexed = 0
        skipped = 0
        errors = 0
        low_battery_detected = False

        logger.info(
            "🔍 Scan démarré : %d racine(s), %d extension(s)",
            len(scan_roots), len(extensions),
        )

        for root in scan_roots:
            if self._stop_event.is_set():
                break
            try:
                for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                    # Filtrer les dossiers exclus sur place (évite de descendre)
                    dirnames[:] = [
                        d for d in dirnames
                        if d.lower() not in exclude_set
                        and not (exclude_hidden and d.startswith("."))
                    ]

                    if self._stop_event.is_set():
                        break

                    for filename in filenames:
                        if exclude_hidden and filename.startswith("."):
                            continue

                        ext = os.path.splitext(filename)[1].lower()
                        if ext not in extensions:
                            continue

                        filepath = os.path.join(dirpath, filename)

                        try:
                            # Vérifier la taille
                            size = os.path.getsize(filepath)
                            if size > max_bytes or size == 0:
                                skipped += 1
                                continue

                            scanned += 1

                            # Vérifier si déjà indexé (via MD5)
                            md5 = self._compute_md5(filepath)
                            if self._store.is_indexed(filepath, md5):
                                skipped += 1
                                continue

                            # Indexer le fichier
                            success = self._index_file(filepath, md5, override_battery)
                            if success:
                                indexed += 1
                            else:
                                # Si échec batterie et retry activé, on compte comme "low_battery"
                                if retry_on_low_battery and self._is_low_battery():
                                    low_battery_detected = True
                                    skipped += 1
                                else:
                                    errors += 1

                        except (PermissionError, OSError) as e:
                            errors += 1
                            logger.debug("Erreur accès %s : %s", filepath, e)
                            continue

            except PermissionError:
                logger.debug("Permission refusée : %s", root)
                continue

        # Mettre à jour les stats
        with self._lock:
            self._stats["last_scan"] = time.time()
            self._stats["total_scanned"] += scanned
            self._stats["total_indexed"] += indexed
            self._stats["total_skipped"] += skipped
            self._stats["total_errors"] += errors

        total = (
            self._store.count_documents()
            if self._store
            else 0
        )

        if low_battery_detected:
            logger.info(
                "⚠ Batterie faible détectée — %d fichiers reportés au prochain cycle",
                scanned - indexed - skipped + errors,
            )

        logger.info(
            "✅ Scan terminé : %d scannés, %d indexés, %d ignorés, "
            "%d erreurs | ChromaDB : %d chunks",
            scanned, indexed, skipped, errors, total,
        )

    # ── Indexation d'un fichier ──

    def _index_file(self, filepath: str, md5: str, override_battery: bool = False) -> bool:
        """
        Parse, chunke, vectorise et stocke un fichier dans ChromaDB.

        Args:
            filepath: Chemin absolu du fichier
            md5: Empreinte MD5 déjà calculée
            override_battery: Ignorer la vérification batterie (défaut: False)

        Retourne:
            True si l'indexation a réussi
        """
        try:
            from ingestion import index_document

            chunk_size = self.rag_config.get("chunk_size", 512)
            overlap = self.rag_config.get("chunk_overlap", 64)

            doc = index_document(filepath, chunk_size=chunk_size, overlap=overlap)
            if doc is None:
                return False

            # Ajouter à ChromaDB
            n_chunks = self._store.add_document_chunks(
                filepath=filepath,
                filename=doc.filename,
                file_hash=md5,
                mime_type=doc.mime_type,
                chunks=doc.chunks,
            )

            # Marquer comme indexé
            self._store.mark_indexed(filepath, md5)

            logger.debug("  ✅ %s → %d chunks", doc.filename, n_chunks)
            return True

        except Exception as e:
            logger.warning("  ⚠ Échec indexation %s : %s", filepath, e)
            return False

    @staticmethod
    def _is_low_battery() -> bool:
        """Vérifie si la batterie est en dessous de 20% (macOS)."""
        try:
            import subprocess
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True, text=True, timeout=5
            )
            if "Battery Power" in result.stdout:
                for line in result.stdout.split("\n"):
                    if "%" in line:
                        try:
                            pct = int(line.split("\t")[-1].split("%")[0])
                            return pct < 20
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass
        return False

    # ── Utilitaires de configuration ──

    @staticmethod
    def _compute_md5(filepath: str) -> str:
        """Calcule l'empreinte MD5 d'un fichier."""
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_extensions(self) -> Set[str]:
        """
        Récupère les extensions à indexer.

        Priorité : indexing.extensions > rag.extensions > défaut.
        """
        exts = self.config.get("extensions", [])
        if not exts:
            exts = self.rag_config.get(
                "extensions", [".pdf", ".docx", ".txt", ".md", ".csv"]
            )
        return {
            ext.lower() if ext.startswith(".") else "." + ext.lower()
            for ext in exts
        }

    def _get_scan_roots(self) -> List[str]:
        """Récupère les racines à scanner, avec expansion de ~."""
        roots = self.config.get("scan_roots", ["~"])
        return [os.path.expanduser(root) for root in roots]

    def _get_exclude_set(self) -> Set[str]:
        """Récupère l'ensemble des noms de dossiers exclus (lowercase)."""
        excludes = self.config.get("exclude_dirs", [])
        return {name.lower() for name in excludes}

    # ── Stats et status ──

    def get_stats(self) -> dict:
        """Retourne les statistiques courantes (thread-safe)."""
        with self._lock:
            stats = dict(self._stats)
        if self._store:
            try:
                stats["chroma_documents"] = self._store.count_documents()
                stats["chroma_corrections"] = self._store.count_corrections()
            except Exception:
                pass
        return stats

    def get_status(self) -> str:
        """Retourne un résumé lisible de l'état."""
        stats = self.get_stats()
        if not stats.get("running"):
            status = "⏹ Arrêté"
        elif stats.get("last_scan", 0) == 0:
            status = "🔄 Premier scan en cours…"
        else:
            last = time.strftime(
                "%H:%M:%S", time.localtime(stats.get("last_scan", 0))
            )
            status = f"🟢 Actif (dernier scan : {last})"

        return (
            f"{status}\n"
            f"  Fichiers scannés : {stats.get('total_scanned', 0)}\n"
            f"  Fichiers indexés : {stats.get('total_indexed', 0)}\n"
            f"  Ignorés          : {stats.get('total_skipped', 0)}\n"
            f"  Erreurs          : {stats.get('total_errors', 0)}\n"
            f"  ChromaDB chunks  : {stats.get('chroma_documents', '?')}\n"
            f"  Intervalle       : {self.config.get('interval_minutes', 30)} min"
        )

    def clear_index(self) -> dict:
        """Vide entièrement la base vectorielle et le cache hash."""
        if not self._ensure_store():
            return {"error": "VectorStore non disponible"}
        try:
            self._store.clear()
            logger.info("🧹 Base vectorielle vidée")
        except Exception as e:
            logger.error("Erreur vidage base : %s", e)
        return self.get_stats()


# ── CLI ──

def _load_config() -> dict:
    """Charge la configuration NURU depuis config/config.yaml."""
    import yaml

    config_path = (
        Path(__file__).parent.parent / "config" / "config.yaml"
    )
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    logger.warning("Fichier de config introuvable : %s", config_path)
    return {}


def _setup_logging(level: str = "INFO"):
    """Configure le logging pour le mode CLI."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Indexeur automatique NURU")
    parser.add_argument("--once", action="store_true", help="Scan unique puis arrêt")
    parser.add_argument("--status", action="store_true", help="Afficher l'état")
    parser.add_argument("--clear", action="store_true", help="Vider l'index")
    parser.add_argument("--log-level", default="INFO", help="Niveau de log (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    config = _load_config()
    if not config:
        print("❌ Impossible de charger la configuration")
        sys.exit(1)

    indexer = IndexerDaemon(config)

    if args.status:
        print(indexer.get_status())
        sys.exit(0)

    if args.clear:
        result = indexer.clear_index()
        print("🧹 Base vidée.")
        print(indexer.get_status())
        sys.exit(0)

    if args.once:
        print("🔍 Scan unique en cours…")
        stats = indexer.scan_once()
        print(indexer.get_status())
        sys.exit(0)

    # Mode daemon : scan puis boucle
    print("🔄 Indexeur NURU démarré (Ctrl+C pour arrêter)")
    indexer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹ Arrêt…")
        indexer.stop()
        print(indexer.get_status())
