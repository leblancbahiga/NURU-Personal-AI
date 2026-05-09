#!/usr/bin/env python3
"""
action_engine.py — Moteur d'actions locales avec confirmation explicite.

Actions supportées :
  - rename_file(src, dst)         : renommer un fichier
  - run_script(path)              : lancer un script Python
  - organize_directory(directory) : organiser les fichiers d'un dossier par type
  - open_file(path)               : ouvrir un fichier avec l'application par défaut

Patterns détectés via regex dans la requête utilisateur :
  'renomme X en Y'         → rename_file
  'lance/execute X'        → run_script
  'organise/range/classer dossier X' → organize_directory
  'ouvre X'                → open_file

Chaque action demande une confirmation explicite avant exécution.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Callable


class ActionEngine:
    """
    Moteur d'actions locales.

    Parse les intentions d'action depuis du texte (réponse LLM),
    demande confirmation à l'utilisateur, puis exécute l'action.
    """

    # ── Patterns de détection ──

    PATTERN_RENAME = re.compile(
        r'(?:renomme|renommer)\s+(?P<src>.+?)\s+(?:en|→|->|=>)\s+(?P<dst>.+)',
        re.IGNORECASE
    )

    PATTERN_RUN = re.compile(
        r'(?:lance|lancer|exécute|execute|exécuter|run)\s+(?:le\s+)?(?:script\s+)?(?P<path>.+)',
        re.IGNORECASE
    )

    PATTERN_ORGANIZE = re.compile(
        r'(?:organise|organiser|range|ranger|classer)\s+(?:le\s+)?(?:dossier\s+)?(?P<dir>.+)',
        re.IGNORECASE
    )

    PATTERN_OPEN = re.compile(
        r'(?:ouvre|ouvrir|open)\s+(?:le\s+)?(?:fichier\s+)?(?P<path>.+)',
        re.IGNORECASE
    )

    def __init__(self, confirm_callback: Optional[Callable] = None):
        """
        Args:
            confirm_callback: Fonction optionnelle (action_name, params) -> bool
                              pour la confirmation. Si None, utilise input().
        """
        self.confirm_callback = confirm_callback

    # ── Parsing ──

    def parse(self, text: str) -> Optional[tuple[str, dict]]:
        """
        Parse un texte et détecte une intention d'action.

        Args:
            text: Texte à analyser (réponse du LLM)

        Returns:
            (action_name, params) ou None si aucune action détectée
        """
        text_clean = text.strip()

        # Renommer : "renomme X en Y"
        m = self.PATTERN_RENAME.search(text_clean)
        if m:
            src = m.group("src").strip().strip('"\'')
            dst = m.group("dst").strip().strip('"\'')
            return ("rename_file", {"src": src, "dst": dst})

        # Lancer / Exécuter : "lance script.py", "execute X"
        m = self.PATTERN_RUN.search(text_clean)
        if m:
            path = m.group("path").strip().strip('"\'')
            return ("run_script", {"path": path})

        # Organiser / Ranger / Classer : "organise dossier X"
        m = self.PATTERN_ORGANIZE.search(text_clean)
        if m:
            directory = m.group("dir").strip().strip('"\'')
            return ("organize_directory", {"directory": directory})

        # Ouvrir : "ouvre X"
        m = self.PATTERN_OPEN.search(text_clean)
        if m:
            path = m.group("path").strip().strip('"\'')
            return ("open_file", {"path": path})

        return None

    # ── Actions ──

    def rename_file(self, src: str, dst: str) -> dict:
        """
        Renomme un fichier.

        Args:
            src: Chemin source (absolu ou relatif)
            dst: Nouveau nom ou chemin

        Returns:
            dict avec status, message, éventuels détails
        """
        src_path = Path(src).expanduser().resolve()
        dst_path = Path(dst).expanduser()

        if not dst_path.is_absolute():
            dst_path = src_path.parent / dst

        if not src_path.exists():
            return {"status": "error", "message": f"Fichier source introuvable : {src_path}"}

        if dst_path.exists():
            return {"status": "error", "message": f"La destination existe déjà : {dst_path}"}

        try:
            src_path.rename(dst_path)
            return {
                "status": "success",
                "message": f"Fichier renommé :\n  {src_path} →\n  {dst_path}",
                "source": str(src_path),
                "destination": str(dst_path),
            }
        except Exception as e:
            return {"status": "error", "message": f"Erreur lors du renommage : {e}"}

    def run_script(self, path: str) -> dict:
        """
        Lance un script Python.

        Args:
            path: Chemin du script

        Returns:
            dict avec status, stdout, stderr
        """
        script_path = Path(path).expanduser().resolve()

        if not script_path.exists():
            return {"status": "error", "message": f"Script introuvable : {script_path}"}

        if script_path.suffix != ".py":
            return {
                "status": "error",
                "message": f"Seuls les scripts Python (.py) sont supportés : {script_path}",
            }

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "message": f"Script terminé (code: {result.returncode})",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Le script a dépassé le temps limite (300s)"}
        except Exception as e:
            return {"status": "error", "message": f"Erreur lors de l'exécution : {e}"}

    def organize_directory(self, directory: str) -> dict:
        """
        Organise les fichiers d'un dossier par type.

        Crée des sous-dossiers : Images/, Documents/, Data/, Autres/
        et déplace les fichiers selon leur extension.

        Args:
            directory: Chemin du dossier à organiser

        Returns:
            dict avec status, détails des déplacements
        """
        dir_path = Path(directory).expanduser().resolve()

        if not dir_path.exists() or not dir_path.is_dir():
            return {"status": "error", "message": f"Dossier introuvable : {dir_path}"}

        # Catégories et extensions associées
        categories = {
            "Images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico", ".tiff", ".tif"},
            "Documents": {".pdf", ".docx", ".doc", ".txt", ".md", ".odt", ".rtf", ".tex", ".pages"},
            "Data": {".csv", ".json", ".xml", ".yaml", ".yml", ".xlsx", ".xls", ".parquet", ".feather", ".arrow"},
        }

        moved: dict[str, list[str]] = {cat: [] for cat in categories}
        moved["Autres"] = []
        errors = []

        for item in sorted(dir_path.iterdir()):
            if item.is_dir():
                continue  # On ne déplace pas les sous-dossiers

            ext = item.suffix.lower()
            target_category = None

            for cat, extensions in categories.items():
                if ext in extensions:
                    target_category = cat
                    break

            if target_category is None:
                target_category = "Autres"

            target_dir = dir_path / target_category
            target_dir.mkdir(exist_ok=True)

            dest = target_dir / item.name
            try:
                shutil.move(str(item), str(dest))
                moved[target_category].append(item.name)
            except Exception as e:
                errors.append(f"{item.name}: {e}")

        # Construction du message
        summary_parts = []
        total_moved = 0
        for cat, files in moved.items():
            if files:
                summary_parts.append(f"  {cat}/ ({len(files)} fichier(s))")
                total_moved += len(files)

        message = f"Dossier organisé : {total_moved} fichier(s) déplacé(s)\n" + "\n".join(summary_parts)
        if errors:
            message += f"\n\nErreurs ({len(errors)}) :\n" + "\n".join(f"  {e}" for e in errors)

        return {
            "status": "success" if not errors else "partial",
            "message": message,
            "moved": {cat: files for cat, files in moved.items() if files},
            "errors": errors,
        }

    def open_file(self, path: str) -> dict:
        """
        Ouvre un fichier avec l'application par défaut du système.

        Args:
            path: Chemin du fichier

        Returns:
            dict avec status et message
        """
        file_path = Path(path).expanduser().resolve()

        if not file_path.exists():
            return {"status": "error", "message": f"Fichier introuvable : {file_path}"}

        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(file_path)], check=True)
            elif sys.platform == "win32":  # Windows
                os.startfile(str(file_path))
            else:  # Linux
                subprocess.run(["xdg-open", str(file_path)], check=True)

            return {"status": "success", "message": f"Fichier ouvert : {file_path}"}
        except Exception as e:
            return {"status": "error", "message": f"Erreur à l'ouverture : {e}"}

    # ── Orchestration ──

    def parse_and_execute(self, text: str, auto_confirm: bool = False) -> Optional[dict]:
        """
        Parse un texte, demande confirmation si nécessaire, et exécute l'action.

        Args:
            text: Texte à analyser (réponse du LLM)
            auto_confirm: Si True, ignore la confirmation (usage interne/test)

        Returns:
            Résultat de l'action (dict), ou None si pas d'action détectée
        """
        parsed = self.parse(text)
        if parsed is None:
            return None

        action_name, params = parsed

        # Demander confirmation (sauf auto_confirm)
        if not auto_confirm:
            confirmed = self._confirm_action(action_name, params)
            if not confirmed:
                return {
                    "status": "cancelled",
                    "action": action_name,
                    "params": params,
                    "message": "Action annulée par l'utilisateur.",
                }

        # Exécuter l'action
        action_method = getattr(self, action_name, None)
        if action_method is None:
            return {"status": "error", "message": f"Action inconnue : {action_name}"}

        print(f"\n  ⚙ Exécution de {action_name}...", file=sys.stderr, flush=True)
        result = action_method(**params)
        return result

    def _confirm_action(self, action_name: str, params: dict) -> bool:
        """
        Demande confirmation à l'utilisateur avant d'exécuter une action.

        Args:
            action_name: Nom de l'action
            params: Paramètres de l'action

        Returns:
            True si l'utilisateur confirme, False sinon
        """
        if self.confirm_callback is not None:
            return self.confirm_callback(action_name, params)

        # Confirmation interactive dans le terminal
        action_labels = {
            "rename_file": "Renommer un fichier",
            "run_script": "Lancer un script Python",
            "organize_directory": "Organiser un dossier",
            "open_file": "Ouvrir un fichier",
        }

        label = action_labels.get(action_name, action_name)
        params_str = "\n  ".join(f"{k}: {v}" for k, v in params.items())

        print(f"\n    🔧 Action détectée : {label}")
        print(f"    {params_str}")

        try:
            response = input("\n    Confirmer l'exécution ? [O/n] ").strip().lower()
            return response in ("", "o", "oui", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            print()
            return False


# ── Test / Demo ──
if __name__ == "__main__":
    engine = ActionEngine()

    test_texts = [
        "renomme mon_fichier.txt en nouveau_nom.txt",
        "lance script.py",
        "exécute mon_script.py",
        "organise dossier ~/Downloads",
        "range le dossier ~/Documents",
        "classer dossier /tmp/tests",
        "ouvre ~/mon_fichier.pdf",
        "ouvrir le fichier notes.md",
    ]

    print("=== Test ActionEngine (parsing) ===\n")
    for text in test_texts:
        result = engine.parse(text)
        if result:
            print(f"  ✓ '{text}'")
            print(f"    → {result[0]}({result[1]})")
        else:
            print(f"  ✗ '{text}' → non détecté")
