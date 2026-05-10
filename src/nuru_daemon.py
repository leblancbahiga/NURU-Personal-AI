#!/usr/bin/env python3
"""
nuru_daemon.py — Orchestrateur + Barre de menus pour NURU.

Machine à états :
  [REPOS] → (Cmd+Espace enfoncé) → [ÉCOUTE]
  → (touche relâchée) → [TRANSCRIPTION] → [ROUTAGE]
  → [GÉNÉRATION] → [TTS] → [REPOS]

Icône barre de menus :
  Gris  = Repos
  Rouge = Écoute
  Bleu  = Réflexion
  Vert  = Parle

Usage :
    python3 src/nuru_daemon.py           # Lancer le daemon complet
    python3 src/nuru_daemon.py --test     # Test de l'interface menu
    python3 src/nuru_daemon.py --state    # Afficher l'état actuel
"""

import sys
import time
import threading
import tempfile
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from audio_tts import speak, speak_streaming, stop as stop_tts
from audio_stt import load_stt_model, transcribe_file, transcribe_microphone, unload_model as unload_stt
from router import Router
from memory import SessionMemory

try:
    from indexer_daemon import IndexerDaemon
    INDEXER_AVAILABLE = True
except ImportError:
    IndexerDaemon = None
    INDEXER_AVAILABLE = False
    logger = logging.getLogger("nuru.daemon")
    logger.warning("indexer_daemon non disponible — indexation automatique désactivée")

try:
    import rumps
except ImportError:
    rumps = None

try:
    import yaml
except ImportError:
    yaml = None


# ── États ──
class State(Enum):
    IDLE = "REPOS"
    LISTENING = "ÉCOUTE"
    TRANSCRIBING = "TRANSCRIPTION"
    ROUTING = "ROUTAGE"
    GENERATING = "GÉNÉRATION"
    SPEAKING = "PAROLE"


STATE_ICONS = {
    State.IDLE: "⚪",        # Gris
    State.LISTENING: "🔴",   # Rouge
    State.TRANSCRIBING: "🔵", # Bleu
    State.ROUTING: "🔵",     # Bleu
    State.GENERATING: "🔵",  # Bleu
    State.SPEAKING: "🟢",    # Vert
}

STATE_COLORS = {
    State.IDLE: "gray",
    State.LISTENING: "red",
    State.TRANSCRIBING: "blue",
    State.ROUTING: "blue",
    State.GENERATING: "blue",
    State.SPEAKING: "green",
}


class NuruDaemon:
    """
    Orchestrateur principal de NURU.

    Gère :
    - La machine à états (Idle → Listening → Transcribing → Routing → Generating → Speaking)
    - Le cycle de vie des modèles (STT chargé/déchargé)
    - La coordination STT → Router → TTS
    - L'icône dans la barre de menus
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.state = State.IDLE
        self.router: Optional[Router] = None
        self.memory = SessionMemory()
        self._stt_loaded = False
        self._shortcut_active = False
        self._current_recording: Optional[str] = None
        self.total_interactions = 0
        self.total_errors = 0

        # Initialisation de l'indexeur si disponible
        self.indexer: Optional[IndexerDaemon] = None
        if INDEXER_AVAILABLE and IndexerDaemon is not None:
            try:
                self.indexer = IndexerDaemon(self.config)
                indexing_cfg = self.config.get("indexing", {})
                if indexing_cfg.get("enabled", True):
                    print("📂 Indexeur NURU initialisé (démarrage auto si activé)")
                    # Scan automatique au démarrage si configuré
                    if indexing_cfg.get("auto_scan_on_startup", True):
                        print("🔍 Lancement d'un scan automatique au démarrage...")
                        # On lance ça en thread pour ne pas bloquer le daemon
                        threading.Thread(target=self._auto_scan_startup, daemon=True).start()
            except Exception as e:
                logger.warning(f"Échec initialisation indexeur : {e}")

        # Chargement de la config audio
        self._tts_rate = self.config.get("audio", {}).get("tts_speed", 200)
        self._stt_model = self.config.get("audio", {}).get("stt_model", "tiny")
        self._language = "fr"

        print(f"🟢 NURU Daemon prêt — mode {self.state.value}")

    def _auto_scan_startup(self):
        """Lance un scan automatique au démarrage (thread dédié)."""
        if self.indexer is None:
            return
        try:
            self.indexer.scan_once()
        except Exception as e:
            logger.warning(f"Échec scan auto démarrage : {e}")

    def _load_config(self, config_path: Optional[str]) -> dict:
        if yaml is None or config_path is None:
            return {}
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    # ── Changement d'état ──

    def _set_state(self, new_state: State):
        old_state = self.state
        self.state = new_state
        icon = STATE_ICONS.get(new_state, "⚪")
        print(f"  {icon} État : {old_state.value} → {new_state.value}")

    # ── Cycle de vie d'une interaction ──

    def process_interaction(self, audio_path: Optional[str] = None,
                            text_input: Optional[str] = None) -> Optional[str]:
        """
        Exécute un cycle complet : STT → Router → TTS

        Args:
            audio_path: Fichier audio pré-enregistré (ou None pour micro)
            text_input: Texte direct (skip STT, pour debug/test)

        Retourne:
            La réponse texte générée
        """
        self.total_interactions += 1

        # ── Initialisation du routeur avec cloud par défaut ──
        if self.router is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            self.router = Router(config_path=str(config_path), memory=self.memory)
            self.router.set_force_cloud(True)  # Cloud par défaut !

        # ── Transcription ──
        if text_input:
            query = text_input
            print(f"  📝 Texte direct : {query[:60]}...")
        else:
            self._set_state(State.LISTENING)
            if audio_path:
                query = transcribe_file(audio_path, self._language)
            else:
                query = transcribe_microphone(
                    duration=15,
                    language=self._language,
                    model_size=self._stt_model,
                )

            if not query or not query.strip():
                self._set_state(State.IDLE)
                return None

            self._set_state(State.TRANSCRIBING)
            # Libérer la RAM du modèle STT après transcription
            unload_stt()

        # ── Détection de feedback / correction ──
        feedback_query = query.strip().lower()
        import re
        correction_text = None

        # Patterns : "Corrige : ...", "Non, la bonne réponse est ...", "Rectifie : ..."
        corr_match = re.search(
            r"(?:corrige|rectifie|correction)\s*:\s*(.+)",
            feedback_query, re.IGNORECASE
        )
        if not corr_match:
            corr_match = re.search(
                r"non,\s*(?:la bonne réponse est|la réponse est|c'est)\s*(.+)",
                feedback_query, re.IGNORECASE
            )

        if corr_match:
            correction_text = corr_match.group(1).strip()
            # Enregistrer la correction avec le dernier échange comme déclencheur
            last_exchanges = self.memory.get_exchanges()
            if last_exchanges:
                last_question = last_exchanges[-1].user
                from feedback import FeedbackManager
                fb = FeedbackManager()
                fb.add_correction(last_question, correction_text)
                reply = f"✅ Correction enregistrée ! Pour '{last_question[:40]}...', je répondrai désormais : {correction_text}"
            else:
                reply = "ℹ️ Correction notée, mais aucun échange précédent à associer."

            self.memory.add(query, reply)
            self._set_state(State.SPEAKING)
            speak_streaming(reply, rate=self._tts_rate)
            self._set_state(State.IDLE)
            return reply

        # ── Routage et génération ──
        self._set_state(State.GENERATING)

        result = self.router.route(query, user_confirmed_cloud=self.router.force_cloud)

        # Gérer la confirmation Cloud
        if result.level == 3 and "CLOUD_NEEDS_CONFIRM:" in result.content:
            print(f"  ⚠ Escalade Cloud nécessaire (mode daemon = fallback local)")
            # En mode daemon, on force local via le routeur
            response_text, model_used = self.router._execute_local(query)
            result.content = response_text

        response_text = result.content
        print(f"  💬 NURU → {response_text[:80]}...")

        # ── Synthèse vocale ──
        if response_text and not response_text.startswith("("):
            self._set_state(State.SPEAKING)
            speak_streaming(response_text, rate=self._tts_rate)

        # Mémoire
        self.memory.add(query, response_text)

        self._set_state(State.IDLE)
        return response_text

    # ── Raccourci clavier (via AppleScript) ──

    def register_shortcut(self):
        """Enregistre le raccourci clavier via AppleScript."""
        # Note : les raccourcis globaux macOS nécessitent soit :
        # 1. Hammerspoon (recommandé)
        # 2. Une app Sandbox avec entitlement
        # 3. AppleScript + Automator
        # Ici on génère un script Automator pour test
        print("⌨️ Pour configurer le raccourci clavier global :")
        print("   1. Ouvrir Automator → Service")
        print(f"   2. Ajouter 'Lancer NURU' → coller le chemin du daemon")
        print(f"   3. Assigner Cmd+Shift+Espace dans Préf. Système → Clavier → Raccourcis")
        print()

    # ── Initialisation du routeur ──

    def _init_router(self):
        """Initialise le routeur avec configuration par défaut."""
        if self.router is not None:
            return
        config_path = str(Path(__file__).parent.parent / "config" / "config.yaml")
        self.router = Router(config_path=config_path, memory=self.memory)
        self.router.set_force_cloud(True)
        print("  🌐 Routeur initialisé (mode Cloud par défaut)")

    def _alert_no_router(self, action_name: str):
        """Affiche une alerte si le routeur n'est pas prêt."""
        rumps.alert(
            title="NURU — Routeur non prêt",
            message=f"Impossible de '{action_name}' : le routeur n'est pas initialisé."
        )

    # ── Menu bar (rumps) ──

    def _build_menu(self) -> list:
        """Construit le menu de la barre de menus."""
        self._state_item = rumps.MenuItem(f"État : {self.state.value}")
        self._interaction_item = rumps.MenuItem(f"Interactions : {self.total_interactions}")
        return [
            self._state_item,
            self._interaction_item,
            None,
            rumps.MenuItem("Mode Avion", callback=self._toggle_airplane),
            rumps.MenuItem("Forcer Cloud", callback=self._toggle_cloud),
            None,
            rumps.MenuItem("Ouvrir le Chat", callback=self._open_chat),
            rumps.MenuItem("Stats", callback=self._show_stats),
            None,
            rumps.MenuItem("Quitter", callback=self._quit),
        ]

    def _toggle_airplane(self, sender):
        if not self.router:
            self._alert_no_router("Mode Avion")
            return
        state = self.router.toggle_airplane_mode()
        sender.state = 1 if state else 0
        print(f"  ✈️ Mode Avion : {'ON' if state else 'OFF'}")

    def _toggle_cloud(self, sender):
        if not self.router:
            self._alert_no_router("Forcer Cloud")
            return
        state = not self.router.force_cloud
        self.router.set_force_cloud(state)
        sender.state = 1 if state else 0
        print(f"  ☁️ Mode Cloud : {'ON' if state else 'OFF'}")

    def _open_chat(self, sender):
        """Ouvre l'interface NURU (overlay natif via NURU.app)."""
        app_path = "/Applications/NURU.app"
        alt_path = str(Path(__file__).parent.parent / "NURU.app")

        bundle = app_path if Path(app_path).exists() else alt_path

        if Path(bundle).exists():
            # Lancer via NURU.app → pas d'icône Python dans le Dock
            subprocess.Popen(
                ["open", bundle, "--args", "--overlay"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Fallback : lancer directement
            subprocess.Popen(
                [sys.executable, str(Path(__file__).parent / "nuru_overlay.py")],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    def _show_stats(self, sender):
        if not self.router:
            self._alert_no_router("Stats")
            return
        stats = self.memory.get_stats()
        rumps.alert(
            title="NURU — Statistiques",
            message=(
                f"Interactions : {self.total_interactions}\n"
                f"Erreurs : {self.total_errors}\n"
                f"Échanges session : {stats['exchanges_count']}\n"
                f"Durée session : {stats['duration_sec']}s\n"
                f"Modèle : {self.router._model_id if self.router else 'N/A'}"
            )
        )

    def _quit(self, sender):
        stop_tts()
        rumps.quit_application()

    def run_menubar(self):
        """Lance l'application de barre de menus."""
        if rumps is None:
            print("⚠ rumps non installé — pas de barre de menus")
            return

        self._init_router()

        app = rumps.App("NURU", "⚪", quit_button=None)
        app.menu = self._build_menu()

        @rumps.timer(0.5)
        def update_loop(sender):
            """Met à jour l'icône et le menu sur le thread principal."""
            icon = STATE_ICONS.get(self.state, "⚪")
            app.title = icon
            if hasattr(self, '_state_item'):
                self._state_item.title = f"État : {self.state.value}"
            if hasattr(self, '_interaction_item'):
                self._interaction_item.title = f"Interactions : {self.total_interactions}"

        app.run()

    def run_cli(self):
        """Mode CLI simple pour le daemon."""
        print(f"\n{'='*50}")
        print("🟡 NURU Daemon — mode CLI")
        print("Commandes : 'écoute' / 'parle [texte]' / 'quit'")
        print("            '/index status' | '/index scan' | '/index clear'")
        print(f"{'='*50}\n")

        self.router = Router(
            config_path=str(Path(__file__).parent.parent / "config" / "config.yaml"),
            memory=self.memory,
        )

        while True:
            try:
                cmd = input("NURU > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not cmd:
                continue
            
            # Commandes indexeur
            if cmd.startswith("/index"):
                if self.indexer is None:
                    print("⚠ Indexeur non disponible")
                else:
                    parts = cmd.split()
                    subcmd = parts[1] if len(parts) > 1 else "status"
                    if subcmd == "status":
                        print(self.indexer.get_status())
                    elif subcmd == "scan":
                        print("🔍 Scan en cours...")
                        self.indexer.scan_once()
                        print(self.indexer.get_status())
                    elif subcmd == "clear":
                        print("🧹 Vidage de l'index...")
                        self.indexer.clear_index()
                        print(self.indexer.get_status())
                    else:
                        print("Commandes : /index status | /index scan | /index clear")
                continue

            if cmd.lower() == "quit":
                break
            if cmd.lower() == "ecoute":
                self.process_interaction()
            else:
                self.process_interaction(text_input=cmd)

        stop_tts()
        print("Au revoir !")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NURU — Daemon Assistant")
    parser.add_argument("--menubar", action="store_true", help="Lancer avec barre de menus")
    parser.add_argument("--cli", action="store_true", help="Mode CLI (défaut)")
    parser.add_argument("--test", action="store_true", help="Test rapide sans micro")
    args = parser.parse_args()

    config_path = str(Path(__file__).parent.parent / "config" / "config.yaml")
    daemon = NuruDaemon(config_path)

    if args.menubar:
        daemon.run_menubar()
    else:
        daemon.run_cli()


if __name__ == "__main__":
    main()
