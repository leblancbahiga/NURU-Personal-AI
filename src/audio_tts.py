#!/usr/bin/env python3
"""
audio_tts.py — Synthèse vocale pour NURU.

Moteurs supportés :
  - say (macOS) — natif, toujours disponible
  - piper — haute qualité, si binaire + modèle présents

Usage :
    python3 src/audio_tts.py "Bonjour"                    # Lecture simple
    python3 src/audio_tts.py "Bonjour" --engine piper      # Forcer Piper
    python3 src/audio_tts.py --stop                        # Arrêter
    python3 src/audio_tts.py --voices                      # Lister voix
"""

import sys
import subprocess
import time
import re
import argparse
import shutil
import json
import urllib.request
import ssl
import tarfile
import threading
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    yaml = None


# ── Chemins ──
ROOT = Path(__file__).parent.parent
PIPER_DIR = ROOT / "data" / "piper"
PIPER_BIN = PIPER_DIR / "piper"
PIPER_VOICES_DIR = PIPER_DIR / "voices"

# ── Lock thread-safe pour _current_process ──
_tts_lock = threading.Lock()

# ── Process TTS en cours (singleton pour interruption) ──
# Pour 'say': pointe vers le processus say.
# Pour 'piper': pointe vers le processus afplay (playback audio).
# Le processus Piper sous-jacent est conservé via _piper_proc.
_current_process: Optional[subprocess.Popen] = None
_piper_proc: Optional[subprocess.Popen] = None  # Processus Piper (pour piper engine)


# ═══════════════════════════════════════════════════════
# Détection des moteurs disponibles
# ═══════════════════════════════════════════════════════

def _get_config() -> dict:
    if yaml is None:
        return {}
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def get_available_engines() -> list[str]:
    """Retourne la liste des moteurs TTS disponibles."""
    engines = ["say"]
    if PIPER_BIN.exists() and PIPER_VOICES_DIR.exists():
        voices = list(PIPER_VOICES_DIR.glob("*.onnx"))
        if voices:
            engines.append("piper")
    return engines


# ═══════════════════════════════════════════════════════
# Moteur : macOS say
# ═══════════════════════════════════════════════════════

def _make_ssl_context():
    """Crée un contexte SSL avec certifi si disponible, sinon mode non-vérifié."""
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx

SAY_VOICES_CACHE = None


def _get_say_voices(language: str = "fr") -> list[dict]:
    """Liste les voix macOS disponibles."""
    global SAY_VOICES_CACHE
    if SAY_VOICES_CACHE is not None:
        return SAY_VOICES_CACHE

    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10)
        voices = []
        for line in result.stdout.split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            lang = parts[1] if len(parts) > 1 else ""
            voices.append({"name": name, "locale": lang})
        SAY_VOICES_CACHE = voices
        return voices
    except Exception:
        return []


def _get_best_say_voice(language: str = "fr") -> str:
    """Meilleure voix française disponible."""
    voices = _get_say_voices(language)
    for preferred in ["Thomas", "Virginie", "Amelie", "Aurelie"]:
        for v in voices:
            if preferred.lower() in v["name"].lower():
                return v["name"]
    for v in voices:
        if language.lower() in v.get("locale", "").lower():
            return v["name"]
    return voices[0]["name"] if voices else "Thomas"


def _say_speak(text: str, rate: int = 200, voice: Optional[str] = None) -> bool:
    """Parle via macOS say."""
    global _current_process
    with _tts_lock:
        stop()
        cmd = ["say"]
        if voice:
            cmd.extend(["-v", voice])
        cmd.extend(["-r", str(rate), text])

        try:
            _current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════
# Moteur : Piper TTS
# ═══════════════════════════════════════════════════════

def _get_piper_voice(language: str = "fr") -> Optional[Path]:
    """Trouve un modèle de voix Piper pour la langue donnée."""
    if not PIPER_VOICES_DIR.exists():
        return None
    models = sorted(PIPER_VOICES_DIR.glob("*.onnx"))
    if not models:
        return None
    # Préférer français
    for m in models:
        if "fr" in m.stem.lower():
            return m
    return models[0]


def _piper_speak(text: str, rate: float = 1.0, voice: Optional[str] = None) -> bool:
    """Parle via Piper TTS."""
    global _current_process, _piper_proc
    with _tts_lock:
        stop()

        model_path = _get_piper_voice()
        if not model_path:
            return False

        config_path = model_path.with_suffix(".json")
        if not config_path.exists():
            return False

        try:
            # 1) Lancer Piper : texte en entrée (stdin), WAV en sortie (stdout)
            piper_args = [
                str(PIPER_BIN), "--model", str(model_path),
                "--config", str(config_path), "--output-type", "wav"
            ]
            # Ajuster la vitesse via length-scale (1.0 = normal)
            length_scale = 1.0 / max(rate, 0.1) if rate != 0 else 1.0
            piper_args.extend(["--length-scale", f"{length_scale:.2f}"])

            piper = subprocess.Popen(
                piper_args,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            _piper_proc = piper  # Sauvegarder la référence pour cleanup

            # 2) Lancer afplay pour jouer le WAV provenant de Piper
            afplay = subprocess.Popen(
                ["afplay", "-"], stdin=piper.stdout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _current_process = afplay  # Pour stop() et is_speaking()

            # 3) Envoyer le texte à Piper (via son stdin, PAS celui d'afplay)
            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()  # Signal EOF -> Piper commence le traitement

            return True
        except Exception:
            # Nettoyage en cas d'erreur
            _cleanup_piper()
            return False


# ═══════════════════════════════════════════════════════
# API publique
# ═══════════════════════════════════════════════════════

def _cleanup_piper():
    """Nettoie le sous-processus Piper orphelin (_piper_proc)."""
    global _piper_proc
    proc = _piper_proc
    _piper_proc = None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def speak(text: str, rate: int = 200, voice: Optional[str] = None,
          engine: Optional[str] = None, wait: bool = False) -> bool:
    """
    Synthèse vocale non-bloquante.

    Args:
        text: Texte à prononcer
        rate: Vitesse (mots/min pour say, ratio pour piper)
        voice: Nom de la voix
        engine: "say", "piper", ou None (auto-détection)
        wait: Attendre la fin

    Retourne:
        True si lancé avec succès
    """
    if engine is None:
        engines = get_available_engines()
        engine = "piper" if "piper" in engines else "say"

    if voice is None and engine == "say":
        voice = _get_best_say_voice()

    if engine == "piper":
        success = _piper_speak(text, rate / 200.0, voice)
    else:
        success = _say_speak(text, rate, voice)

    if success and wait and _current_process:
        try:
            _current_process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            stop()

    return success


def speak_streaming(text: str, rate: int = 200, voice: Optional[str] = None,
                    engine: Optional[str] = None,
                    min_sentence_length: int = 10) -> bool:
    """
    Pseudo-streaming : découpe le texte en phrases et les lit séquentiellement.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return speak(text, rate, voice, engine)

    for sentence in sentences:
        if len(sentence) < min_sentence_length:
            continue

        success = speak(sentence, rate, voice, engine, wait=True)
        if not success:
            return False

    return True


def stop():
    """Arrête la synthèse vocale en cours."""
    global _current_process
    with _tts_lock:
        # Nettoyer le processus Piper sous-jacent (pour piper engine)
        _cleanup_piper()

        if _current_process is not None and _current_process.poll() is None:
            try:
                _current_process.terminate()
                _current_process.wait(timeout=2)
            except Exception:
                try:
                    _current_process.kill()
                except Exception:
                    pass
            _current_process = None


def is_speaking() -> bool:
    """Vérifie si NURU est en train de parler."""
    global _current_process
    with _tts_lock:
        return _current_process is not None and _current_process.poll() is None


def list_voices(engine: str = "say") -> list[str]:
    """Liste les voix disponibles pour un moteur."""
    if engine == "piper":
        if not PIPER_VOICES_DIR.exists():
            return []
        return sorted(p.stem for p in PIPER_VOICES_DIR.glob("*.onnx"))
    else:
        return [v["name"] for v in _get_say_voices()]


def download_piper_voice(language: str = "fr"):
    """
    Télécharge un modèle de voix Piper.
    Source : https://huggingface.co/rhasspy/piper-voices
    """
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Voix françaises disponibles
    FRENCH_VOICES = {
        "fr_FR-siwis-medium": {
            "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
            "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
        }
    }

    voice_name = list(FRENCH_VOICES.keys())[0]
    urls = FRENCH_VOICES[voice_name]

    model_path = PIPER_VOICES_DIR / f"{voice_name}.onnx"
    config_path = PIPER_VOICES_DIR / f"{voice_name}.onnx.json"

    if model_path.exists() and config_path.exists():
        print(f"✅ Voix {voice_name} déjà téléchargée")
        return True

    print(f"⬇️ Téléchargement de la voix {voice_name}...")
    ssl_ctx = _make_ssl_context()
    for name, url in urls.items():
        dest = model_path if url.endswith(".onnx") else config_path
        if dest.exists():
            continue
        try:
            data = urllib.request.urlopen(url, context=ssl_ctx).read()
            dest.write_bytes(data)
            print(f"  ✓ {url.split('/')[-1]}")
        except Exception as e:
            print(f"  ⚠ Échec : {e}")
            return False

    print(f"✅ Voix téléchargée dans {PIPER_VOICES_DIR}")
    return True


def download_piper_binary():
    """Télécharge le binaire Piper pour macOS ARM64."""
    PIPER_DIR.mkdir(parents=True, exist_ok=True)

    url = ("https://github.com/rhasspy/piper/releases/download/2023.11.14-2/"
           "piper_macos_aarch64.tar.gz")

    tarball = PIPER_DIR / "piper_macos_aarch64.tar.gz"
    if PIPER_BIN.exists():
        print(f"✅ Binaire Piper déjà présent")
        return True

    print(f"⬇️ Téléchargement de Piper...")
    ssl_ctx = _make_ssl_context()
    try:
        data = urllib.request.urlopen(url, context=ssl_ctx).read()
        tarball.write_bytes(data)
        with tarfile.open(tarball) as tar:
            tar.extractall(path=PIPER_DIR)
        if PIPER_BIN.exists():
            PIPER_BIN.chmod(0o755)
        tarball.unlink()
        print(f"✅ Piper installé dans {PIPER_DIR}")
        return True
    except Exception as e:
        print(f"  ⚠ Échec téléchargement Piper : {e}")
        return False


def install_piper():
    """Télécharge Piper + une voix française."""
    print("📦 Installation de Piper TTS...")
    if download_piper_binary():
        download_piper_voice()
        if _get_piper_voice():
            print("✅ Piper prêt à l'emploi !")
            return True
    print("⚠ Piper non installé — utilisation du moteur 'say' par défaut")
    return False


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NURU — Synthèse vocale (TTS)")
    parser.add_argument("text", nargs="?", help="Texte à prononcer")
    parser.add_argument("--rate", "-r", type=int, default=200, help="Vitesse")
    parser.add_argument("--voice", "-v", type=str, default=None, help="Voix")
    parser.add_argument("--engine", "-e", type=str, default=None,
                        choices=["say", "piper"], help="Moteur TTS")
    parser.add_argument("--stop", action="store_true", help="Arrêter la lecture")
    parser.add_argument("--voices", action="store_true", help="Lister les voix")
    parser.add_argument("--engines", action="store_true", help="Lister les moteurs")
    parser.add_argument("--install-piper", action="store_true", help="Installer Piper")
    parser.add_argument("--stream", action="store_true", help="Mode streaming")
    args = parser.parse_args()

    if args.stop:
        stop()
        print("⏹ Lecture arrêtée.")
        return

    if args.engines:
        engines = get_available_engines()
        print(f"Moteurs disponibles : {', '.join(engines)}")
        return

    if args.voices:
        for eng in get_available_engines():
            vl = list_voices(eng)
            print(f"{eng} ({len(vl)} voix) :")
            for v in vl[:10]:
                print(f"  • {v}")
        return

    if args.install_piper:
        install_piper()
        return

    if args.text:
        if args.stream:
            speak_streaming(args.text, args.rate, args.voice, args.engine)
        else:
            speak(args.text, args.rate, args.voice, args.engine)


if __name__ == "__main__":
    main()
