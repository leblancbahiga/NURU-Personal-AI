#!/usr/bin/env python3
"""
audio_stt.py — Reconnaissance vocale pour NURU via faster-whisper.

Usage :
    python3 src/audio_stt.py --record      # Enregistre + transcrit via micro
    python3 src/audio_stt.py --file test.wav  # Transcrire un fichier audio existant
    python3 src/audio_stt.py --test       # Test rapide sans micro
"""

import sys
import time
import tempfile
import argparse
import subprocess
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import yaml
except ImportError:
    yaml = None


# ── Modèle (singleton lazy) ──
_model = None


def load_stt_model(model_size: str = "tiny", device: str = "auto") -> "WhisperModel":
    """
    Charge le modèle faster-whisper (lazy, singleton).

    Args:
        model_size: "tiny" (75 Mo) ou "base" (141 Mo)
        device: "auto", "cpu", "cuda"
    """
    global _model
    if _model is None:
        if WhisperModel is None:
            raise ImportError("faster-whisper non installé — pip3 install faster-whisper")

        print(f"🎤 Chargement du modèle STT ({model_size})...", end=" ", flush=True)
        t0 = time.time()

        # Forcer CPU sur Apple Silicon (CTranslate2 est CPU sur M1)
        compute_type = "int8" if model_size == "tiny" else "int8_float16"
        _model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

        print(f"✓ ({time.time() - t0:.1f}s)")
    return _model


def transcribe_file(filepath: str, language: str = "fr") -> str:
    """
    Transcrit un fichier audio.

    Args:
        filepath: Chemin du fichier audio (wav, mp3, m4a...)
        language: Code ISO de la langue (fr, en...)

    Retourne:
        Texte transcrit
    """
    model = load_stt_model()
    print(f"  📝 Transcription...", end=" ", flush=True)
    t0 = time.time()

    segments, info = model.transcribe(filepath, language=language, beam_size=5)

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    result = " ".join(text_parts)
    print(f"✓ ({time.time() - t0:.1f}s)")
    return result


# ── Processus d'enregistrement en cours ──
_record_proc: Optional[subprocess.Popen] = None

def transcribe_microphone(duration: int = 10, language: str = "fr",
                          model_size: str = "tiny") -> Optional[str]:
    """
    Enregistre depuis le micro et transcrit.
    """
    global _record_proc
    model = load_stt_model(model_size)

    recorder = None
    for cmd in ["sox", "ffmpeg", "rec"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode == 0:
            recorder = cmd
            break

    if recorder is None:
        print("⚠ Aucun enregistreur trouvé.")
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        print(f"🎙️ Enregistrement ({duration}s)...")
        
        cmd_args = []
        if recorder == "sox":
            cmd_args = ["sox", "-d", "-r", "16000", "-c", "1", "-b", "16", tmp_path, "trim", "0", str(duration)]
        elif recorder == "ffmpeg":
            cmd_args = ["ffmpeg", "-y", "-f", "avfoundation", "-i", ":0", "-t", str(duration), "-ac", "1", "-ar", "16000", tmp_path]
        elif recorder == "rec":
            cmd_args = ["rec", "-r", "16000", "-c", "1", "-b", "16", tmp_path, "trim", "0", str(duration)]

        _record_proc = subprocess.Popen(cmd_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Attendre la fin ou l'interruption
        try:
            _record_proc.wait(timeout=duration + 2)
        except subprocess.TimeoutExpired:
            stop_recording()

        if Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 1000:
            return transcribe_file(tmp_path, language)
        return None

    except Exception as e:
        print(f"  ✗ Erreur : {e}")
        return None
    finally:
        _record_proc = None
        try: Path(tmp_path).unlink(missing_ok=True)
        except: pass

def stop_recording():
    """Arrête l'enregistrement en cours."""
    global _record_proc
    if _record_proc and _record_proc.poll() is None:
        _record_proc.terminate()
        try: _record_proc.wait(timeout=1)
        except: _record_proc.kill()
    _record_proc = None


def unload_model():
    """Libère le modèle STT de la mémoire."""
    global _model
    if _model is not None:
        _model = None
        import gc
        gc.collect()
        print("  🧹 Modèle STT déchargé.")


def main():
    parser = argparse.ArgumentParser(description="NURU — Reconnaissance vocale (STT)")
    parser.add_argument("--record", action="store_true", help="Enregistrer + transcrire")
    parser.add_argument("--file", "-f", type=str, help="Transcrire un fichier audio")
    parser.add_argument("--duration", "-d", type=int, default=10, help="Durée d'enregistrement (s)")
    parser.add_argument("--model", "-m", choices=["tiny", "base"], default="tiny",
                        help="Taille du modèle STT")
    parser.add_argument("--language", "-l", default="fr", help="Code langue (fr, en...)")
    args = parser.parse_args()

    if args.file:
        text = transcribe_file(args.file, args.language)
        print(f"\n📝 Transcription :\n{text}")

    elif args.record:
        text = transcribe_microphone(args.duration, args.language, args.model)
        if text:
            print(f"\n📝 Transcription :\n{text}")

    else:
        print("Utilisation : python3 src/audio_stt.py --record  (ou --file chemin.wav)")
        print("             python3 src/audio_stt.py --test")


if __name__ == "__main__":
    main()
