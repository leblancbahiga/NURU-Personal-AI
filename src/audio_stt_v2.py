#!/usr/bin/env python3
"""
audio_stt_v2.py — Reconnaissance vocale pour NURU V2 via mlx-whisper.

Upgrade du STT V1 (faster-whisper tiny, WER ~25%) vers mlx-whisper base (WER ~10%).
Utilise le Neural Engine Apple Silicon via MLX pour des performances optimales.

Modèles disponibles :
  - "base"  : 74 MB, WER ~10%, temps réel sur M1 (recommandé)
  - "small" : 244 MB, WER ~5%, quasi temps réel (si RAM dispo)

Usage :
    stt = MLXWhisperSTT(model_size="base")
    text = stt.transcribe("audio.wav")
    text = stt.transcribe_microphone(duration=5)
"""

import sys
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nuru.v2.audio_stt")

try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))


class MLXWhisperSTT:
    """
    Reconnaissance vocale via mlx-whisper sur Apple Silicon.

    Args:
        model_size: Taille du modèle ("base" ou "small"). Défaut: "base".
        language: Langue cible ("fr" pour français). Défaut: "fr".
    """

    def __init__(self, model_size: str = "base", language: str = "fr"):
        if not MLX_WHISPER_AVAILABLE:
            raise ImportError("mlx-whisper non installé — pip3 install mlx-whisper")

        self.model_size = model_size
        self.language = language
        self._model = None
        self._load_time = 0.0
        logger.info("MLXWhisperSTT initialisé (model=%s, lang=%s)", model_size, language)

    def _load_model(self):
        """Charge le modèle mlx-whisper (lazy loading)."""
        if self._model is None:
            t0 = time.time()
            logger.info("Chargement de mlx-whisper/%s...", self.model_size)
            # mlx_whisper utilise un cache HF, le modèle est téléchargé automatiquement
            self._model = True  # Marque comme chargé
            self._load_time = time.time() - t0
            logger.info("Modèle chargé en %.1fs", self._load_time)

    def unload(self):
        """Décharge le modèle pour libérer la RAM."""
        if self._model is not None:
            self._model = None
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
            logger.info("Modèle STT déchargé")

    def transcribe(self, audio_path: str | Path, **kwargs) -> str:
        """
        Transcrit un fichier audio.

        Args:
            audio_path: Chemin vers le fichier audio (WAV, MP3, etc.)

        Retourne:
            Texte transcrit.
        """
        if not MLX_WHISPER_AVAILABLE:
            return "[STT non disponible — mlx-whisper manquant]"

        t0 = time.time()
        try:
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=f"mlx-community/whisper-{self.model_size}",
                language=self.language,
                **kwargs,
            )
            text = result.get("text", "").strip()
            elapsed = time.time() - t0
            logger.debug("Transcription: %.1fs, %d chars", elapsed, len(text))
            return text
        except Exception as e:
            logger.error("Erreur transcription: %s", e)
            return ""

    def transcribe_microphone(self, duration: int = 5, sample_rate: int = 16000) -> str:
        """
        Enregistre depuis le micro et transcrit.

        Args:
            duration: Durée d'enregistrement en secondes.
            sample_rate: Taux d'échantillonnage.

        Retourne:
            Texte transcrit.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Enregistrement via sox ou rec (macOS)
            logger.info("Enregistrement (%ds)...", duration)
            cmd = [
                "rec", "-q", "-r", str(sample_rate), "-c", "1",
                "-b", "16", "-e", "signed-integer",
                temp_path, "trim", "0", str(duration),
            ]
            subprocess.run(cmd, check=True, timeout=duration + 5)

            # Transcription
            text = self.transcribe(temp_path)
            return text

        except FileNotFoundError:
            # Fallback: essayer avec ffmpeg
            try:
                cmd = [
                    "ffmpeg", "-f", "avfoundation", "-i", ":0",
                    "-t", str(duration), "-ar", str(sample_rate),
                    "-ac", "1", temp_path, "-y",
                ]
                subprocess.run(cmd, check=True, timeout=duration + 5, capture_output=True)
                text = self.transcribe(temp_path)
                return text
            except Exception as e2:
                logger.error("Enregistrement impossible: %s", e2)
                return "[Erreur enregistrement microphone]"
        except subprocess.TimeoutExpired:
            logger.warning("Enregistrement interrompu (timeout)")
            return self.transcribe(temp_path)
        except Exception as e:
            logger.error("Erreur enregistrement: %s", e)
            return ""
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def get_stats(self) -> dict:
        """Statistiques du module STT."""
        return {
            "model": self.model_size,
            "language": self.language,
            "loaded": self._model is not None,
            "load_time_s": round(self._load_time, 1),
            "backend": "mlx-whisper",
        }


# ── Singleton ──
_stt_instance: Optional[MLXWhisperSTT] = None


def get_stt(model_size: str = "base", language: str = "fr") -> Optional[MLXWhisperSTT]:
    """Retourne l'instance singleton du STT V2."""
    global _stt_instance
    if _stt_instance is None and MLX_WHISPER_AVAILABLE:
        try:
            _stt_instance = MLXWhisperSTT(model_size=model_size, language=language)
        except Exception as e:
            logger.error("Impossible d'initialiser MLXWhisperSTT: %s", e)
    return _stt_instance


def unload_stt():
    """Décharge le modèle STT."""
    global _stt_instance
    if _stt_instance is not None:
        _stt_instance.unload()
        _stt_instance = None
