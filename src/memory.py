#!/usr/bin/env python3
"""
memory.py — Gestion de la mémoire conversationnelle de NURU.

- Mémoire de session : buffer circulaire des N derniers échanges (RAM)
- Mémoire long-terme : résumé automatique vectorisé (stub pour ChromaDB Phase 3)
"""

import time
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Exchange:
    """Un échange utilisateur ↔ assistant."""
    user: str
    assistant: str
    timestamp: float = field(default_factory=time.time)

    def format(self) -> str:
        """Formate pour injection dans le prompt (format ChatML/Qwen)."""
        return f"<|im_start|>user\n{self.user}<|im_end|>\n<|im_start|>assistant\n{self.assistant}<|im_end|>"


class SessionMemory:
    """
    Mémoire de session : buffer circulaire avec persistance sur disque.
    """

    def __init__(self, buffer_size: int = 10, session_id: Optional[str] = None,
                 storage_dir: Optional[str] = None, restore: bool = True):
        self.buffer_size = buffer_size
        self._buffer: deque[Exchange] = deque(maxlen=buffer_size)
        self._session_id: str = session_id or f"sess_{int(time.time())}"
        self._created_at: float = time.time()
        self._last_activity: float = time.time()

        # Gestion de la persistance
        if storage_dir is None:
            storage_dir = str(Path(__file__).parent.parent / "data" / "sessions")
        self.storage_path = Path(storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        if session_id and restore:
            self.load_session(session_id)

    def add(self, user_msg: str, assistant_msg: str) -> None:
        """Ajoute un échange au buffer circulaire et sauvegarde."""
        self._buffer.append(Exchange(user=user_msg, assistant=assistant_msg))
        self._last_activity = time.time()
        self.save_session()

    def get_exchanges(self) -> list[Exchange]:
        """Retourne la liste des échanges (du plus ancien au plus récent)."""
        return list(self._buffer)

    def get_context(self, include_timestamps: bool = False) -> str:
        """
        Retourne le contexte formaté pour injection dans le prompt.
        """
        parts = []
        for i, ex in enumerate(self._buffer):
            if include_timestamps:
                t = time.strftime("%H:%M", time.localtime(ex.timestamp))
                parts.append(f"<!-- Échange {i+1} à {t} -->")
            parts.append(ex.format())
        return "\n".join(parts)

    def save_session(self) -> bool:
        """Sauvegarde la session en JSON dans data/sessions/<session_id>.json."""
        try:
            file_path = self.storage_path / f"{self._session_id}.json"
            data = {
                "session_id": self._session_id,
                "created_at": self._created_at,
                "last_activity": self._last_activity,
                "exchanges": [asdict(ex) for ex in self._buffer]
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"  ⚠ Erreur sauvegarde session : {e}")
            return False

    def load_session(self, session_id: str) -> bool:
        """Charge une session depuis data/sessions/<session_id>.json."""
        try:
            self._session_id = session_id
            file_path = self.storage_path / f"{session_id}.json"
            if not file_path.exists():
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._created_at = data.get("created_at", time.time())
            self._last_activity = data.get("last_activity", time.time())

            exchanges_data = data.get("exchanges", [])
            self._buffer.clear()
            for ex_data in exchanges_data:
                self._buffer.append(Exchange(**ex_data))

            return True
        except Exception as e:
            print(f"  ⚠ Erreur chargement session : {e}")
            return False

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def clear(self) -> None:
        """Purge le contexte actif et supprime le fichier de session."""
        self._buffer.clear()
        self._last_activity = time.time()
        file_path = self.storage_path / f"{self._session_id}.json"
        if file_path.exists():
            file_path.unlink()

    def get_stats(self) -> dict:
        """Statistiques de la session en cours."""
        return {
            "session_id": self._session_id,
            "duration_sec": int(time.time() - self._created_at),
            "exchanges_count": len(self._buffer),
            "buffer_size": self.buffer_size,
            "idle_sec": int(time.time() - self._last_activity),
        }

    def summarize(self) -> str:
        """Résumé de la session."""
        if not self._buffer:
            return "Session vide."

        lines = [
            f"Session {self._session_id}",
            f"Durée : {self.get_stats()['duration_sec']}s",
            f"Échanges : {len(self._buffer)}",
            "",
        ]
        for i, ex in enumerate(self._buffer, 1):
            user_preview = ex.user[:80] + ("..." if len(ex.user) > 80 else "")
            assistant_preview = ex.assistant[:80] + ("..." if len(ex.assistant) > 80 else "")
            lines.append(f"{i}. U: {user_preview}")
            lines.append(f"   A: {assistant_preview}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SessionMemory(buffer={len(self._buffer)}/{self.buffer_size}, id={self._session_id})"
