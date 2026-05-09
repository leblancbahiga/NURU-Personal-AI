#!/usr/bin/env python3
"""
structured_memory_v2.py — Mémoire structurée persistée en SQLite pour NURU V2.

Remplace le stockage JSON par SQLite avec FTS (Full-Text Search) pour :
  - Recherche plein texte sur les faits
  - Requêtes par catégorie, confiance, date
  - Pas de corruption JSON (transactions ACID)
  - Performance ×10 sur les gros volumes

Usage :
    mem = StructuredMemoryV2(db_path="~/.nuru/memory_v2.db")
    mem.store_fact("nom", "Leblanc", category="person", confidence=0.95)
    facts = mem.search_facts("Leblanc")
    context = mem.get_context()
"""

import json
import sqlite3
import time
import re
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nuru.v2.structured_memory")


class StructuredMemoryV2:
    """
    Mémoire structurée persistée en SQLite avec FTS.
    Rétrocompatible avec l'interface de structured_memory.py original.
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".nuru" / "memory_v2.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        # Cache en mémoire pour accès rapide
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _init_db(self):
        """Crée les tables si elles n'existent pas."""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    key         TEXT NOT NULL,
                    value       TEXT NOT NULL,
                    category    TEXT DEFAULT 'general',
                    confidence  REAL DEFAULT 0.8,
                    status      TEXT DEFAULT 'active',
                    source      TEXT DEFAULT '',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    UNIQUE(key, value)
                );
                CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
                CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
                CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
                CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence);

                CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                    key, value, category, content='facts', content_rowid='id'
                );

                -- Triggers pour maintenir FTS à jour
                CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                    INSERT INTO facts_fts(rowid, key, value, category)
                    VALUES (new.id, new.key, new.value, new.category);
                END;
                CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                    INSERT INTO facts_fts(facts_fts, rowid, key, value, category)
                    VALUES ('delete', old.id, old.key, old.value, old.category);
                END;
                CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                    INSERT INTO facts_fts(facts_fts, rowid, key, value, category)
                    VALUES ('delete', old.id, old.key, old.value, old.category);
                    INSERT INTO facts_fts(rowid, key, value, category)
                    VALUES (new.id, new.key, new.value, new.category);
                END;
            """)
            self._conn.commit()

    def _load_cache(self):
        """Charge tous les faits actifs en mémoire."""
        cursor = self._conn.execute(
            "SELECT key, value, category, confidence, status, created_at, source FROM facts WHERE status='active'"
        )
        for row in cursor.fetchall():
            self._cache[row["key"]] = {
                "value": row["value"],
                "category": row["category"],
                "confidence": row["confidence"],
                "status": row["status"],
                "created_at": row["created_at"],
                "source": row["source"] or "",
            }

    def store_fact(self, key: str, value: str, category: str = "general",
                   confidence: float = 0.8, source: str = "") -> bool:
        """Stocke un fait. Met à jour si existant."""
        now = time.time()
        with self._lock:
            try:
                # Vérifier si le fait existe déjà
                existing = self._conn.execute(
                    "SELECT id, confidence FROM facts WHERE key=? AND value=? AND status='active'",
                    (key, value)
                ).fetchone()

                if existing:
                    # Même valeur → augmenter la confiance
                    new_conf = min(1.0, existing["confidence"] + 0.1)
                    self._conn.execute(
                        "UPDATE facts SET confidence=?, updated_at=? WHERE id=?",
                        (new_conf, now, existing["id"])
                    )
                else:
                    # Ancienne valeur différente → marquer comme remplacée
                    self._conn.execute(
                        "UPDATE facts SET status='replaced', updated_at=? WHERE key=? AND status='active'",
                        (now, key)
                    )
                    # Nouveau fait
                    self._conn.execute(
                        "INSERT INTO facts (key, value, category, confidence, status, source, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
                        (key, value, category, min(max(confidence, 0.0), 1.0), source, now, now)
                    )

                self._conn.commit()
                # Mettre à jour le cache
                self._cache[key] = {
                    "value": value, "category": category,
                    "confidence": min(max(confidence, 0.0), 1.0),
                    "status": "active", "created_at": now, "source": source,
                }
                return True
            except sqlite3.Error as e:
                logger.error("Erreur store_fact: %s", e)
                return False

    def get_fact(self, key: str) -> Optional[dict]:
        """Récupère un fait actif par sa clé."""
        return self._cache.get(key)

    def get_all_facts(self) -> dict[str, dict]:
        """Retourne tous les faits actifs."""
        return dict(self._cache)

    def search_facts(self, query: str, limit: int = 10) -> list[dict]:
        """Recherche plein texte dans les faits via FTS5."""
        if not query or not query.strip():
            return []
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "SELECT f.key, f.value, f.category, f.confidence, f.status, f.created_at, f.source "
                    "FROM facts f JOIN facts_fts fts ON f.id = fts.rowid "
                    "WHERE facts_fts MATCH ? AND f.status='active' "
                    "ORDER BY f.confidence DESC LIMIT ?",
                    (query, limit)
                )
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error:
                # FTS peut échouer sur des requêtes mal formées → fallback LIKE
                cursor = self._conn.execute(
                    "SELECT key, value, category, confidence, status, created_at, source "
                    "FROM facts WHERE status='active' AND (key LIKE ? OR value LIKE ?) "
                    "ORDER BY confidence DESC LIMIT ?",
                    (f"%{query}%", f"%{query}%", limit)
                )
                return [dict(row) for row in cursor.fetchall()]

    def forget(self, key: str) -> bool:
        """Supprime un fait."""
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET status='deleted', updated_at=? WHERE key=? AND status='active'",
                (time.time(), key)
            )
            self._conn.commit()
            self._cache.pop(key, None)
            return True

    def forget_all(self) -> None:
        """Supprime tous les faits."""
        with self._lock:
            now = time.time()
            self._conn.execute(
                "UPDATE facts SET status='deleted', updated_at=? WHERE status='active'", (now,)
            )
            self._conn.commit()
            self._cache.clear()

    def extract_facts(self, text: str) -> dict[str, str]:
        """Extrait des faits d'un texte (compatible structured_memory.py)."""
        if not text:
            return {}

        facts = {}
        patterns = [
            (r"(?:je\s+m['']appelle)\s+(?P<value>[A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+)|[.,!?;]|$)", "nom"),
            (r"(?:mon\s+projet\s+(?:est|s['']appelle|appelle))\s+(?P<value>[A-Za-z0-9À-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+)|[.,!?;]|$)", "projet"),
            (r"(?:je\s+parle)\s+(?P<value>[A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+|couramment|un\s+peu)|[.,!?;]|$)", "langue"),
            (r"(?:mon\s+email\s+(?:est|:))\s*(?P<value>[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "email"),
            (r"(?:j['']habite\s+(?:à|en|dans))\s+(?P<value>[A-Za-zÀ-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+|depuis\s+)|[.,!?;]|$)", "habitation"),
            (r"(?:je\s+travaille\s+(?:chez|sur|dans|pour))\s+(?P<value>[A-Za-z0-9À-ÿ\-\s]+?)(?:\s+(?:et\s+|mon\s+|j['']|je\s+|mais\s+|car\s+|depuis\s+)|[.,!?;]|$)", "travail"),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = re.sub(r"[.,!?;]+$", "", match.group("value").strip()).strip()
                if value and len(value) > 1:
                    facts[key] = value
        return facts

    def extract_and_store(self, text: str) -> dict[str, str]:
        """Extrait des faits et les stocke immédiatement."""
        new_facts = self.extract_facts(text)
        for key, value in new_facts.items():
            existing = self.get_fact(key)
            if not existing or existing["value"] != value:
                self.store_fact(key, value, confidence=0.85)
        return new_facts

    def get_context(self) -> str:
        """Génère le contexte formaté pour injection dans le prompt (compatible V1)."""
        if not self._cache:
            return ""

        labels = {
            "nom": "Nom", "prenom": "Prénom", "profession": "Profession",
            "lieu": "Lieu", "projet": "Projet", "langue": "Langue(s)",
            "email": "Email", "numero": "Numéro", "habitation": "Habitation",
            "travail": "Travail", "aime": "Aime", "prefere": "Préfère",
        }

        lines = ["[Mémoire structurée — informations connues sur l'utilisateur]"]
        for key, entry in self._cache.items():
            label = labels.get(key, key.capitalize())
            lines.append(f"- {label} : {entry['value']}")

        lines.append("Utilise ces informations pour personnaliser tes réponses.")
        return "\n".join(lines)

    def describe_user(self) -> str | None:
        """Génère une description de l'utilisateur."""
        if not self._cache:
            return None

        nom = self._cache.get("nom", {}).get("value", "")
        profession = self._cache.get("profession", {}).get("value", "")
        lieu = self._cache.get("lieu", {}).get("value", "")
        projet = self._cache.get("projet", {}).get("value", "")

        parts = []
        if nom:
            parts.append(f"Tu es {nom}")
            if profession:
                parts[-1] += f", un {profession}"
            if lieu:
                parts[-1] += f" basé en {lieu}" if "en " not in lieu[:4].lower() else f" basé {lieu}"
        if projet:
            parts.append(f"Tu travailles sur le projet {projet}")

        return ". ".join(parts) + "." if parts else None

    def get_stats(self) -> dict:
        """Statistiques de la mémoire structurée."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            active = self._conn.execute("SELECT COUNT(*) FROM facts WHERE status='active'").fetchone()[0]
            by_category = {}
            for row in self._conn.execute("SELECT category, COUNT(*) FROM facts WHERE status='active' GROUP BY category"):
                by_category[row[0]] = row[1]
        return {
            "total": total,
            "active": active,
            "by_category": by_category,
            "db_path": str(self._db_path),
            "cache_size": len(self._cache),
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"StructuredMemoryV2(faits={len(self._cache)}, db={self._db_path.name})"

    def close(self):
        """Ferme la connexion SQLite."""
        self._conn.close()
