#!/usr/bin/env python3
"""
nuru_overlay.py — Interface Cyber-HUD V2 (Premium Edition).
Design futuriste avec monitoring en temps réel et effets visuels avancés.
"""

import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QFrame, QGraphicsDropShadowEffect,
    QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QTextCursor, QKeySequence, QShortcut, QColor, QLinearGradient, QPalette, QBrush, QFont

from router import Router
from memory import SessionMemory

logging.basicConfig(level=logging.WARNING)

# ── Couleurs Cyber-Premium ──
CYAN = "#00F2FF"
DEEP_CYAN = "#008899"
MAGENTA = "#FF00D4"
WHITE = "#F0F0F0"
DARK_BG = "rgba(10, 12, 16, 245)"
BORDER_COLOR = "rgba(0, 242, 255, 60)"
DIM_TEXT = "rgba(255, 255, 255, 120)"

class TelemetryWidget(QFrame):
    """Petit widget de monitoring RAM/MLX intégré au HUD."""
    def __init__(self, router):
        super().__init__()
        self.router = router
        self.setFixedHeight(22)
        self.setStyleSheet("background: transparent; border: none;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        self.ram_label = QLabel("RAM: --")
        self.ram_label.setStyleSheet(f"color: {DIM_TEXT}; font: 10px 'JetBrains Mono', 'Menlo', monospace;")
        
        self.mlx_label = QLabel("MLX: IDLE")
        self.mlx_label.setStyleSheet(f"color: {CYAN}; font: bold 10px 'JetBrains Mono', 'Menlo', monospace;")
        
        layout.addWidget(self.ram_label)
        layout.addWidget(self.mlx_label)
        layout.addStretch()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(2000)
        
    def update_stats(self):
        try:
            if hasattr(self.router, 'resource_manager'):
                stats = self.router.resource_manager.get_stats()
                ram = stats.get("ram_percent", 0)
                threads = stats.get("mlx_threads", 0)
                self.ram_label.setText(f"RAM: {ram:.1f}%")
                self.mlx_label.setText(f"MLX: {threads} THREADS")
        except:
            pass

class InferenceWorker(QThread):
    """Thread d'inférence isolé."""
    token = Signal(str)
    status = Signal(str, str)
    done = Signal()
    error = Signal(str)

    def __init__(self, query: str, router, use_v1: bool = False):
        super().__init__()
        self.query = query
        self.router = router
        self.use_v1 = use_v1

    def run(self):
        try:
            route_fn = self.router.stream_route if self.use_v1 else self.router.stream_v2
            self.status.emit("NEURAL CLASSIFICATION...", CYAN)

            for chunk in route_fn(self.query):
                t = chunk.get("type", "")
                if t == "start":
                    lvl = chunk.get("level_name", "Cognition")
                    self.status.emit(f"NURU // {lvl.upper()}", CYAN)
                elif t == "status":
                    self.status.emit(chunk.get("msg", "").upper(), DIM_TEXT)
                elif t == "token":
                    self.token.emit(chunk.get("token", ""))
                elif t == "error":
                    self.error.emit(chunk.get("msg", "Erreur Système"))
                    return

            self.done.emit()
        except Exception as e:
            self.error.emit(str(e))

class NuruOverlay(QWidget):
    """Interface Cyber-HUD flottante V2."""

    def __init__(self, router, use_v1: bool = False):
        super().__init__()
        self.router = router
        self.use_v1 = use_v1
        self._worker = None
        self._build_window()
        self._build_ui()
        self._build_shortcuts()
        self._set_status("NURU // READY", CYAN)

    def _build_window(self):
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(720, 360)

    def _build_ui(self):
        # Conteneur principal avec effet Glassmorphism
        self._container = QFrame(self)
        self._container.setGeometry(10, 10, 700, 340)
        self._container.setObjectName("MainHUD")
        self._container.setStyleSheet(f"""
            QFrame#MainHUD {{
                background-color: {DARK_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """)

        # Effet de lueur externe
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 242, 255, 40))
        shadow.setOffset(0, 0)
        self._container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)

        # ── Header ──
        header = QHBoxLayout()
        
        self._status = QLabel("NURU // CORE ONLINE")
        self._status.setStyleSheet(f"color: {CYAN}; font: bold 12px 'JetBrains Mono', 'Menlo', monospace; letter-spacing: 1px;")
        
        self.telemetry = TelemetryWidget(self.router)
        
        header.addWidget(self._status)
        header.addStretch()
        header.addWidget(self.telemetry)
        
        layout.addLayout(header)

        # ── Séparateur subtil ──
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: rgba(0, 242, 255, 30); height: 1px; border: none;")
        layout.addWidget(line)

        # ── Console de sortie (Scrollable) ──
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._console.setStyleSheet(f"""
            QTextEdit {{
                background: transparent; border: none;
                color: {WHITE}; font: 14px '.AppleSystemUIFont', 'Helvetica Neue', 'Arial', sans-serif;
                line-height: 1.5;
            }}
        """)
        self._console.setPlaceholderText("Les résultats s'afficheront ici...")
        layout.addWidget(self._console)

        # ── Barre de progression subtile ──
        self._progress = QProgressBar()
        self._progress.setFixedHeight(2)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: rgba(255,255,255,10); border: none; border-radius: 1px; }}
            QProgressBar::chunk {{ background: {CYAN}; }}
        """)
        self._progress.hide()
        layout.addWidget(self._progress)

        # ── Zone de saisie ──
        input_container = QFrame()
        input_container.setStyleSheet(f"""
            QFrame {{
                background: rgba(255,255,255,12);
                border: 1px solid rgba(255,255,255,25);
                border-radius: 8px;
            }}
            QFrame:focus-within {{
                border: 1px solid {CYAN};
                background: rgba(255,255,255,18);
            }}
        """)
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(12, 4, 12, 4)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Demande moi n'importe quoi...")
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background: transparent; border: none;
                color: #FFF; font: 15px '.AppleSystemUIFont', 'Helvetica Neue', 'Arial', sans-serif;
                padding: 10px 0;
            }}
        """)
        self._input.returnPressed.connect(self._trigger)
        
        icon_label = QLabel("⚡")
        icon_label.setStyleSheet("font-size: 16px; background: transparent; border: none;")
        
        input_layout.addWidget(icon_label)
        input_layout.addWidget(self._input)
        layout.addWidget(input_container)

        # Animation d'entrée
        self.setWindowOpacity(0)
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(400)
        self._fade_anim.setStartValue(0)
        self._fade_anim.setEndValue(1)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)

    def show(self):
        super().show()
        self._fade_anim.start()

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.hide)

    def _set_status(self, text, color):
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font: bold 12px 'JetBrains Mono', 'Menlo', monospace;")

    @Slot()
    def _trigger(self):
        query = self._input.text().strip()
        if not query:
            return
        
        # Debounce : empêcher les doubles clics ou doubles signaux
        if self._worker and self._worker.isRunning():
            return

        self._input.setEnabled(False)
        self._console.clear()
        self._progress.show()
        self._progress.setRange(0, 0) # Mode indéterminé (pulse)
        self._set_status("COGNITION IN PROGRESS...", MAGENTA)

        self._worker = InferenceWorker(query, self.router, self.use_v1)
        self._worker.token.connect(self._on_token)
        self._worker.status.connect(self._on_status)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @Slot(str)
    def _on_token(self, text):
        cursor = self._console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self._console.setTextCursor(cursor)
        # Scroll progressif
        self._console.ensureCursorVisible()

    @Slot(str, str)
    def _on_status(self, msg, color):
        self._set_status(msg, color)

    @Slot(str)
    def _on_error(self, msg):
        self._console.setHtml(f'<span style="color:{MAGENTA};"><b>[SYSTEM_ERROR]</b> {msg}</span>')
        self._finish()

    @Slot()
    def _on_done(self):
        self._finish()

    def _finish(self):
        self._input.setEnabled(True)
        self._input.clear()
        self._input.setFocus()
        self._progress.hide()
        self._set_status("NURU // READY", CYAN)
        self._worker = None

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    memory = SessionMemory()
    router = Router(config_path=str(config_path), memory=memory, use_semantic_cache=True)
    router.enable_v2()

    overlay = NuruOverlay(router)

    # Centrer proprement
    screen = app.primaryScreen().availableGeometry()
    x = screen.width() // 2 - overlay.width() // 2
    y = screen.height() - overlay.height() - 80
    overlay.move(x, y)

    overlay.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
