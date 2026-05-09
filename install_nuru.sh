#!/bin/bash
# install_nuru.sh — Installe NURU en un clic
# Usage: bash install_nuru.sh

set -e

NURU_DIR="$HOME/Downloads/Assistant IA"
echo "🚀 Installation de NURU..."

# 1. Vérifier les dépendances
echo "📦 Installation des dépendances (peut prendre 1-2 min)..."
pip3 install -r "$NURU_DIR/requirements.txt"

# 2. Installer le LaunchAgent (démarrage auto)
echo "🔧 Installation du LaunchAgent..."
cp "$NURU_DIR/config/com.nuru.daemon.plist" "$HOME/Library/LaunchAgents/"
launchctl load "$HOME/Library/LaunchAgents/com.nuru.daemon.plist"
echo "  ✅ NURU démarrera automatiquement à chaque connexion"

# 3. Instructions Hammerspoon
echo ""
echo "⌨️  Pour activer le raccourci Option+Espace :"
echo "   1. Installe Hammerspoon : brew install --cask hammerspoon"
echo "   2. Copie le script : cp config/nuru_hammerspoon.lua ~/.hammerspoon/init.lua"
echo "   3. Relance Hammerspoon (icône barre → Reload Config)"
echo ""

# 4. Lancer NURU maintenant
echo "🟢 Lancement de NURU..."
nohup python3 "$NURU_DIR/src/nuru_daemon.py" --menubar > /tmp/nuru-daemon.log 2>&1 &
echo "  ✅ NURU est lancé ! Icône dans la barre de menus."
echo ""
echo "📋 Journal : tail -f /tmp/nuru-daemon.log"
