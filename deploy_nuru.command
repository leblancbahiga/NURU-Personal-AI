#!/usr/bin/env bash
# ============================================================================
# deploy_nuru.command — Déploiement natif de NURU sur macOS
# ============================================================================
# Double-clique sur ce fichier dans le Finder pour tout installer en 1 clic.
# Ce script :
#   1. Installe les dépendances Python
#   2. Copie NURU.app dans /Applications
#   3. Installe le LaunchAgent (démarrage auto à l'ouverture de session)
#   4. Configure le raccourci Option+Espace (Hammerspoon)
#   5. Lance NURU immédiatement
# ============================================================================

set -e

NURU_DIR="$HOME/Downloads/Assistant IA"
APP_SOURCE="$NURU_DIR/NURU.app"
APP_DEST="/Applications/NURU.app"
LOG_FILE="/tmp/nuru-deploy.log"

echo "╔══════════════════════════════════════════════╗"
echo "║        🚀  NURU — Déploiement natif          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Vérifier qu'on est dans le bon dossier ──
if [ ! -d "$NURU_DIR/src" ]; then
    echo "❌ ERREUR : Dossier NURU introuvable dans $NURU_DIR"
    echo "   Place ce script dans le dossier 'Assistant IA' et réessaie."
    exit 1
fi

# ── 1. Dépendances ──
echo "📦 Installation des dépendances Python..."
cd "$NURU_DIR"

# Utiliser le venv existant, ou en créer un
if [ ! -f ".venv/bin/python3" ]; then
    echo "   🔧 Création de l'environnement virtuel..."
    python3 -m venv .venv
fi

.venv/bin/pip3 install --quiet -r requirements.txt 2>&1 | tail -1
echo "   ✅ Dépendances installées"

# ── 2. Copier NURU.app dans /Applications ──
echo ""
echo "📲 Installation de NURU.app dans /Applications..."

# Vérifier que le bundle .app existe avant de copier
if [ ! -d "$APP_SOURCE" ]; then
    echo "   ❌ ERREUR: NURU.app introuvable dans le projet."
    echo "      Le bundle .app doit être créé manuellement :"
    echo "      Structure attendue : NURU.app/Contents/{MacOS,Resources}"
    echo "      Avec Info.plist, MacOS/NURU (launcher), Resources/nuru_icon.icns"
    exit 1
fi

if [ -d "$APP_DEST" ]; then
    rm -rf "$APP_DEST"
fi
cp -R "$APP_SOURCE" "$APP_DEST"

# Résoudre les problèmes Gatekeeper (Quarantine + ad-hoc signature)
echo "   🔐 Configuration des permissions et signature..."
xattr -cr "$APP_DEST" 2>/dev/null || true
codesign --force --deep --sign - "$APP_DEST" 2>/dev/null || true

echo "   ✅ NURU.app installé dans Applications"

# ── 3. LaunchAgent (démarrage auto) ──
echo ""
echo "🔧 Configuration du démarrage automatique..."
PLIST_SRC="$NURU_DIR/config/com.nuru.daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.nuru.daemon.plist"

# Mettre à jour le chemin vers le venv dans le plist
sed "s|/usr/bin/python3|$NURU_DIR/.venv/bin/python3|g" "$PLIST_SRC" > /tmp/com.nuru.daemon.plist
cp /tmp/com.nuru.daemon.plist "$PLIST_DST"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"
echo "   ✅ NURU démarrera automatiquement à chaque connexion"

# ── 4. Raccourci clavier (Hammerspoon) ──
echo ""
echo "⌨️  Configuration du raccourci Option+Espace..."
mkdir -p "$HOME/.hammerspoon"
cp "$NURU_DIR/config/nuru_hammerspoon.lua" "$HOME/.hammerspoon/init.lua"
echo "   ✅ Fichier Hammerspoon configuré"
echo "   ⚠  Tu dois installer Hammerspoon : brew install --cask hammerspoon"
echo "      Puis Reload Config (icône barre de menus → Reload Config)"

# ── 5. Clé API Tavily ──
echo ""
if ! $NURU_DIR/.venv/bin/python3 -c "
from keychain_utils import load_config_service, get_key
k = get_key(load_config_service(), 'tavily')
exit(0 if k else 1)
" 2>/dev/null; then
    echo "🔑 Configuration de la clé API Tavily..."
    echo "   Tu peux la configurer maintenant (ou plus tard avec la commande ci-dessous) :"
    echo ""
    echo "   cd \"$NURU_DIR\" && .venv/bin/python3 src/keychain_utils.py --set tavily"
    echo ""
else
    echo "🔑 Clé API Tavily déjà configurée ✓"
fi

# ── 6. NOTE : Le LaunchAgent a déjà démarré NURU via RunAtLoad.
#    Ne PAS appeler `open /Applications/NURU.app` ici — ça créerait
#    une deuxième instance qui entrerait en conflit avec la première.
echo ""
echo "🟢 NURU est déjà lancé par le LaunchAgent (démarrage auto)."
echo "   Icône dans la barre de menus."
echo ""

echo "╔══════════════════════════════════════════════╗"
echo "║    ✅  NURU prêt — ⌥+Espace pour parler      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "📋 Journal : tail -f /tmp/nuru-daemon.log"
echo ""
