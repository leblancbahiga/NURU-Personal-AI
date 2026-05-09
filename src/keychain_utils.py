#!/usr/bin/env python3
"""
keychain_utils.py — Stockage sécurisé des clés API dans le Keychain macOS.

Utilise exclusivement keyring pour éviter tout fichier .env contenant des clés.
Le service Keychain utilisé est défini dans config.yaml (system.keychain_service).

Usage:
  python3 src/keychain_utils.py --set gemini     # Saisie interactive
  python3 src/keychain_utils.py --set deepseek KEY_VALUE
  python3 src/keychain_utils.py --get gemini     # Affiche masqué
  python3 src/keychain_utils.py --list           # Liste les clés stockées
  python3 src/keychain_utils.py --delete gemini
  python3 src/keychain_utils.py --validate all   # Vérifie que les clés sont accessibles
"""

import sys
import argparse
import getpass
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import keyring
    import keyring.errors
except ImportError:
    print("Erreur : keyring non installé.")
    print("  pip3 install keyring")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None


# ── Services disponibles ──
SERVICES = {
    "gemini": "API Google Gemini (Gemini 2.0 Flash)",
    "deepseek": "API Deepseek (DeepSeek Chat)",
    "openai": "API OpenAI (fallback)",
    "brave": "API Brave Search (recherche web)",
    "tavily": "API Tavily Search (recherche web optimisée LLM)",
}

KEYCHAIN_SERVICE = "com.nuru.assistant"


def load_config_service() -> str:
    """Charge le nom du service Keychain depuis la config."""
    if yaml is None:
        return KEYCHAIN_SERVICE
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        return KEYCHAIN_SERVICE
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("system", {}).get("keychain_service", KEYCHAIN_SERVICE)
    except Exception:
        return KEYCHAIN_SERVICE


def store_key(service_name: str, key_name: str, value: str) -> bool:
    """Stocke une clé API dans le Keychain."""
    try:
        keyring.set_password(service_name, key_name, value)
        _KEY_CACHE[f"{service_name}:{key_name}"] = value  # met à jour le cache
        return True
    except keyring.errors.KeyringError as e:
        print(f"Erreur Keychain : {e}", file=sys.stderr)
        return False


# ── Cache mémoire des clés (évite popup Keychain à chaque appel) ──
_KEY_CACHE: dict[str, str | None] = {}


def get_key(service_name: str, key_name: str) -> str | None:
    """Récupère une clé API depuis le Keychain, avec cache mémoire."""
    cache_key = f"{service_name}:{key_name}"
    if cache_key in _KEY_CACHE:
        return _KEY_CACHE[cache_key]
    try:
        value = keyring.get_password(service_name, key_name)
        _KEY_CACHE[cache_key] = value
        return value
    except keyring.errors.KeyringError as e:
        print(f"Erreur Keychain : {e}", file=sys.stderr)
        _KEY_CACHE[cache_key] = None
        return None


def clear_key_cache(service_name: str | None = None, key_name: str | None = None):
    """Vide le cache mémoire des clés."""
    global _KEY_CACHE
    if service_name and key_name:
        _KEY_CACHE.pop(f"{service_name}:{key_name}", None)
    else:
        _KEY_CACHE = {}


def delete_key(service_name: str, key_name: str) -> bool:
    """Supprime une clé API du Keychain."""
    try:
        keyring.delete_password(service_name, key_name)
        return True
    except keyring.errors.PasswordDeleteError:
        print(f"Clé '{key_name}' introuvable dans le Keychain.", file=sys.stderr)
        return False


def list_keys(service_name: str):
    """Liste les clés disponibles (sans les afficher)."""
    print(f"Clés API stockées dans '{service_name}' :")
    for name, desc in SERVICES.items():
        value = get_key(service_name, name)
        if value:
            masked = value[:4] + "•••" + value[-4:] if len(value) > 8 else "••••••"
            print(f"  ✓ {name:12s} → {masked}  ({desc})")
        else:
            print(f"  ✗ {name:12s} → Non définie  ({desc})")


def validate_keys(service_name: str) -> bool:
    """Valide que les clés nécessaires sont accessibles."""
    all_ok = True
    for name, desc in SERVICES.items():
        value = get_key(service_name, name)
        if value:
            print(f"  ✓ {name}: accessible ({len(value)} chars)")
        else:
            print(f"  ✗ {name}: non définie — {desc}")
            all_ok = False
    return all_ok


def interactive_set(service_name: str, key_name: str):
    """Saisie interactive sécurisée d'une clé API."""
    if key_name not in SERVICES:
        print(f"Service inconnu. Choisissez parmi : {', '.join(SERVICES.keys())}")
        return False

    current = get_key(service_name, key_name)
    if current:
        print(f"Clé actuelle pour '{key_name}' : {current[:4]}•••{current[-4:]}")
        overwrite = input("Voulez-vous la remplacer ? (o/N) : ").strip().lower()
        if overwrite != "o":
            print("Annulé.")
            return True

    print(f"Entrez votre clé API {SERVICES[key_name]} :")
    value = getpass.getpass("> ").strip()
    if not value:
        print("Clé vide — annulé.")
        return False

    if store_key(service_name, key_name, value):
        masked = value[:4] + "•••" + value[-4:] if len(value) > 8 else "••••••"
        print(f"✓ Clé '{key_name}' stockée avec succès ({masked})")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Gestion des clés API via Keychain macOS")
    parser.add_argument("--set", type=str, metavar="SERVICE",
                        help="Définir une clé API (gemini, deepseek, openai)")
    parser.add_argument("--get", type=str, metavar="SERVICE",
                        help="Afficher une clé API")
    parser.add_argument("--delete", type=str, metavar="SERVICE",
                        help="Supprimer une clé API")
    parser.add_argument("--list", action="store_true",
                        help="Lister toutes les clés")
    parser.add_argument("--validate", type=str, nargs="?", const="all",
                        help="Valider les clés (all|gemini|deepseek|openai)")
    parser.add_argument("value", nargs="?", help="Valeur de la clé (optionnel, sinon saisie interactive)")

    args = parser.parse_args()
    service = load_config_service()

    if args.set:
        if args.value:
            if store_key(service, args.set, args.value):
                masked = args.value[:4] + "•••" + args.value[-4:] if len(args.value) > 8 else "••••••"
                print(f"✓ Clé '{args.set}' stockée ({masked})")
        else:
            interactive_set(service, args.set)

    elif args.get:
        value = get_key(service, args.get)
        if value:
            print(value)
        else:
            print(f"Clé '{args.get}' non trouvée dans le Keychain.")

    elif args.delete:
        if delete_key(service, args.delete):
            print(f"✓ Clé '{args.delete}' supprimée.")

    elif args.list:
        list_keys(service)

    elif args.validate:
        validate_keys(service)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
