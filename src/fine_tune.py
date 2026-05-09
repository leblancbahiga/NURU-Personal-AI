#!/usr/bin/env python3
"""
fine_tune.py — Pipeline MLX LoRA pour NURU.

Entraîne NURU sur les conversations collectées (~/.nuru/dataset/conversations.jsonl)
pour améliorer ses réponses.

Étapes :
  1. Collecte (déjà active via chat.py / overlay)
  2. Export → format MLX LoRA (prompt + completion)
  3. Entraînement LoRA sur Qwen 3B
  4. Fusion (fuse) des adapters dans un modèle final
  5. Déploiement

Usage :
  python3 src/fine_tune.py --status     # Voir l'état du dataset
  python3 src/fine_tune.py --export     # Exporter au format MLX
  python3 src/fine_tune.py --train      # Lancer l'entraînement LoRA
  python3 src/fine_tune.py --fuse       # Fusionner les adapters
  python3 src/fine_tune.py --full       # Export → Train → Fuse
"""

import json
import sys
import time
from pathlib import Path

DATASET_DIR = Path.home() / ".nuru" / "dataset"
DATASET_FILE = DATASET_DIR / "conversations.jsonl"
MLX_DATA_DIR = Path(__file__).parent.parent / "mlx_dataset"
ADAPTERS_DIR = Path(__file__).parent.parent / "adapters"
MODELS_DIR = Path(__file__).parent.parent / "models"
BASE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# ── Stats ──

def count_entries() -> int:
    if not DATASET_FILE.exists():
        return 0
    return len(DATASET_FILE.read_text().splitlines())

def stats() -> dict:
    total = count_entries()
    good = bad = 0
    by_model = {}
    if total > 0:
        for line in DATASET_FILE.read_text().splitlines():
            entry = json.loads(line)
            fb = entry.get("feedback")
            if fb == "good": good += 1
            elif fb == "bad": bad += 1
            model = entry.get("model", "?")
            by_model[model] = by_model.get(model, 0) + 1
    return {"total": total, "good": good, "bad": bad, "by_model": by_model}

# ── Export MLX ──

def format_for_mlx(output_dir: str = "mlx_dataset") -> str:
    """Convertit le dataset au format {prompt, completion} pour MLX LoRA."""
    entries = [json.loads(l) for l in DATASET_FILE.read_text().splitlines()]
    valid = [e for e in entries if e.get("assistant") and e.get("user")]
    # Filtrer les feedback "bad"
    valid = [e for e in valid if e.get("feedback") != "bad"]

    out_path = Path(output_dir) / "train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for e in valid:
            prompt = f"<|im_start|>user\n{e['user']}<|im_end|>\n<|im_start|>assistant\n"
            completion = f"{e['assistant']}<|im_end|>"
            f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")

    return str(out_path)

# ── Entraînement LoRA ──

def train_lora(iterations: int = 100, lora_layers: int = 8, lr: float = 1e-4):
    """Lance l'entraînement LoRA via mlx_lm.lora.train_model()."""
    from mlx_lm import lora

    # Préparer le dataset
    train_path = Path(MLX_DATA_DIR) / "train.jsonl"
    if not train_path.exists():
        print("  ⚠ Aucun dataset exporté. Lance d'abord --export")
        return False

    # Charger le modèle de base
    print(f"\n📦 Modèle de base : {BASE_MODEL}")
    print(f"📊 Dataset : {train_path}")
    print(f"📈 Itérations : {iterations}")
    print(f"🔧 Couches LoRA : {lora_layers}")
    print()

    # Créer les arguments d'entraînement
    args = lora.TrainingArgs(
        model=BASE_MODEL,
        data=str(MLX_DATA_DIR),
        train=True,
        seed=42,
        lora_layers=lora_layers,
        batch_size=1,
        iters=iterations,
        val_batches=1,
        steps_per_report=10,
        steps_per_eval=20,
        save_every=50,
        adapter_path=str(ADAPTERS_DIR),
        max_seq_length=2048,
        lr=lr,
    )

    # Lancer l'entraînement
    print("🚀 Démarrage de l'entraînement LoRA...")
    print("=" * 50)
    t0 = time.time()

    try:
        lora.train_model(args)
        elapsed = time.time() - t0
        print(f"\n✅ Entraînement terminé en {elapsed:.0f}s")
        print(f"   Adapters : {ADAPTERS_DIR}")
        return True
    except Exception as e:
        print(f"\n❌ Erreur d'entraînement : {e}")
        return False

# ── Fusion des adapters ──

def fuse_adapters(output_name: str = "nuru-v1"):
    """Fusionne les adapters LoRA dans le modèle de base."""
    from mlx_lm import lora

    output_path = MODELS_DIR / output_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🔗 Fusion des adapters → {output_path}")
    print(f"   Modèle base : {BASE_MODEL}")
    print(f"   Adapters     : {ADAPTERS_DIR}")
    print()

    try:
        lora.load(
            model=BASE_MODEL,
            adapter_path=str(ADAPTERS_DIR),
        )
        # La fusion utilise la CLI fuse intégrée
        import subprocess
        cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", BASE_MODEL,
            "--adapter-path", str(ADAPTERS_DIR),
            "--save-path", str(output_path),
            "--de-quantize",
        ]
        print(f"   Commande : {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✅ Fusion terminée : {output_path}")
            return True
        else:
            print(f"❌ Erreur fusion : {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False

# ── CLI ──

if __name__ == "__main__":
    s = stats()
    n = s["total"]

    print(f"{'='*50}")
    print(f"📊 Dataset NURU — {n} échanges")
    print(f"{'='*50}")
    if n > 0:
        print(f"   👍 Good : {s['good']}  |  👎 Bad : {s['bad']}")
        print(f"   Par modèle : {s['by_model']}")
        print(f"   Fichier : {DATASET_FILE}")
    else:
        print("   (vide — les conversations avec NURU remplissent automatiquement)")
    print()

    if "--export" in sys.argv:
        path = format_for_mlx(str(MLX_DATA_DIR))
        print(f"   ✅ Export MLX : {path}")

    if "--train" in sys.argv:
        train_lora()

    if "--fuse" in sys.argv:
        fuse_adapters()

    if "--full" in sys.argv:
        print("🚀 Pipeline complet : Export → Train → Fuse")
        format_for_mlx(str(MLX_DATA_DIR))
        if train_lora():
            fuse_adapters()
