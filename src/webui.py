#!/usr/bin/env python3
"""
webui.py — Interface holographique NURU (FastAPI).

Serve le tableau de bord cyberpunk (webui_dashboard.html) + API REST.
Usage :
    python3 src/webui.py
    # http://localhost:8080
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import argparse

try:
    import uvicorn
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None

try:
    from rag import VectorStore
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from feedback import FeedbackManager
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False

try:
    from transparency import get_transparency_logger
    TRANSPARENCY_AVAILABLE = True
except ImportError:
    TRANSPARENCY_AVAILABLE = False

try:
    from keychain_utils import get_key, SERVICES, load_config_service
    KEYCHAIN_AVAILABLE = True
except ImportError:
    KEYCHAIN_AVAILABLE = False

# ── Nouveaux modules (dashboard temps réel) ──

try:
    from monitor import get_monitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

try:
    from semantic_cache import SemanticCache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False

try:
    from structured_memory import StructuredMemory
    STRUCTURED_MEMORY_AVAILABLE = True
except ImportError:
    STRUCTURED_MEMORY_AVAILABLE = False

try:
    from router import Router, Level, LEVEL_NAMES
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

try:
    from resource_manager_v2 import get_resource_manager
    from model_pool_v2 import get_model_pool
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# ── Chemins ──
ROOT = Path(__file__).parent.parent
DASHBOARD_PATH = Path(__file__).parent / "dashboard_v2.html"
# Fallback si v2 absent
if not DASHBOARD_PATH.exists():
    DASHBOARD_PATH = Path(__file__).parent / "webui_dashboard.html"
CONFIG_PATH = ROOT / "config" / "config.yaml"

# ── Router singleton (initialisé au premier appel) ──
_router: Optional["Router"] = None
_FORCE_CLOUD = False

def _get_router() -> "Router":
    global _router
    if _router is None:
        from router import Router
        from memory import SessionMemory
        # Utiliser un ID de session stable pour l'interface web pour la persistance
        memory = SessionMemory(session_id="webui_default")
        _router = Router(config_path=str(CONFIG_PATH), memory=memory)
        if _FORCE_CLOUD:
            _router.set_force_cloud(True)
    return _router

app = FastAPI(title="NURU", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    if V2_AVAILABLE:
        try:
            mgr = get_resource_manager()
            mgr.start()
            print("  🚀 NURU V2 Resource Manager démarré")
        except Exception as e:
            print(f"  ⚠ Impossible de démarrer le Resource Manager: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if V2_AVAILABLE:
        try:
            mgr = get_resource_manager()
            mgr.stop()
        except Exception:
            pass

# ── CORS (FastAPI built-in, handles errors) ──

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Tableau de bord holographique."""
    if DASHBOARD_PATH.exists():
        html = DASHBOARD_PATH.read_text(encoding="utf-8")
        # Injecter l'état cloud dans la variable JS state.cloudMode
        if _FORCE_CLOUD:
            html = html.replace(
                "cloudMode: false",
                "cloudMode: true"
            )
            html = html.replace(
                'id="chat-cloud-btn"',
                'id="chat-cloud-btn" class="active"'
            )
        return HTMLResponse(html)
    return HTMLResponse("<h1>NURU</h1><p>Dashboard non trouvé.</p>")


@app.get("/api/stats")
async def api_stats():
    """Statistiques en temps réel."""
    store = None
    if RAG_AVAILABLE:
        try:
            store = VectorStore()
        except Exception:
            pass

    fb = None
    if FEEDBACK_AVAILABLE:
        try:
            fb = FeedbackManager()
        except Exception:
            pass

    return {
        "documents": store.count_documents() if store else 0,
        "corrections": fb.get_stats() if fb else {"total": 0, "active": 0},
        "status": "ok",
        "timestamp": time.time(),
    }


@app.get("/api/state")
async def api_state():
    """État actuel de NURU (pour polling)."""
    return {
        "state": "idle",
        "ram_used_gb": 3.4,
    }


# ── API Dashboard (données réelles) ──


@app.get("/api/perf")
async def api_perf():
    """Statistiques temps réel du monitoring (V1 + V2)."""
    data = {
        "status": "ok",
        "timestamp": time.time(),
        "ram_used": 0.0,
        "v2_enabled": V2_AVAILABLE
    }
    
    # V2 Resource Manager
    if V2_AVAILABLE:
        try:
            mgr = get_resource_manager()
            v2_stats = mgr.get_stats()
            data.update({
                "ram_used": 8.0 - (v2_stats.get("ram_available_gb") or 0.0),
                "ram_available": v2_stats.get("ram_available_gb"),
                "ram_percent": v2_stats.get("ram_percent"),
                "power_mode": v2_stats.get("power_mode"),
                "battery_percent": v2_stats.get("battery_percent"),
                "pressure_level": v2_stats.get("pressure_level", {}).get("label"),
                "mlx_threads": v2_stats.get("mlx_threads"),
            })
        except Exception as e:
            print(f"  [ERROR] V2 Resource Stats: {e}", file=sys.stderr)

    # Legacy monitor (si disponible)
    if MONITOR_AVAILABLE:
        try:
            mon = get_monitor()
            mon_data = mon.get_realtime_stats()
            # Fusionner sans écraser la RAM V2 plus précise
            for k, v in mon_data.items():
                if k not in data or data[k] == 0.0:
                    data[k] = v
        except Exception:
            pass
            
    return data


@app.get("/api/v2/pool")
async def api_v2_pool():
    """État du pool de modèles V2."""
    if not V2_AVAILABLE:
        return {"status": "error", "message": "V2 non disponible"}
    try:
        pool = get_model_pool()
        return {"status": "ok", "data": pool.get_stats()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/timeline")
async def api_timeline():
    """Dernières entrées du journal de transparence."""
    try:
        from transparency import get_transparency_logger
        tl = get_transparency_logger()
        raw_entries = tl.get_entries(20)
        return {"status": "ok", "data": raw_entries}
    except Exception as e:
        print(f"  [ERROR] /api/timeline: {e}", file=sys.stderr)
        return {"status": "error", "message": "Erreur interne"}


@app.get("/api/cache")
async def api_cache():
    """Statistiques du cache sémantique."""
    try:
        if not SEMANTIC_CACHE_AVAILABLE:
            return {"status": "error", "message": "Module cache sémantique non disponible"}
        cache = SemanticCache()
        data = cache.stats()
        return {"status": "ok", "data": data}
    except Exception as e:
        print(f"  [ERROR] /api/cache: {e}", file=sys.stderr)
        return {"status": "error", "message": "Erreur interne"}


@app.get("/api/facts")
async def api_facts():
    """Tous les faits de la mémoire structurée."""
    try:
        if not STRUCTURED_MEMORY_AVAILABLE:
            return {"status": "error", "message": "Module mémoire structurée non disponible"}
        mem = StructuredMemory()
        facts = mem.get_all_facts()
        return {"status": "ok", "data": facts}
    except Exception as e:
        print(f"  [ERROR] /api/facts: {e}", file=sys.stderr)
        return {"status": "error", "message": "Erreur interne"}


async def _route_query(query: str, cloud: bool = False, no_cache: bool = False, web: bool = False) -> dict:
    """Route une requête via le Router et retourne le résultat structuré."""
    if not ROUTER_AVAILABLE:
        return {"status": "error", "message": "Module routeur non disponible"}
    router = _get_router()
    # Save original state to avoid race conditions on shared singleton
    old_cloud = router.force_cloud
    old_web = router.web_search_mode
    old_cache = router.use_semantic_cache
    try:
        if cloud or _FORCE_CLOUD:
            router.set_force_cloud(True)
        if web:
            router.set_web_search(True)
        if no_cache:
            router.use_semantic_cache = False
        result = router.route(query, user_confirmed_cloud=cloud or _FORCE_CLOUD)
    finally:
        router.set_force_cloud(old_cloud)
        router.set_web_search(old_web)
        router.use_semantic_cache = old_cache
    return {
        "response": result.content,
        "level": result.level,
        "level_name": result.level_name or LEVEL_NAMES.get(result.level, f"Niveau {result.level}"),
        "latency_ms": result.latency_ms,
        "model_used": result.model_used or "inconnu",
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    """Chat via POST — corps JSON {query, cloud?, no_cache?, web?}."""
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        cloud = body.get("cloud", False)
        no_cache = body.get("no_cache", False)
        web = body.get("web", False)
        if not query:
            return {"status": "error", "message": "Le champ 'query' est requis"}
        data = await _route_query(query, cloud=cloud, no_cache=no_cache, web=web)
        return {"status": "ok", "data": data}
    except Exception as e:
        print(f"  [ERROR] /api/chat: {e}", file=sys.stderr)
        return {"status": "error", "message": "Erreur interne"}


@app.get("/api/chat/stream")
async def api_chat_stream(q: str, cloud: str = "false", no_cache: str = "false", web: str = "false"):
    """Chat en streaming via Server-Sent Events (SSE)."""
    if not q:
        raise HTTPException(400, "Le paramètre 'q' est requis")
    
    # Conversion explicite des booléens car EventSource envoie des strings
    is_cloud = cloud.lower() in ("true", "1", "yes")
    is_no_cache = no_cache.lower() in ("true", "1", "yes")
    is_web = web.lower() in ("true", "1", "yes")

    import asyncio
    import functools

    router = _get_router()
    # Save original state to avoid race conditions on shared singleton
    old_cloud = router.force_cloud
    old_web = router.web_search_mode
    old_cache = router.use_semantic_cache
    try:
        if is_cloud:
            router.set_force_cloud(True)
        if is_web:
            router.set_web_search(True)
        if is_no_cache:
            router.use_semantic_cache = False

        async def event_generator():
            try:
                print(f"  [STREAM] Requête (Live) : {q[:50]}...", file=sys.stderr)
                loop = asyncio.get_event_loop()
                sync_gen = router.stream_route(q, user_confirmed_cloud=is_cloud)

                def get_next_chunk():
                    try:
                        return next(sync_gen)
                    except StopIteration:
                        return None
                    except Exception as e:
                        return e

                while True:
                    # Exécuter la prochaine itération dans un thread pour ne pas bloquer l'event loop
                    chunk = await loop.run_in_executor(None, get_next_chunk)
                    
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        print(f"  [STREAM ERROR] {chunk}", file=sys.stderr)
                        yield f"data: {json.dumps({'type': 'error', 'message': str(chunk)}, ensure_ascii=False)}\n\n"
                        break
                    
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    # Petit délai pour laisser respirer l'event loop si nécessaire
                    await asyncio.sleep(0.01)

            except Exception as e:
                print(f"  [ERROR] Event Generator: {e}", file=sys.stderr)
                yield f"data: {json.dumps({'type': 'error', 'message': 'Erreur de flux'}, ensure_ascii=False)}\n\n"
    finally:
        router.set_force_cloud(old_cloud)
        router.set_web_search(old_web)
        router.use_semantic_cache = old_cache

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/query")
async def api_query(request: Request):
    """Chat via GET — paramètres ?q=&cloud=1&no_cache=1&web=1."""
    try:
        query = request.query_params.get("q", "").strip()
        cloud = request.query_params.get("cloud", "0") in ("1", "true", "yes")
        no_cache = request.query_params.get("no_cache", "0") in ("1", "true", "yes")
        web = request.query_params.get("web", "0") in ("1", "true", "yes")
        if not query:
            return {"status": "error", "message": "Le paramètre 'q' est requis"}
        data = await _route_query(query, cloud=cloud, no_cache=no_cache, web=web)
        return {"status": "ok", "data": data}
    except Exception as e:
        print(f"  [ERROR] /api/query: {e}", file=sys.stderr)
        return {"status": "error", "message": "Erreur interne"}


# ── Pages de gestion ──

def _page_template(title: str, content: str) -> str:
    """Template minimal pour les pages de gestion."""
    return f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title} — NURU</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:#0a0a0f; color:#c0d0e0; padding:32px;
  }}
  a {{ color:#00b4ff; text-decoration:none; font-size:13px; }}
  a:hover {{ color:#00d4aa; }}
  h1 {{ font-family:'Orbitron',sans-serif; font-size:18px; color:#00d4aa; margin-bottom:20px; }}
  .card {{ background:rgba(16,20,35,0.7); border:1px solid rgba(0,180,255,0.08); border-radius:12px; padding:20px; margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:10px 12px; text-align:left; border-bottom:1px solid rgba(0,180,255,0.06); }}
  th {{ color:rgba(0,180,255,0.4); font-size:11px; text-transform:uppercase; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:6px; font-size:11px; }}
  .badge.ok {{ background:rgba(0,212,170,0.1); color:#00d4aa; }}
  .badge.warn {{ background:rgba(255,170,0,0.1); color:#ffaa00; }}
  .btn {{ padding:6px 14px; border-radius:6px; border:none; cursor:pointer; font-size:12px; background:rgba(0,212,170,0.15); color:#00d4aa; }}
  .btn.danger {{ background:rgba(255,68,68,0.15); color:#ff4444; }}
  input,textarea,select {{ width:100%; padding:10px; border-radius:8px; border:1px solid rgba(0,180,255,0.1); background:rgba(0,0,0,0.3); color:#fff; font-size:13px; margin-bottom:12px; }}
  label {{ font-size:11px; color:rgba(0,180,255,0.4); margin-bottom:4px; display:block; }}
  pre {{ background:rgba(0,0,0,0.3); padding:16px; border-radius:8px; font-size:12px; overflow-x:auto; }}
  .nav {{ display:flex; gap:12px; margin-bottom:24px; }}
  .nav a {{ padding:6px 12px; border-radius:6px; background:rgba(0,180,255,0.03); border:1px solid rgba(0,180,255,0.05); }}
  .nav a:hover {{ background:rgba(0,180,255,0.08); }}
</style></head>
<body>
  <div class="nav">
    <a href="/">⬅ Tableau de bord</a>
    <a href="/corrections">Corrections</a>
    <a href="/keychain">Clés API</a>
    <a href="/config">Configuration</a>
  </div>
  <h1>{title}</h1>
  {content}
</body></html>"""


@app.get("/corrections", response_class=HTMLResponse)
async def corrections_page():
    fb = FeedbackManager() if FEEDBACK_AVAILABLE else None
    corrections = fb.get_corrections(include_disabled=True) if fb else []

    rows = ""
    for c in corrections:
        disabled = c.get("disabled", False)
        status = "🔇 Désactivée" if disabled else "✅ Active"
        rows += f"""<tr>
            <td style="font-family:monospace;font-size:11px;">{c['id'][:16]}..</td>
            <td>{c['query'][:40]}...</td>
            <td>{c['correction'][:40]}...</td>
            <td>{status}</td>
            <td>
                <a href="/api/corrections/toggle/{c['id']}" class="btn">{'Réactiver' if disabled else 'Désactiver'}</a>
                <a href="/api/corrections/delete/{c['id']}" class="btn danger">🗑</a>
            </td>
        </tr>"""

    return _page_template("Corrections", f"""
    <div class="card">
        <table>
            <thead><tr><th>ID</th><th>Requête</th><th>Correction</th><th>Statut</th><th>Actions</th></tr></thead>
            <tbody>{rows or '<tr><td colspan="5" style="text-align:center;color:rgba(0,180,255,0.2);">Aucune correction</td></tr>'}</tbody>
        </table>
    </div>
    <div class="card">
        <form action="/api/corrections/add" method="post">
            <label>Requête déclencheuse</label>
            <input type="text" name="query" placeholder="Ex: Quel est mon langage préféré ?" required>
            <label>Réponse correcte</label>
            <textarea name="correction" rows="3" placeholder="Ex: Python." required></textarea>
            <button type="submit" class="btn">➕ Ajouter</button>
        </form>
    </div>
    """)


@app.get("/keychain", response_class=HTMLResponse)
async def keychain_page():
    if not KEYCHAIN_AVAILABLE:
        return _page_template("Clés API", "<p>Keychain non disponible.</p>")
    service = load_config_service()
    rows = ""
    for name, desc in SERVICES.items():
        value = get_key(service, name)
        status = '<span class="badge ok">Configurée</span>' if value else '<span class="badge warn">Non définie</span>'
        masked = (value[:4] + "••••" + value[-4:]) if value and len(value) > 8 else "••••••" if value else "—"
        rows += f"<tr><td style='font-family:monospace'>{name}</td><td style='color:rgba(0,180,255,0.5)'>{desc}</td><td>{masked}</td><td>{status}</td></tr>"
    return _page_template("Clés API", f"""
    <div class="card">
        <table><thead><tr><th>Service</th><th>Description</th><th>Clé</th><th>Statut</th></tr></thead>
        <tbody>{rows}</tbody></table>
    </div>
    <div class="card">
        <p style="color:rgba(0,180,255,0.3);font-size:12px;">
        python3 src/keychain_utils.py --set gemini<br>
        python3 src/keychain_utils.py --set deepseek
        </p>
    </div>
    """)


@app.get("/config", response_class=HTMLResponse)
async def config_page():
    cfg = {}
    if CONFIG_PATH.exists() and yaml:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    raw = yaml.dump(cfg, default_flow_style=False) if yaml else str(cfg)
    return _page_template("Configuration", f'<div class="card"><pre>{raw}</pre></div>')


# ── API ──

@app.post("/api/corrections/add")
async def api_add_correction(request: Request):
    form = await request.form()
    query = form.get("query")
    correction = form.get("correction")
    if not query or not correction:
        raise HTTPException(400, "query et correction requis")
    fb = FeedbackManager() if FEEDBACK_AVAILABLE else None
    if fb:
        fb.add_correction(query, correction)
        return HTMLResponse('<script>window.location.href="/corrections"</script>')
    raise HTTPException(500, "Feedback non disponible")


@app.get("/api/corrections/delete/{corr_id}")
@app.get("/api/corrections/toggle/{corr_id}")
async def api_correction_action(corr_id: str, request: Request):
    fb = FeedbackManager() if FEEDBACK_AVAILABLE else None
    action = "delete" if "delete" in request.url.path else "toggle"
    if fb:
        if action == "delete":
            fb.delete_correction(corr_id)
        else:
            fb.toggle_correction(corr_id)
        return HTMLResponse('<script>window.location.href="/corrections"</script>')
    raise HTTPException(500, "Feedback non disponible")


def main():
    global _FORCE_CLOUD
    if not FASTAPI_AVAILABLE:
        print("⚠ FastAPI/Uvicorn non installé")
        return

    parser = argparse.ArgumentParser(description="NURU — Interface Holographique (WebUI)")
    parser.add_argument("--cloud", "-c", action="store_true", help="Forcer le cloud (Deepseek/Gemini)")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port d'écoute (défaut: 8080)")
    args = parser.parse_args()

    if args.cloud:
        _FORCE_CLOUD = True
        print("  ☁️  Mode cloud forcé activé")

    port = args.port
    print(f"\n{'='*50}")
    print(f"  ⚡ NURU — Interface Holographique")
    print(f"  🌐 http://localhost:{port}")
    print(f"{'='*50}")
    print(f"  Dashboard    → http://localhost:{port}/")
    print(f"  Corrections  → http://localhost:{port}/corrections")
    print(f"  Clés API     → http://localhost:{port}/keychain")
    print(f"  Config       → http://localhost:{port}/config")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
