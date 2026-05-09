import httpx
import os
import time
from dataclasses import dataclass

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


# ─────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────

@dataclass
class SearchResult:
    title: str
    url: str
    description: str


@dataclass
class BraveSearchResponse:
    success: bool
    results: list[SearchResult]
    raw_context: str
    error: str | None = None


# ─────────────────────────────────────────────
# Client Brave Search
# ─────────────────────────────────────────────

class BraveSearchClient:
    # Cache TTL pour éviter les appels redondants (5 minutes)
    _CACHE_TTL = 300  # secondes
    _cache: dict[str, tuple[float, BraveSearchResponse]] = {}

    def __init__(
        self,
        api_key: str = "",
        max_results: int = 5,
        country: str = "fr",
        language: str = "fr",
        timeout: float = 8.0,
        max_retries: int = 2,
    ):
        self.api_key     = api_key
        self.max_results = max_results
        self.country     = country
        self.language    = language
        self.timeout     = timeout
        self.max_retries = max_retries

    def _clear_expired_cache(self):
        """Nettoie les entrées expirées du cache."""
        now = time.time()
        expired = [q for q, (ts, _) in self._cache.items() if now - ts > self._CACHE_TTL]
        for q in expired:
            del self._cache[q]

    def search(self, query: str) -> BraveSearchResponse:
        # ── Vérifier le cache TTL ──
        self._clear_expired_cache()
        if query in self._cache:
            ts, response = self._cache[query]
            print(f"  📦 Brave Search : réponse en cache ({int(time.time() - ts)}s)", file=__import__('sys').stderr)
            return response

        if not self.api_key:
            return BraveSearchResponse(
                success=False, results=[], raw_context="",
                error="BRAVE_API_KEY manquant. Configurez la clé API Brave."
            )

        # ── Tentatives avec retry ──
        last_error = None
        for attempt in range(1, self.max_retries + 2):  # +1 pour la première tentative
            try:
                response = httpx.get(
                    BRAVE_ENDPOINT,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self.api_key,
                    },
                    params={
                        "q":                query,
                        "count":            self.max_results,
                        "country":          self.country,
                        "search_lang":      self.language,
                        "text_decorations": False,
                        "spellcheck":       True,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                web_results = data.get("web", {}).get("results", [])
                results = [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        description=r.get("description", ""),
                    )
                    for r in web_results
                ]

                raw_context = self._format_for_prompt(query, results)

                resp = BraveSearchResponse(
                    success=True,
                    results=results,
                    raw_context=raw_context,
                )

                # Mettre en cache
                self._cache[query] = (time.time(), resp)
                return resp

            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError) as e:
                last_error = e
                if attempt <= self.max_retries:
                    sleep_time = 1.5 ** attempt  # backoff exponentiel: 1.5s, 2.25s
                    print(f"  ⚠ Brave retry {attempt}/{self.max_retries} dans {sleep_time:.1f}s : {e}",
                          file=__import__('sys').stderr)
                    time.sleep(sleep_time)
                else:
                    break
            except Exception as e:
                last_error = e
                break

        error_msg = f"Timeout après {self.max_retries + 1} tentatives" if isinstance(last_error, httpx.TimeoutException) else f"Erreur : {str(last_error)}"
        return BraveSearchResponse(
            success=False, results=[], raw_context="",
            error=error_msg
        )

    def _format_for_prompt(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return "Aucun résultat trouvé pour cette recherche."

        lines = [f"Résultats de recherche web pour : « {query} »\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.title}\n"
                f"    URL : {r.url}\n"
                f"    Résumé : {r.description}\n"
            )
        return "\n".join(lines)
