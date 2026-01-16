# utils/services.py
from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from urllib.parse import urlparse

import google.generativeai as genai

DDG_BACKENDS_DEFAULT = ("api", "lite", "html")

try:
    from thefuzz import fuzz, process  # type: ignore
except Exception:
    fuzz = None
    process = None

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
DEFAULT_TIMEOUT_SECONDS = 10
MAX_SCRAPED_CHARS = 6000
DDG_PAUSE_SECONDS = 0.35

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}


# --- HTTP session with retries ---
_session: Optional[requests.Session] = None

def get_http_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session

    s = requests.Session()
    s.headers.update(HEADERS)

    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    except Exception:
        # If retry plumbing is unavailable, proceed without it.
        pass

    _session = s
    return s

def _ddg_text_with_fallback(
    query: str,
    region: str,
    timelimit: str,
    max_results: int,
    backends: Sequence[str],
    debug_log: Optional[List[str]] = None,
):
    last_err = None

    for backend in backends:
        try:
            # DDGS works better as a context manager in hosted environments
            with DDGS() as ddgs:
                res = list(
                    ddgs.text(
                        keywords=query,
                        region=region,
                        timelimit=timelimit,
                        max_results=max_results,
                        backend=backend,  # key change
                    )
                )

            if debug_log is not None:
                debug_log.append(
                    f"DDG backend='{backend}' query='{query}' -> {len(res)} results"
                )

            if res:
                return res

        except Exception as e:
            last_err = e
            if debug_log is not None:
                debug_log.append(
                    f"DDG backend='{backend}' query='{query}' ERROR: {repr(e)}"
                )

    # If everything failed, raise last error (caller will log)
    if last_err is not None:
        raise last_err

    return []

# --- SEARCH ENGINE ---
def run_duckduckgo_search(
    queries: Sequence[str],
    num_results: int = 10,
    filter_func: Optional[Callable[[str, str, str], bool]] = None,
    region: str = "in-en",
    timelimit: str = "y",
    pause_seconds: float = DDG_PAUSE_SECONDS,
    debug_log: Optional[List[str]] = None,
    ddg_backends: Sequence[str] = DDG_BACKENDS_DEFAULT,  # NEW
) -> pd.DataFrame:
    """DuckDuckGo search with optional relevance filter + backend fallback."""
    all_results: List[dict] = []

    for q in queries:
        query = (q or "").strip()
        if not query:
            continue

        try:
            results = _ddg_text_with_fallback(
                query=query,
                region=region,
                timelimit=timelimit,
                max_results=num_results + 10,
                backends=ddg_backends,
                debug_log=debug_log,
            )

            for res in results or []:
                title = res.get("title", "") or ""
                snippet = res.get("body", "") or ""
                link = res.get("href", "") or ""
                if not link:
                    continue

                if _is_junk(title, snippet):
                    continue

                if filter_func and not filter_func(title, snippet, query):
                    continue

                all_results.append(
                    {
                        "Title": title,
                        "Link": link,
                        "Snippet": snippet,
                        "Source Query": query,
                        "Domain": urlparse(link).netloc,
                    }
                )

            if pause_seconds:
                time.sleep(pause_seconds)

        except Exception as e:
            msg = f"Search failure for '{query}': {repr(e)}"
            logger.exception(msg)
            if debug_log is not None:
                debug_log.append(msg)

    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df.drop_duplicates(subset=["Link"]).reset_index(drop=True)
    return df

def _is_junk(title: str, snippet: str) -> bool:
    text = (str(title) + " " + str(snippet)).lower()
    negative_terms = [
        "javascript",
        "css",
        "job description",
        "hiring",
        "driver",
        "software error",
    ]
    return any(term in text for term in negative_terms)

# --- SCRAPER ---
def scrape_url(
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_chars: int = MAX_SCRAPED_CHARS,
) -> Optional[str]:
    """Best-effort content extraction for a single URL."""

    try:
        session = get_http_session()
        r = session.get(url, timeout=timeout_seconds)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.extract()

        text = " ".join(soup.get_text(" ").split())
        if not text:
            return None

        return text[:max_chars]

    except Exception:
        return None

def fetch_content_batch(
    urls_data: Sequence[dict],
    lite_mode: bool = True,
    max_workers: int = 5,
) -> List[dict]:
    """Fetches content for a list of {Link, Title, Snippet} rows.

    Returns list of {title, content, link} suitable for AI processing.
    """

    results: List[dict] = []

    if lite_mode:
        for item in urls_data:
            results.append(
                {
                    "title": item.get("Title", ""),
                    "link": item.get("Link", ""),
                    "content": (
                        f"Title: {item.get('Title','')}\n"
                        f"Snippet: {item.get('Snippet','')}\n"
                        f"Link: {item.get('Link','')}\n"
                        "(Lite Mode: snippet-only)"
                    ),
                }
            )
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(scrape_url, item.get("Link", "")): item for item in urls_data
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            link = item.get("Link", "")
            try:
                content = future.result()
                if not content or len(content) < 200:
                    content = (
                        f"Title: {item.get('Title','')}\n"
                        f"Snippet: {item.get('Snippet','')}\n"
                        f"Link: {link}\n"
                        "(Fallback: scrape failed or content too short)"
                    )
            except Exception:
                content = (
                    f"Title: {item.get('Title','')}\n"
                    f"Snippet: {item.get('Snippet','')}\n"
                    f"Link: {link}\n"
                    "(Fallback: exception during scrape)"
                )

            results.append({"title": item.get("Title", ""), "link": link, "content": content})

    return results

# --- AI ENGINE ---
_last_genai_key: Optional[str] = None


def _ensure_genai_configured(api_key: str) -> None:
    global _last_genai_key
    if not api_key:
        return
    if _last_genai_key == api_key:
        return
    genai.configure(api_key=api_key)
    _last_genai_key = api_key

def extract_entities_with_ai(
    api_key: str,
    content_list: Sequence[dict],
    entity_type: str,
    model_name: str = "gemini-3-flash-preview",
) -> Tuple[List[str], str]:
    """Returns (entities, raw_model_text).

    The model is instructed to output strict JSON.
    """

    _ensure_genai_configured(api_key)

    combined_text = ""
    for i, item in enumerate(content_list):
        combined_text += (
            f"\n--- SOURCE {i+1}: {item.get('title','')} ---\n"
            f"{item.get('content','')}\n"
        )

    prompt = f"""
You are a Financial Market Intelligence Analyst.

Task:
Identify all {entity_type} in India mentioned in the source text.

Rules:
- Extract specific brand / platform names only (e.g., \"GoldenPi\", \"Strata\", \"Wint Wealth\").
- Ignore generic terms (e.g., \"banks\", \"government securities\", \"property\").
- Return unique names.

Output format (STRICT):
Return ONLY valid JSON with this schema:
{{
  \"entities\": [\"Name 1\", \"Name 2\"]
}}
No markdown. No extra text.

Source Text:
{combined_text}
""".strip()

    model = genai.GenerativeModel(model_name)

    try:
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        entities = _parse_entities_json_or_fallback(raw)
        return entities, raw
    except Exception as e:
        raw = f"Error: {str(e)}"
        return [], raw

def _parse_entities_json_or_fallback(raw: str) -> List[str]:
    """Parse strict JSON first; fall back to robust text parsing."""

    if not raw:
        return []

    # 1) Try strict JSON
    try:
        obj = json.loads(raw)
        ents = obj.get("entities", []) if isinstance(obj, dict) else []
        if isinstance(ents, list):
            cleaned = [_clean_entity(e) for e in ents]
            return _unique_keep_order([e for e in cleaned if e])
    except Exception:
        pass

    # 2) Try to extract a JSON object inside extra text
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start : end + 1])
            ents = obj.get("entities", []) if isinstance(obj, dict) else []
            if isinstance(ents, list):
                cleaned = [_clean_entity(e) for e in ents]
                return _unique_keep_order([e for e in cleaned if e])
    except Exception:
        pass

    # 3) Fallback: comma/newline/bullet parsing
    text = raw
    text = re.sub(r"^```.*?$", "", text, flags=re.MULTILINE)
    text = text.replace("\n", ",")

    parts: List[str] = []
    for p in text.split(","):
        p = p.strip()
        if not p:
            continue
        # remove common bullet prefixes
        p = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", p).strip()
        parts.append(p)

    cleaned = [_clean_entity(x) for x in parts]
    cleaned = [x for x in cleaned if x and x.lower() != "none"]
    return _unique_keep_order(cleaned)


def _clean_entity(value: object) -> str:
    s = str(value).strip()
    s = s.strip('"\'')
    s = re.sub(r"\s+", " ", s).strip()
    # Drop extremely short / generic fragments
    if len(s) < 2:
        return ""
    return s

def _unique_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

# --- VERIFICATION ---
def build_registry_index(registry_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a normalized name column for stable matching."""
    if registry_df is None or registry_df.empty:
        return pd.DataFrame(columns=["Entity Name", "SEBI ID", "Website", "Source", "__norm_name"])  # type: ignore

    df = registry_df.copy()
    if "Entity Name" not in df.columns:
        return df

    df["__norm_name"] = (
        df["Entity Name"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return df

def _coerce_url(url: str) -> str:
    u = (url or "").strip()
    if not u or u == "-":
        return "-"
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "https://" + u

def _extract_first_id(text: str, id_regexes: Sequence[str]) -> Optional[str]:
    hay = (text or "")
    for pat in id_regexes:
        try:
            m = re.search(pat, hay, flags=re.IGNORECASE)
            if m:
                return m.group(0)
        except re.error:
            continue
    return None

def generic_sebi_verifier(
    platform_name: str,
    registry_df: pd.DataFrame,
    search_pattern_suffix: str,
    id_regexes: Optional[Sequence[str]] = None,
    min_score: int = 71,
) -> Tuple[str, str, str]:
    """Generic verifier used by modules.

    Returns: (status, id, link)

    - Status is one of: Verified (Official), Verified (Web with ID), Web Mention (No ID), Not Found.
    - Link is either the entity website from official list (if available) or a web-search URL.
    """

    name = (platform_name or "").strip()
    if not name:
        return "❓ Not Found", "-", "-"

    id_regexes = list(id_regexes or [])

    # 1) Official list match
    df = build_registry_index(registry_df)
    if not df.empty and "__norm_name" in df.columns:
        p_clean = re.sub(r"\s+", " ", name.lower()).strip()

        # Fuzzy match if available
        if process is not None and fuzz is not None:
            choices = df["__norm_name"].astype(str).tolist()
            match, score = process.extractOne(p_clean, choices, scorer=fuzz.token_set_ratio)
            if match and score >= min_score:
                row = df[df["__norm_name"] == match].iloc[0]
                website = _coerce_url(str(row.get("Website", "-")))
                return "✅ Verified (Official List)", str(row.get("SEBI ID", "-")).strip(), website

        # Basic containment fallback
        mask = df["__norm_name"].astype(str).str.contains(p_clean, na=False)
        if mask.any():
            row = df[mask].iloc[0]
            website = _coerce_url(str(row.get("Website", "-")))
            return "✅ Verified (Official List)", str(row.get("SEBI ID", "-")).strip(), website

    # 2) Web fallback
    query = f"{name} {search_pattern_suffix}".strip()
    try:
        results = DDGS().text(keywords=query, region="in-en", max_results=3)
        for res in results or []:
            title = res.get("title", "") or ""
            body = res.get("body", "") or ""
            href = res.get("href", "") or "-"

            id_found = None
            if id_regexes:
                id_found = _extract_first_id(title + " " + body, id_regexes)

            if id_found:
                return "✅ Verified (Web Search)", id_found, href

            # If the link is relevant but we cannot extract an ID, mark as a hint.
            if href and href != "-":
                return "⚠️ Web Mention (No ID)", "-", href

    except Exception:
        pass

    return "❓ Not Found", "-", "-"