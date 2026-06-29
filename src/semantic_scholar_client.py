import requests
import re
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
def enrich_paper(arxiv_id: str) -> dict:
    """
    Query Semantic Scholar to enrich metadata with citation counts, TL;DR, venue, and year.
    Free API, no key needed. Strips version suffix from arXiv ID automatically.
    """
    clean_id = re.sub(r'v\d+$', '', arxiv_id.strip())
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}"
    fields = "citationCount,influentialCitationCount,tldr,venue,year"
    try:
        resp = requests.get(url, params={"fields": fields}, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Semantic Scholar API enrich returned status {resp.status_code} for arXiv:{clean_id}")
            return {}
        d = resp.json()
        return {
            "citation_count": d.get("citationCount", 0),
            "influential_citation_count": d.get("influentialCitationCount", 0),
            "tldr": (d.get("tldr") or {}).get("text", ""),
            "venue": d.get("venue", ""),
            "year": d.get("year"),
        }
    except Exception as e:
        logger.error(f"Failed to query Semantic Scholar for arXiv:{clean_id}: {e}")
        return {}

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
def get_recommendations(arxiv_id: str, limit: int = 3) -> list:
    """
    Gets recommended papers from Semantic Scholar based on the given arXiv ID.
    """
    clean_id = re.sub(r'v\d+$', '', arxiv_id.strip())
    url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/arXiv:{clean_id}"
    try:
        resp = requests.get(url, params={"limit": limit, "fields": "title,externalIds"}, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Semantic Scholar API recs returned status {resp.status_code} for arXiv:{clean_id}")
            return []
        results = []
        for p in resp.json().get("recommendedPapers", []):
            ext = p.get("externalIds", {})
            if "ArXiv" in ext:
                results.append({"arxiv_id": ext["ArXiv"], "title": p.get("title", "")})
        return results
    except Exception as e:
        logger.error(f"Failed to query recommendations for arXiv:{clean_id}: {e}")
        return []
