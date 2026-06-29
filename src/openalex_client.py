import requests
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

class OpenAlexClient:
    """
    v2.3.3: OpenAlex Global Citation Integration.
    Fetches seminal papers, their references (upstream), and citations (downstream)
    from the OpenAlex open API without requiring an API key.
    """
    def __init__(self):
        self.base_url = "https://api.openalex.org"
        self.headers = {"User-Agent": "AcademicRAG/v2.3.3 (mailto:developer@academicrag.local)"}

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _get_with_retry(self, url: str, params: dict = None, timeout: int = 10):
        res = requests.get(url, params=params, headers=self.headers, timeout=timeout)
        res.raise_for_status()
        return res

    def get_citation_tree(self, concept: str, strategy: str = "relevance") -> dict:
        """
        Searches for the concept, finds the seminal paper based on strategy,
        and builds a citation tree (upstream, core, downstream) of real paper titles.
        """
        tree = {
            "upstream": "OpenAlex 網路連線失敗或找不到相關文獻。",
            "core": f"無法在 OpenAlex 上找到關於 '{concept}' 的核心權威文獻。",
            "downstream": "OpenAlex 網路連線失敗或找不到相關文獻。"
        }
        
        try:
            # 1. Search for top paper related to the concept
            base_filter = "is_retracted:false,is_paratext:false"
            
            search_params = {
                "search": concept,
                "per-page": 1
            }
            
            if strategy == "impact":
                search_params["sort"] = "cited_by_count:desc"
                search_params["filter"] = f"{base_filter},cited_by_count:>100"
            elif strategy == "recent":
                search_params["sort"] = "publication_year:desc"
                search_params["filter"] = f"{base_filter},has_doi:true,cited_by_count:>0"
            else:
                search_params["filter"] = f"{base_filter},cited_by_count:>10"
            
            res = self._get_with_retry(f"{self.base_url}/works", params=search_params, timeout=10)
            data = res.json()
            
            if not data.get("results"):
                return tree
                
            seminal_paper = data["results"][0]
            seminal_id = seminal_paper["id"]
            seminal_title = seminal_paper.get("title", "")
            
            if not seminal_title:
                return tree
                
            pub_year = seminal_paper.get('publication_year', 'N/A')
            tree["core"] = f"[Global Seminal Work] {seminal_title} ({pub_year}, Citations: {seminal_paper.get('cited_by_count', 0)})"
            
            # 2. Get upstream (References)
            referenced_works = seminal_paper.get("referenced_works", [])
            if referenced_works:
                upstream_titles = []
                for ref_id in referenced_works[:3]: # limit to 3 to save time
                    try:
                        ref_res = self._get_with_retry(ref_id, timeout=5)
                        ref_data = ref_res.json()
                        title = ref_data.get("title")
                        if title:
                            upstream_titles.append(f"• {title} ({ref_data.get('publication_year', '')})")
                    except Exception as ref_err:
                        logger.warning(f"Failed to fetch reference {ref_id}: {ref_err}")
                if upstream_titles:
                    tree["upstream"] = "Foundational references:\n" + "\n".join(upstream_titles)
                else:
                    tree["upstream"] = "找不到這篇核心論文的參考文獻標題。"
            else:
                tree["upstream"] = "這篇核心論文沒有提供參考文獻列表。"
                
            # 3. Get downstream (Citations)
            cite_params = {
                "filter": f"cites:{seminal_id.split('/')[-1]}",
                "per-page": 3
            }
            if strategy == "impact":
                cite_params["sort"] = "cited_by_count:desc"
            elif strategy == "recent":
                cite_params["sort"] = "publication_year:desc"
                
            try:
                cite_res = self._get_with_retry(f"{self.base_url}/works", params=cite_params, timeout=10)
                cite_data = cite_res.json()
                downstream_titles = []
                for doc in cite_data.get("results", []):
                    title = doc.get("title")
                    if title:
                        downstream_titles.append(f"• {title} ({doc.get('publication_year', '')})")
                
                if downstream_titles:
                    tree["downstream"] = "Succeeding works:\n" + "\n".join(downstream_titles)
                else:
                    tree["downstream"] = "目前沒有找到相關的後續引用文獻。"
            except Exception as cite_err:
                logger.warning(f"Failed to fetch citations for seminal work: {cite_err}")
            
            return tree
            
        except Exception as e:
            logger.error(f"OpenAlex API error: {e}")
            return tree
