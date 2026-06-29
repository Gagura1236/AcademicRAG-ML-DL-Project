import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def search_arxiv(keyword: str, max_results: int = 3) -> str:
    """
    Search ArXiv for the given keyword and return the top abstracts.
    """
    try:
        query = urllib.parse.quote(keyword)
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', namespace)
        
        if not entries:
            return "No recent papers found on ArXiv for this keyword."
            
        results = []
        for i, entry in enumerate(entries):
            title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
            results.append(f"[{i+1}] Title: {title}\nAbstract: {summary}\n")
            
        return "\n---\n".join(results)
    except Exception as e:
        return f"Error fetching from ArXiv: {e}"
