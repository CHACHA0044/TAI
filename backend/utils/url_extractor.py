import trafilatura
from bs4 import BeautifulSoup
import requests

def extract_content(url_or_text):
    """
    Determines if input is a URL or raw text.
    Extracts content if URL, otherwise returns text.
    """
    if url_or_text.startswith("http://") or url_or_text.startswith("https://"):
        print(f"Extracting content from URL: {url_or_text}")
        downloaded = trafilatura.fetch_url(url_or_text)
        if downloaded:
            content = trafilatura.extract(downloaded)
            if content:
                return content
        
        # Fallback to BeautifulSoup if trafilatura fails
        try:
            response = requests.get(url_or_text, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Basic cleanup
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except:
            return None
    return url_or_text

def get_claims(text, max_claims=3):
    """
    Simplistic claim extraction (first few sentences).
    In a real system, this would use a tailored model.
    """
    # Just take the first few distinct sentences
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    return sentences[:max_claims]
