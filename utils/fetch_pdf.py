import requests
from io import BytesIO


def fetch_pdf_content(url: str):
    """Fetches PDF content from a specified URL."""
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)
