import os
from typing import Any, Dict, List, Tuple
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Define the base URL for the Custom Search JSON API
BASE_URL = "https://www.googleapis.com/customsearch/v1"

GOOGLE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "google_search",
        "description": "Perform a Google search given a search string and a number of search results to fetch",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to be used",
                },
                "num_results": {
                    "type": "integer",
                    "description": "The number of search results to fetch. Defaults to 10.",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
}
PARSE_TEXTS_TOOL = {
    "type": "function",
    "function": {
        "name": "parse_texts_from_webpage",
        "description": "Extracts and returns all text content from a webpage",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage",
                },
            },
            "required": ["url"],
        },
    },
}
PARSE_ANCHORS_TOOL = {
    "type": "function",
    "function": {
        "name": "parse_anchors_from_webpage",
        "description": "Extracts anchor text and href links from a webpage",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage",
                },
            },
            "required": ["url"],
        },
    },
}


# Retry decorator settings
retry_settings = {
    "wait": wait_exponential(multiplier=1, min=4, max=10),
    "stop": stop_after_attempt(3),
}


@retry(**retry_settings)
def parse_texts_from_webpage(url: str) -> str:
    """
    Extracts and returns all text content from a webpage.

    Args:
        url (str): The URL of the webpage to extract text from.

    Returns:
        str: Concatenated text content of the webpage.
    """
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract the text content of the webpage
            texts = " ".join(soup.get_text().split())

            return texts

        else:
            print(f"Failed to retrieve content. Status code: {response.status_code}")
            return ""

    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


@retry(**retry_settings)
def parse_anchors_from_webpage(url: str) -> Dict[str, str]:
    """
    Extracts anchor text and href links from a webpage.

    Args:
        url (str): The URL of the webpage to extract anchor links from.

    Returns:
        Dict[str, str]: A dictionary containing anchor text as keys and href links as values.
            If the href URL is within the same site, only the path without the scheme and hostname is kept.
            The return dictionary also contains a key "_host" with the hostname of the input URL as its value.
            If the href points to a different site, it is kept intact.

    Example:
        >>> extract_anchor_links('https://example.com')
        {
            'Link 1': '/page1',
            'Link 2': '/page2',
            'Link 3': 'https://external-site.com/page3',
            '_host': 'https://example.com'
        }
    """
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract anchor tags with links separately
            anchor_tags = soup.find_all("a")
            parsed_url = urlparse(url)
            anchors = {"_host": f"{parsed_url.scheme}://{parsed_url.hostname}"}

            for tag in anchor_tags:
                key = tag.text.strip()
                href = tag.get("href")
                parsed_href = urlparse(href)

                if key and href and not parsed_href.fragment:
                    hostname = parsed_href.hostname
                    if hostname:
                        if hostname == parsed_url.hostname:
                            anchors[key] = parsed_href.path
                        else:
                            anchors[key] = href
                    else:
                        anchors[key] = href

            return anchors
        else:
            print(f"Failed to retrieve content. Status code: {response.status_code}")
            return {}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def parse_search_item(search_item: Dict[str, Any]) -> Dict[str, str]:
    try:
        long_description = search_item["pagemap"]["metatags"][0]["og:description"]
    except KeyError:
        long_description = "N/A"
    title = search_item.get("title")
    description = search_item.get("snippet")
    link = search_item.get("link")
    return {
        "title": title,
        "description": description,
        "long_description": long_description,
        "link": link,
    }


@retry(**retry_settings)
def search(params: Dict[str, str | int]) -> Tuple[List[Dict[str, str]], int]:
    try:
        # Make the HTTP GET request to the Custom Search JSON API
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad responses

        # Extract search results
        search_results = response.json().get("items", [])
        results = [parse_search_item(search_item) for search_item in search_results]
        next_page = response.json().get("queries", {}).get("nextPage", [])
        next_index = next_page[0].get("startIndex", -1) if next_page else -1

        return results, next_index

    except requests.RequestException as e:
        # Retry on network errors
        print(f"Error making Google search: {e}")
        raise e


def google_search(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Perform a Google search using the Custom Search JSON API.

    Args:
        query (str): The search query to be used.
        num_results (int, optional): The number of search results to fetch.
            Defaults to 10.

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing individual search results.
            Each dictionary contains the following keys:
            - 'title': The title of the search result.
            - 'description': A short description or snippet of the search result.
            - 'long_description': A longer description obtained from metadata, or 'N/A' if not available.
            - 'link': The URL link to the search result.
    """
    results = []
    index = 1
    try:
        while len(results) < num_results and index > 0:
            params = {
                "key": API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "q": query,
                "start": index,
            }
            sub_results, index = search(params)
            results.extend(sub_results)

        results = results[:num_results]
    except Exception as e:
        print(f"Error making Google search: {e}")

    return results


# Example usage
if __name__ == "__main__":
    search_query = "global billionaires tax"
    results = google_search(search_query, 150)
    print(f"{len(results)}")

    if results:
        for idx, result in enumerate(results, start=1):
            print("=" * 10, f"Result #{idx}", "=" * 10)
            print("Title:", result["title"])
            print("Description:", result["description"])
            print("Long description:", result["long_description"])
            print("URL:", result["link"], "\n")
    else:
        print("No search results found.")
