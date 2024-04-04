import requests
from bs4 import BeautifulSoup


def scrape_website(url):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract the text content of the webpage
            texts = " ".join(soup.get_text().split())

            # Extract anchor tags with links separately
            anchor_tags = soup.find_all("a")
            anchors = [(tag.text.strip(), tag.get("href")) for tag in anchor_tags]

            return texts, anchors

        else:
            print(f"Failed to retrieve content. Status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


# Example usage:
url = "https://edition.cnn.com/2024/03/02/politics/jill-biden-2024-campaign/index.html"
texts, anchors = scrape_website(url)

if texts and anchors:
    print(texts)
    print("=" * 20)
    for anchor in anchors:
        print(anchor)
