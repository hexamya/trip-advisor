import requests


def map_search_api(term: str, lat: str, lng: str) -> str:

    url = "https://app.radeai.com/tools/neshan/search/"

    payload = {
        'term': term,
        'lat': lat,
        'lng': lng
    }

    files = []
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files, verify=False)

    return response.text


def google_search_api(query: str, include_content: bool = True) -> str:
    url = "https://app.radeai.com/tools/google/search/"

    payload = {
        'query': query,
        'include_content': include_content,
    }

    files = []
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files, verify=False)

    return response.text
