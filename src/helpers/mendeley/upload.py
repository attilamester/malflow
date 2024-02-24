import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

# https://api.mendeley.com/apidocs/docs
ACCESS_TOKEN = ""


def sanitize_title(title):
    return title.replace(" ", "_").replace(":", "_").replace("-", "_").replace(",", "_").replace("/", "_").replace(
        "\\", "_").replace("?", "_").replace("*", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace(
        "|", "_")


def search_scihub(title):
    download_path = f"./papers/{sanitize_title(title)}.pdf"
    if os.path.exists(download_path):
        return download_path
    res = requests.post("https://sci-hub.se/", data={"request": title}, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    })
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, "html.parser")
        try:
            embed = soup.find(id="article").find("embed")
        except Exception as e:
            print(f"Error: could not download sci-hub paper for {title} | {e}")
            return None

        url = embed.get("src").strip()
        url = url.strip("/")
        url = f"https://{url}"
        for i in range(3):
            try:
                urllib.request.urlretrieve(url, download_path)
                return download_path
            except Exception as e:
                print(f"Error: could not download sci-hub paper for {title} | {e}")
                time.sleep(1)
                pass
        return None
    return None


def display_response(response: requests.Response):
    print(f"{response.request.method} {response.request.url} : {response.status_code}")
    if response.status_code >= 300:
        print(response.content)


def get_paper_tags(paper):
    tags = []
    for tag in paper.get("algorithms", []):
        tags.append(f"ALGORITHM:{tag}")
    for tag in paper.get("features", []):
        tags.append(f"FEATURE:{tag}")
    if "objective" in paper:
        tags.append(f"OBJECTIVE:{paper['objective']}")
    if "dset_benign" in paper:
        tags.append(f"DATASET:benign_{paper['dset_benign']}")
    if "dset_malicious" in paper:
        tags.append(f"DATASET:malicious_{paper['dset_malicious']}")
    return tags


def process_and_upload_to_mendeley(paper):
    if "title" not in paper:
        print(f"Title not found in paper: {paper}")
        return

    headers = {"Authorization": "Bearer " + ACCESS_TOKEN}
    file_upload_success = False

    download_path = search_scihub(paper["title"])
    if download_path:
        files = {"upload_file": open(download_path, "rb")}
        res = requests.post("https://api.mendeley.com/documents", files=files, headers={
            **headers, **{"Content-Disposition": f"attachment; filename='{os.path.basename(download_path)}'"}
        })
        display_response(res)
        if res.status_code == 201:
            response = res.json()
            file_upload_success = True

    paper_data = {
        "title": paper["title"],
        "type": "journal",

        "year": paper["year"],
        "tags": paper["algorithms"],
        "keywords": paper["algorithms"]
    }

    headers = {**headers, **{"Content-Type": "application/vnd.mendeley-document.1+json"}}
    if not file_upload_success:
        res = requests.post("https://api.mendeley.com/documents", json=paper_data, headers=headers)
        display_response(res)
    else:
        paper_data = {
            "tags": get_paper_tags(paper),
        }
        res = requests.patch(f"https://api.mendeley.com/documents/{response['id']}", json=paper_data,
                             headers=headers)
        display_response(res)


def upload_from_json(path):
    """
    https://api.mendeley.com/apidocs/docs#!/documents/createDocument
    {
      "authors": [
        {
          "first_name": "",
          "last_name": "",
        }
      ],
      "keywords": [
        ""
      ],
      "publisher": "",
      "tags": [
        ""
      ],
      "title": "",
      "year": 0
    }
    :param path:
    :return:
    """

    with open(path, "r") as f:
        data = json.load(f)
    i = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_and_upload_to_mendeley, data):
            i += 1
            print(f"Done: {i}")
