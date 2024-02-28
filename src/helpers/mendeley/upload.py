import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup

# https://api.mendeley.com/apidocs/docs
ACCESS_TOKEN = ""


# ==================
# Model
# ==================
@dataclass
class MendeleyPaper:
    title: str
    sanitized_title: str
    year: int
    tags: List[str]

    def __init__(self, title, year, tags):
        self.title = title
        self.sanitized_title = MendeleyPaper.sanitize_title(title)
        self.year = year
        self.tags = tags

    @staticmethod
    def sanitize_title(title):
        return (title.replace(" ", "_").replace(":", "_").replace("-", "_").replace(",", "_").replace("/", "_")
                .replace("\\", "_").replace("?", "_").replace("*", "_").replace("\"", "_").replace("<", "_")
                .replace(">", "_").replace("|", "_"))


# ==================
# Util
# ==================
def send_mendeley_api_request(method, endpoint, **kwargs):
    request_method = requests.get
    if method == "patch":
        request_method = requests.patch
    elif method == "post":
        request_method = requests.post
    elif method == "get":
        request_method = requests.get
    if not request_method:
        raise Exception(f"Invalid method: {method}")
    # res = request_method("https://api.mendeley.com" + ("" if endpoint.startswith("/") else "/") + endpoint, **kwargs)
    # display_response(res)
    # return res


def display_response(response: requests.Response):
    print(f"{response.request.method} {response.request.url} : {response.status_code}")
    if response.status_code >= 300:
        print(response.content)


def upload_papers(papers: List[MendeleyPaper]):
    i = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_and_upload_to_mendeley, papers):
            i += 1
            print(f"Done: {i}")


def mendeley_xml_export_to_csv(path):
    soup = BeautifulSoup(open(path, "r").read(), "xml")
    sources = soup.find_all("Source")

    def get_field(source, field):
        try:
            return source.find(field).text
        except:
            return ""

    for i, source in enumerate(sources):
        title = get_field(source, "Title")
        year = get_field(source, "Year")
        tag = get_field(source, "Tag")
        print(f"{year};{title};{tag}")


# ==================
# SciHub
# ==================
def search_scihub(title):
    download_path = f"./papers/{MendeleyPaper.sanitize_title(title)}.pdf"
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


def get_paper_tags(json_paper):
    tags = []
    for key, tag_key in [("algorithms", "ALGORITHM"), ("features", "FEATURE"), ("objectives", "OBJECTIVE")]:
        if key in json_paper:
            if isinstance(json_paper[key], str):
                tags.append(f"{tag_key}:{json_paper[key]}")
            elif isinstance(json_paper[key], list):
                for tag in json_paper[key]:
                    tags.append(f"{tag_key}:{tag}")

    if "dset_benign" in json_paper:
        tags.append(f"DATASET:benign_{json_paper['dset_benign']}")
    if "dset_malicious" in json_paper:
        tags.append(f"DATASET:malicious_{json_paper['dset_malicious']}")
    return tags


def process_and_upload_to_mendeley(paper: MendeleyPaper):
    headers = {"Authorization": "Bearer " + ACCESS_TOKEN}
    file_upload_success = False

    download_path = search_scihub(paper.title)
    if download_path:
        files = {"upload_file": open(download_path, "rb")}
        res = send_mendeley_api_request("post", "/documents", files=files, headers={
            **headers, **{"Content-Disposition": f"attachment; filename='{os.path.basename(download_path)}'"}
        })
        if res.status_code == 201:
            response = res.json()
            file_upload_success = True

    paper_data = {
        "title": paper.title,
        "type": "journal",

        "year": paper.year,
        "tags": paper.tags,
    }

    headers = {**headers, **{"Content-Type": "application/vnd.mendeley-document.1+json"}}
    if not file_upload_success:
        send_mendeley_api_request("post", "/documents", json=paper_data, headers=headers)
    else:
        paper_data = {
            "title": paper.title,
            "year": paper.year,
            "tags": paper.tags,
        }
        send_mendeley_api_request("patch", f"/documents/{response['id']}", json=paper_data, headers=headers)


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

    titles = {}
    papers = []
    for paper in data:
        if not "title" in paper:
            print(f"No title found in paper {paper}")
            continue

        sanitized_title = MendeleyPaper.sanitize_title(paper["title"])
        if sanitized_title in titles:
            print(f"Duplicate: {paper['title']} "
                  f"\n\tEXISTING {titles[sanitized_title]}"
                  f"\n\tNEW      {paper}")
        else:
            titles[sanitized_title] = paper
            papers.append(MendeleyPaper(paper["title"], paper["year"], get_paper_tags(paper)))
    upload_papers(papers)


def upload_from_buffer():
    buff = """
"""
    papers = []
    for line in buff.splitlines():
        line = line.strip()
        if not line:
            continue
        year, title = line.split(maxsplit=1)
        year = int(year.strip("[").strip("]"))
        papers.append(MendeleyPaper(title, year, []))

    upload_papers(papers)


if __name__ == "__main__":
    pass
    # upload_from_json("/home/amester/Data/__master_thesis_ti-clustering/bibliography/research_papers_latex.json")
    # upload_from_buffer()
