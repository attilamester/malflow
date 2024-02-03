import pprint
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

FELIXCLOUTIER_URL = "https://www.felixcloutier.com/"
FELIXCLOUTIER_URL_X86 = "https://www.felixcloutier.com/x86/"


def parse_felixcloutier():
    response = requests.get(FELIXCLOUTIER_URL_X86)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")
    info = {}
    mnemonics_to_process = []
    for row in rows:
        cols = row.find_all("td")
        if not cols:  # header
            continue
        mnemonic = cols[0].text.strip().lower()
        link = cols[0].find("a")["href"]

        mnemonics_to_process.append((mnemonic, link))

    with ThreadPoolExecutor(max_workers=8) as executor:
        i = 0
        for res in executor.map(lambda p: parse_felixcloutier_mnemonic(*p), [(m, l) for m, l in mnemonics_to_process]):
            i += 1
            print(f"{i}: {res}")
            info.update(res)

    return info


def parse_felixcloutier_mnemonic(mnemonic: str, link: str) -> dict:
    link = link.strip("/")
    url = f"{FELIXCLOUTIER_URL}{link}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find_all("table")[1]
    rows = table.find_all("tr")
    operand_count = 0

    operand_column_indices_weight = []
    for i, th in enumerate(rows[0].find_all("th")):
        if "Operand" in th.text and not "Operands" in th.text:
            operand_column_indices_weight.append((i, 1))
        elif "Operands" in th.text:
            if "—" in th.text:
                tokens = th.text.replace("Operands", "").strip().split("—")
                weight = int(tokens[1]) - int(tokens[0]) + 1
                operand_column_indices_weight.append((i, weight))

    for row in rows[1:]:
        cols = row.find_all("td")
        operands = 0
        for i, w in operand_column_indices_weight:
            if cols[i].text.strip() not in {"N/A", "", None}:
                operands += w

        operand_count = max(operand_count, operands)

    return {mnemonic: {"max_operands": operand_count}}


if __name__ == "__main__":
    mnemonic_info = parse_felixcloutier()
    pprint.pprint(mnemonic_info)
