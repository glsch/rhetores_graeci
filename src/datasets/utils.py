import copy
import json
import os
import re

from tqdm import tqdm

from bs4 import BeautifulSoup, Tag
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from ..PathManager import PathManager

text_children = {}

def prepare_attribution_df():
    pass

def prepare_verification_df():
    pass

def read_card(tag: Tag) -> dict:
    work_dict = {"title": None, "edition": None}

    title = tag.find("p", class_="WorkInfo")

def parse_tr(tr: Tag, level_names: list, n_cols: int = 3) -> dict:
    global text_children

    text_row_dict = {"text": []}
    cells = []
    for t in tr.children:
        if isinstance(t, Tag):
            cells.append(t)

    for i, c in enumerate(cells):
        if c.name == "td":
            if c.has_attr("class") and c["class"] == ["H"]:
                heads = c.find_all(re.compile(r"h\d"))
                for j, h in enumerate(heads):
                    text_row_dict.update({f"{h.name}": h.text})

            elif c.has_attr("class") and c["class"] in (["K"], ["R"]):
                continue

            elif c.has_attr("class") and c["class"] == ["M"]:
                for s in c.stripped_strings:
                    text_strings = [ts.replace("[", "\[").replace("]", "\]") for ts in s.split("\n") if
                                    ts.strip()]
                    for ts in text_strings:
                        text_row_dict["text"] = text_row_dict.get("text", []) + ["<speaker>" + ts + "</speaker>"]

            # elif c.has_attr("class"):
            #     print("Unknown class", c["class"])

            else:
                if c.has_attr("class"):
                    text_children["unknown_class"] = text_children.get("unknown_class", [])
                    if not c["class"] in text_children["unknown_class"]:
                        # print(f"Unknown class: {c['class']}")
                        text_children["unknown_class"] += [c["class"]]

                for t in c.descendants:
                    if isinstance(t, Tag):
                        text_children[t.name] = set([atr for atr in t.attrs])

                for s in c.stripped_strings:
                    text_strings = [ts.replace("[", "\[").replace("]", "\]") for ts in s.split("\n") if ts.strip()]
                    text_row_dict["text"] = text_row_dict.get("text", []) + text_strings


        elif c.name == "th":
            continue

    # print(text_row_dict)
    return text_row_dict

def main():
    global text_children

    service = ChromeService(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=service, options=options)
    tlg_path = os.path.join(PathManager.data_path, "TLG-E")
    index_path = os.path.join(tlg_path, "indices", "index.htm")
    index_path_html_file_path = os.path.abspath(index_path)

    driver.get(f'file://{index_path_html_file_path}')
    rendered_html = driver.page_source


    soup = BeautifulSoup(rendered_html, "html.parser")

    author_metadata = []
    for tr in soup.find_all("tr", class_="Cont"):
        print(tr)

        author_id = tr.find("td", class_="CoG").get_text(strip=True).split()[1]
        author_name = tr.find("td", class_="CoW").find('a', class_="CoT").get_text(strip=True)

        if tr.find("td", class_="CoW").find('a', class_="CoEpi") is not None:
            epithet = tr.find("td", class_="CoW").find('a', class_="CoEpi").get_text(strip=True)
        else:
            epithet = None

        if tr.find("a", class_="CanonGeo") is not None:
            region = tr.find("a", class_="CanonGeo").get_text(strip=True)
        else:
            region = None

        if tr.find("a", class_="CanonDat") is not None:
            date = tr.find("a", class_="CanonDat").get_text(strip=True)
        else:
            date = None
        n_works = tr.find("a", class_="CoC").get_text(strip=True)

        author_metadata.append({"author_id": author_id, "author_name": author_name, "epithet": epithet, "region": region, "date": date, "n_works": n_works})

    author_metadata_df = pd.DataFrame(author_metadata)
    author_metadata_df.to_csv(os.path.join(PathManager.preprocessed_path, "tlg_preprocessed", "tlg_author_metadata.csv"), index=False)

    dirs = [d for d in os.listdir(tlg_path) if re.match(r"TLG\d+", d)]
    dirs = sorted(dirs, key=lambda x: int(x[3:]) if x[3:].isdigit() else x)

    dirs = [os.path.join(tlg_path, d) for d in dirs]

    for _,  d in tqdm(enumerate(dirs), total=len(dirs)):
        # if _ == 10: break
        found_work = False
        found_index = False
        dir_counts = {"index": 0, "work": 0, "other": 0}
        for f in sorted(os.listdir(d)):

            sec_count = 0
            if not f.endswith(".htm"):
                print(f"Skipping {f} in {d}")
                continue
            work_file = re.match(r"\d{4}_\d{3}\.htm", f)
            index_file = re.match(r"\d{4}\.htm", f)
            if work_file:
                found_work = True
                dir_counts["work"] += 1
            elif index_file:
                found_index = True
                dir_counts["index"] += 1
            else:
                dir_counts["other"] += 1

            if not index_file:
                work_id = f.split(".")[0].split("_")[1]
                author_id = f.split(".")[0].split("_")[0]
                html_file_path = os.path.abspath(os.path.join(d, f))
                print(html_file_path)

                driver.get(f'file://{html_file_path}')
                rendered_html = driver.page_source
                soup = BeautifulSoup(rendered_html, "html.parser")

                abbr = soup.find("h6")
                if not abbr is None:
                    abbr = abbr.get_text(strip=True)
                else:
                    abbr = work_id

                # print(abbr)

                # sections_pattern = re.compile(r"""^var\s*IL\s*=\s*(\[.[^><]\]);$""")
                sections_pattern = re.compile(r'var\s+IL\s*=\s*(\[.*?\]);', re.DOTALL)
                sections_desc = soup.find(string=sections_pattern)
                if not sections_desc is None:
                    sections = json.loads(re.search(sections_pattern, sections_desc).group(1))
                    num_levels = len(sections)

                else:
                    raise ValueError(f"Could not find sections in {html_file_path}")

                card_found = False
                text_started = False

                rows = []
                n_cols = 3
                for tr in soup.find_all("tr"):
                    # print(tr)
                    if not card_found or not text_started:
                        tds = tr.find_all("td")
                        if len(tds) == 1 and tds[0].has_attr("class") and tds[0]["class"] == ["Card"]:
                            # print("Find card", tds[0])
                            card_found = True
                            continue

                        elif len(tds) == 1 and tds[0].has_attr("class") and tds[0]["class"] == ["qS"]:
                            n_cols = int(tds[0]["colspan"])
                            if not n_cols == 3:
                                print(f"N cols is not 3 but {n_cols}: {html_file_path}")
                            continue

                        elif card_found and not text_started:
                            # print("Now here")
                            if len(tds) > 0 and tds[0].has_attr("class") and tds[0]["class"] == ["H"]:
                                text_started = True
                                sec_count += 1
                                # print("Text found")
                                row_dict = parse_tr(tr, level_names=sections, n_cols=n_cols)
                            else:
                                continue

                        else:
                            continue

                    else:
                        sec_count += 1
                        row_dict = parse_tr(tr, level_names=sections, n_cols=n_cols)

                    row_dict["n"] = len(rows) + 1
                    row_dict["author_id"] = author_id
                    row_dict["work_id"] = work_id
                    rows.append(row_dict)


                """
                {'unknown_class': [['Y'], ['QE'], ['QH'], ['J0'], ['G'], ['B'], ['JC'], 
                ['JK'], ['QU'], ['Qk'], ['J4'], ['JO'], 
                ['JG'], ['J8'], ['JS'], ['J6'], ['B', 'Q6'], ['J5'], ['Qo'], 
                ['QK'], ['J7'], ['QS'], ['Qs'], ['Qy'], ['QO'], 
                ['Q2'], ['B', 'Q2'], ['QW'], ['Qi'], ['J1'], ['J3'], ['QG'], 
                ['Qa'], ['QC'], ['QM'], ['QQ'], ['Qq'], ['Qc'], ['QI'], ['Qg'], 
                ['JJ'], ['JI'], ['JH'], ['Jj'], ['Jn'], ['J9'], ['JW'], ['Ja'], 
                ['Ji'], ['Je'], ['Jy'], ['Jm'], ['Ju'], ['V3'], ['V0'], ['V1'], ['Jw'], 
                ['J2'], ['JE'], ['Jc'], ['V5'], ['JQ'], ['V2'], ['Jk'], ['Jo'], ['Jq'], 
                ['JM'], ['JA'], ['JU'], ['JY'], ['Q6'], ['QA'], ['Jg'], ['Qz'], ['JL'], 
                ['JT'], ['JD'], ['JV'], ['JN'], ['Qr'], ['JR'], ['JP'], ['Jd'], ['JF'], 
                ['Jf'], ['JZ'], ['JX'], ['JB'], ['Jh'], ['Jb'], ['QT'], ['QP'], ['Qf'], ['QX'], 
                ['Qw'], ['Qn'], ['Qv'], ['Qj'], ['Jl'], ['Js'], ['QY'], ['V4'], 
                ['Section'], ['V9'], ['QD'], ['Q9'], ['Q3'], ['Qb'], ['QL'], ['V6'], ['V7'], ['V8'], ['VA']], 
                'div': {'class'}, 'a': {'class'}, 'span': {'class'}, 'img': {'src', 'alt'}, 'p': {'class'}, 'hr': set()}
"""

                line_rows = []
                for r in rows:
                    r_dict = copy.deepcopy(r)
                    if len(r["text"]) > 0:
                        for i, t in enumerate(r["text"]):
                            line_dict = copy.deepcopy(r_dict)
                            line_dict["line"] = i + 1
                            line_dict["text"] = t
                            line_rows.append(line_dict)

                df = pd.DataFrame(line_rows)

                levels = [c for c in df.columns if re.match(r"h\d",c)]
                levels = sorted(levels, key=lambda x: int(x[1:]),reverse=False)

                for i, s in enumerate(sections):
                    df[f"h{i+1}_name"] = s

                otp_path = os.path.join(PathManager.preprocessed_path, "tlg_preprocessed")
                os.makedirs(otp_path, exist_ok=True)
                df.to_csv(os.path.join(otp_path, f"{author_id}_{work_id}.csv"), index=False)



if __name__ == "__main__":
    main()


