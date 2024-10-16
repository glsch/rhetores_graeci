import argparse
import copy
import json
import os
import re
from tqdm import tqdm
from typing import List
import zipfile

from bs4 import BeautifulSoup, Tag, NavigableString
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from ..PathManager import PathManager

text_children = {}

def has_child_title(tag: Tag):
    return tag.find_all(("a", "span"), class_=['T'])

def sort_key(fname):
    parts = re.findall(r'\d+', fname)
    return tuple(map(int, parts))


def compress_csv_files_to_zip(file_to_zip, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in file_to_zip:
            fname = os.path.basename(f)
            zipf.write(f, arcname=fname)

def read_card(tag: Tag) -> dict:
    work_dict = {"title": None, "edition": None}

    title = tag.find("p", class_="WorkInfo")
    edition = tag.find("p", class_="EditionInfo")

    if title is not None:
        work_dict["title"] = title.get_text(strip=True)

    if edition is not None:
        work_dict["edition"] = edition.get_text(strip=True)

    return work_dict

def _parse_row(row: Tag, level_names: List[str]) -> dict:
    # children = row.find_all("td")
    children = (row.children)

    row_dict = {"text": [], "levels": []}
    for i, c in enumerate(children):
        if isinstance(c, Tag):
            if c.name == "td":
                if c.has_attr("class"):
                    if c["class"] in (["R"], ["K"], ["G"], ["Y"]): # skipping right margins regardless
                        continue
                    if c["class"] == ["M"]: # left margins are more complex as they might contain names of speakers
                        # if td.get_text(strip=True) == "":
                        #     continue
                        # else:
                        #     row_dict["text"].append(td.get_text(strip=True))
                        continue # alternatively, just skipping left margins, too
                    if c["class"] == ["H"]:
                        heads = c.find_all(re.compile(r"h\d"))
                        if len(heads) > 0:
                            for h in heads:
                                level_name_idx = int(h.name[1:]) - 1
                                if level_name_idx == 5:
                                    level_name_idx = "T"
                                row_dict["levels"] = row_dict.get("levels", []) + [[h.get_text(strip=True), level_names[level_name_idx] if level_name_idx != "T" else "abbr"]]

                    else:
                        if has_child_title(c):
                            row_dict["text"].append(f"<title>{c.get_text(strip=True)}</title>")
                            continue

                        row_dict["text"].append(c.get_text(strip=True))

                else:
                    row_dict["text"].append(c.get_text(strip=True))

            # elif c.name == "th":
            #     level_name_idx = 0
            #     row_dict["levels"] = row_dict.get("levels", []) + [{"level": c.get_text(strip=True), "name": level_names[level_name_idx]}]


    # if not row_dict["levels"]:
    #     text = "\n".join(row_dict["text"])
    #     for l, t in enumerate(text.split("\n")):
    #         row_dict["levels"] = row_dict.get("levels", []) + [[t, level_names[0]]]

    if isinstance(row_dict["levels"], list) and len(row_dict["levels"]) == 0:
        row_dict["levels"] = None

    return row_dict

def main():
    parser = argparse.ArgumentParser("Preparing data_df for Authorship Attribution Transformers for Ancient Greek")

    parser.add_argument("--input", type=str, default="TLG-E", help="Path to the TLG-E dataset.")
    parser.add_argument("--overwrite", type=str, nargs="+", default=None, choices=["all", "metadata", "texts"], help="Overwrite or not.")
    parser.add_argument("--output_path", type=str, default="tlg_preprocessed", help="Output path for the preprocessed data_df.")
    parser.add_argument("--zip", action="store_true", default=False, help="Whether or not to zip the output folder.")
    parser.add_argument("--author_metadata_fname", type=str, default="tlg_author_metadata.csv", help="Whether or not to zip the output folder.")
    parser.add_argument("--stop_after", type=int, default=None, help="Stop after.")
    parser.add_argument("--reimport_author", type=str, default=None, help="Reimport author.")
    parser.add_argument("--reimport_work", type=str, default=None, help="Reimport work.")

    args = parser.parse_args()
    # creating paths
    if args.output_path == "tlg_preprocessed":
        args.output_path = os.path.join(PathManager.preprocessed_path, "tlg_preprocessed")
        os.makedirs(args.output_path, exist_ok=True)
    if args.input == "TLG-E":
        args.input = os.path.join(PathManager.data_path, "TLG-E")


    raw_data_path = args.input
    author_metadata_path = os.path.join(args.output_path, args.author_metadata_fname)

    # since htm files are not rendered, we open the in a headless selenium to get the rendered html
    # initializing the driver
    service = ChromeService(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=service, options=options)

    # first we get the metadata of the authors, if needed
    if not os.path.exists(author_metadata_path) or "all" in args.overwrite or "metadata" in args.overwrite:
        index_path = os.path.join(raw_data_path, "indices", "index.htm")
        index_path_html_file_path = os.path.abspath(index_path)

        driver.get(f'file://{index_path_html_file_path}')
        rendered_html = driver.page_source

        soup = BeautifulSoup(rendered_html, "html.parser")
        author_metadata = []
        for tr in soup.find_all("tr", class_="Cont"):

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

            author_metadata.append(
                {"author_id": author_id, "author_name": author_name, "epithet": epithet, "region": region, "period": date,
                 "n_works": n_works})

        author_metadata_df = pd.DataFrame(author_metadata)
        author_metadata_df.to_csv(author_metadata_path, index=False)

    else:
        if os.path.exists(author_metadata_path):
            author_metadata_df = pd.read_csv(author_metadata_path)

        else:
            raise FileNotFoundError(f"Could not find author metadata file at {author_metadata_path}")


    # now we get the texts
    if not os.path.exists(args.output_path) or "all" in args.overwrite or "texts" in args.overwrite:
        dirs = [d for d in os.listdir(raw_data_path) if re.match(r"TLG\d+", d)]
        dirs = sorted(dirs, key=lambda x: int(x[3:]) if x[3:].isdigit() else x)

        dirs = [os.path.join(raw_data_path, d) for d in dirs]

        for _, d in tqdm(enumerate(dirs), total=len(dirs)):
            if args.stop_after is not None and _ > args.stop_after:
                break

            # skipping if we are importing specific author
            if args.reimport_author is not None and not d.endswith(f"TLG{str(args.reimport_author)}"):
                # print(f"Not importing {d} as it is not the author we are looking for: {args.reimport_author}")
                continue

            # counting files for each directory
            for f in sorted(os.listdir(d)):
                if not f.endswith(".htm"):
                    print(f"Skipping {f} in {d}")
                    continue

                if args.reimport_work is not None and not f.split(".")[0].endswith(f"_{str(args.reimport_work)}"):
                    # print(f"Not importing {f} as it is not the author we are looking for: {args.reimport_work}")
                    continue

                # reconginzing the type of file
                work_file = re.match(r"\d{4}_\d{3}\.htm", f)
                index_file = re.match(r"\d{4}\.htm", f)

                if not index_file:
                    work_id = f.split(".")[0].split("_")[1]
                    author_id = f.split(".")[0].split("_")[0]
                    edition = None
                    title = None

                    # getting absolute path of the file with text to load it in the driver
                    html_file_path = os.path.abspath(os.path.join(d, f))

                    # if not html_file_path == "/Users/glebschmidt/code/language_models_for_greek_rhetoricians/data_df/TLG-E/TLG0002/0002_002.htm":
                    #     continue

                    # print(html_file_path)
                    driver.get(f'file://{html_file_path}')
                    rendered_html = driver.page_source
                    soup = BeautifulSoup(rendered_html, "html.parser")

                    abbr = soup.find("h6") # abbreviation of the work, if any
                    if not abbr is None:
                        abbr = abbr.get_text(strip=True)
                    else:
                        abbr = 1 # merely 1, if no abbreviation is found

                    # getting the list of section names of the work
                    # it looks like
                    # <script>
                    # var IL = ["Line", "Epigram", "Book"];
                    # document.body.onload=f();
                    # </script>
                    sections_pattern = re.compile(r'var\s+IL\s*=\s*(\[.*?\]);', re.DOTALL)
                    sections_desc = soup.find(string=sections_pattern)
                    if not sections_desc is None:
                        sections = json.loads(re.search(sections_pattern, sections_desc).group(1))

                    else:
                        raise ValueError(f"Could not find sections in {html_file_path}")

                    # since there is a metadata card at the beginning, we need to keep track of the type of the data_df we are parsing
                    card_found = False
                    text_started = False

                    rows = []
                    for tr in soup.find_all("tr"):
                        if not card_found or not text_started:
                            tds = tr.find_all("td")
                            if len(tds) == 1 and tds[0].has_attr("class") and tds[0]["class"] == ["Card"]:
                                card_found = True
                                card_info = read_card(tds[0])
                                edition = card_info["edition"]
                                title = card_info["title"]
                                continue

                            elif len(tds) == 1 and tds[0].has_attr("class") and tds[0]["class"] == ["qS"]:
                                n_cols = int(tds[0]["colspan"])
                                if not n_cols == 3:
                                    pass
                                text_started = True
                                continue
                            else:
                                continue

                        row_dict = _parse_row(tr, level_names=sections)

                        row_dict["row"] = len(rows) + 1
                        row_dict["author_id"] = author_id
                        row_dict["text"] = "\n".join(row_dict["text"])
                        row_dict["title"] = title
                        # row_dict["edition"] = edition
                        # row_dict["author_name"] = author_metadata_df[author_metadata_df["author_id"] == author_id]["author_name"].values[0]
                        row_dict["work_id"] = work_id

                        if not len(row_dict["text"].strip()) == 0:
                            rows.append(row_dict)

                        # print(row_dict)

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

                    df = pd.DataFrame(rows)

                    df["levels"] = df["levels"].fillna(method="ffill")

                    df.to_csv(os.path.join(args.output_path, f"{author_id}_{work_id}.csv"), index=False, encoding='utf-8')

                    del df
                    del soup

    def concat_csvs_from_folder(folder_path, exclude_fname: List[str]):
        print("concat_csvs_from_folder")
        # Initialize an empty list to hold the DataFrames
        dfs = []

        # Iterate over all files in the specified folder
        for filename in tqdm(os.listdir(folder_path), total=len(os.listdir(folder_path))):
            if filename.endswith('.csv'):
                if filename in exclude_fname:
                    continue
                # Create the full file path
                file_path = os.path.join(folder_path, filename)
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                # Append the DataFrame to the list
                dfs.append(df)

        # Concatenate all DataFrames in the list
        concatenated_df = pd.concat(dfs, ignore_index=True)

        return concatenated_df

    dataset = concat_csvs_from_folder(args.output_path, exclude_fname=[args.author_metadata_fname])

    dataset.to_csv(os.path.join(args.output_path, "dataset.csv"), index=False)

    if args.zip:
        compress_csv_files_to_zip([os.path.join(args.output_path, "dataset.csv"), author_metadata_path], os.path.join(args.output_path, "tlg_dataset.zip"))

if __name__ == "__main__":
    main()