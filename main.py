from wikidata.client import Client
import json
from pathlib import Path
from collections import defaultdict
import os
from lxml import etree
import time
import glob
from typer import Typer
import wikipediaapi
from tqdm import tqdm
from matplotlib import pyplot as plt
from urllib.error import HTTPError

app = Typer()

SUMMARY_SECTIONS = {
        "de": ["Plot", "Handlung", "Zusammenfassung", "Synopsis", "Inhalt", "Handlungsübersicht", "Zusammenfassung der Handlung"],
        "en": ["Plot", "Summary", "Synopsis", "Plot summary", "Plot Summary", "Story", "Storyline", "Content"],
        "fr": ["Synopsis", "Résumé de l'œuvre", "Résumé", "Le roman", "Synopsis détaillé", "Histoire"],
        "es": ["Trama", "Sinopsis", "Argumento", "Argument"],
        "it": ["Trama", "Contenuto", "Tema"],
}

#
# While Synposis does contain some sort of summary it is not typically what we are looking for. Therefore we exclude them
#

LANGS = ["dewiki", "enwiki", "frwiki", "eswiki", "itwiki"]

@app.command()
def scrape_wikidata():
    c = Client()
    work_ids = open("query.csv")
    next(work_ids)


    i = 0
    for line in tqdm(work_ids, total=484054):
        wid = line.strip().split("/")[-1]
        os.makedirs(f"data/wikidata/{wid[1:3]}/", exist_ok=True)
        file_name = f"data/wikidata/{wid[1:3]}/{wid[1:]}.json"
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            continue
        i += 1
        out_file = open(file_name, "w")
        try:
            entity = c.get(wid, load=True)
        except HTTPError as e:
            if e.code == 404:
                print("404")
                continue
            else:
                print("HTTP ERROR:", e)
                break
        json.dump(entity.data, out_file)
        out_file.close()
        time.sleep(0.1)


def build_clients():
    return {lang: wikipediaapi.Wikipedia(lang.removesuffix("wiki"), extract_format=wikipediaapi.ExtractFormat.HTML) for lang in LANGS}

@app.command()
def scrape_wikipedia(target_langs: list[str] = LANGS[2:]):
    # Sometimes summary is in an h2 e.g. english version of this: https://www.wikidata.org/wiki/Q1194637
    clients = build_clients()
    for work_file in tqdm(glob.glob("data/wikidata/*/*.json")):
        data = json.load(open(work_file))
        if len(set(data["sitelinks"].keys()) & {"dewiki", "enwiki"}) == 2:
            path = f"data/wikipedia/{data['title'][1:3]}/{data['title'][1:]}"
            os.makedirs(path, exist_ok=True)
            for lang in target_langs:
                if not os.path.exists(path + f"/{lang}.json"):
                    title = data["sitelinks"].get(lang, {}).get("title")
                    if title is not None:
                        page = clients[lang].page(title)
                        json.dump({s.title : s.full_text() for s in page.sections}, open(path + f"/{lang}.json", "w"))


def extract_summary(sections, lang):
    for section_name in SUMMARY_SECTIONS[lang]:
        text = sections.get(section_name)
        if text is not None and len(text) > 100:
            return section_name, text


def html_to_plain(markup, parser=etree.HTMLParser(recover=True)):
    markup = "<root>" + markup + "</root>"
    root = etree.fromstring(markup, parser=parser)
    elements = root.xpath(".//root/*[self::h3 or self::h2 or self::p]")
    out_text = []
    for el in elements:
        if el.tag == "p":
            out_text.append("".join(el.itertext()) or "")
        elif el.tag == "h2":
            out_text.append(("# " + el.text) or "")
        elif el.tag == "h3":
            out_text.append(("## " + el.text) or "")
        elif el.tag == "h1":
            print("H1", el.text or "")
        else:
            print("Unknown tag")
    return " ".join(out_text)


@app.command()
def extract_summaries():
    for work_dir in tqdm(glob.glob("data/wikipedia/*/*")):
        wikidata_info = json.load(open(work_dir.replace("/wikipedia/", "/wikidata/") + ".json"))
        wikidata_id = wikidata_info["title"]
        summary_dict = {}
        summary_section_dict = {}
        for file_name in os.listdir(work_dir):
            path = Path(work_dir) / file_name
            data = json.load(open(path))
            lang = path.name.removesuffix("wiki.json")
            summary_dict[lang], summary_section_dict[lang] = extract_summary(data, lang)
            if lang in ["es", "fr", "it"] and summary_dict[lang] is None:
                print("Url", wikidata_info["sitelinks"][lang + "wiki"]["url"])
                print("Sections", data.keys())
                breakpoint()
        if  len([text for text in summary_dict.values() if text is not None]) >= 2:
            labels = wikidata_info.get("labels", {})
            title_en = labels.get("en", {}).get("value")
            title_de = labels.get("de", {}).get("value")
            description_en = wikidata_info.get("descriptions", {}).get("en", {}).get("value")
            title = title_en or labels.get(list(labels.keys())[0])
            out = {
                    "wikidata_id": wikidata_id,
                    "title_en": title_en,
                    "title_de": title_de,
                    "titles": {k : labels.get(k) for k in summary_dict.keys()},
                    "title": title,
                    "description": description_en,
                    "summaries": {k: html_to_plain(v) for k, v in summary_dict.items() if v is not None},
                    "summary_sections": summary_section_dict,
            }
            out_file_path = f"data/summaries/{wikidata_id[1:3]}/{wikidata_id[1:]}.json"
            if not os.path.exists(out_file_path):
                os.makedirs(Path(out_file_path).parent, exist_ok=True)
                json.dump(out, open(out_file_path, "w"))


def get_all_summaries(limit=None):
    summaries = defaultdict(list)
    ids = []
    i = 0
    for file_name in glob.glob("data/summaries/*/*.json"):
        i += 1
        data = json.load(open(file_name))
        if data["summaries"]["de"] == "":
            continue
        if data["summaries"]["en"] == "":
            continue
        summaries["en"].append(data["summaries"]["en"])
        summaries["de"].append(data["summaries"]["de"])
        ids.append(data["wikidata_id"])
        if limit is not None and i + 1 == limit:
            break
    return summaries, ids

@app.command()
def analyze_similarities():
    from bert_score import score
    all_scores = []
    summaries, ids = get_all_summaries()
    (_, _, f1s) = score(summaries["de"], summaries["en"], lang="other")
    print(f1s)
    print(f1s.mean())
    maxes = f1s.topk(20).indices
    for m in maxes:
        print(ids[m])
        # f1s[m]
    # max_pos = f1s.argmax()
    plt.hist(f1s, bins=40)
    plt.show()


@app.command()
def sbert():
    import sentence_transformers
    summaries, ids = get_all_summaries(limit=1200)
    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    de = model.encode(summaries["de"])
    en = model.encode(summaries["en"])
    sims = sentence_transformers.util.cos_sim(de, en)
    print(sum([1 if a == b else 0 for a, b in enumerate(sims.max(1).indices)]) / len(de))

if __name__ == "__main__":
    app()
