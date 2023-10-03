from wikidata.client import Client
import json
import itertools
from pathlib import Path
import random
from collections import defaultdict
import re
import time
import os
import bert_score
from lxml import etree
import time
import glob
from typer import Typer
import wikipediaapi
from tqdm import tqdm
from matplotlib import pyplot as plt
from urllib.error import HTTPError
import sklearn
import sentence_transformers
import ersatz
import torch
from itertools import groupby
import sentence_transformers

import dataset

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
            entity = None
            entity = c.get(wid, load=True)
        except HTTPError as e:
            if e.code == 404:
                print("404")
                continue
            else:
                print("HTTP ERROR:", e)
                break
        finally:
            if entity is not None:
                json.dump(entity.data, out_file)
            else:
                json.dump({}, out_file)
            out_file.close()
        time.sleep(0.1)


def build_clients():
    return {lang: wikipediaapi.Wikipedia(language=lang.removesuffix("wiki"), extract_format=wikipediaapi.ExtractFormat.HTML, user_agent='StorySummaryBot (hans.ole.hatzel@uni-hamburg.de)') for lang in LANGS}

@app.command()
def scrape_wikipedia(target_langs: list[str] = LANGS):
    # Sometimes summary is in an h2 e.g. english version of this: https://www.wikidata.org/wiki/Q1194637
    clients = build_clients()
    for work_file in tqdm(glob.glob("data/wikidata/*/*.json")):
        data = json.load(open(work_file))
        if len(data) == 0:
            continue
        if len(set(data.get("sitelinks", {}).keys()) & {"dewiki", "enwiki"}) == 2:
            path = f"data/wikipedia/{data['title'][1:3]}/{data['title'][1:]}"
            os.makedirs(path, exist_ok=True)
            for lang in target_langs:
                if not os.path.exists(path + f"/{lang}.json"):
                    title = data["sitelinks"].get(lang, {}).get("title")
                    if title is not None:
                        page = clients[lang].page(title)
                        json.dump({s.title : s.full_text() for s in page.sections}, open(path + f"/{lang}.json", "w"))
                        time.sleep(0.1)


def extract_summary(sections, lang):
    for section_name in SUMMARY_SECTIONS[lang]:
        text = sections.get(section_name)
        if text is not None and len(text) > 100:
            return section_name, text


def html_to_plain(markup, parser=etree.HTMLParser(recover=True)):
    markup = "<root>" + markup + "</root>"
    root = etree.fromstring(markup, parser=parser)
    elements = root.xpath(".//root/*[self::h3 or self::h2 or self::p or self::dd or self::dl]")
    out_text = []
    for el in elements:
        if el.tag in ["p", "dl", "dd"]:
            out_text.append("".join(el.itertext()).strip() + "\n")
        elif el.tag == "h2":
            out_text.append(("# " + el.text.strip() + "\n") if el.text is not None else "")
        elif el.tag == "h3":
            out_text.append(("## " + el.text.strip() + "\n") if el.text is not None else "")
        elif el.tag == "h1":
            print("H1", el.text or "")
        else:
            print("Unknown tag")
    return "".join(out_text).strip()


@app.command()
def extract_summaries():
    for work_dir in tqdm(glob.glob("data/wikipedia/*/*")):
        try:
            wikidata_info = json.load(open(work_dir.replace("/wikipedia/", "/wikidata/") + ".json"))
        except FileNotFoundError:
            continue
        wikidata_id = wikidata_info["title"]
        summary_dict = {}
        summary_section_dict = {}
        for file_name in os.listdir(work_dir):
            path = Path(work_dir) / file_name
            data = json.load(open(path))
            lang = path.name.removesuffix("wiki.json")
            summary_pair = extract_summary(data, lang)
            if summary_pair is not None:
                summary_dict[lang], summary_section_dict[lang] = summary_pair
                if lang in ["es", "fr", "it"] and summary_dict[lang] is None:
                    print("Url", wikidata_info["sitelinks"][lang + "wiki"]["url"])
                    print("Sections", data.keys())
        if  len([text for text in summary_dict.values() if text is not None]) >= 2:
            labels = wikidata_info.get("labels", {})
            title_en = labels.get("en", {}).get("value")
            title_de = labels.get("de", {}).get("value")
            description_en = wikidata_info.get("descriptions", {}).get("en", {}).get("value")
            title = title_en or labels.get(list(labels.keys())[0])
            plain_text_summaries = {k: html_to_plain(v) for k, v in summary_section_dict.items() if v is not None}
            out = {
                    "wikidata_id": wikidata_id,
                    "title_en": title_en,
                    "title_de": title_de,
                    "titles": {k : labels.get(k) for k in summary_dict.keys()},
                    "title": title,
                    "description": description_en,
                    "summaries": plain_text_summaries,
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
        if data["summaries"].get("de", "") == "":
            continue
        if data["summaries"].get("en", "") == "":
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
    plt.hist(recalls, bins=41)
    plt.savefig("sims.pdf")


def sentence_score(embs_a, embs_b):
    """
    Best-match sentence embedding similarity.
    """
    sims = sentence_transformers.util.cos_sim(embs_a, embs_b)
    recall = sims.max(dim=0)[0].mean()
    precision = sims.max(dim=1)[0].mean()
    print(recall, precision)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def similarity_scores(sentencized_docs, sbert_model):
    embeddings = sbert_model.encode(list(itertools.chain.from_iterable(sentencized_docs)))
    score_matrix = torch.zeros((len(sentencized_docs), len(sentencized_docs)), dtype=torch.float)
    for i, sents in enumerate(sentencized_docs):
        start_index = len(list(itertools.chain.from_iterable(sentencized_docs[:i])))
        end_index = start_index + len(sents)
        doc_a_embeddings = embeddings[start_index:end_index]
        for j, sents in enumerate(sentencized_docs):
            if i == j:
                continue
            start_index_j = len(list(itertools.chain.from_iterable(sentencized_docs[:j])))
            end_index_j = start_index_j + len(sents)
            doc_b_embeddings = embeddings[start_index_j:end_index_j]
            if len(doc_a_embeddings) > 0 and len(doc_b_embeddings) > 0:
                score_matrix[i][j] = sentence_score(doc_a_embeddings, doc_b_embeddings)[-1]
    return score_matrix



@app.command()
def add_similarity_rating():
    """
    Adds sentence boundaries and similarity rating.
    """
    sbert_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    for file_name in tqdm(glob.glob("data/summaries/*/*.json"), desc="Splitting work summaries"):
        data = json.load(open(file_name))
        temp_file_path = os.path.splitext(file_name)[0] + ".temp"
        texts = []
        data["split_into_sents"] = {}
        if data["summaries"].get("en") is not None:
            split_original = ersatz.split_text(text=data["summaries"]["en"], model="en") or []
            data["split_into_sents"]["en"] = split_original
            texts.append(split_original)
        for key, item in data.get("en_translated_summaries", {}).items():
            split_original = ersatz.split_text(text=item["text"], model="en") or []
            item["sentences"] = split_original
            texts.append(split_original)
        scores = similarity_scores(texts, sbert_model)
        data["similarity"] = {
            "indexes": (["en"] if data["summaries"].get("en") is not None else []) + list(data.get("en_translated_summaries", {}).keys()),
            "similarities":  scores.tolist(),
        }
        json.dump(data, open(temp_file_path, "w"))
        os.replace(temp_file_path, file_name)

@app.command()
def test_annotated(annotated_tsv: str):
    sbert_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    label_dict = {}
    for line in open(annotated_tsv):
        id_, label = line.strip().split("\t")
        label_dict[id_[1:]] = label == "True"
    ds = dataset.SummaryDataset("data", only_include=label_dict.keys())
    dev, test = ds.stratified_split(label_dict)
    data = test
    scores = []
    translated_texts, original_texts = [], []
    for label, summary in data:
        translated, original = summary.summaries_translated["de"], summary.summaries_original["en"]
        translated_texts.append(translated)
        original_texts.append(original)
        p, r, f1 = sentence_score(translated, original, sbert_model)
        scores.append(f1)
    # p, r, f1 = bert_score.score(translated_texts, original_texts, lang="en")
    f1s = torch.tensor(scores)
    boundary = 0.6
    print("Boundary", boundary)
    metrics = sklearn.metrics.classification_report([label for label, _ in data], f1s > boundary)
    print(metrics)


@app.command()
def split_docs():
    line_ids = defaultdict(list)
    os.makedirs("data/to_translate/", exist_ok=True)
    out_files = {lang: open(f"data/to_translate/{lang}.txt", "w") for lang in SUMMARY_SECTIONS.keys()}
    out_line_ids = {lang: open(f"data/to_translate/{lang}.tsv", "w") for lang in SUMMARY_SECTIONS.keys()}
    for file_name in tqdm(glob.glob("data/summaries/*/*.json"), desc="Splitting work summaries"):
        data = json.load(open(file_name))
        for lang, summary in data["summaries"].items():
            paragraphs = []
            wiki_id = os.path.splitext(os.path.basename(file_name))[0]
            for i, p in enumerate(summary.strip().split("\n")):
                if len(p.strip()) == 0:
                    continue
                split = ersatz.split_text(model=lang.replace("it", "default-multilingual"), text=p)
                out_files[lang].writelines([s + "\n" for s in split])
                out_line_ids[lang].writelines([f"{wiki_id}_{i}\n" for _ in range(len(split))])


@app.command()
def add_translations(lang, translated_file: str, line_mapping_file: str):
    line_iterator = zip([info.strip().split("_") for info in open(line_mapping_file)], open(translated_file))
    grouped_by_wiki_id = groupby(line_iterator, key=lambda line: line[0][0])
    translations = {}
    for wiki_id, paragraphs in grouped_by_wiki_id:
        grouped_by_paragraph = groupby(paragraphs, key=lambda line: line[0][1])
        text = "\n".join([" ".join([line[1].strip() for line in lines]) for k, lines in grouped_by_paragraph])
        translations[wiki_id] = text
    for wiki_id, text in translations.items():
        basename = f"data/summaries/{wiki_id[:2]}/{wiki_id}"
        data_file_path = basename + ".json"
        temp_file_path = basename + ".temp"
        data = json.load(open(data_file_path))
        if lang not in data["summaries"].keys():
            # This is a fix for summaries that were removed after the translations were made
            continue
        data["en_translated_summaries"] = data.get("en_translated_summaries", {})
        with open(temp_file_path, "w") as temp_file:
            data["en_translated_summaries"][lang] = {"text": text, "translation_tool": "nllb-200-3.3B", "source_lang": lang}
            json.dump(data, temp_file)
        os.replace(temp_file_path, data_file_path)


@app.command()
def build_subset(out_path: str = "sample.txt"):
    ds = dataset.SummaryDataset("data")
    stories = random.choices(list(ds.stories.values()), k=100)
    for story in stories:
        out_file = open(out_path, "w")
        for story in stories:
            out_file.write("=======================================Summary Pair\n")
            out_file.write(story.title + "\t" + story.description + "\n")
            for summary in story.get_all_summaries_en():
                out_file.write("====================\n")
                out_file.write(summary + "\n")


@app.command()
def stats():
    ds = dataset.SummaryDataset("data")
    # results = ds.get_metadata_stats()
    # genre_count_file = open("data/genre_counts.csv", "w")
    # for genre_id, count in results["genres"]:
    #     genre_count_file.write(f"{genre_id},{count}\n")
    results = ds.get_lang_stats(sentence_lengths=False)
    print(results)
    print(json.dumps(results))
    json.dump(results, open("data/stats.json", "w"))

@app.command()
def sbert():
    summaries, ids = get_all_summaries(limit=1200)
    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    de = model.encode(summaries["de"])
    en = model.encode(summaries["en"])
    sims = sentence_transformers.util.cos_sim(de, en)
    print(sum([1 if a == b else 0 for a, b in enumerate(sims.max(1).indices)]) / len(de))

if __name__ == "__main__":
    app()
