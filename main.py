from wikidata.client import Client
import json
import string
from typing import Dict, Tuple, List, Optional, Optional
import itertools
from pathlib import Path
import random
from collections import defaultdict, Counter
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
import requests
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from coref import add_coref
import csv
import spacy_flair
import numpy as np

import dataset
from name_db import NameDB

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
def genre_tsne():
    from sklearn.manifold import TSNE
    from itertools import repeat
    genres_with_id = {
        #"Q860626": "romcom", 
        "Q1054574": "romance",
        "Q200092": "horror",
        "Q157443": "comedy" 
    }
    ds = dataset.SummaryDataset("data")
    genre_stories = defaultdict(list)
    for _, story in ds.stories.items():
        overlap = set(genres_with_id.keys()) & set(story.genres)
        if len(overlap) == 1:
            genre = list(overlap)[0]
            summary = story.summaries_original.get("en")
            if summary is not None:
                genre_stories[genres_with_id[genre]].append(summary)
    for texts in genre_stories.values():
        random.shuffle(texts)
    model = sentence_transformers.SentenceTransformer("sentence-transformers/sentence-t5-large")
    embedding_list = []
    labels = []
    per_genre = 20
    for label, story_list in genre_stories.items():
        encoded = model.encode(story_list[:per_genre])
        embedding_list.append(encoded)
        labels.extend([label] * len(encoded))
    embeddings = np.concatenate(embedding_list)
    reducer = TSNE(n_components=2, init="random", perplexity=len(embeddings) // 2)
    reduced = reducer.fit_transform(embeddings)
    out_file = open("tsne.csv", "w")
    for label, (x, y) in zip(labels, reduced):
        print(x, y, label, sep=",", file=out_file)

    

@app.command()
def stats():
    ds = dataset.SummaryDataset("data")
    results = {
        "metadata": ds.get_metadata_stats(),
        "langauges": ds.get_lang_stats(sentence_lengths=False),
    }
    # genre_count_file = open("data/genre_counts.csv", "w")
    # for genre_id, count in results["genres"]:
    #     genre_count_file.write(f"{genre_id},{count}\n")
    json.dump(results, open("data/stats.json", "w"))

@app.command()
def sbert():
    summaries, ids = get_all_summaries(limit=1200)
    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    de = model.encode(summaries["de"])
    en = model.encode(summaries["en"])
    sims = sentence_transformers.util.cos_sim(de, en)
    print(sum([1 if a == b else 0 for a, b in enumerate(sims.max(1).indices)]) / len(de))


@app.command()
def splits():
    ds = dataset.SummaryDataset("data")
    splits = ds.perform_splits()
    print(splits)
    print({len(v) for k, v in splits.items()})


def texts_to_entities(texts):
    from flair.nn import Classifier
    from flair.data import Sentence
    tagger = Classifier.load('ner')
    all_entities = []
    for sentences in tqdm(texts):
        sents = [Sentence(s) for s in sentences]
        tagger.predict(sents)
        entities = []
        for s in sents:
            for label in s.get_labels():
                for token in label.data_point.tokens:
                    entities.append(token.text.lower())
        all_entities.append(entities)
    return all_entities


@app.command()
def entities_test_remakes(remake_like: bool = False, our_data: bool = False):
    from chaturvedi import MovieSummaryDataset
    from sentence_transformers.util import cos_sim
    from sklearn.feature_extraction.text import TfidfVectorizer
    if our_data:
        ds = dataset.SummaryDataset("data")
        texts, labels, in_test_set = ds.chaturvedi_like_split()
    else:
        ds = MovieSummaryDataset(Path(os.environ["MOVIE_REMAKE_PATH"]) / "movieRemakesManuallyCleaned.tsv", Path(os.environ["MOVIE_REMAKE_PATH"]) / "testInstances.csv")
        texts = [summary.text for summary in ds]
        labels = [summary.cluster_id for summary in ds]
        in_test_set = [s.movie_id in ds.test_movies for s in ds]
    print(in_test_set)
    all_labeled = zip(texts, labels)
    split_texts = []
    for text in texts:
        split_text = ersatz.split_text(text=text, model="en") or []
        split_texts.append(split_text)
    all_entities = texts_to_entities(split_texts)
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)    
    encoded = vectorizer.fit_transform(all_entities).todense()
    similarities = cos_sim(encoded, encoded)
    similarities.fill_diagonal_(0)
    matches = similarities.argmax(1)
    correct = 0
    total = 0
    test_labels = np.array(labels)[in_test_set]
    for source, match in enumerate(matches[in_test_set]):
        if test_labels[source].item() == labels[match]:
            correct += 1
        total += 1
    print("P@1", correct / total)



@app.command()
def entities_test(use_anonymized: bool = False, min_length: int = 0, remake_like: bool = False):
    from sentence_transformers.util import cos_sim
    from sklearn.feature_extraction.text import TfidfVectorizer
    ds = dataset.SummaryDataset("data")
    splits = ds.perform_splits()
    all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.sentences.values()) for i, v in enumerate(list(splits["test"].stories.values()))]))
    all_labeled = [(id_, sents) for id_, sents in all_labeled if min_length is None or len(sents) >= min_length]
    all_entities = texts_to_entities([t for _, t in all_labled])
    labels, texts = zip(*all_labeled)
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)    
    encoded = vectorizer.fit_transform(all_entities).todense()
    similarities = cos_sim(encoded, encoded)
    similarities.fill_diagonal_(0)
    matches = similarities.argmax(1)
    correct = 0
    total = 0
    for source, match in enumerate(matches):
        if labels[source] == labels[match]:
            correct += 1
        total += 1
    print("P@1", correct / total)


@app.command()
def sbert_test(use_anonymized: bool = False, min_length: int = 0):
    from sentence_transformers.util import cos_sim
    from sentence_transformers import SentenceTransformer, models

    ds = dataset.SummaryDataset("data")
    splits = ds.perform_splits()
    if use_anonymized:
        all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.get_anonymized(min_sentences=min_length).values()) for i, v in enumerate(splits["test"].stories.values())]))
    else:
        all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.get_all_summaries_en(min_sentences=min_length)[1]) for i, v in enumerate(splits["test"].stories.values())]))
    all_labeled = [(id_, text) for id_, text in all_labeled if min_length is None or len(text) >= min_length]
    labels, texts = zip(*all_labeled)
    print("Num summaries", len(texts))
    print("Num stories", len(splits["test"].stories.values()))
    print("Num stories (after length filtering)", len(set(labels)))
    model_names = ["all-mpnet-base-v2", "sentence-transformers/sentence-t5-large", "finetuned-LaBSE-narrative",  "sentence-transformers/LaBSE"]
    out_file = open("sbert-test.csv", "w")
    for model_name in model_names:
        model = sentence_transformers.SentenceTransformer(model_name)
        encoded = model.encode(texts, show_progress_bar=True)
        similarities = cos_sim(encoded, encoded)
        similarities.fill_diagonal_(0)
        matches = similarities.argmax(1)
        correct = 0
        total = 0
        for source, match in enumerate(matches):
            if labels[source] == labels[match]:
                correct += 1
            total += 1
        print(model_name, "P@1", correct / total)
        print(model_name, correct / total, sep=",", file=out_file)


@app.command()
def chaturvedi_comparison(use_anonymized: bool=False):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    ds = dataset.SummaryDataset("data")
    summaries, labels, in_test_set = ds.chaturvedi_like_split(use_anonymized=use_anonymized)
    model_names = ["all-mpnet-base-v2", "sentence-transformers/sentence-t5-large", "finetuned-LaBSE-narrative",  "sentence-transformers/LaBSE"]
    for model_name in model_names:
        model = SentenceTransformer(model_name)
        encoded = model.encode(summaries, show_progress_bar=True)
        similarities = cos_sim(encoded, encoded)
        similarities.fill_diagonal_(0)
        matches = similarities.argmax(1)
        correct = 0
        total = 0
        for included, (source, match) in zip(in_test_set, enumerate(matches)):
            if not included:
                continue
            if labels[source] == labels[match]:
                correct += 1
            total += 1
        print(model_name, "P@1", correct / total)


@app.command()
def get_coreference_chaturvedi(coref_model_url):
    from chaturvedi import MovieSummaryDataset
    dataset = MovieSummaryDataset(Path(os.environ["MOVIE_REMAKE_PATH"]) / "movieRemakesManuallyCleaned.tsv", Path(os.environ["MOVIE_REMAKE_PATH"]) / "testInstances.csv")
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.add_pipe("flair_ner")
    name_db = NameDB("data/baby-names.csv")
    out = []
    writer = csv.writer(open("chaturvedi.csv", "w"))
    for i, summary in tqdm(enumerate(dataset)):
        doc = nlp(summary.text)
        tokenized_sentences = [[token.text for token in sentence] for sentence in doc.sents]
        coref_path = f"data/coref-remakes/{i}.json"
        if os.path.exists(coref_path):
            coref_info = json.load(open(coref_path))
        else:
            coref_info = request_coref(coref_model_url, tokenized_sentences)
            json.dump(coref_info, open(coref_path, "w"))
        add_coref(doc, coref_info)
        replaced = coref_replace(doc, name_db)
        writer.writerow([summary.cluster_id, replaced])
        out.append((summary.cluster_id, replaced))



@app.command()
def get_coreferences(coref_model_url):
    # No need for the trf model here
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.add_pipe("flair_ner")
    ds = dataset.SummaryDataset("data")
    for id_ in tqdm(ds.stories.keys()):
        lang_ids, translations = ds[id_].get_all_summaries_en()
        for lang_id, text in zip(lang_ids, translations):
            spacy_path = f"data/spacy/{id_[:2]}/{id_}_{lang_id}.spacy"
            coref_path = f"data/coref/{id_[:2]}/{id_}_{lang_id}.json"
            os.makedirs(f"data/coref/{id_[:2]}", exist_ok=True)
            os.makedirs(f"data/spacy/{id_[:2]}", exist_ok=True)
            if os.path.exists(spacy_path) and os.path.exists(coref_path):
                continue
            if not os.path.exists(spacy_path):
                doc = nlp(text)
                doc.to_disk(spacy_path)
            else:
                doc = Doc(Vocab()).from_disk(spacy_path)
            tokenized_sentences = [[token.text for token in sentence] for sentence in doc.sents]
            if not os.path.exists(coref_path):
                try:
                    print(len(doc))
                    print(doc)
                    coref_data = request_coref(coref_model_url, tokenized_sentences)
                    json.dump(coref_data, open(coref_path, "w"))
                except ValueError as e:
                    print("Error", e)


def request_coref(coref_model_url, tokenized_sentences):
    payload = {"tokenized_sentences": tokenized_sentences, "output_format": "list"}
    resp = requests.post(coref_model_url, json=payload)
    if not resp.ok:
        raise ValueError("Coref failed", resp.json())
    return resp.json()


@app.command()
def add_renamed_texts():
    import spacy
    from spacy.tokens import Doc
    from spacy.vocab import Vocab
    ds = dataset.SummaryDataset("data")
    name_db = NameDB("data/baby-names.csv")
    for id_ in tqdm(ds.stories.keys()):
        lang_ids, translations = ds[id_].get_all_summaries_en()
        renamed = {}
        for lang_id, text in zip(lang_ids, translations):
            spacy_path = f"data/spacy/{id_[:2]}/{id_}_{lang_id}.spacy"
            coref_path = f"data/coref/{id_[:2]}/{id_}_{lang_id}.json"
            doc = Doc(Vocab()).from_disk(spacy_path)
            if len(doc) == 0:
                continue
            coref_info = json.load(open(coref_path))
            add_coref(doc, coref_info)
            renamed[lang_id] = coref_replace(doc, name_db)
        summary_path = f"data/summaries/{id_[:2]}/{id_}.json"
        data = json.load(open(summary_path))
        data["anonymized"] = renamed
        temp_file_path = summary_path + "_temp"
        json.dump(data, open(temp_file_path, "w"))
        os.replace(temp_file_path, summary_path)


def guess_text_span_gender(text, name_db):
    sexes = []
    for name in text.split(" "):
        if sex := name_db.get_sex_for_name(name):
            sexes.append(sex)
    try:
        sex = Counter(sexes).most_common(1)[0][0]
    except IndexError:
        sex = None 
    return sex


def get_cluster_name(cluster, used_names, name_db):
    resp = None
    first_time = True
    counter = 0
    while (first_time or (resp in used_names)) and counter < 10:
        first_time = False
        counter += 1
        if cluster.ner_label == "PER":
            sexes = []
            for span in cluster.spans:
                for name in span.text.split(" "):
                    if sex := name_db.get_sex_for_name(name):
                        sexes.append(sex)
            try:
                sex = Counter(sexes).most_common(1)[0][0]
            except IndexError:
                sex = None 
            resp = name_db.random_name_with_sex(sex)
        elif cluster.ner_label == "LOC":
            resp = f"Location {string.ascii_uppercase[cluster.id % 26]}"
        elif cluster.ner_label == "ORG":
            resp = f"Organization {string.ascii_uppercase[cluster.id % 26]}"
        elif cluster.ner_label == "MISC":
            resp = f"Entity {string.ascii_uppercase[cluster.id % 26]}"
        else:
            resp = None
    return resp


def get_replacement_text(tag, text, name_db, used_names, performed_replacements):
    out = None
    counter = 0
    if already_replaced := performed_replacements.get((tag, text)):
        return already_replaced, performed_replacements
    while out is None or out in used_names and counter < 100:
        counter += 1
        if tag == "PER":
            sex = guess_text_span_gender(text, name_db)
            out = name_db.random_name_with_sex(sex)
        elif tag == "LOC":
            out = f"Location {random.choice(string.ascii_uppercase)}"
        elif tag == "ORG":
            out = f"Organization {random.choice(string.ascii_uppercase)}"
        elif tag == "MISC":
            out = f"Entity {random.choice(string.ascii_uppercase)}"
    if counter == 100:
        out = text
    performed_replacements.update({(tag, text): out})
    return out, performed_replacements


def coref_replace(doc, name_db):
    posessives = set(["my", "our", "your", "his", "her", "its", "their", "whose"])
    pronouns = set(["I", "you", "he", "she", "it", "we", "you", "they", "me", "you", "him", "her", "it", "us", "you", "them"]) | posessives
    replacements = []
    used_names = set()
    performed_singleton_replacements = {}
    for cluster in doc._.coref_clusters:
        cluster_name = get_cluster_name(cluster, used_names, name_db)
        if cluster_name is not None:
            used_names.add(cluster_name)
        previous_span = None
        for span in cluster.spans:
            # If it's very recent we continue using the possesssives
            if (previous_span is not None) and (previous_span.start + 8 >= span.start) and (span.text.lower().strip() in pronouns):
                replacement_text = span.text
            elif (span.text.endswith("'s") or span.text.lower() in posessives) and cluster_name is not None:
                replacement_text = cluster_name + "'s"
            else:
                replacement_text = cluster_name
            previous_span = span
            replacements.append((span, replacement_text))
    for span in doc.ents:
        if span._.has_coref:
            continue
        # We can assume it to be a singleton.
        replace, performed_singleton_replacements = get_replacement_text(
            span.label_,
            span.text,
            name_db,
            used_names,
            performed_singleton_replacements
        )
        replacements.append((span, replace))

    sorted_replacements = sorted(replacements, key=lambda rep: rep[0].start_char)
    texts = []
    current_pos = 0
    for span, replacement_text in sorted_replacements:
        if replacement_text is None:
            texts.append(doc.text[current_pos:span.end_char])
            current_pos = span.end_char
            continue
        if span.start_char < current_pos:
            continue
        texts.append(doc.text[current_pos:span.start_char])
        texts.append(replacement_text)
        current_pos = span.end_char
    texts.append(doc.text[current_pos:])
    return "".join(texts)

def get_replacements(doc, clusters):
    for cluster in clusters:
        pass

def get_ner(doc, cluster):
    doc.ner_
    pass

if __name__ == "__main__":
    app()
