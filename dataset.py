import glob
import os
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
import json
from sklearn.model_selection import StratifiedKFold
import random
import itertools
from datasets import Dataset

TRANSLATION_SCORES = {
    "en": 100,
    "fr": 68.1,
    "de": 67.4,
    "it": 61.2,
    "es": 59.1,
}


def get_genres(wikidata_dict):
    genres = wikidata_dict["claims"].get("P136", [])
    genre_ids = [e["mainsnak"]["datavalue"]["value"]["id"] for e in genres if e["mainsnak"]["snaktype"] != "novalue"]
    return genre_ids

@dataclass
class Story:
    wikidata_id: str  
    description: str
    titles: Dict[str, str]
    title: str
    summaries_original: Dict[str, str]
    summaries_translated: Dict[str, str]
    anonymized: Optional[Dict[str, str]]
    similarities: torch.Tensor
    similarities_labels: List[str]
    num_sentences: Dict[str, int]
    sentences: Dict[str, List[str]]
    genres: List[str]

    @classmethod
    def from_dict(cls, data, wikidata_data=None):
        if wikidata_data is not None:
            genres = get_genres(wikidata_data)
        else:
            genres = []
        sentences = {k: s["sentences"] for k, s in data.get("en_translated_summaries", {}).items()}
        num_sentences = {k: len(s["sentences"]) for k, s in data.get("en_translated_summaries", {}).items()}
        if "en" in (sents := data.get("split_into_sents")):
            num_sentences.update({"en": len(sents["en"])})
            sentences.update({"en": sents["en"]})
        return cls(
            wikidata_id=data["wikidata_id"],
            titles={k: (v or {}).get("value") for k, v in data.get("titles", {}).items()},
            title=data["title"],
            description=data["description"],
            summaries_original=data["summaries"],
            summaries_translated={k: s["text"] for k, s in data.get("en_translated_summaries", {}).items()},
            similarities_labels=data.get("similarity", {}).get("indexes"),
            similarities=torch.tensor(data.get("similarity", {}).get("similarities", [])),
            anonymized=data.get("anonymized"),
            num_sentences=num_sentences,
            sentences=sentences,
            genres=genres,
        )

    def remove_duplicates(self, threshold=0.6):
        out = {}
        sorted_labels = sorted(self.similarities_labels, key=lambda x: TRANSLATION_SCORES[x])
        sorted_similarities = [[v.item() for k, v in sorted(
            zip(self.similarities_labels, sim),
            key=lambda kv: TRANSLATION_SCORES[kv[0]],
            reverse=True
        )] for sim in self.similarities]
        for i, (lang, text) in enumerate(sorted(self.summaries_translated.items(), key=lambda kv: TRANSLATION_SCORES[kv[0]], reverse=True)):
            try:
                index = (sorted_labels or []).index(lang)
            except ValueError:
                breakpoint()
                print(lang)
                index = None
            try:
                max_value = max(sorted_similarities[index][:i])
            except ValueError:
                max_value = 0
            if index is not None and max_value > threshold:
                pass
            else:
                out[lang] = text
        return out

    def get_anonymized(self, min_sentences=0):
        return {lang : text for lang, text in self.anonymized.items() if self.num_sentences[lang] >= min_sentences}

    def get_all_summaries_en(self, max_similarity=0.6, min_sentences=0):
        en = self.summaries_original.get("en")
        summaries = []
        ids = []
        if en is not None:
            summaries.append(en)
            ids.append("en")
        no_dups = self.remove_duplicates() 
        summaries += [e for e in no_dups.values()]
        ids += [e for e in no_dups.keys()]
        ids = [id_ for id_ in ids if self.num_sentences[id_] >= min_sentences]
        summaries = [s for (id_, s) in zip(ids, summaries) if self.num_sentences[id_] >= min_sentences]
        return ids, summaries

    def __repr__(self):
        return f"<Story title='{self.title}' description='{self.description}'>"


class SummaryDataset():
    def __init__(self, data_path, only_include=[], stories=None):
        self.stories = stories or {}
        self.force_test_ids = set(open("data/test_ids.csv").readlines())
        if len(self.stories) > 0:
            return
        for file_name in glob.glob("data/summaries/*/*.json"):
            wikidata_id = os.path.splitext(os.path.basename(file_name))[0]
            wikidata_data = json.load(open(f"data/wikidata/{wikidata_id[:2]}/{wikidata_id}.json"))
            if len(only_include) > 0 and (wikidata_id not in only_include):
                continue
            else:
                self.stories[wikidata_id] = Story.from_dict(json.load(open(file_name)), wikidata_data)

    def __getitem__(self, i):
        return self.stories[i]

    def __len__(self):
        return len(self.stories)

    def __iter__(self):
        yield from self.stories.values()

    def perform_splits(self):
        test_stories = {id_: story for id_, story in self.stories.items() if id_ in self.force_test_ids}
        train_len = int(len(self.stories) / 100 * 80)
        dev_len = int(len(self.stories) / 100 * 10)
        test_len = len(self.stories) - dev_len - train_len
        to_split = list(set(self.stories.keys()) - set(test_stories.keys()))
        randomizer = random.Random(42)
        randomizer.shuffle(to_split)
        train_stories = {k: self.stories[k] for k in to_split[:train_len]}
        dev_stories = {k: self.stories[k] for k in to_split[train_len:train_len + dev_len]}
        test_stories.update({k: self.stories[k] for k in to_split[train_len + dev_len:]})
        return {k: self.__class__(data_path=None, stories=s) for k, s in [("train", train_stories), ("dev", dev_stories), ("test", test_stories)]}

    def chaturvedi_like_split(self, use_anonymized: bool = False, seed=1337):
        target_length_count = {
            2: 235,
            3: 20,
            4: 10,
            5: 1 # TODO: this should in all likelihood be 1 rather than 7, rerun experiments?
        }
        randomizer = random.Random(seed)
        ids = list(self.stories.keys())
        randomizer.shuffle(ids)
        by_length = defaultdict(list)
        out_file = open("in_chaturvedi_test.csv", "w")
        all_summaries = []
        all_summaries_test = []
        labels = []
        labels_test = []
        included = []
        for id_ in ids:
            if use_anonymized:
                summaries = self.stories[id_].anonymized.values()
            else:
                _, summaries = self.stories[id_].get_all_summaries_en()
            if len(by_length.get(len(summaries), [])) < target_length_count.get(len(summaries), 0):
                in_test_set = [True if randomizer.random() <= 0.8 else False for _ in range(len(summaries))]
                included.extend(in_test_set)
                test_summaries = [s for t, s in zip(in_test_set, summaries) if t]
                for i, (is_in, summary) in enumerate(zip(in_test_set, summaries)):
                    if is_in:
                        out_file.write(id_ + f"_{i}\n")
                labels.extend([id_] * len(summaries))
                labels_test.extend([id_] * len(test_summaries))
                all_summaries.extend(summaries)
                all_summaries_test.extend(summaries)
                by_length[len(summaries)].append(summaries)
        out_file.close()
        return all_summaries, labels, included
        

    def stratified_split(self, label_dict, seed=2):
        splitter = StratifiedKFold(n_splits=2, random_state=seed, shuffle=True)
        splits = list(splitter.split(list(label_dict.keys()), list(label_dict.values())))
        ids = list(label_dict.keys())
        return [[(label_dict[ids[i]], self[ids[i]]) for i in split] for split in splits[0]]

    def get_metadata_stats(self):
        book_count, movie_count, both_count = 0, 0, 0
        has_gutenberg = 0
        has_isbn = 0
        genre_counter = Counter()
        count = 0
        neither_count = 0
        for _, story in tqdm(self.stories.items()):
            data = json.load(open(f"data/wikidata/{story.wikidata_id[1:3]}/{story.wikidata_id[1:]}.json"))
            genres = data["claims"].get("P136", [])
            genre_ids = [e["mainsnak"]["datavalue"]["value"]["id"] for e in genres if e["mainsnak"]["snaktype"] != "novalue"]
            gutenberg = data["claims"].get("P2034", [])
            gutenberg_ids = [e["mainsnak"]["datavalue"]["value"] for e in gutenberg if e["mainsnak"]["snaktype"] != "novalue"]
            isbn = data["claims"].get("P212", [])
            isbns = [e["mainsnak"]["datavalue"]["value"] for e in isbn if e["mainsnak"]["snaktype"] != "novalue"]
            if len(gutenberg_ids) > 0:
                has_gutenberg += 1
            if len(isbns) > 0:
                has_isbn += 1
                print(has_isbn)
            if len(genres) > 0:
                genre_counter.update(genre_ids)
            else:
                genre_counter.update([None])
            is_instance_claims = data["claims"]["P31"]
            is_instance_target_ids = [e["mainsnak"]["datavalue"]["value"]["id"] for e in is_instance_claims]
            is_movie = "Q11424" in is_instance_target_ids
            is_book = "Q7725634" in is_instance_target_ids
            if is_book:
                book_count += 1
            if is_movie:
                movie_count += 1
            if is_movie and is_book:
                both_count += 1
            if not is_movie and not is_book:
                neither_count += 1
                print(story.description)
                print(story.wikidata_id)
            count += 1
        return {
            "neither_count": neither_count,
            "story_count": count,
            "num_books": book_count,
            "num_movies": movie_count,
            "num_both": both_count,
            "genres": genre_counter.most_common(),
            "has_gutenberg": has_gutenberg,
            "has_isbn": has_isbn
        }

    def get_lang_stats(self, sentence_lengths=True):
        counter = Counter()
        counter_no_duplicates = Counter()
        length_counter = defaultdict(Counter)
        i = 0
        for story in tqdm(self.stories.values()):
            counter_no_duplicates.update(story.remove_duplicates().keys())
            counter.update(story.summaries_original.keys())
            if sentence_lengths:
                import ersatz
                for lang, summary in story.summaries_original.items():
                    sentences = ersatz.split_text(text=summary, model=lang.replace("it", "default-multilingual"))
                    if sentences is not None:
                        length_counter[lang].update([len(sentences)])
            i += 1
        return {
            "languages": dict(counter),
            "languages_direct_translations_removed": dict(counter_no_duplicates),
            "lengths_per_language": dict(length_counter),
        }


def pair_combinations(iterable):
    out = []
    for i, a in enumerate(iterable):
        for j, b in enumerate(iterable):
            if i >= j:
                continue
            else:
                out.append((a, b))
    return out


def adjacent_pairs(iterable):
    out = []
    data = list(iterable)
    for i in range(len(data)):
        if i < len(iterable) - 1:
            out.append((data[i], data[i + 1]))
        else:
            out.append((data[i], data[i - 1]))
    return out


def shuffle_into(shuffle_from, shuffle_into, randomizer=random):
    ratio = len(shuffle_from) / (len(shuffle_into) + len(shuffle_from))
    out = []
    i = 0
    j = 0
    while (len(shuffle_from) + len(shuffle_into)) != len(out):
        if i < len(shuffle_from) and j < len(shuffle_into):
            if randomizer.random() < ratio:
                out.append(shuffle_from[i])
                i += 1
            else:
                out.append(shuffle_into[j])
                j += 1
        elif i < len(shuffle_from) and j == len(shuffle_into):
            out.append(shuffle_from[i])
            i += 1
        elif i == len(shuffle_from) and j < len(shuffle_into):
            out.append(shuffle_into[j])
            j += 1
    return out


def remove_duplicates(x):
    breakpoint()
    return x

class SimilarityDataset():
    def __init__(self, path, anonymized=True, min_sentences=0, negative_sample_scale=1.0, seed=42, clusters_together: bool = False):
        self.summary_dataset = SummaryDataset(path)
        splits = self.summary_dataset.perform_splits()
        self.summaries = {}
        randomizer = random.Random(seed)
        self.splits = {}
        for split in ["train", "dev", "test"]:
            if anonymized:
                summaries_getter = lambda x, min_sentences: x.get_anonymized(min_sentences=min_sentences).values()
            else:
                summaries_getter = lambda x, min_sentences: x.get_all_summaries_en(min_sentences=min_sentences)[1]
            stories = list(splits[split].stories.values())
            random.shuffle(stories)
            combination_getter = lambda x: pair_combinations(x)
            if clusters_together:
                combination_getter = lambda x: adjacent_pairs(x)
            positive_samples = list(
                itertools.chain.from_iterable(
                    [
                        [(story.wikidata_id, pair) for pair in combination_getter(summaries_getter(story, min_sentences))]
                        for story in stories
                    ]
                )
            )
            num_negative_samples = int(len(positive_samples) * negative_sample_scale)
            negative_samples = []
            for _ in range(num_negative_samples):
                story_a = randomizer.choice(stories)
                story_b = None
                while story_b == story_a or story_b is None:
                    story_b = randomizer.choice(stories)
                negative_samples.append(([story_a.wikidata_id, story_b.wikidata_id], (
                    randomizer.choice(list(story_b.get_anonymized().values())),
                    randomizer.choice(list(story_b.get_anonymized().values()))
                )))
            negative_samples = [{"text_a": sample[0], "text_b": sample[1], "label": -1, "text_ids": ids} for (ids, sample) in negative_samples]
            positive_samples = [{"text_a": sample[0], "text_b": sample[1], "label": 1, "text_ids": [id_, id_]} for (id_, sample) in positive_samples]
            if clusters_together:
                random.shuffle(negative_samples)
                samples = shuffle_into(negative_samples, positive_samples, randomizer)
            else:
                samples = negative_samples + positive_samples
                randomizer.shuffle(samples)
            self.splits[split] = Dataset.from_list(samples)

    def __getitem__(self, split):
        return self.splits[split]


class Split():
    def __init__(self, items):
        self.samples = items

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        yield from self.samples

    def __getitem__(self, i):
        return self.samples[i]
