import glob
import os
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict
from tqdm import tqdm
import json
from sklearn.model_selection import StratifiedKFold


@dataclass
class Story:
    wikidata_id: str  
    description: str
    titles: Dict[str, str]
    title: str
    summaries_original: Dict[str, str]
    summaries_translated: Dict[str, str]

    @classmethod
    def from_dict(cls, data):
        return cls(
            wikidata_id=data["wikidata_id"],
            titles={k: (v or {}).get("value") for k, v in data.get("titles", {}).items()},
            title=data["title"],
            description=data["description"],
            summaries_original=data["summaries"],
            summaries_translated={k: s["text"] for k, s in data.get("en_translated_summaries", {}).items()},
        )

    def __repr__(self):
        return f"<Story title='{self.title}' description='{self.description}'>"


class SummaryDataset():
    def __init__(self, data_path, only_include=[]):
        self.stories = {}
        for file_name in glob.glob("data/summaries/*/*.json"):
            wikidata_id = os.path.splitext(os.path.basename(file_name))[0]
            if len(only_include) > 0 and (wikidata_id not in only_include):
                continue
            else:
                self.stories[wikidata_id] = Story.from_dict(json.load(open(file_name)))

    def __getitem__(self, i):
        return self.stories[i]

    def __len__(self):
        return len(self.stories)

    def stratified_split(self, label_dict, seed=2):
        splitter = StratifiedKFold(n_splits=2, random_state=seed, shuffle=True)
        splits = list(splitter.split(list(label_dict.keys()), list(label_dict.values())))
        ids = list(label_dict.keys())
        return [[(label_dict[ids[i]], self[ids[i]]) for i in split] for split in splits[0]]

    def get_metadata_stats(self):
        book_count, movie_count, both_count = 0, 0, 0
        genre_counter = Counter()
        for _, story in tqdm(self.stories.items()):
            data = json.load(open(f"data/wikidata/{story.wikidata_id[1:3]}/{story.wikidata_id[1:]}.json"))
            genres = data["claims"].get("P136", [])
            genre_ids = [e["mainsnak"]["datavalue"]["value"]["id"] for e in genres if e["mainsnak"]["snaktype"] != "novalue"]
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
        return {"num_books": book_count, "num_movies": movie_count, "num_both": both_count, "genres": genre_counter.most_common()}

    def get_lang_stats(self, sentence_lengths=True):
        counter = Counter()
        length_counter = defaultdict(Counter)
        i = 0
        for story in tqdm(self.stories.values()):
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
            "lengths_per_language": dict(length_counter),
        }
