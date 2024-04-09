import os
import csv
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# from replacement import ReplacementDataset
from torch.utils.data.dataset import Dataset
from pathlib import Path

def main():
    out_file = open("results.csv", "w")
    for model_name in ["sentence-transformers/sentence-t5-large", "all-mpnet-base-v2", "finetuned-LaBSE-narrative",  "sentence-transformers/LaBSE"]:
        hits_at_1 = eval_model(model_name)
        out_file.write(f"{model_name}\t{hits_at_1}\n")


def eval_model(model_name):
    dataset = MovieSummaryDataset(Path(os.environ["MOVIE_REMAKE_PATH"]) / "movieRemakesManuallyCleaned.tsv", Path(os.environ["MOVIE_REMAKE_PATH"]) / "testInstances.csv")
    model = SentenceTransformer(model_name)
    texts = [s.text for s in dataset]
    cluster_ids = [s.cluster_id for s in dataset]
    movie_ids = [s.movie_id for s in dataset]
    encoded = model.encode(texts, show_progress_bar=True)
    similarities = cos_sim(encoded, encoded)
    similarities.fill_diagonal_(0)
    match = similarities.argmax(1)
    # ds = ReplacementDataset(texts[5:8])
    # breakpoint()
    total = 0
    correct = 0
    for a, b in enumerate(match):
        if movie_ids[a] not in dataset.test_movies:
            continue
        if cluster_ids[a] == cluster_ids[b]:
            correct += 1
        total += 1
    return correct / total


@dataclass
class Summary:
    title: str
    text: str
    text_anonymized: str
    cluster_id: int
    movie_id: int


class MovieSummaryDataset(Dataset):
    def __init__(self, path, test_instances_path, csv_anon_path):
        data = {}
        self.summaries = []
        in_file = open(path)
        csv_reader = csv.reader(in_file, delimiter="\t")
        if csv_anon_path:
            csv_reader_anon = csv.reader(open(csv_anon_path), delimiter=",")
        else:
            csv_reader_anon = None
        test_instances_file = open(test_instances_path)
        next(test_instances_file)
        self.test_movies = set([int(l[1]) for l in csv.reader(test_instances_file)])
        for line in csv_reader:
            print(line)
            cluster_id, *fields = line
            fields = [f.strip() for f in fields]
            cluster = list(split_into_three(fields))
            data[cluster_id] = cluster
        lengths = []
        for k, v in data.items():
            #lengths.append(len([e for e in v if int(e[0]) in self.test_movies]))
            lengths.append(len([e for e in v ]))
            if len(v) == 7:
                print(*[film[1] for film in v])
        print("Mean length", sum(lengths) / len(lengths))
        print("Max length", max(lengths))
        for n in range(1, 8):
            print("Out of", len(lengths), "there are", len([l for l in lengths if l == n]), f"clusters with length {n}")
        for cluster_id, summaries in data.items():
            for id_, title, text in summaries:
                if csv_reader_anon is not None:
                    text_anon = next(csv_reader_anon)[1]
                else:
                    text_anon = None
                self.summaries.append(
                    Summary(
                        text=text,
                        text_anonymized=text_anon,
                        cluster_id=int(cluster_id),
                        title=title,
                        movie_id=int(id_),
                    )
                )

    def __getitem__(self, i):
        return self.summaries[i]

    def __len__(self):
        return len(self.summaries)


def split_into_three(fields):
    for x in range(0, 1000, 3):
        if x >= len(fields):
            return
        yield fields[x:x+3]


if __name__ == "__main__":
    main()
