from collections import defaultdict
import json

import requests
import os
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    MistralConfig,
    MistralModel,
    TrainerCallback,
)
from typer import Typer
from peft import PeftModel, PeftConfig
import peft
from sentence_transformers.util import cos_sim
import itertools
from chaturvedi import MovieSummaryDataset
import dataset
from datetime import datetime
import more_itertools
from tqdm import tqdm
import torch
from torch import nn
from dataclasses import dataclass
from gradient_cache_trainer import (
    GradientCacheTrainer,
    LLM2VecGradientCacheTrainer,
    get_repr,
)
from news_sim import SemEvalDataset
from scipy.stats import pearsonr
from sklearn.metrics import classification_report, silhouette_score

from roc_stories import ROCStoriesDataset
import tell_me_again

MAX_INPUT_LENGTH = 5000
# MAX_INPUT_LENGTH = 2500
# MAX_INPUT_LENGTH = 8192
QUERY_PREFIX = "Retrieve stories with a similar narrative to the given story: "
# QUERY_PREFIX = "Retrieve stories with similar narrative to the given story: "

app = Typer()


def last_token_pooling(texts, last_hidden_state):
    # This is okay if the padding side is left
    assert (texts.attention_mask[:, -1] == 1).all()
    return last_hidden_state[:, -1]


class ContrastiveLlama(MistralModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_loss = nn.CosineEmbeddingLoss()
        self.post_init()

    def __call__(self, texts_a, texts_b, labels):
        encoded_a = self.forward(**texts_a)
        emb_a = last_token_pooling(texts_a, encoded_a.last_hidden_state)
        encoded_b = self.forward(**texts_b)
        emb_b = last_token_pooling(texts_b, encoded_b.last_hidden_state)
        loss = self.sim_loss(emb_a, emb_b, labels)
        return loss, emb_a, emb_b


def get_model(base_model_name, adapter_model_name, for_training=True):
    base_model = AutoModel.from_pretrained(
        base_model_name, device_map="auto"
    )  # , torch_dtype=torch.float16)
    if for_training:
        base_model.eval()
    if adapter_model_name is None:
        peft_config = peft.LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        print(peft_config)
        base_model.add_adapter(peft_config)
    else:
        base_model.load_adapter(adapter_model_name)
    return base_model


@app.command()
def news():
    """
    Test SemEval 2022 Task 8 correlations.

    We expect the similarity to correlate highly with the narrative dimension and ideally not much else.

    Values with our checkpoint: ../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9
    geography 0.48460
    entities  0.50900
    time      0.16039
    narrative 0.47906
    overall   __0.51600__
    style     0.42895
    tone      0.45270

    Interestingly this is geography biased, maybe this is because we do not anonymize countries.
    """
    base_model_name = "mistralai/Mistral-7B-v0.1"
    news_similarity_path = Path(os.environ["NEWS_SIM_PATH"])
    dataset_train = SemEvalDataset(
        news_similarity_path / "train.csv", news_similarity_path / "all_data"
    )
    dataset = SemEvalDataset(
        news_similarity_path / "eval.csv", news_similarity_path / "all_data"
    )
    model = get_model(
        base_model_name,
        "../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9",
        for_training=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    train, dev = dataset_train.random_split(0.8)
    predicted_sims = []
    i = 0
    sim_dict = defaultdict(list)
    for pair in tqdm(dev):
        with torch.no_grad():
            encoded_a = model(
                **tokenizer(
                    pair.article_1.text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_INPUT_LENGTH,
                ).to("cuda:0")
            )
            encoded_b = model(
                **tokenizer(
                    pair.article_2.text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_INPUT_LENGTH,
                ).to("cuda:0")
            )
            sim = get_repr(encoded_a).squeeze() @ get_repr(encoded_b).squeeze()
            predicted_sims.append(sim.item())
            for k, v in pair.get_similarity_dict().items():
                sim_dict[k].append(v)
            i += 1
            if i > 1000:
                break
    for k, v in sim_dict.items():
        print(k, pearsonr(predicted_sims, v))


@app.command()
def chaturvedi_doc2vec(anonymized: bool = False):
    from tokenizers.pre_tokenizers import BertPreTokenizer

    tokenizer = BertPreTokenizer()
    dataset = MovieSummaryDataset(
        Path(os.environ["MOVIE_REMAKE_PATH"]) / "movieRemakesManuallyCleaned.tsv",
        Path(os.environ["MOVIE_REMAKE_PATH"]) / "testInstances.csv",
        Path(os.environ["MOVIE_REMAKE_PATH"]) / "remakes-anon.csv",
    )
    texts = [s.text_anonymized if anonymized else s.text for s in dataset]
    cluster_ids = [s.cluster_id for s in dataset]
    movie_ids = [s.movie_id for s in dataset]
    encoded = []
    print(dataset.test_movies)
    for text in tqdm(texts):
        tokens = [text for text, _span in tokenizer.pre_tokenize_str(text)]
        resp = requests.post("http://localhost:5000/embed", json={"tokens": tokens})
        encoded.append(torch.tensor(resp.json()["vector"]))
    encoded = torch.stack(encoded)
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
    print(correct / total)
    gold_sim_matrix = label_list_to_matrix(torch.tensor(cluster_ids))[
        [x in dataset.test_movies for x in movie_ids]
    ]
    test_similarities = similarities[[x in dataset.test_movies for x in movie_ids]]
    eval_result = eval_similarities(gold_sim_matrix, test_similarities)
    print(eval_result)
    print_eval_result(eval_result)


@app.command()
def chaturvedi(
    model_path: str = "e5-mistral-7b-instruct-adapters",
    anonymized: bool = False,
    encode_query_separately: bool = False,
):
    """
    Checkpoint  Score
    ../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9 0.8047210300429185
    e5-mistral-7b-instruct-adapters                                               0.45278969957081544
    e5-mistral-7b-isntruct-adapters + query_prefix                                0.44206008583690987

    anonymized:
    ../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9 0.8068669527896996
    e5-mistral-7b-instruct-adapters                                               0.16738197424892703
    e5-mistral-7b-instruct-adapters + query prefix                                0.16523605150214593
    """
    # TOdo: for e5 we need the task prefix to make it fair :/ (maybe, does it do symmetric embeddings)
    dataset = MovieSummaryDataset(
        Path(os.environ["MOVIE_REMAKE_PATH"]) / "movieRemakesManuallyCleaned.tsv",
        Path(os.environ["MOVIE_REMAKE_PATH"]) / "testInstances.csv",
        Path(os.environ["MOVIE_REMAKE_PATH"]) / "remakes-anon.csv",
    )
    base_model_name = "mistralai/Mistral-7B-v0.1"
    model = get_model(base_model_name, model_path, for_training=False).to("cuda:0")
    # model = get_model(base_model_name, "../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9", for_training=False).to("cuda:0")
    model = model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    texts = [s.text_anonymized if anonymized else s.text for s in dataset]
    cluster_ids = [s.cluster_id for s in dataset]
    movie_ids = [s.movie_id for s in dataset]
    encoded = []
    encoded_queries = []
    # query_prefix = ""
    print(QUERY_PREFIX)
    with torch.no_grad():
        for text in tqdm(texts):
            batch = tokenizer(QUERY_PREFIX + text, return_tensors="pt")
            output = model(**batch.to("cuda:0"))
            encoded.append(output.last_hidden_state[:, -1].to("cpu").squeeze())
            if encode_query_separately:
                batch_query = tokenizer(QUERY_PREFIX + text, return_tensors="pt")
                output = model(**batch_query.to("cuda:0"))
                encoded_queries.append(
                    output.last_hidden_state[:, -1].to("cpu").squeeze()
                )
    encoded = torch.stack(encoded)
    if encode_query_separately:
        encoded_queries = torch.stack(encoded_queries)
        similarities = cos_sim(encoded, encoded_queries)
    else:
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
    print(correct / total)


@app.command()
def token_lengths(use_anonymized: bool = True, min_length: int = 0, split: str = "dev"):
    base_model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    ds = dataset.SimilarityDataset("data", negative_sample_scale=0.0)
    lengths = defaultdict(int)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    ds = dataset.SummaryDataset("data")
    splits = ds.perform_splits()
    if use_anonymized:
        all_labeled = list(
            itertools.chain.from_iterable(
                [
                    zip(
                        itertools.repeat(i),
                        v.get_anonymized(min_sentences=min_length).values(),
                    )
                    for i, v in enumerate(splits[split].stories.values())
                ]
            )
        )
    else:
        all_labeled = list(
            itertools.chain.from_iterable(
                [
                    zip(
                        itertools.repeat(i),
                        v.get_all_summaries_en(min_sentences=min_length)[1],
                    )
                    for i, v in enumerate(splits[split].stories.values())
                ]
            )
        )
    labels, texts = zip(*all_labeled)
    for text in tqdm(texts):
        tokenized = tokenizer(text)
        lengths[len(tokenized["input_ids"])] += 1
    max_length = max(lengths.keys())
    for x in range(max_length + 1):
        print(x, lengths[x], sep=",")
    # The results basically tell us: 3k should be plenty


def eval_similarities(gold_sims, predicted_sims):
    """
    Given NxN matrices returns a dict with all relevant eval metrics.
    """
    from torchmetrics.retrieval import (
        RetrievalMAP,
        RetrievalNormalizedDCG,
        RetrievalRPrecision,
    )

    results = {}
    map_scorer = RetrievalMAP("error")
    ndcg_scorer = RetrievalNormalizedDCG("error")
    r_precision_scorer = RetrievalRPrecision("error")
    query_id = (
        torch.arange(gold_sims.shape[0]).unsqueeze(1).repeat(1, gold_sims.shape[1])
    )
    predicted_sims = predicted_sims.fill_diagonal_(0)
    results["MAP"] = map_scorer(
        predicted_sims.flatten(),
        gold_sims.flatten().bool(),
        indexes=query_id.flatten().long(),
    ).item()
    results["ndcg"] = ndcg_scorer(
        predicted_sims.flatten(),
        gold_sims.flatten().bool(),
        indexes=query_id.flatten().long(),
    ).item()
    results["r-precision"] = r_precision_scorer(
        predicted_sims.flatten(),
        gold_sims.flatten().bool(),
        indexes=query_id.flatten().long(),
    ).item()
    max_cluster_size = gold_sims.sum(1).max()
    p_at_n = []
    for k in range(1, max_cluster_size.int().item()):
        is_in_topk = torch.zeros_like(predicted_sims)
        _top_k_values, top_k_indices = predicted_sims.topk(k, dim=1)
        rows = torch.arange(gold_sims.size(0)).unsqueeze(1)
        is_in_topk[rows, top_k_indices] = 1
        at_n = gold_sims.sum(1) == k
        p_at_k = (is_in_topk.int() & gold_sims.int()).sum(1) / k
        p_at_n.append(p_at_k.masked_select(at_n))
        if k == 1:
            results[f"P@{k}"] = p_at_k.mean().item()
    results["P@N"] = torch.cat(p_at_n).mean().item()
    return results


def print_eval_result(eval_result):
    print(",".join(eval_result.keys()))
    print(",".join([f"{v * 100:.2f}" for v in eval_result.values()]))


def label_list_to_matrix(labels):
    out = torch.zeros(len(labels), len(labels))
    for x in range(labels.max() + 1):
        matches = (labels == x).to(int)
        out += matches.unsqueeze(0) * matches.unsqueeze(1)
    out.fill_diagonal_(0)
    return out


@app.command()
def test(
    adapter_paths: List[str],
    use_anonymized: bool = True,
    min_length: int = 0,
    batch_size: int = 1,
    split: str = "dev",
    quick: bool = False,
    collect_lengths: bool = False,
    save_embeddings: Optional[str] = None,
):
    base_model_name = "mistralai/Mistral-7B-v0.1"
    # model = get_model(base_model_name, "sim-trainer2024-02-21T19:26:14.519764/checkpoint-1500/", for_training=False).to("cuda:0"): 0.88
    ds = tell_me_again.StoryDataset()
    splits = ds.perform_splits()
    if use_anonymized:
        all_labeled = list(
            itertools.chain.from_iterable(
                [
                    zip(
                        itertools.repeat(i),
                        itertools.repeat(v.wikidata_id),
                        v.get_anonymized(min_sentences=min_length).values(),
                    )
                    for i, v in enumerate(splits[split].stories.values())
                    if len(v.get_anonymized(min_sentences=min_length)) > 1
                ]
            )
        )
    else:
        all_labeled = list(
            itertools.chain.from_iterable(
                [
                    zip(
                        itertools.repeat(i),
                        itertools.repeat(v.wikidata_id),
                        v.get_all_summaries_en(min_sentences=min_length)[1],
                    )
                    for i, v in enumerate(splits[split].stories.values())
                    if len(v.get_anonymized(min_sentences=min_length)) > 1
                ]
            )
        )
    labels, wiki_ids, texts = zip(*all_labeled)
    if quick:
        labels = labels[:11]
        texts = texts[:11]
    results_csv = open("results.csv", "w")
    for adapter_path in adapter_paths:
        model = get_model(base_model_name, adapter_path, for_training=False).to(
            "cuda:0"
        )
        model = model.to(torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        batches = []
        texts = [QUERY_PREFIX + t for t in texts]  # let's not make them too long
        for texts_for_batch in tqdm(more_itertools.chunked(texts, batch_size)):
            tokenized = tokenizer(
                texts_for_batch, return_tensors="pt", max_length=MAX_INPUT_LENGTH
            )
            batches.append(tokenized)
        model.eval()
        batches_encoded = []
        with torch.no_grad():
            for batch in tqdm(batches):
                output = model(**batch.to("cuda:0"))
                batches_encoded.append(output.last_hidden_state[:, -1].to("cpu"))
        encoded = torch.cat(batches_encoded)
        similarities = cos_sim(encoded, encoded)
        similarities.fill_diagonal_(0)
        matches = similarities.argmax(1)
        correct = 0
        total = 0
        for source, match in enumerate(matches):
            if labels[source] == labels[match]:
                correct += 1
            total += 1
        print(adapter_path, str(correct / total), sep="\t", file=results_csv)
        print("P@1", correct / total)
        eval_result = eval_similarities(
            label_list_to_matrix(torch.tensor(labels)), similarities
        )
        print(eval_result)
        print_eval_result(eval_result)
        if save_embeddings:
            out_file = open(save_embeddings, "wb")
            out_file_ids = open(save_embeddings + ".json", "w")
            torch.save(encoded, out_file)
            breakpoint()
            json.dump(wiki_ids, out_file_ids)
        # With anonymization: 0.139
        # W/o anonymiztaion: 0.596
        # These are not competitive with e.g. sentence-T5


@app.command()
def test_doc2vec(
    use_anonymized: bool = True,
    min_length: int = 0,
    quick: bool = False,
    split: str = "dev",
):
    from tokenizers.pre_tokenizers import BertPreTokenizer

    tokenizer = BertPreTokenizer()
    ds = tell_me_again.StoryDataset()
    splits = ds.perform_splits()
    if use_anonymized:
        all_labeled = list(
            itertools.chain.from_iterable(
                [
                    zip(
                        itertools.repeat(i),
                        v.get_anonymized(min_sentences=min_length).values(),
                    )
                    for i, v in enumerate(splits[split].stories.values())
                    if len(v.get_anonymized(min_sentences=min_length)) > 1
                ]
            )
        )
    else:
        all_labeled = list(
            itertools.chain.from_iterable(
                [
                    zip(
                        itertools.repeat(i),
                        v.get_all_summaries_en(min_sentences=min_length)[1],
                    )
                    for i, v in enumerate(splits[split].stories.values())
                    if len(v.get_anonymized(min_sentences=min_length)) > 1
                ]
            )
        )
    all_labeled = [
        (id_, text)
        for id_, text in all_labeled
        if min_length is None or len(text) >= min_length
    ]
    labels, texts = zip(*all_labeled)
    if quick:
        labels = labels[:1002]
        texts = texts[:1002]
    encoded = []
    with torch.no_grad():
        for text in tqdm(texts):
            tokens = [text for text, _span in tokenizer.pre_tokenize_str(text)]
            resp = requests.post("http://localhost:5000/embed", json={"tokens": tokens})
            encoded.append(torch.tensor(resp.json()["vector"]))
    encoded = torch.stack(encoded)
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
    eval_result = eval_similarities(
        label_list_to_matrix(torch.tensor(labels)), similarities
    )
    print(eval_result)
    print_eval_result(eval_result)


@dataclass
class DataCollatorForSimilarityModeling:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, samples):
        texts_a = self.tokenizer(
            [sample["text_a"] for sample in samples],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        texts_b = self.tokenizer(
            [sample["text_b"] for sample in samples],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        ids = torch.tensor([[int(id_[1:]) for id_ in s["text_ids"]] for s in samples])
        ids_a = ids[:, 0]
        ids_b = ids[:, 1]
        target = ids_a[:, None] == ids_b
        # labels = torch.tensor([sample["label"] for sample in samples])
        # length = max(texts_a["input_ids"].shape()[-1], texts_b["input_ids"].shape()[-1])
        return {"texts_a": texts_a, "texts_b": texts_b, "labels": target}


def clip_texts(item):
    item.update(
        {
            "text_a": QUERY_PREFIX + item["text_a"],
            "text_b": QUERY_PREFIX + item["text_b"],
            "length": max([len(item["text_a"]), len(item["text_b"])]),
        }
    )
    return item


class GradientCacheCallbacks(TrainerCallback):
    pass


@app.command()
def train():
    timestamp = datetime.utcnow().isoformat()
    ds = dataset.SimilarityDataset(
        "data",
        negative_sample_scale=0.0,
        min_sentences=10,
        max_sentences=50,
        clusters_together=True,
    )
    base_model_name = "mistralai/Mistral-7B-v0.1"
    # model = get_model(base_model_name, None)
    model = get_model(base_model_name, "e5-mistral-7b-instruct-adapters")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    effective_batch_size = 256
    base_lr = 1e-5
    # effective_lr = base_lr * effective_batch_size / 1024 # Recommended in https://arxiv.org/pdf/2304.12210.pdf
    optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if "lora" in name], lr=base_lr
    )
    num_steps = (len(ds["train"]) // effective_batch_size) + 1
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, effective_lr, total_steps=num_steps, pct_start=0.1)
    training_args = TrainingArguments(
        "sim-trainer" + timestamp,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=effective_batch_size,
        logging_steps=1,
        save_steps=1,
        max_grad_norm=3,
        fp16=True,
        num_train_epochs=1,
        # length_column_name="length",
        # group_by_length=True, # This can substantially speed up training
    )
    trainer = GradientCacheTrainer(
        model,
        training_args,
        train_dataset=ds["train"].map(clip_texts),
        eval_dataset=ds["dev"].map(clip_texts),
        data_collator=DataCollatorForSimilarityModeling(tokenizer),
        optimizers=[optimizer, None],
    )
    trainer.train()


@app.command()
def train_llm2vec():
    timestamp = datetime.utcnow().isoformat()
    ds = dataset.SimilarityDataset(
        "data",
        negative_sample_scale=0.0,
        min_sentences=10,
        max_sentences=50,
        clusters_together=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
    )
    config = AutoConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        trust_remote_code=True,
        config=config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
        is_trainable=True,
        device_map="auto",
    )
    # model = model.merge_and_unload()
    # effective_batch_size = 2
    effective_batch_size = 1024
    base_lr = 4e-5
    # model.to("cuda:0")
    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if "lora" in name], lr=base_lr
    )
    training_args = TrainingArguments(
        "sim-trainer" + timestamp,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=effective_batch_size,
        logging_steps=1,
        save_steps=1,
        max_grad_norm=3,
        fp16=True,
        num_train_epochs=1,
    )
    trainer = LLM2VecGradientCacheTrainer(
        model,
        training_args,
        train_dataset=ds["train"].map(clip_texts),
        eval_dataset=ds["dev"].map(clip_texts),
        data_collator=DataCollatorForSimilarityModeling(tokenizer),
        optimizers=[optimizer, None],
    )
    trainer.train()


def prepare_model_and_tokenizer(base_model_name, adapter_name):
    model = get_model(base_model_name, adapter_name, for_training=False)
    model = model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    return model, tokenizer


@app.command()
def roc_stories(
    split: str = "dev",
    adapter_name: str = "../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9",
):
    base_model_name = "mistralai/Mistral-7B-v0.1"
    # model = get_model(base_model_name, "e5-mistral-7b-instruct-adapters").to("cuda:0")
    model, tokenizer = prepare_model_and_tokenizer(base_model_name, adapter_name)
    ds = ROCStoriesDataset("../roc_stories", split)
    labels = []
    predictions = []
    with torch.no_grad():
        for item in tqdm(ds):
            anchor = QUERY_PREFIX + " ".join(item.sentences)
            choices = [anchor + " " + s for s in item.candidate_endings]
            batch = tokenizer([anchor] + choices, return_tensors="pt", padding=True).to(
                "cuda:0"
            )
            encoded = model(**batch)
            sim = get_repr(encoded) @ get_repr(encoded).transpose(0, 1)
            prediction = (sim[0][1] < sim[0][2]).int()
            predictions.append(prediction)
            labels.append(item.label)
    if split == "dev":
        labels = torch.tensor(labels)
        predictions = torch.tensor(predictions)
        report = classification_report(labels, predictions)
        print(report)
        report = classification_report(labels, predictions, output_dict=True)
        print(report)
    elif split == "test":
        predictions = [p + 1 for p in predictions]
        in_file = open(
            "../roc_stories/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv",
            "r",
        )
        out_file = open("roc-stories-test.csv", "w")
        out_file.write(next(in_file).strip() + ",AnswerRightEnding\r\n")
        i = 0
        for line in in_file:
            out_file.write(line.strip() + "," + str(predictions[i].item()) + "\r\n")
            i += 1


@app.command()
def genres():
    """
    Distinguishing between crime and thriller is more or less impossible for our models
    e5: 0.017751
    ../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9: 0.042648430

    Crime vs romance:
    e5: 0.03778897459761644
    ../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9: 0.11215

    This can be considered a bit of an improvement, but we will need to check tsne tbh.
    Either way: there is a slight improvement but it is not enough for any usecase. Hardly surprising as we are essentially also training on this (but keep in mind there are many overlapping samples in the training data).

    This function is somehow not quite deterministic i.e. I had different sizes for the categories.
    """
    base_model_name = "mistralai/Mistral-7B-v0.1"
    adapter_name = (
        "../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9"
    )
    # adapter_name = "e5-mistral-7b-instruct-adapters"
    model, tokenizer = prepare_model_and_tokenizer(base_model_name, adapter_name)
    genre_to_id = {
        x.split(",")[0]: x.split(",")[1]
        for x in open("data/genres/joined.csv").readlines()
    }
    id_to_genre = {v: k for k, v in genre_to_id.items()}
    genres = ["crime film", "romance film"]
    banned_genres = []
    genre_ids = [genre_to_id[g_id] for g_id in genres]
    banned_genre_ids = [genre_to_id[g_id] for g_id in banned_genres]
    # We need to require that they are only in exactly one of the pair
    stories = tell_me_again.StoryDataset().perform_splits()["dev"]
    stories_per_genre = defaultdict(list)
    to_embed = []
    labels = []
    for story in stories:
        shared_genres = set(story.genres) & set(genre_ids)
        if (
            len(shared_genres) == 1
            and len(set(story.genres) & set(banned_genre_ids)) == 0
        ):
            genre = list(shared_genres)[0]
            stories_per_genre[genre].append(story)
            try:
                to_embed.append(story.summaries_original["en"])
            except KeyError:
                to_embed.append(
                    list(
                        sorted(
                            story.summaries_translated.values(), key=lambda x: len(x)
                        )
                    )[0]
                )
            labels.append(genre == genre_to_id["crime film"])
    for k, v in stories_per_genre.items():
        print(id_to_genre[k], len(v))
    encoded = []
    with torch.no_grad():
        for s in tqdm(to_embed):
            batch = tokenizer(
                s,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
            ).to("cuda:0")
            embedding = get_repr(model(**batch))
            encoded.append(embedding.squeeze())
    encoded = torch.stack(encoded)
    sims = encoded @ encoded.transpose(0, 1)
    dists = 1 - sims
    dists.fill_diagonal_(0.0)
    score = silhouette_score(dists.cpu(), labels, metric="precomputed")
    print(score)


@app.command()
def sim_delta():
    """
    Find texts with a very large delta in similarity from one model to another.
    """
    stories = tell_me_again.StoryDataset().perform_splits()["dev"]
    for story in stories:
        en_summary = story.summaries_original.get("en")
        if en_summary is not None:
            en_summary
    # Output some html here


@app.command()
def retellings(
    model_path: str = "e5-mistral-7b-instruct-adapters",
    anonymized: bool = False,
    encode_query_separately: bool = False,
):
    dataset = MovieSummaryDataset(
        "data/retellings/movieRemakesManuallyCleaned.tsv",
        "data/retellings/testInstances.csv",
        None,
    )
    base_model_name = "mistralai/Mistral-7B-v0.1"
    model = get_model(base_model_name, model_path, for_training=False).to("cuda:0")
    model = model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    texts = [s.text_anonymized if anonymized else s.text for s in dataset]
    cluster_ids = [s.cluster_id for s in dataset]
    movie_ids = [s.movie_id for s in dataset]
    encoded = []
    encoded_queries = []
    # query_prefix = ""
    print(QUERY_PREFIX)
    with torch.no_grad():
        for text in tqdm(texts):
            batch = tokenizer(QUERY_PREFIX + text, return_tensors="pt")
            output = model(**batch.to("cuda:0"))
            encoded.append(output.last_hidden_state[:, -1].to("cpu").squeeze())
            if encode_query_separately:
                batch_query = tokenizer(QUERY_PREFIX + text, return_tensors="pt")
                output = model(**batch_query.to("cuda:0"))
                encoded_queries.append(
                    output.last_hidden_state[:, -1].to("cpu").squeeze()
                )
    encoded = torch.stack(encoded)
    if encode_query_separately:
        encoded_queries = torch.stack(encoded_queries)
        similarities = cos_sim(encoded, encoded_queries)
    else:
        similarities = cos_sim(encoded, encoded)
    similarities.fill_diagonal_(0)
    match = similarities.argmax(1)
    total = 0
    correct = 0
    for a, b in enumerate(match):
        if movie_ids[a] not in dataset.test_movies:
            continue
        if cluster_ids[a] == cluster_ids[b]:
            correct += 1
        total += 1
    print(correct / total)


if __name__ == "__main__":
    app()
