from collections import defaultdict
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase, PreTrainedModel, MistralConfig, MistralModel, TrainerCallback
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
from gradient_cache_trainer import GradientCacheTrainer, get_repr
from news_sim import SemEvalDataset
from scipy.stats import pearsonr
from sklearn.metrics import classification_report

from roc_stories import ROCStoriesDataset

#MAX_INPUT_LENGTH = 5000
MAX_INPUT_LENGTH = 50

app = Typer()
    
def last_token_pooling(texts, last_hidden_state):
    # This is okay if the padding side is left
    assert (texts.attention_mask[:,-1] == 1).all()
    return last_hidden_state[:,-1]



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
    base_model = AutoModel.from_pretrained(base_model_name)#, device_map="auto") #, torch_dtype=torch.float16)
    if for_training:
        base_model.eval()
    #peft_model = peft.get_peft_model(base_model, peft_config)
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
    dataset_train = SemEvalDataset(news_similarity_path / "train.csv", news_similarity_path / "all_data")
    dataset = SemEvalDataset(news_similarity_path / "eval.csv", news_similarity_path / "all_data")
    model = get_model(base_model_name, "../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9", for_training=False).to("cuda:0")
    model = model.to(torch.float16)
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
            encoded_a = model(**tokenizer(pair.article_1.text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_INPUT_LENGTH).to("cuda:0"))
            encoded_b = model(**tokenizer(pair.article_2.text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_INPUT_LENGTH).to("cuda:0"))
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
def chaturvedi(anonymized: bool = False, encode_query_separately: bool = False):
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
    dataset = MovieSummaryDataset(Path(os.environ["MOVIE_REMAKE_PATH"]) / "movieRemakesManuallyCleaned.tsv", Path(os.environ["MOVIE_REMAKE_PATH"]) / "testInstances.csv", Path(os.environ["MOVIE_REMAKE_PATH"]) / "remakes-anon.csv")
    base_model_name = "mistralai/Mistral-7B-v0.1"
    model = get_model(base_model_name, "e5-mistral-7b-instruct-adapters", for_training=False).to("cuda:0")
    # model = get_model(base_model_name, "../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9", for_training=False).to("cuda:0")
    model = model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    texts = [s.text_anonymized if anonymized else s.text for s in dataset]
    cluster_ids = [s.cluster_id for s in dataset]
    movie_ids = [s.movie_id for s in dataset]
    encoded = []
    encoded_queries = []
    query_prefix = "Given a story summary, find a summary of a remake: "
    with torch.no_grad():
        for text in tqdm(texts):
            batch = tokenizer(text, return_tensors="pt")
            output = model(**batch.to("cuda:0"))
            encoded.append(output.last_hidden_state[:,-1].to("cpu").squeeze())
            if encode_query_separately:
                batch_query = tokenizer(query_prefix + text, return_tensors="pt")
                output = model(**batch_query.to("cuda:0"))
                encoded_queries.append(output.last_hidden_state[:,-1].to("cpu").squeeze())
    encoded = torch.stack(encoded)
    encoded_queries = torch.stack(encoded_queries)
    if encode_query_separately:
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
        all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.get_anonymized(min_sentences=min_length).values()) for i, v in enumerate(splits[split].stories.values())]))
    else:
        all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.get_all_summaries_en(min_sentences=min_length)[1]) for i, v in enumerate(splits[split].stories.values())]))
    labels, texts = zip(*all_labeled)
    for text in tqdm(texts):
        tokenized = tokenizer(text)
        lengths[len(tokenized["input_ids"])] += 1
    max_length = max(lengths.keys())
    for x in range(max_length + 1):
        print(x, lengths[x], sep=",")
    # The results basically tell us: 3k should be plenty


@app.command()
def llama_test(use_anonymized: bool = True, min_length: int = 0, batch_size: int = 1, split: str = "dev", quick: bool=False, collect_lengths: bool = False):
    base_model_name = "mistralai/Mistral-7B-v0.1"
    #model = get_model(base_model_name, "sim-trainer2024-02-21T19:26:14.519764/checkpoint-1500/", for_training=False).to("cuda:0"): 0.88
    model = get_model(base_model_name, "../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9", for_training=False).to("cuda:0")
    model = model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    ds = dataset.SummaryDataset("data")
    splits = ds.perform_splits()
    if use_anonymized:
        all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.get_anonymized(min_sentences=min_length).values()) for i, v in enumerate(splits[split].stories.values())]))
    else:
        all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.get_all_summaries_en(min_sentences=min_length)[1]) for i, v in enumerate(splits[split].stories.values())]))
    all_labeled = [(id_, text) for id_, text in all_labeled if min_length is None or len(text) >= min_length]
    labels, texts = zip(*all_labeled)
    if quick:
        labels = labels[:1000]
        texts = texts[:1000]
    batches = []
    texts = ["Retrieve stories with a similar narrative." + t for t in texts] # let's not make them too long
    for texts_for_batch in more_itertools.chunked(texts, batch_size):
        tokenized = tokenizer(texts_for_batch, return_tensors="pt", max_length=4096)
        batches.append(tokenized)
    model.eval()
    batches_encoded = []
    with torch.no_grad():
        for batch in tqdm(batches):
            output = model(**batch.to("cuda:0"))
            batches_encoded.append(output.last_hidden_state[:,-1].to("cpu"))
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
    print("P@1", correct / total)
    # With anonymization: 0.139
    # W/o anonymiztaion: 0.596
    # These are not competitive with e.g. sentence-T5

@dataclass
class DataCollatorForSimilarityModeling:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, samples):
        texts_a = self.tokenizer([sample["text_a"] for sample in samples], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_INPUT_LENGTH)
        texts_b = self.tokenizer([sample["text_b"] for sample in samples], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_INPUT_LENGTH)
        ids = torch.tensor([[int(id_[1:]) for id_ in s["text_ids"]] for s in samples])
        ids_a = ids[:, 0]
        ids_b = ids[:, 1]
        target = (ids_a[:, None] == ids_b)
        #labels = torch.tensor([sample["label"] for sample in samples])
        return {"texts_a": texts_a, "texts_b": texts_b, "labels": target}


def clip_texts(item):
    item.update({
        "text_a": "Retrieve stories with a similar narrative. " + item["text_a"],
        "text_b": "Retrieve stories with a similar narrative. " + item["text_b"],
    })
    return item


class GradientCacheCallbacks(TrainerCallback):
    pass

@app.command()
def train():
    timestamp = datetime.utcnow().isoformat()
    ds = dataset.SimilarityDataset("data", negative_sample_scale=0.0, min_sentences=10, clusters_together=True)
    base_model_name = "mistralai/Mistral-7B-v0.1"
    # model = get_model(base_model_name, None)
    model = get_model(base_model_name, "e5-mistral-7b-instruct-adapters")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    effective_batch_size = 100
    base_lr = 5e-5
    effective_lr = base_lr * effective_batch_size / 1024 # Recommended in https://arxiv.org/pdf/2304.12210.pdf
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if "lora" in name], lr=effective_lr)
    training_args = TrainingArguments(
        "sim-trainer" + timestamp,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=effective_batch_size,
        logging_steps=1,
        save_steps=1,
        max_grad_norm=0.1,
        fp16=True,
    )
    trainer = GradientCacheTrainer(
        model,
        training_args,
        train_dataset=ds["train"].map(clip_texts),
        eval_dataset=ds["dev"].map(clip_texts),
        data_collator=DataCollatorForSimilarityModeling(tokenizer),
        optimizers=[optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)],
    )
    trainer.train()


@app.command()
def roc_stories(split: str = "dev"):
    base_model_name = "mistralai/Mistral-7B-v0.1"
    # model = get_model(base_model_name, "e5-mistral-7b-instruct-adapters").to("cuda:0")
    model = get_model(base_model_name, "../sim-trainer-checkpoints/sim-trainer2024-03-14T12:55:28.479641/checkpoint-9", for_training=False).to("cuda:0")
    model = model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    ds = ROCStoriesDataset("../roc_stories", split)
    labels = []
    predictions = []
    with torch.no_grad():
        for item in tqdm(ds):
            anchor = " ".join(item.sentences)
            choices = [anchor + " " + s for s in item.candidate_endings]
            batch = tokenizer([anchor] + choices, return_tensors="pt", padding=True).to("cuda:0")
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
    elif split == "test":
        predictions = [p + 1 for p in predictions]
        in_file = open("../roc_stories/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv", "r")
        out_file = open("roc-stories-test.csv", "w")
        out_file.write(next(in_file).strip() + ",AnswerRightEnding\r\n")
        i = 0
        for line in in_file:
            out_file.write(line.strip() + "," + str(predictions[i].item()) + "\r\n")
            i += 1


if __name__ == "__main__":
    app()
