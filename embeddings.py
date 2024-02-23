from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase, PreTrainedModel, MistralConfig, MistralModel, TrainerCallback
from typer import Typer
from peft import PeftModel, PeftConfig
import peft
from sentence_transformers.util import cos_sim
import itertools
import dataset
from datetime import datetime
import more_itertools
from tqdm import tqdm
import torch
from torch import nn
from dataclasses import dataclass
from gradient_cache_trainer import GradientCacheTrainer

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
    base_model = AutoModel.from_pretrained(base_model_name) #, torch_dtype=torch.float16)
    if for_training:
        base_model.eval()
    #peft_config = peft.LoraConfig(
    #    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    #)
    #peft_model = peft.get_peft_model(base_model, peft_config)
    base_model.load_adapter(adapter_model_name)
    return base_model


@app.command()
def llama_test(use_anonymized: bool = True, min_length: int = 0, batch_size: int = 1, split: str = "dev", quick: bool=False):
    base_model_name = "mistralai/Mistral-7B-v0.1"
    model = get_model(base_model_name, "sim-trainer2024-02-21T19:26:14.519764/checkpoint-1500/", for_training=False).to("cuda:0")
    model = model.to(torch.float16)
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
    texts = ["Retrieve semantically similar text. " + t for t in texts] # let's not make them too long
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
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
        texts_a = self.tokenizer([sample["text_a"] for sample in samples], return_tensors="pt", padding="max_length", truncation=True, max_length=1500)
        texts_b = self.tokenizer([sample["text_b"] for sample in samples], return_tensors="pt", padding="max_length", truncation=True, max_length=1500)
        labels = torch.tensor([sample["label"] for sample in samples])
        return {"texts_a": texts_a, "texts_b": texts_b, "labels": labels}


def clip_texts(item):
    item.update({
        "text_a": "Retrieve semantically similar stories. " + item["text_a"],
        "text_b": "Retrieve semantically similar stories. " + item["text_b"],
    })
    return item


class GradientCacheCallbacks(TrainerCallback):
    pass

@app.command()
def train():
    timestamp = datetime.utcnow().isoformat()
    ds = dataset.SimilarityDataset("data", negative_sample_scale=0.0)
    base_model_name = "mistralai/Mistral-7B-v0.1"
    model = get_model(base_model_name, "e5-mistral-7b-instruct-adapters")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if "lora" in name], lr=0.00005)
    training_args = TrainingArguments(
        "sim-trainer" + timestamp,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=16,
        logging_steps=10,
        max_grad_norm=0.1,
        fp16=True,
        # gradient_checkpointing=True
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



if __name__ == "__main__":
    app()
