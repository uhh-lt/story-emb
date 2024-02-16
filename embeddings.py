from transformers import AutoTokenizer, AutoModel
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
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase, PreTrainedModel, MistralConfig
from dataclasses import dataclass

app = Typer()
    
def last_token_pooling(texts, last_hidden_state):
    # This is okay if the padding side is left
    assert (texts.attention_mask[:,-1] == 1).all()
    return last_hidden_state[:,-1]



class ContrastiveLlama(PreTrainedModel):
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    config_class = MistralConfig

    def __init__(self, llama):
        super().__init__(llama.config)
        self.model = llama
        self.model.use_chache = False
        self.use_cache = False

        self.sim_loss = nn.CosineEmbeddingLoss()
        self.post_init()

    def forward(self, texts_a, texts_b, labels):
        encoded_a = self.model(**texts_a)
        encoded_b = self.model(**texts_b)
        emb_a = last_token_pooling(texts_a, encoded_a.last_hidden_state)
        emb_b = last_token_pooling(texts_b, encoded_b.last_hidden_state)
        loss = self.sim_loss(emb_a, emb_b, labels)
        return loss, emb_a, emb_b


    def __call__(self, texts_a, texts_b, labels):
        return self.forward(texts_a, texts_b, labels)


def get_model(model_name):
    base_model = ContrastiveLlama.from_pretrained(model_name)
    peft_config = peft.LoraConfig(
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    peft_model = peft.get_peft_model(base_model.model, peft_config)
    return base_model, peft_model


@app.command()
def llama_test(use_anonymized: bool = True, min_length: int = 0, batch_size: int = 1, split: str = "dev"):
    model = ContrastiveLlama.from_pretrained("sim-trainer2024-02-15T16:25:03.618780/checkpoint-8500/", local_files_only=True).model
    ds = dataset.SummaryDataset("data")
    splits = ds.perform_splits()
    if use_anonymized:
        all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.get_anonymized(min_sentences=min_length).values()) for i, v in enumerate(splits["dev"].stories.values())]))
    else:
        all_labeled = list(itertools.chain.from_iterable([zip(itertools.repeat(i), v.get_all_summaries_en(min_sentences=min_length)[1]) for i, v in enumerate(splits["dev"].stories.values())]))
    all_labeled = [(id_, text) for id_, text in all_labeled if min_length is None or len(text) >= min_length]
    labels, texts = zip(*all_labeled)
    labels = labels[:100]
    texts = texts[:100]
    batches = []
    texts = [clip_texts(t) for t in texts] # let's not make them too long
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
    for texts_for_batch in more_itertools.chunked(texts, batch_size):
        tokenized = tokenizer(texts_for_batch, return_tensors="pt")
        batches.append(tokenized)
    model.eval()
    batches_encoded = []
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
        texts_a = self.tokenizer([sample["text_a"] for sample in samples], return_tensors="pt", padding=True)
        texts_b = self.tokenizer([sample["text_b"] for sample in samples], return_tensors="pt", padding=True)
        labels = torch.tensor([sample["label"] for sample in samples])
        return {"texts_a": texts_a, "texts_b": texts_b, "labels": labels}


def clip_texts(item):
    item.update({
        "text_a": "Retrieve semantically similar stories. " + item["text_a"][:1000],
        "text_b": "Retrieve semantically similar stories. " + item["text_b"][:1000],
    })
    return item

@app.command()
def train():
    timestamp = datetime.utcnow().isoformat()
    ds = dataset.SimilarityDataset("data")
    # inner_model, peft_model = get_model("meta-llama/Llama-2-7b-hf")
    inner_model, model = get_model("intfloat/e5-mistral-7b-instruct")
    #inner_model, peft_model = get_model("TinyLlama/TinyLlama-1.1B-python-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
    #tokenizer.pad_token = "[PAD]"
    #tokenizer.padding_side = "left"
    training_args = TrainingArguments(
        "sim-trainer" + timestamp,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=4,
        fp16=True,
        # gradient_checkpointing=True
    )
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=0.0001)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=ds["train"].map(clip_texts),
        eval_dataset=ds["dev"].map(clip_texts),
        data_collator=DataCollatorForSimilarityModeling(tokenizer),
        optimizers=[optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)],
    )
    trainer.save_checkpoint("test-checkpoint")
    return
    trainer.train()



if __name__ == "__main__":
    app()
