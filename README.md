# Story Embeddings

This repository contains the code for both the papers "Tell Me Again! a Large-Scale Dataset of Multiple Summaries for the Same Story" and "Story Embeddings — Narrative-Focused Representations of Fictional Stories".

The code is licensed under MIT

When using any of our code we kindly ask that you reference our paper:
```
@inproceedings{hatzel-biemann-2024-story-embeddings,
    title = "Story Embeddings -- Narrative-Focused Representations of Fictional Stories",
    author = "Hatzel, Hans Ole and Biemann, Chris",
    booktitle = "Proceedings of the 62st Annual Meeting of the Association for Computational Linguistics",
    year = "2024",
    address = "Miami, Florida",
    publisher = "Association for Computational Linguistics",
}
```

## Quick Overview

The most important things you could be looking for are listed below:

* Contrastive learning for story embeddings: `embeddings.py`
* Retelling Dataset: `data/retellings/`
* Dataset scraping: `main.py` unless you know what you are doing just `pip install tell-me again` instead!

## Tell Me Again Data

You shouldn't need to scrape the Tell Me Again data yourself (unless you want to build a new version of the dataset).
Instead just use [the package](https://github.com/uhh-lt/tell-me-again) `tell-me-again`.


## Model Snapshots

To be relased, if you see this and need them feel free to write.
