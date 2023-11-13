from spacy.tokens import Span, Doc, Token
import spacy

from spacy.language import Language
from flair.data import Sentence
from flair.nn import Classifier

@Language.factory("flair_ner")
def statementFactory(nlp, name):
    return FlairTagger()

class FlairTagger:
    def __init__(self):
        self.tagger = Classifier.load('ner')

    def __call__(self, doc):
        ner_spans = []
        flair_sents = []
        for sent in doc.sents:
            flair_sent = Sentence(sent.text)
            flair_sents.append(flair_sent)
        self.tagger.predict(flair_sents)
        for flair_sent, sent in zip(flair_sents, doc.sents):
            for label in flair_sent.get_labels():
                start_pos = label.data_point.tokens[0].start_position
                end_pos = label.data_point.tokens[-1].end_position
                span = doc.char_span(start_pos + sent.start_char, end_pos + sent.start_char, label=label.value)
                if span is not None:
                    ner_spans.append(span)
        doc.set_ents(entities=ner_spans)
        return doc


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    nlp.add_pipe("flair_ner")
    doc = nlp("Her name is Hanna. His name is Hans.")
    for sent in doc.sents:
        print(sent.ents)