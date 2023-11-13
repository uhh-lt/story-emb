from dataclasses import dataclass
from collections import Counter
from typing import List
import itertools
from spacy.tokens import Span, Doc, Token

def __is_coref(tokens):
    return all(map(lambda t: t._.in_coref, tokens))

def __has_coref(tokens):
    return any(map(lambda t: t._.in_coref, tokens))

def _corefs(tokens):
    if len(tokens) == 0:
        return []
    elif len(tokens) == 1:
        return tokens[0]._.corefs
    result = tokens[0]._.corefs
    for t in tokens:
        for ref in t._.corefs:
            if ref not in result:
                result.append(ref)
    return result


Token.set_extension("in_coref", default=False)
Token.set_extension("in_coref_clusters", default=[])
Token.set_extension("mention_ids", default=[])
Span.set_extension("is_coref", getter=__is_coref)
Span.set_extension("has_coref", getter=__has_coref)
Span.set_extension("corefs", getter=_corefs)
Doc.set_extension("coref_clusters", default=[])

@dataclass
class Cluster():
    '''A cluster consists of usually two or more mentions referring to the same entity'''
    longest_mention: Span
    ner_label: str
    spans: List[Span]
    id: int

    def __repr__(self):
        return f'Cluster(i={self.id}: {self.longest_mention.text} {self.ner_label})'



def add_coref(doc, pred):
    clusters = []
    for i, cluster in enumerate(pred):
        new_cluster_spans = []
        for m, mention in enumerate(cluster):
            new_cluster_spans.append(doc[mention[0] : mention[1] + 1])
        longest_span = sorted(new_cluster_spans, key=lambda x: len(x.text))[-1]
        all_labels = list(itertools.chain.from_iterable([ent.label_ for ent in span.ents if ent.label_ is not None] for span in new_cluster_spans))
        ner_label_counter = Counter(all_labels)
        if len(ner_label_counter) > 0:
            most_common_label, count = ner_label_counter.most_common(1)[0]
        else:
            most_common_label, count = (None, 0)
        new_cluster = Cluster(
            longest_mention=longest_span,
            ner_label=most_common_label,
            spans=new_cluster_spans,
            id=i,
        )
        for span in new_cluster.spans:
            for t in span:
                t._.in_coref = True
                t._.in_coref_clusters.append(new_cluster)
                t._.mention_ids.append(new_cluster.id)
        clusters.append(new_cluster)
    doc._.set('coref_clusters', clusters)
    return doc