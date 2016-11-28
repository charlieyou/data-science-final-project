# coding: utf-8

import re
import string

from spacy.en import English, STOPWORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

STOPLIST = STOPWORDS
STOPLIST |= ENGLISH_STOP_WORDS
STOPLIST |= set(stopwords.words('english'))
STOPLIST |= set(["n't", "'s", "'m", "ca"])

SYMBOLS = set(" ".join(string.punctuation).split(" ")) |\
    set(["-----", "---", "...", "“", "”", "'ve"])

nlp = English(parser=False, matcher=False)


def preprocess(doc):
    doc = doc.lower().strip()
    doc = re.sub(ur'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<URL>', doc)
    doc = re.sub(ur'#\S+', '<HASHTAG>', doc)
    doc = re.sub(ur'[-+]?[.\d]*[\d]+[:,.\d]*', '<NUMBER>', doc)
    doc = re.sub(ur'@\w+', '<USER>', doc)
    doc = doc.replace(u'\n', ' ')
    doc = doc.replace(u'\r', ' ')
    doc = doc.replace(u'/', ' / ')
    doc = re.sub(ur'\s{2,}', ' ', doc)

    defined_tags = set([u'USER', u'URL', u'HASHTAG', u'NUMBER'])
    return [tok for tok in [tok.lemma_ if tok.lemma_ != u'-PRON-' and
        tok.orth_ not in defined_tags else tok.orth_ for tok in nlp(doc)] if
        tok not in STOPLIST | SYMBOLS]
