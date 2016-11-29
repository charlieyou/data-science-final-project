# coding: utf-8

import cPickle as pickle
import re
import string

from spacy.en import English
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

STOPLIST = English.Defaults().stop_words
STOPLIST |= ENGLISH_STOP_WORDS
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


def debates():
    clinton1 = [preprocess(doc) for doc in pickle.load(open('clinton_Debate_1.p', 'rb'))]
    clinton2 = [preprocess(doc) for doc in pickle.load(open('clinton_Debate_2.p', 'rb'))]
    clinton3 = [preprocess(doc) for doc in pickle.load(open('clinton_Debate_3.p', 'rb'))]
    trump1 = [preprocess(doc) for doc in pickle.load(open('trump_Debate_1.p', 'rb'))]
    trump2 = [preprocess(doc) for doc in pickle.load(open('trump_Debate_2.p', 'rb'))]
    trump3 = [preprocess(doc) for doc in pickle.load(open('trump_Debate_3.p', 'rb'))]
    pickle.dump((clinton1, clinton2, clinton3, trump1, trump2, trump3),
            open('debates_preprocessed', 'wb'))


def tweets():
    clinton = [(i, preprocess(doc)) for i, doc in pickle.load(open('clinton_tweets.p', 'rb'))]
    trump = [(i, preprocess(doc)) for i, doc in pickle.load(open('trump_tweets.p', 'rb'))]
    pickle.dump((clinton, trump), open('tweets_preprocessed', 'wb'))


if __name__ == '__main__':
    debates()
    tweets()
