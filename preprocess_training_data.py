# coding: utf-8

import cPickle as pickle

import pandas as pd

from preprocess import preprocess


if __name__ == '__main__':
    df = pd.read_csv('data/training_data.csv', nrows=100000, encoding='utf-8', error_bad_lines=False)

    raw = df['SentimentText'].tolist()
    X = [preprocess(doc) for doc in raw]
    y = df['Sentiment'].tolist()

    pickle.dump((X, y), open('training_data.p', 'wb'))
