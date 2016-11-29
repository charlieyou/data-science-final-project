import joblib
import cPickle as pickle

import pandas as pd


def tweets(clf):
    pass


def debates(clf):
    pass


if __name__ == '__main__':
    clf = joblib.load('clf/clf.p')

    tweets(clf)
    debates(clf)
