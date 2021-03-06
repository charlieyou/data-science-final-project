import joblib
import cPickle as pickle

import pandas as pd


def tweets(clf):
    clinton, trump = pickle.load(open('tweets_preprocessed', 'rb'))
    c_result = clf.predict([' '.join(doc) for _, doc in clinton])
    t_result = clf.predict([' '.join(doc) for _, doc in trump])
    df = pd.DataFrame([[c_result.sum(), len(c_result)],
        [t_result.sum(), len(t_result)]],
        index=('Clinton', 'Trump'), columns=('Positive', 'Total'))
    df.to_csv('twitter_results.csv')


def time_series(clf):
    clinton, trump = pickle.load(open('tweets_preprocessed', 'rb'))

    c_dates = [date for date, _ in clinton]
    c_result = clf.predict([' '.join(doc) for _, doc in clinton])

    t_dates = [date for date, _ in trump]
    t_result = clf.predict([' '.join(doc) for _, doc in trump])

    c = pd.Series(c_result, index=c_dates)
    t = pd.Series(t_result, index=t_dates)

    c = c.groupby(pd.TimeGrouper(freq='D')).mean()
    t = t.groupby(pd.TimeGrouper(freq='D')).mean()

    c = pd.rolling_mean(c, 14)
    t = pd.rolling_mean(t, 14)

    df = pd.concat([c, t], axis=1)
    df.columns = ('Clinton', 'Trump')

    df.to_pickle('time_series.p')


def debates(clf):
    names = ('clinton1', 'clinton2', 'clinton3', 'trump1', 'trump2', 'trump3')
    results = []
    for name, obj in zip(names, pickle.load(open('debates_preprocessed', 'rb'))):
        temp = [x for x in obj if x != []]
        results.append(clf.predict([' '.join(doc) for doc in temp]))

    for i, result in enumerate(results):
        results[i] = (result.sum(), len(result), float(result.sum()) / len(result))

    df = pd.DataFrame(results, index=names, columns=('positive', 'total', 'percent'))
    df.to_csv('debate_results.csv')

if __name__ == '__main__':
    clf = joblib.load('clf/clf.p')

    time_series(clf)
    # tweets(clf)
    # debates(clf)
