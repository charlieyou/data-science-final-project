import cPickle as pickle
import pandas as pd


if __name__ == '__main__':
    fnames = set(['clinton_tweets.json', 'trump_tweets.json'])
    for fname in fnames:
        df = pd.read_json('data/' + fname)
        df = df.transpose()
        df = df['text']
        pickle.dump([(i, v) for i, v in zip(df.index, df.values)], open(fname, 'wb'))
