import cPickle as pickle
import pandas as pd


if __name__ == '__main__':
    fnames = set(['Debate_1.json', 'Debate_2.json', 'Debate_3.json'])
    for fname in fnames:
        df = pd.read_json('data/' + fname)
        clinton = df[df['speaker'] == 'CLINTON']
        pickle.dump(clinton['sentence'].tolist(), open('clinton_' + fname, 'wb'))
        trump = df[df['speaker'] == 'TRUMP']
        pickle.dump(trump['sentence'].tolist(), open('trump_' + fname, 'wb'))
