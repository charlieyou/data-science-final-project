import cPickle as pickle

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import cross_val_score

from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier


class EnsembleClassifier():
    def fit(self, X, y):
        self._models = [NB(), PassiveAggressiveClassifier(), SGDClassifier(
            loss='modified_huber')]
        for model in self._models:
            model.fit(X, y)

        return self

    def predict(self, X):
        weighted_predictions = [0, 0]
        for model in self._models:
            weighted_predictions[model.predict(X)] += 1

        return max(enumerate(weighted_predictions), key=lambda x: x[1])[0]

    def score(self, X, Y):
        return sum([1 if self.predict(x) == y else 0 for x, y in zip(X, Y)])

    def get_params(self, deep=False):
        return {}


if __name__ == '__main__':
    print 'unpickling data...'
    raw, y = pickle.load(open('data.p', 'rb'))

    print 'fitting...'

    cv = 3
    print 'NB:'
    X = HashingVectorizer(ngram_range=(1, 3)).transform(raw)
    score = cross_val_score(NB(), X, y, cv=cv, n_jobs=-1)
    print score.mean()
    print score.std() * 2

    print 'PA:'
    # X = HashingVectorizer(ngram_range=(1, 3)).transform(raw)
    # X = HashingVectorizer().transform(raw)
    score = cross_val_score(PassiveAggressiveClassifier(), X, y, cv=cv,
        n_jobs=-1)
    print score.mean()
    print score.std() * 2

    print 'SGD, Modified Huber: '
    X = HashingVectorizer().transform(raw)
    score = cross_val_score(SGDClassifier(loss='modified_huber'), X, y, cv=cv,
        n_jobs=-1)
    print score.mean()
    print score.std() * 2
