import cPickle as pickle
import joblib

from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn.naive_bayes import MultinomialNB as NB


if __name__ == '__main__':
    raw, y = pickle.load(open('training_data.p', 'rb'))
    X = [' '.join(doc) for doc in raw]

    vec = Vectorizer(ngram_range=(1, 1))
    clf = NB()
    vec_clf = Pipeline([('vectorizer', vec), ('clf', clf)])
    print cross_val_score(vec_clf, X, y)

    vec_clf.fit(X, y)
    joblib.dump(vec_clf, 'clf/clf.p')
