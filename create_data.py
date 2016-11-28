import sys
import cPickle as pickle

from data_helpers import preprocess


print 'getting tweets...'
tweets = get_tweets(sys.argv[1])
print 'found %i tweets' % len(tweets)

print 'preprocessing...'
processed = [preprocess(tweet) for tweet in tweets]

print 'creating data from topic models...'
# y = [np.argmax(np.array(model[doc])) for doc in corpus]
X = [' '.join(doc) for doc in processed]
y = [max(model[doc], key=lambda x: x[1])[0] for doc in corpus]
print y[:10]

print 'pickling...'
pickle.dump((X, y), open('data.p', 'wb'))
