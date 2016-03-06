import sframe, math, string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import numpy as np

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

def remove_punctuation(text):
    return text.translate(None, string.punctuation) 
    
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

products = sframe.SFrame('amazon_baby.gl/')
products['review_clean'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)


#######################################################################
#
# When dealing with text data, "bag-of-words" method indicates
# vectorization of text string into a matrix
#
# number of rows = number of text strings in input
# number of cols = number of distinct words in all texts from input
#
# http://stackoverflow.com/questions/22920801/can-i-use-countvectorizer-in-scikit-learn-to-count-frequency-of-documents-that-w
#
#######################################################################
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
     # Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix


train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

lr = LogisticRegression()
sentiment_model = lr.fit(X = train_matrix, y = train_data['sentiment'])

NonNegWeights = sum(x > 0 for x in sentiment_model.coef_)
print type(NonNegWeights)
print sum(NonNegWeights)

sample_test_data = test_data[10:13]
print sample_test_data

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
print sample_test_matrix
# decision_function() is the function of decision boundary - calculates
# the score for determining which class an input belongs to 
scores = sentiment_model.decision_function(sample_test_matrix)
print scores
classLable = sentiment_model.predict(sample_test_matrix)
print classLable
    
for s in scores:
    print sigmoid(s)
    
classLable = sentiment_model.predict(test_matrix)
scores = sentiment_model.decision_function(test_matrix)
sorted_testing_score_index = np.argsort(scores)

for i in range(len(sorted_testing_score_index) - 21, len(sorted_testing_score_index) - 1):
    print test_data[sorted_testing_score_index[i]]['name']
    
accuracy = sum(classLable == test_data['sentiment'])/len(test_data['sentiment'])

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

simple_model = lr.fit(X = train_matrix_word_subset, y = train_data['sentiment'])

# numpy.ndarray.flatten() - return the nd array collapsed into 1-D
# no matter how many dimensions there was
simple_model_coef_table = sframe.SFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})
simple_model_coef_table.print_rows(num_rows=20, num_columns=2)

simple_classLable = simple_model.predict(test_matrix_word_subset)