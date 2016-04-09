import sframe
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

products = sframe.SFrame('amazon_baby.gl/')

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 
    
# method for printing out confusion matrix
# In the case of binary classification, the confusion matrix is a 2-by-2 matrix
def print_confusion_matrix(y, y_hat, classifier):
    from sklearn.metrics import confusion_matrix
    # use the same order of class as the LR model
    cmat = confusion_matrix(y_true = y, y_pred = y_hat, labels = classifier.classes_)
    print ' target_label | predicted_label | count '
    print '--------------+-----------------+-------'
    # Print out the confusion matrix.    
    for i, target_label in enumerate(classifier.classes_):
        for j, predicted_label in enumerate(classifier.classes_):
            print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j])
            
def apply_threshold(probabilities, threshold):
    return probabilities.applymap(lambda x : 1 if x >= threshold else -1) ## !!
    
    
products['review_clean'] = products['review'].apply(remove_punctuation)

# ignore all reviews with rating = 3, since they tend to have a neutral sentiment
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)

# Use this token pattern to keep single-letter words
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

model = LogisticRegression()
model.fit(X = train_matrix, y = train_data['sentiment'].to_numpy())

# calculate model accuracy
accuracy = accuracy_score(y_true=test_data['sentiment'].to_numpy(), y_pred=model.predict(test_matrix))
print "Test Accuracy: %s" % accuracy

# majority class classifier's accuracy as base line
baseline = len(test_data[test_data['sentiment'] == 1])/len(test_data)
print "Baseline accuracy (majority class classifier): %s" % baseline

print_confusion_matrix(test_data['sentiment'].to_numpy(), model.predict(test_matrix), model)

# Precision
from sklearn.metrics import precision_score
precision = precision_score(y_true=test_data['sentiment'].to_numpy(), 
                            y_pred=model.predict(test_matrix))
print "Precision on test data: %s" % precision

# Recall
from sklearn.metrics import recall_score
recall = recall_score(y_true=test_data['sentiment'].to_numpy(),
                      y_pred=model.predict(test_matrix))
print "Recall on test data: %s" % recall

print model.classes_
# column ordering of output matrix from predict_proba() is the same as output from model.classes_
score_after_sigmoid = pd.DataFrame(model.predict_proba(test_matrix))

threshold_values = np.linspace(0.5, 1, num=100)

precision_all = []
recall_all = []

for threshold in threshold_values:
    prediction = apply_threshold(pd.DataFrame(model.predict_proba(test_matrix)[:,1]), threshold)
    
    precision_all.append(precision_score(y_true=test_data['sentiment'].to_numpy(), 
        y_pred = prediction.as_matrix()[:,0]))

    recall_all.append(recall_score(y_true=test_data['sentiment'].to_numpy(),
        y_pred = prediction.as_matrix()[:,0]))

prediction_98 = apply_threshold(pd.DataFrame(model.predict_proba(test_matrix)[:,1]), 0.98)
print_confusion_matrix(test_data['sentiment'].to_numpy(), prediction_98.as_matrix()[:,0], model)


baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]
baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities = model.decision_function(baby_matrix)

