import sframe
import json
import numpy as np
from math import sqrt

products = sframe.SFrame('amazon_baby_subset.gl/')

# For this assignment, we eliminated class imbalance by choosing a 
# subset of the data with a similar number of positive and negative reviews.
print '# of positive reviews =', len(products[products['sentiment']==1])
print '# of negative reviews =', len(products[products['sentiment']==-1])

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

# the following function extracts columns from an SFrame and converts
# them into a NumPy array
#
# the feature matrix includes an additional column 'intercept'
# to take account of the intercept term - all 1s
def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)
    
'''
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    # score_vector = w * h(x) - matrix multiplication
    score_vector = feature_matrix.dot(coefficients)
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = np.apply_along_axis(lambda s : 1/(1+np.exp(-s)), 0, score_vector)
    return predictions
    
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp
    
def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = errors.dot(feature)
    return derivative

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    # initialization - as coefficients will change along iterations
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_i,w) using predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute value for indicator function (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient/feature
            # feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it as variable "derivative"
            derivative = feature_derivative(errors, feature_matrix[:, j])
            
            # add the step size times the derivative to the current coefficient
            coefficients[j] = coefficients[j] + step_size * derivative
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
    
with open('important_words.json', 'r') as f: # Reads the list of most frequent words
    important_words = json.load(f)
important_words = [str(s) for s in important_words]
#print important_words

products['review_clean'] = products['review'].apply(remove_punctuation)

# build "bag-of-words" matrix
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))    
#print products.head()
    
# new column set to TRUE if the count of the word perfect 
# (stored in column "perfect") is >= 1
products['contains_perfect'] = products['perfect'].apply(lambda p : p >= 1)
print sum(products['contains_perfect'])

# Warning: This may take a few minutes...
feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
print feature_matrix.shape

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
                                   step_size=1e-7, max_iter=301)
# make predictions using the logistic regression model trained above ^
scores = np.dot(feature_matrix, coefficients)
predicted_class_with_0 = (scores > 0).astype(np.int64)
print sum(predicted_class_with_0)
'''
class is represented by 1/-1, the above class array uses 1/0

np.where() uses a position matrix, replacement value, target array
position matrix indicates the position in target array that needs to be modified
replacement value indicates the values to be used for update
'''
predicted_class = np.where(predicted_class_with_0 == 0, -1, predicted_class_with_0)

num_mistakes = sum(sentiment != predicted_class)
accuracy = (len(sentiment) - num_mistakes) / len(sentiment)
print "-----------------------------------------------------"
print '# Reviews   correctly classified =', len(products) - num_mistakes
print '# Reviews incorrectly classified =', num_mistakes
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Accuracy = %.2f' % accuracy

coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)


def unitTest():
    dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
    dummy_coefficients = np.array([1., 3., -1.])

    correct_scores = np.array( [ 1.*1. + 2.*3. + 3.*(-1.), 1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
    correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

    print 'The following outputs must match '
    print '------------------------------------------------'
    print 'correct_predictions           =', correct_predictions
    print 'output of predict_probability =', predict_probability(dummy_feature_matrix, dummy_coefficients)
    
    
    dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
    dummy_coefficients = np.array([1., 3., -1.])
    dummy_sentiment = np.array([-1, 1])

    correct_indicators  = np.array( [ -1==+1, 1==+1 ] )
    correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),                     1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
    correct_first_term  = np.array( [ (correct_indicators[0]-1)*correct_scores[0],  (correct_indicators[1]-1)*correct_scores[1] ] )
    correct_second_term = np.array( [ np.log(1. + np.exp(-correct_scores[0])),      np.log(1. + np.exp(-correct_scores[1])) ] )

    correct_ll          =      sum( [ correct_first_term[0]-correct_second_term[0], correct_first_term[1]-correct_second_term[1] ] ) 

    print 'The following outputs must match '
    print '------------------------------------------------'
    print 'correct_log_likelihood           =', correct_ll
    print 'output of compute_log_likelihood =', compute_log_likelihood(dummy_feature_matrix, dummy_sentiment, dummy_coefficients)
    
#unitTest()