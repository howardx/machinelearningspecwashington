import sframe
import numpy as np
loans = sframe.SFrame('lending-club-data.gl/')

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

loans_data_features = loans_data[features]

# one-hot encoding implementation, normalize categorical features 
# to additional columns with column names as feature indicators
#
# sklearn.preprocessing.OneHotEncoder ONLY takes integer type matrix
#
# sklearn.preprocessing.LabelEncoder converts string type into integer type 
# But only 1 column at a time
'''
sf = graphlab.SFrame({'id': [1,2,3],
                      'wc': [{'a': 1}, {'b': 2}, {'a': 1, 'b': 2}]})
+----+------------------+
| id |        wc        |
+----+------------------+
| 1  |     {'a': 1}     |
| 2  |     {'b': 2}     |
| 3  | {'a': 1, 'b': 2} |
+----+------------------+
    
sframe.unpack('wc')
+----+------+------+
| id | wc.a | wc.b |
+----+------+------+
| 1  |  1   | None |
| 2  | None |  2   |
| 3  |  1   |  2   |
+----+------+------+
'''
def one_hot_normalize_to_columns(sfData):
    categorical_features = []
    for feature_name, feature_type in zip(sfData.column_names(), sfData.column_types()):
        if feature_type == str:
            categorical_features.append(feature_name)

    for feature in categorical_features:
        data_one_hot_encoded = sfData[feature].apply(lambda x : {x : 1})
        data_unpacked = data_one_hot_encoded.unpack(column_name_prefix = feature)

        # Change None's to 0's
        for column in data_unpacked.column_names():
            data_unpacked[column] = data_unpacked[column].fillna(0)

        sfData.remove_column(feature)
        sfData.add_columns(data_unpacked)
    return sfData

def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0    
    # Count the number of 1's (safe loans)
    positive_count = sum((labels_in_node == 1))
    # Count the number of -1's (risky loans)
    negative_count = sum((labels_in_node == -1))
    # Return the number of mistakes that the majority classifier makes.
    if positive_count > negative_count:
        return negative_count
    else:
        return positive_count
        
def best_splitting_feature(data, features, predictor):
    target_values = data[predictor]
    best_feature = None # Keep track of the best feature 
    # Note: Since error is always <= 1, intialize it with value larger than 1.
    best_error = 10     # Keep track of the best error so far 
   
    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        # The left split/node have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        # The right split/node will have all data points where the feature value is 1
        right_split = data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split[predictor])    
        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[predictor])
            
        # Compute the classification error of this split/level in tree/stump
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, update
        # the feature as best_feature and the error as best_error
        if error < best_error:
            best_error = error 
            best_feature = feature
    return best_feature # Return the best feature we found
    
def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None, # feature this node splits on
            'left' : None, # dictionary corresponding to left sub tree
            'right' : None, # dictionary corresponding to right sub tree
            'prediction' : 0, # +1 or -1
            'is_leaf': True}
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    # For the leaf node, set the prediction to be the majority class.
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1
    return leaf
    
def decision_tree_create(data, features, predictor, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[predictor]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    # Stopping condition 1
    # Check if there are mistakes at current node, if no then leaf
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached."     
        # If no mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 - see if there are remaining features to consider splitting on
    if len(remaining_features) == 0:
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, features, predictor)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # remove the splitting feature from making further splitting/entropy
    remaining_features.remove(splitting_feature)
    
    print "Split on feature %s. (%s, %s)" % (\
        splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[predictor])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[predictor])

    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, predictor, current_depth + 1, max_depth)
    right_tree = decision_tree_create(right_split, remaining_features, predictor, current_depth + 1, max_depth)
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

def classify(tree, testingSet, annotate = True):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = testingSet[tree['splitting_feature']]
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], testingSet, annotate)
        else:
            return classify(tree['right'], testingSet, annotate)

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda testingData: classify(tree, testingData, False))
    
    # calculate the classification error
    error_count = sum(np.not_equal(prediction, data[target]))
    return float(error_count)/float(len(prediction))
     
def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))     
   
   
   
loans_data_1_hot = one_hot_normalize_to_columns(loans_data_features)
features_to_split_on = loans_data_1_hot.column_names()

loans_data_1_hot[target] = loans_data[target]

train_data, test_data = loans_data_1_hot.random_split(.8, seed=1)

my_decision_tree = decision_tree_create(train_data, features_to_split_on, target, 0, 6)
print "\n\nDONE PLANTING THE TREE\n\n"

print 'Predicted class: %s ' % classify(my_decision_tree, test_data[0])
print evaluate_classification_error(my_decision_tree, test_data)

print_stump(my_decision_tree)





def unitTests():
    # Test case 1
    example_labels = sframe.SArray([-1, -1, 1, 1, 1])
    if intermediate_node_num_mistakes(example_labels) == 2:
        print 'Test passed!'
    else:
        print 'Test 1 failed... try again!'

    # Test case 2
    example_labels = sframe.SArray([-1, -1, 1, 1, 1, 1, 1])
    if intermediate_node_num_mistakes(example_labels) == 2:
        print 'Test passed!'
    else:
        print 'Test 3 failed... try again!'
    
    # Test case 3
    example_labels = sframe.SArray([-1, -1, -1, -1, -1, 1, 1])
    if intermediate_node_num_mistakes(example_labels) == 2:
        print 'Test passed!'
    else:
        print 'Test 3 failed... try again!'
        
#unitTests()