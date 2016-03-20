import sframe
import sklearn
from sklearn import tree
import numpy as np

loans = sframe.SFrame('lending-club-data.gl/')

# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)

# One way to combat class imbalance is to undersample the larger class
# until the class distribution is approximately half and half
#
# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

# For scikit-learn's decision tree implementation, it requires
# numerical values for it's data matrix. This means you will
# have to turn categorical variables into binary features via one-hot encoding
#
# sort to like "bag of words" approach
categorical_variables = []
for feat_name, feat_type in zip(loans_data.column_names(), loans_data.column_types()):
    if feat_type == str:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)
    
train_data, validation_data = loans_data.random_split(.8, seed=1)


def removeTargetColumnFromDatasetAndReturnNumpy(sframeData, targetCol):
    target_data = sframeData[targetCol] # result of predictor in train/test set
    target_np = target_data.to_numpy() 

    sframeData.remove_column(targetCol) # remove column doesn't return copy
    feature_only_np = sframeData.to_numpy()
    return feature_only_np, target_np


train_np, train_target_np = removeTargetColumnFromDatasetAndReturnNumpy(train_data, target)

validation_np = validation_data.to_numpy()

clf_6 = tree.DecisionTreeClassifier(max_depth = 6)
decision_tree_model = clf_6.fit(train_np, train_target_np)

clf_2 = tree.DecisionTreeClassifier(max_depth = 2)
small_model = clf_2.fit(train_np, train_target_np)

# consider two positive and two negative examples from the validation
# set and see what the model predicts
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_np, sample_validation_target_np = removeTargetColumnFromDatasetAndReturnNumpy(sample_validation_data, target)

model_prediction = decision_tree_model.predict(sample_validation_np)
model_prediction_p = decision_tree_model.predict_proba(sample_validation_np)

small_model_prediction = small_model.predict(sample_validation_np)
small_model_prediction_p = small_model.predict_proba(sample_validation_np)

# evaluate the accuracy of the small_model and decision_tree_model on the entire validation_data
validation_np, validation_target_np = removeTargetColumnFromDatasetAndReturnNumpy(validation_data, target)

print decision_tree_model.score(validation_np, validation_target_np)
print small_model.score(validation_np, validation_target_np)

'''
False negatives: Loans that were actually safe but were predicted to be risky.
This results in an oppurtunity cost of loosing a loan that would have otherwise been accepted.

False positives: Loans that were actually risky but were predicted to be safe.
These are much more expensive because it results in a risky loan being given.

Correct predictions: All correct predictions don't typically incur any cost.
'''
model_prediction = decision_tree_model.predict(validation_np)
print model_prediction
print validation_target_np

error_terms = np.not_equal(model_prediction, validation_target_np)
false_negative = sum(validation_target_np[error_terms] == 1)
false_positive = sum(validation_target_np[error_terms] == -1)

