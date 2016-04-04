import sframe
import sklearn
import sklearn.ensemble
import numpy as np

loans = sframe.SFrame('lending-club-data.gl/')

# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]
loans, loans_with_na = loans[[target] + features].dropna_split()

# Count the number of rows with missing data
num_rows_with_na = loans_with_na.num_rows()
num_rows = loans.num_rows()
print 'Dropping %s observations; keeping %s ' % (num_rows_with_na, num_rows)

loans = loans[[target] + features].dropna()

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)


loans_data = risky_loans.append(safe_loans)

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

loans_data.column_names()

train_data, validation_data = loans_data.random_split(.8, seed=1)

features = loans_data.column_names()
features.remove(target)

train_target_np = train_data[target].to_numpy()
train_feature_np = train_data[features].to_numpy()

validation_target_np = validation_data[target].to_numpy()
validation_feature_np = validation_data[features].to_numpy()

# instantiate and Fit/Train Gradient Boosted Tree
model_5 = sklearn.ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=5)
model_5.fit(X=train_feature_np, y=train_target_np)

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
print model_5.predict_proba(sample_validation_data[features].to_numpy())

validation_output = model_5.predict_proba(validation_data[features].to_numpy())
          
# Calculate the number of false positive prediction
positive_prediction_boolean_mask = validation_output[:,1] > validation_output[:,0]
validation_predictor_result = validation_data[target].to_numpy()

positive_prediction_index = np.where(positive_prediction_boolean_mask == True)
positive_prediction = validation_predictor_result[positive_prediction_index]

num_false_positive = sum(positive_prediction == -1)

# calculate the number of false negative prediction
negative_prediction_boolean_mask = validation_output[:,1] < validation_output[:,0]

negative_prediction_index = np.where(negative_prediction_boolean_mask == True)
negative_prediction = validation_predictor_result[negative_prediction_index]

num_false_negative = sum(negative_prediction == 1)
