import sframe
loans = sframe.SFrame('lending-club-data.gl/')

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

# Subsample dataset to make sure classes are balanced
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
    
loans_data_1_hot = one_hot_normalize_to_columns(loans_data_features)
train_data, validation_data = loans_data.random_split(.8, seed=1)