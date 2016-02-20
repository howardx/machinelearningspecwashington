import pandas as pd
import numpy as np

class knnRegression:
  # data type mapping for training set CSV
  dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
  'sqft_living15':float, 'grade':int, 'yr_renovated':int,
  'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
  'sqft_lot15':float, 'sqft_living':float, 'floors':float,
  'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
  'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

  def get_numpy_data(self, trainingSet, features, predictor):
    df = pd.read_csv(trainingSet, dtype = self.dtype_dict)

    # pick regressors needed
    feature_matrix = df[features]

    # add constant/intercept term (1s) as the first col of feature matrix
    intercept = pd.Series(1, index = feature_matrix.index)
    feature_matrix.insert(0, 'constant', intercept)

    # take out predictor values from training set
    predictor_val = df[predictor]

    return feature_matrix, predictor_val

  def predict_output(self, feature_matrix, weights):
    # y = h(x) * w - matrix multiplication, ignoring error term
    predictions = feature_matrix.dot(weights)
    return predictions

  def normalize_features(self, features):
    norms = np.linalg.norm(features, axis = 0) # compute normalizer
    normalized_features = features/norms # element-wise division
    return normalized_features, norms
    
  def euclid_dist(self, vectorA, vectorB):
    # double start means "power of" operation
    return np.sqrt(np.sum((vectorA - vectorB)**2))
    
  '''
  the following method calculates euclidean distance between
  a vector again another k vectors wrapped in a matrix.
  It uses numpy vectorizaion for the k-vector matrix
  '''
  def compute_distances(self, predictor, features_instances):
    # numpy Vectorization #
    diff = features_instances - predictor
    distances = np.sqrt(np.sum(diff**2, axis = 1))
    return distances
    
  def k_nearest_neighbors(self, k, feature_train, features_query):
    distances = self.compute_distances(features_query, feature_train)
    sorted_d_index = np.argsort(distances)
    return sorted_d_index


knn = knnRegression()
  
feature_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'waterfront', 'view', 'condition',
                'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                'yr_renovated', 'lat', 'long', 'sqft_living15',
                'sqft_lot15']  
  
train, predictor_output = knn.get_numpy_data("kc_house_data_small_train.csv", feature_list, "price")
test, predictor = knn.get_numpy_data("kc_house_data_small_test.csv", feature_list, "price")
validation, actualV = knn.get_numpy_data("kc_house_data_validation.csv", feature_list, "price")
  
features_train, norms = knn.normalize_features(train)
features_test = test / norms
features_valid = validation / norms
  
# for pandas DF .iloc to explicity support ONLY integer indexing,
# and .loc to explicity support ONLY label/customized/user-defined indexing
#
# .as_matrix() converts pandas DF to numpy nd array
queryHouse = features_test.iloc[[0]].as_matrix()
  
for i in range(0, 10):
  print i
  print knn.euclid_dist(queryHouse, features_train.as_matrix()[i])
   
queryHouse = features_test.iloc[[2]].as_matrix()
euclid = knn.compute_distances(queryHouse, features_train.as_matrix())
oneNN_distance = min(euclid)
oneNN = np.where(euclid == oneNN_distance)
print predictor_output[382]

print "\n\nKNN REGRESSION\n\n"
knn_index = knn.k_nearest_neighbors(4, features_train.as_matrix(), queryHouse)
knn_values = predictor_output[knn_index]
print np.mean(knn_values[0:4,])