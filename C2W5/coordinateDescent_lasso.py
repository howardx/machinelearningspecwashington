import pandas as pd
import numpy as np

class lasso:
  # data type mapping for training set CSV
  dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
    'sqft_living15':float, 'grade':int, 'yr_renovated':int,
    'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
    'sqft_lot15':float, 'sqft_living':float, 'floors':str,
    'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
    'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

  def get_numpy_data(self, trainingSet, features, predictor):
    df = pd.read_csv(trainingSet, dtype = self.dtype_dict)
    dim = df.shape

    # training set row count indicates number of observations
    observations = dim[0]

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


class unitTests:
  las = lasso()
  def test_normalization(self):
    feature, output = self.las.get_numpy_data("kc_house_train_data.csv",
      ["sqft_living"], "price")
    ##print self.las.normalize_features(feature)


def main():
  # testing
  ut = unitTests()
  ut.test_normalization()
  
  # quiz problem solving
  las = lasso()

if __name__ == "__main__":
  main()
