import pandas as pd
import numpy as np

class ridgeRegression:
  # data type mapping for training set CSV
  dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
    'sqft_living15':float, 'grade':int, 'yr_renovated':int,
    'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
    'sqft_lot15':float, 'sqft_living':float, 'floors':str,
    'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
    'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

  def Prep_FeatureMatrix_OutputVec(self, trainingSet, features, predictor):
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

  def feature_derivative_ridge(self, error_vector, feature_j, weight_j,
		  l2_penalty, feature_is_constant):
    RSS_gradient_wrt_j = 2 * feature_j.dot(error_vector)
    L2Norm_gradient_wrt_j = 2 * l2_penalty * weight_j

    # NO regularization for constant/intercept term
    if feature_is_constant:
      return RSS_gradient_wrt_j
    else:
      return RSS_gradient_wrt_j + L2Norm_gradient_wrt_j

  def ridge_regression_gradient_descent(self, feature, predictor,
		  initial_weights, step_size, l2_penalty, max_iterations = 100):
    weights = np.array(initial_weights)

    i = 0
    while i <= max_iterations:
      predictions = self.predict_output(feature, weights)
      errors = predictions - predictor

      # compute partial derivative wrt each weight
      for j in xrange(len(weights)):
        if j == 0:
	  feature_constant = True
	else:
	  feature_constant = False

	partial_wrt_wj = self.feature_derivative_ridge(errors, feature.ix[:,j],
            weights[j], l2_penalty, feature_constant)
	weights[j] = weights[j] - step_size * partial_wrt_wj

      i = i + 1
    return weights


class unitTests:
  rr = ridgeRegression()

  def test_feature_derivative_ridge(self):
    example_features, example_output = self.rr.Prep_FeatureMatrix_OutputVec(
      "kc_house_train_data.csv", ["sqft_living"], "price")
    myWeights = np.array([1., 10.])
    test_predictions = self.rr.predict_output(example_features, myWeights)
    errors = test_predictions - example_output # prediction error/residual vector

    # the following 2 print statements should generate the same values
    derivative_wrt_feature = self.rr.feature_derivative_ridge(errors,
        example_features['sqft_living'], myWeights[1], 1, False)
    print derivative_wrt_feature
    derivative_wrt_feature_test = np.sum(errors * example_features['sqft_living']) * 2 + 20.
    print derivative_wrt_feature_test

    # the following 2 print statements should generate the same values
    derivative_wrt_constant = self.rr.feature_derivative_ridge(errors,
        example_features['constant'], myWeights[0], 1, True)
    print derivative_wrt_constant
    derivative_wrt_constant_test = np.sum(errors)*2.
    print derivative_wrt_constant_test

    print "test result"
    print derivative_wrt_feature == derivative_wrt_feature_test
    print derivative_wrt_constant == derivative_wrt_constant_test


def main():
  # testing
  ut = unitTests()
  ut.test_feature_derivative_ridge()
  
  # quiz problem solving
  rr = ridgeRegression()

  features = ["sqft_living"]
  predictor = "price"
  step_size = 1e-12
  max_iterations = 1000
  initial_weights = np.zeros(len(features) + 1)

  simple_feature, output = rr.Prep_FeatureMatrix_OutputVec(
      "kc_house_train_data.csv", features, predictor)
  simple_test_feature, test_output = rr.Prep_FeatureMatrix_OutputVec(
      "kc_house_test_data.csv", features, predictor)

  simple_weights_0_penalty = rr.ridge_regression_gradient_descent(
      simple_feature, output, initial_weights, step_size, 0, max_iterations)
  print simple_weights_0_penalty
  
  simple_weights_high_penalty = rr.ridge_regression_gradient_descent(
      simple_feature, output, initial_weights, step_size, 1e11, max_iterations)
  print simple_weights_high_penalty

if __name__ == "__main__":
  main()
