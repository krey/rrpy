import numpy as np
import sklearn.datasets
import sklearn.linear_model

from .. import ReducedRankRidge

def test_reduced_rank_regression_intercept():
    X, Y = sklearn.datasets.make_regression(n_samples=100, n_features=50, n_targets=50, random_state=0, n_informative=25)
    estimator = ReducedRankRidge(alpha=10, fit_intercept=True).fit(X, Y)
    residuals = Y-estimator.predict(X)
    np.testing.assert_almost_equal(np.sum(residuals, axis=0), 0)
    
def test_reduced_rank_regresssion_no_intercept():
    X, Y = sklearn.datasets.make_regression(n_samples=100, n_features=50, n_targets=50, random_state=0, n_informative=25)
    estimator = ReducedRankRidge(fit_intercept=False).fit(X, Y)
    np.testing.assert_almost_equal(estimator.intercept_, 0)
    
def test_reduced_rank_regression_rank():
    X, Y = sklearn.datasets.make_regression(n_samples=100, n_features=50, n_targets=50, random_state=0, n_informative=25)
    estimator = ReducedRankRidge(fit_intercept=True).fit(X, Y)
    np.testing.assert_almost_equal(estimator.rank_, 25)
    
def test_reduced_rank_regression_vs_ridge():
    X, Y = sklearn.datasets.make_regression(n_samples=100, n_features=50, n_targets=50, random_state=0, n_informative=50)
    ridge = sklearn.linear_model.Ridge().fit(X, Y)
    rrr = ReducedRankRidge(rank=max(X.shape)).fit(X, Y)
    np.testing.assert_almost_equal(ridge.coef_, rrr.coef_)
    rrr = ReducedRankRidge(rank=5).fit(X, Y)
    np.testing.assert_array_less(rrr.score(X, Y), ridge.score(X, Y))