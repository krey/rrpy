import numpy as np
import sklearn.datasets
import sklearn.linear_model

from rrpy import ReducedRankRidge

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
    estimator = ReducedRankRidge(fit_intercept=True, rank=20).fit(X, Y)
    assert estimator.rank_ <= 20, f"{estimator.rank_} > 20"

def test_reduced_rank_regression_vs_ridge():
    X, Y = sklearn.datasets.make_regression(n_samples=500, n_features=5, n_targets=5, random_state=0, n_informative=2)
    alpha = 1e5
    ridge = sklearn.linear_model.Ridge(alpha=alpha).fit(X, Y)
    rrr = ReducedRankRidge(rank=min(X.shape[1],Y.shape[1]), alpha=alpha).fit(X, Y)
    np.testing.assert_almost_equal(ridge.coef_, rrr.coef_)
    rrr = ReducedRankRidge(rank=1, alpha=alpha).fit(X, Y)
    np.testing.assert_array_less(rrr.score(X, Y), ridge.score(X, Y))

