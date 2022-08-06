# rrpy

**rrpy** is a scikit-learn compatible Python implementation of reduced rank ridge regression.
It is based on the `rrs.fit` method of the R package **rrpack**, which is in turn based on [[1]](#1).

## Installation

```
pip install git+https://github.com/krey/rrpy.git
```

## Usage

This implementation does not support missing values, though such a feature could be added using https://github.com/aksarkar/wlra.

The `ReducedRankRidge` estimator has a `memory` parameter which allows rapid tuning of the `rank` parameter:
```python
import sklearn.datasets
import joblib
from rrpy import ReducedRankRidge
X, Y = sklearn.datasets.make_regression(n_samples=1000, n_features=500, n_targets=50, random_state=1, n_informative=25)
memory = joblib.Memory(location='/tmp/rrpy-test/', verbose=2)
estimator = ReducedRankRidge(memory=memory, rank=10)
estimator.fit(X, Y)
estimator.rank = 20
estimator.fit(X, Y) # cached
memory.clear(warn=False)
```

## References
<a id="1">[1]</a> 
Mukherjee, A. and Zhu, J. (2011)
Reduced rank ridge regression and its kernel extensions.
