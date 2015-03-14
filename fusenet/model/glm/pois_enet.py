import warnings

import numpy as np

import path
from ..base import LinearModel


class PoissonENet(LinearModel):
    """Poisson regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::

            (yXw - exp(Xw) +
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * L1 + b * L2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.

    Parameters
    ----------
    alpha : float
        Constant that multiplies the penalty terms. Defaults to 1.0

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.

    max_iter : int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    tol: float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    verbose : boolean
        An indicator of verbosity.

    Attributes
    ----------
    coef_ : array, shape = (n_features,)
        parameter vector (w in the cost function formula)

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, copy_X=True,
                 tol=1e-4, warm_start=False, random_state=np.random.RandomState(),
                 selection='cyclic', verbose=False):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.intercept_ = 0.0
        self.random_state = random_state
        self.selection = selection
        self.verbose = verbose

    def fit(self, X, y):
        """Fit model with coordinate descent.

        Parameters
        -----------
        X : ndarray (n_samples, n_features)
            Data

        y : ndarray, shape = (n_samples,)
            Target

        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        if self.alpha == 0:
            warnings.warn("With alpha=0 there is no regularization.", stacklevel=2)

        n_samples, n_features = X.shape

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or self.coef_ is None:
            coef_ = np.zeros(n_features, dtype=np.float64, order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        _, this_coef, this_iter = path.poisson_path(X, y, l1_ratio=self.l1_ratio, eps=None,
                                                    n_alphas=None, alphas=[self.alpha],
                                                    copy_X=True, verbose=self.verbose,
                                                    tol=self.tol, return_n_iter=True,
                                                    coef_init=coef_, max_iter=self.max_iter,
                                                    random_state=self.random_state,
                                                    selection=self.selection)
        self.n_iter_ = this_iter
        self.coef_ = np.squeeze(this_coef)

        return self

    def decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)

        Returns
        -------
        T : array, shape = (n_samples,)
            The predicted decision function
        """
        return super(PoissonENet, self).decision_function(X)