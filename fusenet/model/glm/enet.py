import warnings

import numpy as np

from ..base import LinearModel, _pre_fit
import path


class ElasticNet(LinearModel):
    """Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2 +
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
        See the notes for the exact mathematical meaning of this
        parameter.

    n_alphas : float
        Number of values tried for alpha parameter. If not None (default) then
        the grid of alpha values for network inference is defined.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept: bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.

    precompute : True | False | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.

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

    positive: bool, optional
        When set to ``True``, forces the coefficients to be positive.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.

    Attributes
    ----------
    coef_ : array, shape = (n_features,)
        parameter vector (w in the cost function formula)

    intercept_ : float
        independent term in decision function.

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """
    def __init__(self, alpha=1.0, n_alphas=None, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=np.random.RandomState(), selection='cyclic'):
        self.alpha = alpha
        self.n_alphas = None
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.intercept_ = 0.0
        self.random_state = random_state
        self.selection = selection

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

        X, y, X_mean, y_mean, X_std, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=True)

        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or self.coef_ is None:
            coef_ = np.zeros(n_features, dtype=np.float64, order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        self.n_iter_ = []

        model = path.enet_path(X, y, l1_ratio=self.l1_ratio, eps=None, n_alphas=self.n_alphas,
                               alphas=[self.alpha], precompute=precompute, Xy=None,
                               fit_intercept=False, normalize=False, copy_X=True, verbose=False,
                               tol=self.tol, positive=self.positive, X_mean=X_mean,
                               X_std=X_std, return_n_iter=True, coef_init=coef_,
                               max_iter=self.max_iter, random_state=self.random_state,
                               selection=self.selection)
        this_alpha, this_coef, this_dual_gap, this_iter = model
        coef_ = this_coef
        alpha_ = this_alpha
        dual_gap_ = this_dual_gap
        self.n_iter_ = this_iter

        self.alpha_ = np.squeeze(alpha_)
        self.coef_ = np.squeeze(coef_)
        self.dual_gap_ = np.squeeze(dual_gap_)
        self._set_intercept(X_mean, y_mean, X_std)

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
        return super(ElasticNet, self).decision_function(X)