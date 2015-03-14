from abc import abstractmethod

import numpy as np


def _net_alpha_grid(X, l1_ratio=1.0, eps=1e-3, n_alphas=100):
    """ Compute the grid of alpha values for elastic net parameter search

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    l1_ratio : float
        The elastic net mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. ``For
        l1_ratio = 1`` it is an L1 penalty.  For ``0 < l1_ratio <
        1``, the penalty is a combination of L1 and L2.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path
    """
    n_samples = X.shape[0]

    alpha_max = (np.sqrt(np.sum(X ** 2, axis=1)).max() / (n_samples * l1_ratio))
    alphas = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), num=n_alphas)[::-1]
    return alphas


def _alpha_grid(X, y, Xy=None, l1_ratio=1.0, fit_intercept=True,
                eps=1e-3, n_alphas=100, normalize=False, copy_X=True):
    """Compute the grid of alpha values for network inference parameter search

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    y : ndarray, shape = (n_samples,)
        Target values

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed.

    l1_ratio : float
        The elastic net mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. ``For
        l1_ratio = 1`` it is an L1 penalty.  For ``0 < l1_ratio <
        1``, the penalty is a combination of L1 and L2.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    fit_intercept : bool
        Fit or not an intercept

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    n_samples = X.shape[0]

    if Xy is None:
        X, y, _, _, _ = center_data(X, y, fit_intercept, normalize, copy=False)
        Xy = np.dot(X.T, y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]
    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /(n_samples * l1_ratio))
    alphas = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), num=n_alphas)[::-1]
    return alphas


def _pre_fit(X, y, Xy, precompute, normalize, fit_intercept, copy):
    """Aux function used at beginning of fit in linear models"""
    n_samples, n_features = X.shape
    # copy was done in fit if necessary
    X, y, X_mean, y_mean, X_std = center_data(X, y, fit_intercept, normalize, copy=copy)

    if hasattr(precompute, '__array__') \
            and not np.allclose(X_mean, np.zeros(n_features)) \
            and not np.allclose(X_std, np.ones(n_features)):
        # recompute Gram
        precompute = 'auto'
        Xy = None

    # precompute if n_samples > n_features
    if precompute == 'auto':
        precompute = (n_samples > n_features)

    if precompute is True:
        precompute = np.dot(X.T, X)

    if not hasattr(precompute, '__array__'):
        Xy = None  # cannot use Xy if precompute is not Gram

    if hasattr(precompute, '__array__') and Xy is None:
        Xy = np.dot(X.T, y)

    return X, y, X_mean, y_mean, X_std, precompute, Xy


def center_data(X, y, fit_intercept, normalize=False, copy=True):
    """
    Centers data to have mean zero along axis 0. This is here because
    nearly all linear models will want their data to be centered.

    """
    X = np.copy(X, copy)
    if fit_intercept:
        X_mean = np.average(X, axis=0)
        X -= X_mean
        if normalize:
            # XXX: currently scaled to variance=n_samples
            X_std = np.sqrt(np.sum(X ** 2, axis=0))
            X_std[X_std == 0] = 1
            X /= X_std
        else:
            X_std = np.ones(X.shape[1])
        y_mean = np.average(y, axis=0)
        y = y - y_mean
    else:
        X_mean = np.zeros(X.shape[1])
        X_std = np.ones(X.shape[1])
        y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
    return X, y, X_mean, y_mean, X_std


class LinearModel():
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def decision_function(self, X):
        """Decision function of the linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return np.dot(X, self.coef_.T) + self.intercept_

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return self.decision_function(X)

    _center_data = staticmethod(center_data)

    def _set_intercept(self, X_mean, y_mean, X_std):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.
