import warnings

import numpy as np

import path


class MultinomialNet():
    """Network inference via estimation of a nonfactorized multinomial graphical model.

    We minimize::

        - 1/n sum(r=1 ...p : sum(i=1 ...n : log(P(X_r = x_r^(i) | X_\r = x_\r^(i))))) +
        + alpha * l12_ratio * || Theta_\r ||_1,2

    where::

        || Theta_\r ||_1,2 = sum(u in X.shape[1], u neq r : || theta_ru ||_2

    and:

        theta_ru = [theta_ru;jk for j in range(m-1) for k in range(m-1)].

    Parameters
    ----------
    alpha : float
        Constant that multiplies the penalty terms. Defaults to 1.0
        See the notes for the exact mathematical meaning of this
        parameter.

    n_alphas : float
        Number of values tried for alpha parameter. If not None (default) then
        the grid of alpha values for network inference is defined.

    l12_ratio : float, optional
        float between 0 and 1 denoting group sparse regularizer.

    max_iter : int, optional
        The maximum number of iterations

    copy_X : boolean, optional}, default True
        If ``True``, X will be copied; else, it may be overwritten.

    tol: float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    verbose : boolean
        An indicator of verbosity.

    Attributes
    ----------
    D_ : ndarray, shape (n_features, n_features)
        Feature dependency structure.

    theta_ : ndarray, shape (n_features, n_values, n_alphas)
        Coefficients along the path.

    Theta_ : ndarray, shape (n_features, n_features, n_values, n_values, n_alphas)
        Coefficients along the path

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """
    def __init__(self, alpha=1.0, n_alphas=None, l12_ratio=0.5, max_iter=1000, copy_X=True,
                 tol=1e-4, warm_start=False, verbose=False):
        self.alpha = alpha
        self.n_alphas = None
        self.l12_ratio = l12_ratio
        self.theta_ = None
        self.Theta_ = None
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, m=None):
        """Fit model with coordinate descent.

        Parameters
        -----------
        X : ndarray (n_samples, n_features)
            Data.

        m : {array-like}
            Discrete values that define the domain of multinomial distribution.

        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        if m is None:
            warnings.warn("Domain of multinomial distribution is not provided."
                          "Domain will be estimated from data", stacklevel=2)
            self.m = np.array(set(X.flatten().tolist()))
        else:
            self.m = m
        n_m = len(self.m)
        if self.alpha == 0:
            warnings.warn("With alpha=0 there is no regularization.", stacklevel=2)

        n_samples, n_features = X.shape

        sc = 1.
        if not self.warm_start or self.theta_ is None:
            theta_ = sc * np.random.rand(n_features, n_m)
            theta_ = np.asfortranarray(theta_)
        else:
            theta_ = self.theta_

        if not self.warm_start or self.Theta_ is None:
            Theta_ = sc * np.random.rand(n_features, n_features, n_m, n_m)
            Theta_ = np.asfortranarray(Theta_)
        else:
            Theta_ = self.Theta_

        model = path.multinomial_path(X, m=m, l12_ratio=self.l12_ratio, eps=None,
                                      n_alphas=self.n_alphas, alphas=[self.alpha],
                                      copy_X=True, theta_init=theta_, Theta_init=Theta_,
                                      verbose=self.verbose, return_n_iter=True,
                                      max_iter=self.max_iter, tol=self.tol)
        alpha, theta, Theta, n_iter = model

        self.n_iter_ = n_iter

        # multiply appropriately to obtain feature dependency structure
        D = np.zeros((n_features, n_features, len(alpha)))
        xidx, yidx = np.triu_indices(n_features, 1)
        for i_alpha, alpha in enumerate(alpha):
            for r, t in zip(xidx, yidx):
                val = np.linalg.norm(Theta_[r, t, ...], 2)
                D[r, t, i_alpha] = D[t, r, i_alpha] = val

        self.alpha_ = np.squeeze(alpha)
        self.theta_ = np.squeeze(theta)
        self.Theta_ = np.squeeze(Theta)
        self.D_ = np.squeeze(D)

        return self
