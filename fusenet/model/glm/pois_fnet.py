import warnings

import numpy as np

import path


class PoissonFNet():
    """Network inference via estimation of a factorized Poisson graphical model.

    We minimize::

       sum_s sum_i X_s^i <U_\s W^T W u_s, X_\s^i> - exp(<U_\s W^T W u_s, X_\s^i>)
                   + alpha norm(T, 1) + beta norm(T, 2)^2

    where::

       T = U_\s^T W^T W u_s

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
    D_ : ndarray, shape (n_features, n_features)
        Feature dependency structure.

    U_ : ndarray, shape (k, n_features, n_alphas)
        Coefficients along the path.

    W_ : ndarray, shape (k, k, n_alphas)
        Coefficients along the path.

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """
    def __init__(self, alpha=1.0, n_alphas=None, l1_ratio=0.5, max_iter=1000, copy_X=True,
                 tol=1e-4, warm_start=False, random_state=np.random.RandomState(),
                 selection='cyclic', verbose=False):
        self.alpha = alpha
        self.n_alphas = None
        self.l1_ratio = l1_ratio
        self.U_ = None
        self.W_ = None
        self.Theta_ = None
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection
        self.verbose = verbose

    def fit(self, X, k=None):
        """Fit model with coordinate descent.

        Parameters
        -----------
        X : ndarray (n_samples, n_features)
            Data.

        k : int, optional
            Latent model dimensionality (5 by default).

        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        if not k:
            warnings.warn("Latent dimensionality, i.e. factorization rank is"
                          "not set. Default option of k=5 will be used.", stacklevel=2)
            self.k = 5
        else:
            self.k = k
        if self.alpha == 0:
            warnings.warn("With alpha=0 there is no regularization.", stacklevel=2)

        n_samples, n_features = X.shape

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        sc = 1e-1
        if not self.warm_start or self.U_ is None:
            U_ = sc * np.random.rand(self.k, n_features) #- 0.5 * sc
            U_ = np.asfortranarray(U_)
        else:
            U_ = self.U_

        if not self.warm_start or self.W_ is None:
            W_ = sc * np.random.rand(self.k, self.k) #- 0.5 * sc
            W_ = np.asfortranarray(W_)
        else:
            W_ = self.W_

        alpha, U, W, n_iter = path.poisson_net_path(X, k=k,
                      l1_ratio=self.l1_ratio, eps=None,
                      n_alphas=self.n_alphas, alphas=[self.alpha],
                      copy_X=True, U_init=U_, W_init=W_,
                      verbose=self.verbose, tol=self.tol, return_n_iter=True,
                      max_iter=self.max_iter, random_state=self.random_state,
                      selection=self.selection)

        self.n_iter_ = n_iter

        # multiply appropriately to obtain feature dependency structure
        D = np.zeros((n_features, n_features, len(self.alpha_)))
        for i_alpha, alpha in enumerate(self.alpha_):
            for ii in range(n_features):
                idxs = list(range(n_features))
                idxs.remove(ii)
                tmp = np.dot(W[..., i_alpha], U[:, idxs, i_alpha])
                theta = np.dot(U[:, ii, i_alpha].T, np.dot(W[..., i_alpha].T, tmp))
                D[ii, idxs, i_alpha] = theta

        self.alpha_ = alpha
        self.U_ = np.squeeze(U)
        self.W_ = np.squeeze(W)
        self.D_ = np.squeeze(D)

        return self
