import warnings

import numpy as np

import path


class MultinomialFNet():
    """Network inference via estimation of a factorized multinomial graphical model.

    We minimize::

        - 1/n sum(r=1 ...p : sum(i=1 ...n : log(P(X_r = x_r^(i) | X_\r = x_\r^(i))))) +
        + alpha * l12_ratio * || Theta_\r ||_1,2

    where::

        || Theta_\r ||_1,2 = sum(u in X.shape[1], u neq r : || theta_ru ||_2

    and::

        theta_ru = [theta_ru;jk for j in range(m-1) for k in range(m-1)]

    and::

        theta_ru;jk = U_r^T V_rj W^T W V_uk U_u.

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

    U_ : ndarray, shape (k, n_features, n_alphas)
        Coefficients along the path.

    V_ : ndarray, shape (n_features, m, n_alphas)
        Coefficients along the path.

    W_ : ndarray, shape (k, k, n_alphas)
        Coefficients along the path.

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """
    def __init__(self, alpha=1.0, n_alphas=None, l12_ratio=0.5, max_iter=1000, copy_X=True,
                 tol=1e-4, warm_start=False, verbose=False):
        self.alpha = alpha
        self.n_alphas = None
        self.l12_ratio = l12_ratio
        self.bias_theta_ = None
        self.U_ = None
        self.V_ = None
        self.W_ = None
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, m=None, k=None):
        """Fit model with coordinate descent.

        Parameters
        -----------
        X : ndarray (n_samples, n_features)
            Data.

        m : {array-like}
            Discrete values that define the domain of multinomial distribution.

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

        sc = 1e-1
        if not self.warm_start or self.bias_theta_ is None:
            bias_theta_ = sc * np.random.rand(n_features, n_m)
            bias_theta_ = np.asfortranarray(bias_theta_)
        else:
            bias_theta_ = self.bias_theta_

        if not self.warm_start or self.U_ is None:
            U_ = sc * np.random.rand(self.k, n_features)
            U_ = np.asfortranarray(U_)
        else:
            U_ = self.U_

        if not self.warm_start or self.V_ is None:
            V_ = sc * np.random.rand(n_features, n_m)
            V_ = np.asfortranarray(V_)
        else:
            V_ = self.V_

        if not self.warm_start or self.W_ is None:
            W_ = sc * np.random.rand(self.k, self.k)
            W_ = np.asfortranarray(W_)
        else:
            W_ = self.W_

        model = path.multinomial_net_path(X, m=m, k=k, l12_ratio=self.l12_ratio, eps=None,
                                      n_alphas=self.n_alphas, alphas=[self.alpha],
                                      copy_X=True, bias_theta_init=bias_theta_,
                                      U_init=U_, V_init=V_, W_init=W_,
                                      verbose=self.verbose, return_n_iter=True,
                                      max_iter=self.max_iter, tol=self.tol)
        alpha, bias_theta, U, V, W, n_iter = model
        bias_theta = np.asarray(bias_theta)

        self.n_iter_ = n_iter
        self.alpha_ = np.squeeze(alpha)

        # multiply appropriately to obtain feature dependency structure
        D = np.zeros((n_features, n_features, len(self.alpha_)))
        xidx, yidx = np.triu_indices(n_features, 1)
        for i_alpha, alpha in enumerate(self.alpha_):
            for r, t in zip(xidx, yidx):
                tmp1 = np.dot(W[..., i_alpha], U[:, t, i_alpha]).reshape(k, 1)
                tmp2 = np.dot(U[:, r, i_alpha].T, np.dot(W[..., i_alpha].T, tmp1))
                vec = np.sum(np.dot(V[r, :, i_alpha].T, V[t, :, i_alpha])) * tmp2
                val = np.linalg.norm(vec, 2)
                D[r, t, i_alpha] = D[t, r, i_alpha] = val

        self.bias_theta_ = np.squeeze(bias_theta)
        self.U_ = np.squeeze(U)
        self.V_ = np.squeeze(V)
        self.W_ = np.squeeze(W)
        self.D_ = np.squeeze(D)

        return self
