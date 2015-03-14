import sys
import warnings

import numpy as np

from ..base import _alpha_grid, _net_alpha_grid, _pre_fit
from .. import cd_fast


def multinomial_path(X, m, l12_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              copy_X=True, theta_init=None, Theta_init=None, verbose=False,
              return_n_iter=False, **params):
    """Compute multinomial nonfactorized net path with coordinate descent

    The optimization function is::

        - 1/n sum(r=1 ...p : sum(i=1 ...n : log(P(X_r = x_r^(i) | X_\r = x_\r^(i))))) +
        + alpha * l12_ratio * || Theta_\r ||_1,2

    where::

        || Theta_\r ||_1,2 = sum(u in X.shape[1], u neq r : || theta_ru ||_2

    and:

        theta_ru = [theta_ru;jk for j in range(m-1) for k in range(m-1)].

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    m : {array-like}
        Discrete values that define the domain of multinomial distribution.

    l12_ratio : float, optional
        float between 0 and 1 denoting group sparse regularizer.

    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    theta_init : ndarray, shape (n_features, n_m) | None
        The initial values of the coefficients (n_m is the cardinality of the
        multinomial distribution).

    Theta_init : ndarray, shape (n_features, n_features, n_m, n_m) | None
        The initial values of the coefficients (n_m is the cardinality of the
        multinomial distribution).

    verbose : boolean
        An indicator of verbosity.

    params : kwargs
        keyword arguments passed to the coordinate descent solver.

    return_n_iter : bool
        whether to return the number of iterations or not.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    thetas : array, shape (n_features, n_m, n_alphas)
        Coefficients along the path.

    Thetas : array, shape (n_features, n_features, n_m, n_m, n_alphas)
        Coefficients along the path

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
    """
    if verbose: print 'Entering multinomial net path'
    X = np.array(X, dtype=np.float64, order='F', copy=copy_X)
    n_samples, n_features = X.shape

    if alphas is None:
        alphas = _net_alpha_grid(X, l1_ratio=l12_ratio, eps=eps, n_alphas=n_alphas)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    n_iters = []

    rng = params.get('random_state', np.random.RandomState())

    n_m = len(m)
    thetas = np.empty((n_features, n_m, n_alphas), dtype=np.float64)
    Thetas = np.empty((n_features, n_features, n_m, n_m, n_alphas), dtype=np.float64)

    sc = 1e-1
    if theta_init is None:
        theta_ = sc * np.random.rand(n_features, n_m)
        theta_ = np.asfortranarray(theta_)
    else:
        theta_ = np.asfortranarray(theta_init)

    if Theta_init is None:
        Theta_ = sc * np.random.rand(n_features, n_features, n_m, n_m)
        Theta_ = np.asfortranarray(Theta_)
    else:
        Theta_ = np.asfortranarray(Theta_init)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l12_ratio * n_samples
        l2_reg = alpha * (1.0 - l12_ratio) * n_samples
        model = cd_fast.multinomial_descent(theta_, Theta_, m, l1_reg, l2_reg, X,
                                             max_iter, tol, rng, verbose=verbose)
        theta_, Theta_, n_iter_ = model
        theta_ = np.array(theta_)
        Theta_ = np.array(Theta_)

        thetas[..., i] = theta_
        Thetas[..., i] = Theta_
        n_iters.append(n_iter_)
        if verbose: print('Path: %03i out of %03i' % (i, n_alphas))

    if return_n_iter:
        return alphas, thetas, Thetas, n_iters
    else:
        return alphas, thetas, Thetas


def multinomial_net_path(X, m, k=5, l12_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              copy_X=True, bias_theta_init=None, U_init=None, V_init=None,
              W_init=None, verbose=False, return_n_iter=False, **params):
    """Compute multinomial factorized net path with coordinate descent

    The optimization function is::

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
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    m : {array-like}
        Discrete values that define the domain of multinomial distribution.

    k : int, optional
        Dimensionality of latent factor model.

    l12_ratio : float, optional
        float between 0 and 1 denoting group sparse regularizer.

    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    bias_theta_init : ndarray, shape (n_features, n_m) | None
        The initial values of the coefficients (n_m is the cardinality of the
        multinomial distribution).

    U_init : ndarray, shape (k, n_features) | None
        The initial values of the coefficients.

    V_init : ndarray, shape (n_features, n_m) | None
        The initial value of the coefficients (n_m is the cardinality of the
        multinomial distribution).

    W_init : ndarray, shape (k, k) | None
        The initial value of the coefficients.

    verbose : boolean
        An indicator of verbosity.

    params : kwargs
        keyword arguments passed to the coordinate descent solver.

    return_n_iter : bool
        whether to return the number of iterations or not.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    Us : array, shape (k, n_features, n_alphas)
        Coefficients along the path.

    Vs : array, shape (n_features, m, n_alphas)
        Coefficients along the path

    Ws : array, shape (k, k, n_alphas)
        Coefficients along the path

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
    """
    if verbose: print 'Entering multinomial factorized net path'
    X = np.array(X, dtype=np.float64, order='F', copy=copy_X)
    n_samples, n_features = X.shape

    if alphas is None:
        alphas = _net_alpha_grid(X, l1_ratio=l12_ratio, eps=eps, n_alphas=n_alphas)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    n_iters = []

    rng = params.get('random_state', np.random.RandomState())

    n_m = len(m)
    bias_thetas = np.empty((n_features, n_m, n_alphas), dtype=np.float64)
    Us = np.empty((k, n_features, n_alphas), dtype=np.float64)
    Vs = np.empty((n_features, n_m, n_alphas), dtype=np.float64)
    Ws = np.empty((k, k, n_alphas), dtype=np.float64)

    sc = 1e-1
    if bias_theta_init is None:
        bias_theta_ = sc * np.random.rand(n_features, n_m)
        bias_theta_ = np.asfortranarray(bias_theta_)
    else:
        bias_theta_ = np.asfortranarray(bias_theta_init)

    if U_init is None:
        U_ = sc * np.random.rand(k, n_features)
        U_ = np.asfortranarray(U_)
    else:
        U_ = np.asfortranarray(U_init)

    if V_init is None:
        V_ = sc * np.random.rand(n_features, n_m)
        V_ = np.asfortranarray(V_)
    else:
        V_ = np.asfortranarray(V_init)

    if W_init is None:
        W_ = sc * np.random.rand(k, k)
        W_ = np.asfortranarray(W_)
    else:
        W_ = np.asfortranarray(W_init)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l12_ratio * n_samples
        l2_reg = alpha * (1.0 - l12_ratio) * n_samples
        model = cd_fast.factorized_multinomial_descent(bias_theta_, U_, W_, V_, m, l1_reg, l2_reg,
                                                       X, max_iter, tol, rng, verbose=verbose)
        bias_theta_, U_, V_, W_, n_iter_ = model
        bias_theta_ = np.array(bias_theta_)
        U_ = np.array(U_)
        V_ = np.array(V_)
        W_ = np.array(W_)

        bias_thetas[..., i] = bias_theta_
        Us[..., i] = U_
        Vs[..., i] = V_
        Ws[..., i] = W_
        n_iters.append(n_iter_)
        if verbose: print('Path: %03i out of %03i' % (i, n_alphas))

    if return_n_iter:
        return alphas, bias_thetas, Us, Vs, Ws, n_iters
    else:
        return alphas, bias_thetas, Us, Vs, Ws


def poisson_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              copy_X=True, coef_init=None, verbose=False, return_n_iter=False, **params):
    """Compute Poisson net path with coordinate descent

    The optimization function is::

        (yXw - exp(Xw)) +
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples,)
        Target values

    l1_ratio : float, optional
        float between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso

    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.

    verbose : boolean
        An indicator of verbosity.

    params : kwargs
        keyword arguments passed to the coordinate descent solver.

    return_n_iter : bool
        whether to return the number of iterations or not.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
    """
    X = np.array(X, dtype=np.float64, order='F', copy=copy_X)
    y = np.array(y, dtype=np.float64, order='F', copy=True)

    n_samples, n_features = X.shape

    if alphas is None:
        # No need to normalize fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=None, l1_ratio=l1_ratio,
                             fit_intercept=False, eps=eps, n_alphas=n_alphas,
                             normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    n_iters = []

    rng = params.get('random_state', np.random.RandomState())
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    coefs = np.empty((n_features, n_alphas), dtype=np.float64)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1]))
    else:
        coef_ = np.asfortranarray(coef_init)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        model = cd_fast.poisson_coordinate_descent(coef_, l1_reg, l2_reg, X, y,
                                                     max_iter, tol, rng, random, verbose)
        coef_, n_iter_ = model
        coefs[..., i] = coef_
        n_iters.append(n_iter_)
        if verbose: print('Path: %03i out of %03i' % (i, n_alphas))

    if return_n_iter:
        return alphas, coefs, n_iters
    else:
        return alphas, coefs


def poisson_net_path(X, k=5, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              copy_X=True, U_init=None, W_init=None, verbose=False,
              return_n_iter=False, **params):
    """Compute Poisson net path with coordinate descent.

    We minimize::

       sum_s sum_i X_s^i <U_\s W^T W u_s, X_\s^i> - exp(<U_\s W^T W u_s, X_\s^i>)
                   + alpha norm(T, 1) + beta norm(T, 2)^2

    where::

       T = U_\s^T W^T W u_s

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    k : int, optional
        Dimensionality of latent factor model.

    l1_ratio : float, optional
        float between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso.

    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    U_init : ndarray, shape (k, n_features) | None
        The initial values of the coefficients.

    W_init : ndarray, shape (k, k) | None
        The initial values of the coefficients.

    verbose : boolean
        An indicator of verbosity.

    params : kwargs
        keyword arguments passed to the descent solver.

    return_n_iter : bool
        whether to return the number of iterations or not.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    Us : array, shape (k, n_features, n_alphas)
        Coefficients along the path.

    Ws : array, shape (k, k, n_alphas)
        Coefficients along the path

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the descent optimizer to
        reach the specified tolerance for each alpha.
    """
    if verbose: print 'Entering Poisson factorized net path'
    X = np.array(X, dtype=np.float64, order='F', copy=copy_X)
    n_samples, n_features = X.shape

    if alphas is None:
        alphas = _net_alpha_grid(X, l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    n_iters = []

    rng = params.get('random_state', np.random.RandomState())
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    Us = np.empty((k, n_features, n_alphas), dtype=np.float64)
    Ws = np.empty((k, k, n_alphas), dtype=np.float64)

    if U_init is None:
        U_ = 1e-1 * (np.random.rand(k, n_features) - 0.5)
        U_ = np.asfortranarray(U_)
    else:
        U_ = np.asfortranarray(U_init)

    if W_init is None:
        W_ = 1e-1 * (np.random.rand(k, k) - 0.5)
        W_ = np.asfortranarray(W_)
    else:
        W_ = np.asfortranarray(W_init)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        model = cd_fast.factorized_poisson_descent(U_, W_, l1_reg, l2_reg,
                                                    X, max_iter, tol, rng, random, verbose)
        U_, W_, n_iter_ = model
        Us[..., i] = U_
        Ws[..., i] = W_
        n_iters.append(n_iter_)
        if verbose: print('Path: %03i out of %03i' % (i, n_alphas))

    if return_n_iter:
        return alphas, Us, Ws, n_iters
    else:
        return alphas, Us, Ws


def enet_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              precompute=False, Xy=None, fit_intercept=True,
              normalize=False, copy_X=True, coef_init=None,
              verbose=False, return_n_iter=False, positive=False, **params):
    """Compute elastic net path with coordinate descent

    The elastic net optimization function is::

        1 / (2 * n_samples) * ||y - Xw||^2_2 +
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples,)
        Target values

    l1_ratio : float, optional
        float between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso

    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically

    precompute : True | False | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    fit_intercept : bool
        Fit or not an intercept.

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.

    verbose : boolean
        An indicator of verbosity.

    params : kwargs
        keyword arguments passed to the coordinate descent solver.

    return_n_iter : bool
        whether to return the number of iterations or not.

    positive : bool, default False
        If set to True, forces coefficients to be positive.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
    """
    if fit_intercept is None:
        fit_intercept = True

    X = np.array(X, dtype=np.float64, order='F', copy=copy_X and fit_intercept)

    n_samples, n_features = X.shape

    X, y, X_mean, y_mean, X_std, precompute, Xy = _pre_fit(X, y, Xy, precompute, normalize, fit_intercept, copy=False)
    if alphas is None:
        # No need to normalize of fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=l1_ratio,
                             fit_intercept=False, eps=eps, n_alphas=n_alphas,
                             normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = params.get('random_state', np.random.RandomState())
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    coefs = np.empty((n_features, n_alphas), dtype=np.float64)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1]))
    else:
        coef_ = np.asfortranarray(coef_init)

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        if isinstance(precompute, np.ndarray):
            model = cd_fast.enet_coordinate_descent_gram(
                coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter, tol, rng, random, positive)
        elif precompute is False:
            model = cd_fast.enet_coordinate_descent(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive)
        else:
            raise ValueError("Precompute should be one of True, False, or array-like")
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)
        if dual_gap_ > eps_:
            warnings.warn('Objective did not converge')
        if verbose: print('Path: %03i out of %03i' % (i, n_alphas))

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    else:
        return alphas, coefs, dual_gaps