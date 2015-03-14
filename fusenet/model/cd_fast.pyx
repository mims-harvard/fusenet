from libc.math cimport fabs, sqrt
cimport numpy as np
import numpy as np
import numpy.linalg as linalg
from collections import defaultdict
from sklearn import metrics
from itertools import product

cimport cython
from cpython cimport bool
import warnings

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t

np.import_array()

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef double abs_max(int n, double* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef double max(int n, double* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef double diff_abs_max(int n, double* a, double* b) nogil:
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef double m = fabs(a[0] - b[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m


cdef extern from "cblas.h":
    enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        AtlasConj=114

    void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
                             double *Y, int incY) nogil
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY
                             ) nogil
    double dasum "cblas_dasum"(int N, double *X, int incX) nogil
    void dger "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                double *X, int incX, double *Y, int incY, double *A, int lda) nogil
    void dgemv "cblas_dgemv"(CBLAS_ORDER Order,
                      CBLAS_TRANSPOSE TransA, int M, int N,
                      double alpha, double *A, int lda,
                      double *X, int incX, double beta,
                      double *Y, int incY) nogil
    double dnrm2 "cblas_dnrm2"(int N, double *X, int incX) nogil
    void dcopy "cblas_dcopy"(int N, double *X, int incX, double *Y, int incY) nogil
    void dscal "cblas_dscal"(int N, double alpha, double *X, int incX) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def factorized_poisson_descent(np.ndarray[DOUBLE, ndim=2] U, np.ndarray[DOUBLE, ndim=2] W,
                               double alpha, double beta, np.ndarray[DOUBLE, ndim=2] X,
                               int max_iter, double tol, object rng, bint random=0, bint verbose=0):
    """Cython version of the descent algorithm for factorized poisson log-linear regression

    We minimize

        sum_s sum_i X_s^i <U_\s W^T W u_s, X_\s^i> - exp(<U_\s W^T W u_s, X_\s^i>)
                + alpha norm(T, 1) + beta norm(T, 2)^2

    where

        T = U_\s^T W^T W u_s
    """
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]
    cdef unsigned int k = U.shape[0]

    cdef np.ndarray[DOUBLE, ndim=2] grad_u = np.mat(np.empty((k, 1)))
    cdef np.ndarray[DOUBLE, ndim=2] tmp_u = np.mat(np.empty((k, 1)))
    cdef np.ndarray[DOUBLE, ndim=2] tmp1_u = np.mat(np.empty((k, 1)))
    cdef np.ndarray[DOUBLE, ndim=2] tmp2_u = np.mat(np.empty((k, 1)))
    cdef np.ndarray[DOUBLE, ndim=2] tmp3_u = np.mat(np.empty((k, 1)))

    cdef np.ndarray[DOUBLE, ndim=2] grad_W = np.mat(np.empty((k, k)))
    cdef np.ndarray[DOUBLE, ndim=2] tmp_W = np.mat(np.empty((k, k)))
    cdef np.ndarray[DOUBLE, ndim=2] tmp1_W = np.mat(np.empty((k, k)))

    cdef double tmp
    cdef double tmp2
    cdef double d_u_max
    cdef double u_max
    cdef double d_u_ii
    cdef double d_u_tol = tol
    cdef unsigned int ii
    cdef unsigned int i
    cdef unsigned int n_out_iter
    cdef unsigned int f_iter
    cdef unsigned int n_outer_loop = max_iter
    cdef unsigned int soft_thr_iter = 2
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    W = np.mat(W)
    U = np.mat(U)
    X = np.mat(X)

    for n_out_iter in range(n_outer_loop):
        w_max = 0.0
        d_w_max = 0.0
        total_loss = 0.0

        for ii in range(n_features):
            idxs = list(range(n_features))
            idxs.remove(ii)
            theta = np.dot(U[:, ii].T, np.dot(W.T, np.dot(W, U[:, idxs])))
            yhat = np.exp(np.dot(X[:, idxs], theta.T))
            loss = np.sum(np.abs(-X[:, ii] + yhat))
            if verbose: print 'Part of Theta:', np.array(theta)[:5, :5]
            total_loss += loss
        if verbose: print 'Total Poisson loss (iter: %d): %5.4f' % (n_out_iter, total_loss)

        for f_iter in range(n_features):
            if random:
                ii = rand_int(n_features, rand_r_state)
            else:
                ii = f_iter

            u_ii = U[:, ii]

            if verbose: print 'Derivative of l_ii with respect to U_ii'
            grad_u_ii = np.mat(np.zeros((k, 1)))
            for i in range(n_samples):
                tmp_u = np.mat(np.zeros((k, 1)))
                for j in range(n_features):
                    if j == ii:
                        continue
                    tmp_u += X[i, j] * np.dot(W.T, np.dot(W, U[:, j]))
                tmp1_u = X[i, ii] * tmp_u

                tmp = 0
                for j in range(n_features):
                    if j == ii:
                        continue
                    tmp += X[i, j] * np.dot(U[:, j].T, np.dot(W.T, np.dot(W, U[:, ii])))
                tmp = np.exp(tmp)

                grad_u_ii += 1. / n_samples * (- tmp1_u + tmp * tmp_u)

            if verbose: print 'Derivative of l_t with respect tu U_ii'
            grad_u_t = np.mat(np.zeros((k, 1)))
            for t in range(n_features):
                if t == ii:
                    continue
                for i in range(n_samples):
                    tmp2_u = X[i, t] * X[i, ii] * np.dot(W.T, np.dot(W, U[:, t]))

                    tmp = 0
                    for j in range(n_features):
                        if j == t:
                            continue
                        tmp += X[i, j] * np.dot(U[:, j].T, np.dot(W.T, np.dot(W, U[:, t])))
                    tmp = np.exp(tmp)

                    tmp3_u = X[i, ii] * np.dot(W.T, np.dot(W, U[:, t]))

                    grad_u_t += 1. / n_samples * (- tmp2_u + tmp * tmp3_u)

            # Derivative of l with respect to U_ii is the sum of derivative of l_ii with respect
            # to U_ii and the sum of derivative of L_t with respect to U_ii (for all t)
            grad_u = grad_u_ii + grad_u_t
            U[:, ii] -= 1e-2 / n_samples * grad_u

            # Update the maximum absolute coefficient update
            d_u_ii = np.linalg.norm(U[:, ii] - u_ii)
            if d_u_ii > d_u_max:
                d_u_max = d_u_ii

            tmp = np.linalg.norm(U[:, ii])
            if tmp > u_max:
                u_max = tmp

        if verbose: print 'Derivative with respect to interaction matrix'
        grad_W = np.mat(np.zeros((k, k)))
        for ii in range(n_features):
            tmp_W = np.mat(np.zeros((k, k)))
            for i in range(n_samples):
                tmp1_W = np.mat(np.zeros((k, k)))
                for j in range(n_features):
                    if j == ii:
                        continue
                    tmp1_W += np.dot(W, X[i, j] * np.dot(U[:, j], U[:, ii].T) + X[i, j] * np.dot(U[:, ii], U[:, j].T))

                tmp_W -= X[i, ii] * tmp1_W

                tmp2 = 0.
                for j in range(n_features):
                    if j == ii:
                        continue
                    tmp2 += X[i, j] * np.dot(U[:, j].T, np.dot(W.T, np.dot(W, U[:, ii])))
                tmp2 = np.exp(tmp2)

                tmp_W += tmp2 * tmp1_W

            grad_W += 1. / n_samples * tmp_W
        W -= 1e-2 / n_samples * grad_W

        if verbose: print 'Soft thresholding of latent vectors in U'
        if n_out_iter % soft_thr_iter == 0 and n_out_iter > 1:
            for ii in range(n_features):
                for i in range(k):
                    U[i, ii] = fsign(U[i, ii]) * fmax(fabs(U[i, ii]) - alpha, 0) / (1. + beta)

        if verbose: print 'Soft thresholding of interaction matrix W'
        if n_out_iter % soft_thr_iter == 0 and n_out_iter > 1:
            for i in range(k):
                for j in range(k):
                    W[i, j] = fsign(W[i, j]) * fmax(fabs(W[i, j]) - alpha, 0) / (1. + beta)

        if u_max == 0.0 or d_u_max / u_max < d_u_tol or n_out_iter == max_iter - 1:
            break

    return U, W, n_out_iter + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def factorized_multinomial_descent(np.ndarray[DOUBLE, ndim=2] bias,
                                   np.ndarray[DOUBLE, ndim=2] U, np.ndarray[DOUBLE, ndim=2] W,
                                   np.ndarray[DOUBLE, ndim=2] V, np.ndarray[np.int32_t, ndim=1] m, double alpha, double beta,
                                   np.ndarray[DOUBLE, ndim=2] X, int max_iter, double tol,
                                   object rng, bint verbose=0):
    """Cython version of the descent algorithm for multinomial log-linear regression,
    a variant of multi-class logistic regression program.
    """
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]
    # assume that values in m lie on the interval 0 ... (m-1)
    cdef unsigned int n_values = len(m)

    cdef double [:, :] bias_theta = bias
    cdef double [:, :, :, :] Theta = np.empty((n_features, n_features, n_values, n_values))

    cdef double d_theta_tol = tol
    cdef unsigned int n_out_iter
    cdef unsigned int f_iter
    cdef unsigned int i, r, t, j, k
    cdef unsigned int soft_thr_iter = 2
    cdef unsigned int n_outer_loop = max_iter

    Xm = np.zeros((n_values, n_samples, n_features))
    for k in m:
        Xm[k, ...] = np.array(X == k, dtype=np.int)

    for n_out_iter in range(n_outer_loop):
        WtW = np.dot(W.T, W)
        for r, t, j, k in product(range(n_features), range(n_features), range(n_values), range(n_values)):
            Theta[r, t, j, k] = np.dot(U[:, r].T * V[r, j], np.dot(WtW, U[:, t] * V[t, k]))

        if verbose: print 'Multinomial gradient descent (iter: %d)' % n_out_iter

        if verbose: print 'Derivative of theta_r;j'
        for i, r, j in product(range(n_samples), range(n_features), range(n_values)):
            tmp = np.sum(np.multiply(Theta[r, :, j, :], Xm[:, i, :].T)) - \
                  np.sum(np.multiply(Theta[r, r, j, :], Xm[:, i, r].T))
            enum = np.exp(bias_theta[r, j] + tmp)

            denom = 0
            for l in m:
                tmp = np.sum(np.multiply(Theta[r, :, l, :], Xm[:, i, :].T)) - \
                      np.sum(np.multiply(Theta[r, r, l, :], Xm[:, i, r].T))
                tmp = np.exp(bias_theta[r, l] + tmp)
                denom += tmp

            grad = - 1. + enum / (denom + 1.)
            bias_theta[r, j] -= 1e-1 / n_samples * grad

            if n_out_iter % soft_thr_iter == 0 and n_out_iter > 1:
                bias_theta[r, j] = fsign(bias_theta[r, j]) * fmax(fabs(bias_theta[r, j]) - alpha, 0) / (1. + beta)

        if verbose: print 'Precomputing parts of derivative of theta_rt;jk'
        precomputed1 = {}
        for i, r, j in product(range(n_samples), range(n_features), m):
            tmp = np.sum(np.multiply(Theta[r, :, j, :], Xm[:, i, :].T)) - \
                  np.sum(np.multiply(Theta[r, r, j, :], Xm[:, i, r].T))
            precomputed1[i, r, j] = np.exp(bias_theta[r, j] + tmp)

        precomputed2 = {}
        for i, r in product(range(n_samples), range(n_features)):
            tmp1 = 0
            for l in m:
                tmp = np.sum(np.multiply(Theta[r, :, l, :], Xm[:, i, :].T)) - \
                      np.sum(np.multiply(Theta[r, r, l, :], Xm[:, i, r].T))
                tmp = np.exp(bias_theta[r, l] + tmp)
                tmp1 += tmp
            precomputed2[i, r] = tmp1

        precomputed3 = {}
        for i, t in product(range(n_samples), range(n_features)):
            tmp1 = 0
            for l in m:
                tmp = np.sum(np.multiply(Theta[t, :, l, :], Xm[:, i, :].T)) - \
                      np.sum(np.multiply(Theta[t, t, l, :], Xm[:, i, t].T))
                tmp1 += np.exp(bias_theta[t, l] + tmp)
            precomputed3[i, t] = tmp1

        precomputed_grad_part = defaultdict(int)
        for i, r, t, j, k in product(range(n_samples), range(n_features), range(n_features), m, m):
            tmp1 = Xm[k, i, t] + Xm[j, i, r]

            enum2 = precomputed1[i, r, j]
            denom2 = precomputed2[i, r]
            tmp2 = enum2 * Xm[k, i, t] / (denom2 + 1.)

            enum3 = precomputed3[i, t] * Xm[j, i, r]
            denom3 = precomputed3[i, t] + 1.
            tmp3 = enum3 / (denom3 + 1.)

            precomputed_grad_part[r, t, j, k] += - tmp1 + tmp2 + tmp3

        if verbose: print 'Derivative of U_r'
        for r, t, j, k in product(range(n_features), range(n_features), m, m):
            grad = precomputed_grad_part[r, t, j, k] * np.dot(V[r, j] * WtW, V[t, k] * U[:, t])
            U[:, r] -= 1e-1 / n_samples * grad

        if verbose: print 'Derivative of U_t'
        for r, t, j, k in product(range(n_features), range(n_features), m, m):
            grad = precomputed_grad_part[r, t, j, k] * np.dot(V[t, k] * WtW, V[r, j] * U[:, r])
            U[:, t] -= 1e-1 / n_samples * grad

        if verbose: print 'Derivative of V_rj'
        for r, t, j, k in product(range(n_features), range(n_features), m, m):
            grad = precomputed_grad_part[r, t, j, k] * np.dot(U[:, r].T, np.dot(WtW, V[t, k] * U[:, t]))
            V[r, j] -= 1e-1 / n_samples * grad

        if verbose: print 'Derivative of V_tk'
        for r, t, j, k in product(range(n_features), range(n_features), m, m):
            grad = precomputed_grad_part[r, t, j, k] * np.dot(U[:, r].T * V[r, j], np.dot(WtW, V[t, k] * U[:, t]))
            V[t, k] -= 1e-1 / n_samples * grad

        if verbose: print 'Derivative of W'
        for r, t, j, k in product(range(n_features), range(n_features), m, m):
            tmpW1 = np.dot(V[r, j] * U[:, r], V[t, k] * U[:, t].T)
            tmpW2 = np.dot(V[t, k] * U[:, t], V[r, j] * U[:, r].T)
            tmpW = np.dot(W, tmpW1 + tmpW2)
            grad = precomputed_grad_part[r, t, j, k] * tmpW
            W -= 1e-1 / n_samples * grad

        if n_out_iter % soft_thr_iter == 0 and n_out_iter > 1:
            if verbose: print 'Soft thresholding of U'
            for i, j in product(range(U.shape[0]), range(U.shape[1])):
                U[i, j] = fsign(U[i, j]) * fmax(fabs(U[i, j]) - alpha, 0) / (1. + beta)
            if verbose: print 'Soft thresholding of V'
            for i, j in product(range(V.shape[0]), range(V.shape[1])):
                V[i, j] = fsign(V[i, j]) * fmax(fabs(V[i, j]) - alpha, 0) / (1. + beta)
            if verbose: print 'Soft thresholding of W'
            for i, j in product(range(W.shape[0]), range(W.shape[1])):
                W[i, j] = fsign(W[i, j]) * fmax(fabs(W[i, j]) - alpha, 0) / (1. + beta)

        print 'U', np.array(U[:2, :2])
        print 'W', np.array(W[:5, :5])
        print 'V', np.array(V[:5, :5])

        if verbose: print 'Log-likelihood computation'
        WtW = np.dot(W.T, W)
        for r, t, j, k in product(range(n_features), range(n_features), range(n_values), range(n_values)):
            Theta[r, t, j, k] = np.dot(U[:, r].T * V[r, j], np.dot(WtW, U[:, t] * V[t, k]))
        lkl = 0.
        for i, r in product(range(n_samples), range(n_features)):
            def log_enum(l):
                tmp = np.sum(np.multiply(Theta[r, :, l, :], Xm[:, i, :].T)) \
                      - np.sum(np.multiply(Theta[r, r, l, :], Xm[:, i, r].T))
                return bias_theta[r, l] + tmp

            enum = log_enum(int(X[i, r]))
            denom = np.sum([np.exp(log_enum(l)) for l in m])
            tmp = enum - np.log(denom + 1.)
            lkl += 1. / n_samples * tmp
        if verbose: print 'Multiclass logistic log-likelihood: %5.4f' % lkl

    return bias_theta, U, V, W, n_out_iter + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def multinomial_descent(np.ndarray[DOUBLE, ndim=2] np_theta, np.ndarray[DOUBLE, ndim=4] np_Theta,
                        np.ndarray[np.int32_t, ndim=1] m, double alpha, double beta,
                        np.ndarray[DOUBLE, ndim=2] X, int max_iter, double tol,
                        object rng, bint verbose=0):
    """Cython version of the descent algorithm for multinomial log-linear regression,
    a variant of multi-class logistic regression program.

    We minimize the program defined in Eq. 5 and Eq. 6 in Jalali et al., AISTATS, 2011.
    """
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]
    # assume that values in m lie on the interval 0 ... (m-1)
    cdef unsigned int n_values = len(m)

    cdef double [:, :] theta = np_theta
    cdef double [:, :, :, :] Theta = np_Theta

    cdef double d_theta_tol = tol
    cdef unsigned int n_out_iter
    cdef unsigned int f_iter
    cdef unsigned int i, r, t, j, k
    cdef unsigned int soft_thr_iter = 2
    cdef unsigned int n_outer_loop = max_iter

    Xm = np.zeros((n_values, n_samples, n_features))
    for k in m:
        Xm[k, ...] = np.array(X == k, dtype=np.int)

    for n_out_iter in range(n_outer_loop):
        if verbose: print 'Multinomial gradient descent (iter: %d)' % n_out_iter

        if verbose: print 'Derivative of theta_r;j'
        for i, r, j in product(range(n_samples), range(n_features), range(n_values)):
            tmp = np.sum(np.multiply(Theta[r, :, j, :], Xm[:, i, :].T)) - \
                  np.sum(np.multiply(Theta[r, r, j, :], Xm[:, i, r].T))
            enum = np.exp(theta[r, j] + tmp)

            denom = 0
            for l in m:
                tmp = np.sum(np.multiply(Theta[r, :, l, :], Xm[:, i, :].T)) - \
                      np.sum(np.multiply(Theta[r, r, l, :], Xm[:, i, r].T))
                tmp = np.exp(theta[r, l] + tmp)
                denom += tmp

            grad = - 1. + enum / (denom + 1.)
            theta[r, j] -= 1e-1 / n_samples * grad

            if n_out_iter % soft_thr_iter == 0 and n_out_iter > 1:
                theta[r, j] = fsign(theta[r, j]) * fmax(fabs(theta[r, j]) - alpha, 0) / (1. + beta)

        if verbose: print 'Precomputing parts of derivative of theta_rt;jk'
        precomputed1 = {}
        for i, r, j in product(range(n_samples), range(n_features), m):
            tmp = np.sum(np.multiply(Theta[r, :, j, :], Xm[:, i, :].T)) - \
                  np.sum(np.multiply(Theta[r, r, j, :], Xm[:, i, r].T))
            precomputed1[i, r, j] = np.exp(theta[r, j] + tmp)

        precomputed2 = {}
        for i, r in product(range(n_samples), range(n_features)):
            tmp1 = 0
            for l in m:
                tmp = np.sum(np.multiply(Theta[r, :, l, :], Xm[:, i, :].T)) - \
                      np.sum(np.multiply(Theta[r, r, l, :], Xm[:, i, r].T))
                tmp = np.exp(theta[r, l] + tmp)
                tmp1 += tmp
            precomputed2[i, r] = tmp1

        precomputed3 = {}
        for i, t in product(range(n_samples), range(n_features)):
            tmp1 = 0
            for l in m:
                tmp = np.sum(np.multiply(Theta[t, :, l, :], Xm[:, i, :].T)) - \
                      np.sum(np.multiply(Theta[t, t, l, :], Xm[:, i, t].T))
                tmp1 += np.exp(theta[t, l] + tmp)
            precomputed3[i, t] = tmp1

        if verbose: print 'Derivative of theta_rt;jk'
        for i, r, t, j, k in product(range(n_samples), range(n_features), range(n_features), m, m):
            tmp1 = Xm[k, i, t] + Xm[j, i, r]

            enum2 = precomputed1[i, r, j]
            denom2 = precomputed2[i, r]
            tmp2 = enum2 * Xm[k, i, t] / (denom2 + 1.)

            enum3 = precomputed3[i, t] * Xm[j, i, r]
            denom3 = precomputed3[i, t] + 1.
            tmp3 = enum3 / (denom3 + 1.)

            grad = - tmp1 + tmp2 + tmp3
            Theta[r, t, j, k] -= 1e-1 / n_samples * grad

            if n_out_iter % soft_thr_iter == 0 and n_out_iter > 1:
                # L1/L2: collate the L2 norms of the groups and compute their overall L1 norm
                Theta[r, t, j, k] /= (1. + beta)
                if j == n_values - 1 and k == n_values - 1:
                    gamma_jk = np.linalg.norm(Theta[r, t, :, :], 2)
                    for j1, k1 in product(m, m):
                        Theta[r, t, j1, k1] = fsign(Theta[r, t, j1, k1]) * \
                                              fmax(fabs(Theta[r, t, j1, k1]) - alpha * gamma_jk, 0)

        print 'Theta', np.array(Theta[:2, :2])
        print 'theta', np.array(theta[:5, :5])

        if verbose: print 'Log-likelihood computation'
        lkl = 0.
        for i, r in product(range(n_samples), range(n_features)):
            def log_enum(l):
                tmp = np.sum(np.multiply(Theta[r, :, l, :], Xm[:, i, :].T)) \
                      - np.sum(np.multiply(Theta[r, r, l, :], Xm[:, i, r].T))
                return theta[r, l] + tmp

            enum = log_enum(int(X[i, r]))
            denom = np.sum([np.exp(log_enum(l)) for l in m])
            tmp = enum - np.log(denom + 1.)
            lkl += 1. / n_samples * tmp
        if verbose: print 'Multiclass logistic log-likelihood: %5.4f' % lkl

        if verbose: print 'Model evaluation'
        X_pred = np.zeros((n_samples, n_features, n_values))
        X_max = np.zeros((n_samples, n_features))
        for i, r in product(range(n_samples), range(n_features)):
            def exp_enum(l):
                tmp = np.sum(np.multiply(Theta[r, :, l, :], Xm[:, i, :].T)) - \
                      np.sum(np.multiply(Theta[r, r, l, :], Xm[:, i, r].T))
                return np.exp(theta[r, l] + tmp)

            enums = np.array([exp_enum(mm) for mm in m])
            X_pred[i, r, :] = enums / (np.sum(enums) + 1.)
            X_max[i, r] = np.max(X_pred[i, r, :])
        aucs = []
        for r, mm in product(range(n_features), m):
            y_true = X[:, r] == mm
            y_pred = X_pred[:, r, mm]
            y_pred = np.array([y_pred[i] / X_max[i, r] for i in range(n_samples)])
            auc = metrics.roc_auc_score(y_true, y_pred)
            aucs.append(auc)
        print 'Mean AUC across all columns and classes: %5.4f' % np.mean(aucs)

    return theta, Theta, n_out_iter + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def poisson_coordinate_descent(np.ndarray[DOUBLE, ndim=1] w, double alpha, double beta,
                               np.ndarray[DOUBLE, ndim=2] X, np.ndarray[DOUBLE, ndim=1] y,
                               int max_iter, double tol, object rng, bint random=0, bint verbose=0):
    """Cython version of the coordinate descent algorithm
    for poisson log-linear regression

    We minimize

        y(X w) - exp(X w) + alpha norm(w, 1) + beta norm(w, 2)^2
    """
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    cdef double w_ii
    cdef double d_w_max
    cdef double w_max
    cdef double d_w_ii
    cdef double d_w_tol = tol
    cdef unsigned int ii
    cdef unsigned int i
    cdef unsigned int n_out_iter
    cdef unsigned int n_in_iter
    cdef unsigned int f_iter
    cdef unsigned int n_outer_loop = max_iter
    cdef unsigned int n_inner_loop = 100
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    for n_out_iter in range(n_outer_loop):
        yhat = np.exp(np.dot(X, w))
        if verbose: print 'Poisson loss (iter %d): %f' % (n_out_iter, np.sum(np.abs(-y+yhat)))
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_features):  # Loop over coordinates
            if random:
                ii = rand_int(n_features, rand_r_state)
            else:
                ii = f_iter

            w_ii = w[ii]

            for n_in_iter in xrange(n_inner_loop):
                grad = np.sum([X[i, ii] * (-y[i] + np.exp(np.dot(X[i,:], w))) for i in range(n_samples)])
                w[ii] -= 1e-3 / (n_samples * (n_out_iter + 1)) * grad

            # see Friedman et al., 2010 paper, Eq. 6, soft thresholding
            w[ii] = fsign(w[ii]) * fmax(fabs(w[ii]) - alpha, 0)/(1. + beta)

            # update the maximum absolute coefficient update
            d_w_ii = fabs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if fabs(w[ii]) > w_max:
                w_max = fabs(w[ii])

        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_out_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller
            # than the tolerance
            break

    return w, n_out_iter + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def enet_coordinate_descent(np.ndarray[DOUBLE, ndim=1] w, double alpha, double beta,
                            np.ndarray[DOUBLE, ndim=2] X, np.ndarray[DOUBLE, ndim=1] y,
                            int max_iter, double tol, object rng, bint random=0, bint positive=0):
    """Cython version of the coordinate descent algorithm for Elastic-Net regression

    We minimize

        norm(y - X w, 2)^2 + alpha norm(w, 1) + beta norm(w, 2)^2
    """
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    # compute norms of the columns of X
    cdef np.ndarray[DOUBLE, ndim=1] norm_cols_X = (X**2).sum(axis=0)

    # initial value of the residuals
    cdef np.ndarray[DOUBLE, ndim=1] R = np.empty(n_samples)

    cdef np.ndarray[DOUBLE, ndim=1] XtA = np.empty(n_features)
    cdef double tmp
    cdef double w_ii
    cdef double d_w_max
    cdef double w_max
    cdef double d_w_ii
    cdef double gap = tol + 1.0
    cdef double d_w_tol = tol
    cdef double dual_norm_XtA
    cdef double R_norm2
    cdef double w_norm2
    cdef double l1_norm
    cdef unsigned int ii
    cdef unsigned int i
    cdef unsigned int n_iter
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    with nogil:

        # R = y - np.dot(X, w)
        for i in range(n_samples):
            R[i] = y[i] - ddot(n_features,
                               <DOUBLE*>(X.data + i * sizeof(DOUBLE)),
                               n_samples, <DOUBLE*>w.data, 1)

        # tol *= np.dot(y, y)
        tol *= ddot(n_samples, <DOUBLE*>y.data, 1,
                    <DOUBLE*>y.data, 1)

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if norm_cols_X[ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # R += w_ii * X[:,ii]
                    daxpy(n_samples, w_ii,
                          <DOUBLE*>(X.data + ii * n_samples * sizeof(DOUBLE)), 1, <DOUBLE*>R.data, 1)

                # tmp = (X[:,ii]*R).sum()
                # tmp is the usual least squares estimate
                tmp = ddot(n_samples,
                           <DOUBLE*>(X.data + ii * n_samples * sizeof(DOUBLE)), 1, <DOUBLE*>R.data, 1)

                if positive and tmp < 0:
                    w[ii] = 0.0
                else:
                    w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0)
                             / (norm_cols_X[ii] + beta))

                if w[ii] != 0.0:
                    # R -=  w[ii] * X[:,ii] # Update residual
                    daxpy(n_samples, -w[ii],
                          <DOUBLE*>(X.data + ii * n_samples * sizeof(DOUBLE)),
                          1, <DOUBLE*>R.data, 1)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            if (w_max == 0.0
                    or d_w_max / w_max < d_w_tol
                    or n_iter == max_iter - 1):
                # the biggest coordinate update of this iteration was smaller
                # than the tolerance: check the duality gap as ultimate
                # stopping criterion

                # XtA = np.dot(X.T, R) - beta * w
                for i in range(n_features):
                    XtA[i] = ddot(
                        n_samples,
                        <DOUBLE*>(X.data + i * n_samples *sizeof(DOUBLE)),
                        1, <DOUBLE*>R.data, 1) - beta * w[i]

                if positive:
                    dual_norm_XtA = max(n_features, <DOUBLE*>XtA.data)
                else:
                    dual_norm_XtA = abs_max(n_features, <DOUBLE*>XtA.data)

                # R_norm2 = np.dot(R, R)
                R_norm2 = ddot(n_samples, <DOUBLE*>R.data, 1, <DOUBLE*>R.data, 1)

                # w_norm2 = np.dot(w, w)
                w_norm2 = ddot(n_features, <DOUBLE*>w.data, 1, <DOUBLE*>w.data, 1)

                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * (const ** 2)
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                l1_norm = dasum(n_features, <DOUBLE*>w.data, 1)

                # np.dot(R.T, y)
                gap += (alpha * l1_norm - const * ddot(
                            n_samples,
                            <DOUBLE*>R.data, 1,
                            <DOUBLE*>y.data, 1)
                        + 0.5 * beta * (1 + const ** 2) * (w_norm2))

                if gap < tol:
                    # return if we reached desired tolerance
                    break

    return w, gap, tol, n_iter + 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def enet_coordinate_descent_gram(double[:] w, double alpha, double beta,
                                 double[:, :] Q, double[:] q, double[:] y,
                                 int max_iter, double tol, object rng,
                                 bint random=0, bint positive=0):
    """Cython version of the coordinate descent algorithm for Elastic-Net regression

    We minimize

        w^T Q w - q^T w + alpha norm(w, 1) + beta norm(w, 2)^2

    which amount to the Elastic-Net problem when

        Q = X^T X (Gram matrix)
        q = X^T y
    """
    # get the data information into easy vars
    cdef unsigned int n_samples = y.shape[0]
    cdef unsigned int n_features = Q.shape[0]

    # initial value "Q w" which will be kept of up to date in the iterations
    cdef double[:] H = np.dot(Q, w)

    cdef double[:] XtA = np.zeros(n_features)
    cdef double tmp
    cdef double w_ii
    cdef double d_w_max
    cdef double w_max
    cdef double d_w_ii
    cdef double gap = tol + 1.0
    cdef double d_w_tol = tol
    cdef double dual_norm_XtA
    cdef unsigned int ii
    cdef unsigned int n_iter
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    cdef double y_norm2 = np.dot(y, y)
    cdef double* Q_ptr = &Q[0, 0]
    cdef double* H_ptr = &H[0]
    cdef double* XtA_ptr = &XtA[0]
    tol = tol * y_norm2

    with nogil:
        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if Q[ii, ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # H -= w_ii * Q[ii]
                    daxpy(n_features, -w_ii, Q_ptr + ii * n_features, 1, H_ptr, 1)

                tmp = q[ii] - H[ii]

                if positive and tmp < 0:
                    w[ii] = 0.0
                else:
                    w[ii] = fsign(tmp) * fmax(fabs(tmp) - alpha, 0) \
                        / (Q[ii, ii] + beta)

                if w[ii] != 0.0:
                    # H +=  w[ii] * Q[ii] # Update H = X.T X w
                    daxpy(n_features, w[ii], Q_ptr + ii * n_features, 1, H_ptr, 1)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
                # the biggest coordinate update of this iteration was smaller than
                # the tolerance: check the duality gap as ultimate stopping
                # criterion

                # q_dot_w = np.dot(w, q)
                # Note that increment in q is not 1 because the strides
                # vary if q is sliced from a 2-D array.
                q_dot_w = ddot(n_features, &w[0], 1, &q[0], 1)

                for ii in range(n_features):
                    XtA[ii] = q[ii] - H[ii] - beta * w[ii]
                if positive:
                    dual_norm_XtA = max(n_features, XtA_ptr)
                else:
                    dual_norm_XtA = abs_max(n_features, XtA_ptr)

                # temp = np.sum(w * H)
                tmp = 0.0
                for ii in range(n_features):
                    tmp += w[ii] * H[ii]
                R_norm2 = y_norm2 + tmp - 2.0 * q_dot_w

                # w_norm2 = np.dot(w, w)
                w_norm2 = ddot(n_features, &w[0], 1, &w[0], 1)

                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * (const ** 2)
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                # The call to dasum is equivalent to the L1 norm of w
                gap += (alpha * dasum(n_features, &w[0], 1) -
                        const * y_norm2 +  const * q_dot_w +
                        0.5 * beta * (1 + const ** 2) * w_norm2)

                if gap < tol:
                    # return if we reached desired tolerance
                    break

    return np.asarray(w), gap, tol, n_iter + 1