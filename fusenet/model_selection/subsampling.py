import numpy as np

class Subsampling():
    """Data sub-sampler.

    Parameters
    ----------
    m : int
        Number of observations being sampled

    X : ndarray, optional
        Complete data set.

    B : int, optional
        The number of sub-samples (1 by default).

    verbose : boolean
        An indicator of verbosity.

    Attributes
    ----------
    m : int
        Number of observations being sampled

    X : ndarray, optional
        Complete data set.

    B : int, optional
        The number of sub-samples (1 by default).
    """
    def __init__(self, X=None, m=None, B=1, verbose=True):
        # recommended value for m is floor(10 * sqrt(X.shape[0]))
        self.m = m
        self.X = X
        self.B = B
        self.verbose = True

    def subsample(self, X=None, m=None, B=1):
        if X is None and self.X is None:
            raise ValueError('Data set is missing.')
        else:
            X = X if X is not None else self.X
        if m is None and self.m is None:
            m = int(np.floor(10 * np.sqrt(X.shape[0])))
        else:
            m = m if m is not None else self.m
        Ss = []
        if self.verbose:
            print 'Subsampling %d data sets (examples: %d)' % (B, m)
        for b in xrange(B):
            idxs = np.random.randint(0, X.shape[0], m)
            Ss.append(X[idxs, :].copy())
        return Ss
