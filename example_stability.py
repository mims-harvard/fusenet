from collections import defaultdict

import numpy as np
from scipy import sparse as sp

from fusenet.model_selection import stability
from fusenet.model_selection import subsampling


n_examples = 100
n_features = 20
n_samples = 20

# Create a random dataset `data`
sbs = subsampling.Subsampling()
data = np.random.rand(n_examples, n_features)
# Sample `n_samples` datasets based on `X`
data_samples = sbs.subsample(data, n_samples)

# Select optimal regularization via stability
stb = stability.Stability()
candidate_regularization = np.logspace(-5, 0, num=100, base=10)[::-1]

sample2models = defaultdict(list)
for i, X_sample in enumerate(data_samples):
    for candidate in candidate_regularization:
        # In real data, dependency matrix is inferred by a model
        # that sets regularization strength to `reg`
        dependency_matrix = sp.rand(n_features, n_features, density=0.9).todense()
        sample2models[i].append(dependency_matrix)

rho_opt = stb.optimal_reg(candidate_regularization, sample2models)
