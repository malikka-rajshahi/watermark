import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky
from scipy.stats import norm

def generate_gaussian_samples(token_embeddings, num_samples=1000, beta=1.0):
    print("Made it to Gaussian samples")
    print("Step 2: Calculate pairwise Euclidean distances l(i, j) between token embeddings")
    distances = squareform(pdist(token_embeddings, metric='euclidean'))

    print("Step 3: Compute the theoretical covariance matrix cov(i, j) = e^(-beta * l(i, j))")
    cov_matrix = np.exp(-beta * distances)

    print("Step 4: Cholesky decomposition of the covariance matrix")
    L = cholesky(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]), lower=True)

    print("Step 5: Generate multiple random Gaussian vectors")
    gaussian_samples = np.dot(L, np.random.randn(cov_matrix.shape[0], num_samples))

    print("Map Gaussian samples to [0, 1] using the CDF")
    cdf_samples = norm.cdf(gaussian_samples)

    return cdf_samples

def gumbel_mod_key_func(generator, n, vocab_size, token_embeddings, eff_vocab_size=None, beta=1):
    print("Made it to key func")
    if eff_vocab_size is None:
        eff_vocab_size = vocab_size

    pi = torch.arange(eff_vocab_size)

    # Generate Gaussian samples and map to [0, 1]
    samples = generate_gaussian_samples(token_embeddings, num_samples=n, beta=beta)
    
    # Convert the samples to a PyTorch tensor
    xi = torch.tensor(samples.T, dtype=torch.float32)  # Transpose to match (n, eff_vocab_size)

    return xi, pi, token_embeddings, beta
