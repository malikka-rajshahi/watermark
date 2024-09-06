import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky
from scipy.stats import norm
from sklearn.decomposition import PCA

def reduce_dimensionality(embeddings, n_components=100):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings.astype(np.float32)

def norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def generate_gaussian_samples(token_embeddings, generator, out_file, num_samples=1000, beta=1.0, pca_components=10):
    print('generating samples')
    file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
    file.write('generating samples\n')
    file.close()
    file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
    reduced_embeddings = reduce_dimensionality(token_embeddings, n_components=pca_components)
    
    distances = pdist(reduced_embeddings, metric='euclidean')
    
    print('matrix stuff')
    file.write('matrix stuff\n')
    file.close()
    file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
    cov_matrix = np.exp(-beta * distances).astype(np.float32)
    cov_matrix = squareform(cov_matrix)
    np.fill_diagonal(cov_matrix, 1.0)
    
    print('cholesky')
    file.write('cholesky\n')
    file.close()
    file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
    L = cholesky(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0], dtype=np.float32), lower=True)
    print('it happened!')
    file.write('it happened!\n')
    file.close()

    # Step 1: Generate random samples in PyTorch using the provided generator
    torch_samples = torch.randn((L.shape[1], num_samples), generator=generator, dtype=torch.float32)
    # Step 2: Convert the random samples to a NumPy array
    np_samples = torch_samples.numpy()
    # Step 3: Perform matrix multiplication in NumPy
    gaussian_samples_np = np.dot(L, np_samples)
    # Apply the CDF using SciPy in NumPy
    cdf_samples_np = norm.cdf(gaussian_samples_np)
    # Step 4: Convert the final result back to a PyTorch tensor
    cdf_samples = torch.from_numpy(cdf_samples_np)

    #gaussian_samples = np.dot(L, np.random.randn(cov_matrix.shape[0], num_samples).astype(np.float32))
    #L = torch.from_numpy(L).float()  # Convert L to a PyTorch tensor
    # Generate random samples from a standard normal distribution using the provided generator
    #torch_samples = torch.randn((L.shape[1], num_samples), generator=generator, dtype=torch.float32)
    # Perform matrix multiplication using PyTorch
    #gaussian_samples = torch.matmul(L, torch_samples)

    #gaussian_samples = np.dot(L, np.random.randn(cov_matrix.shape[0], num_samples).astype(np.float32))
    
    #cdf_samples = norm_cdf(gaussian_samples)
    
    return cdf_samples

def gumbel_mod_key_func(generator, n, vocab_size, token_embeddings, out_file, eff_vocab_size=None, beta=1):
    print('gumbel key function')
    file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
    file.write('gumbel key function\n')
    file.close()
    
    # Convert PyTorch tensor to NumPy array
    print(token_embeddings.shape)
    #token_embeddings_np = token_embeddings.cpu().numpy()

    if eff_vocab_size is None:
        eff_vocab_size = vocab_size

    pi = np.arange(eff_vocab_size)

    samples = generate_gaussian_samples(token_embeddings, generator, out_file, num_samples=n, beta=beta)
    
    xi = samples.T
    print('converting results')
    file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
    file.write('converting results\n')
    file.close()
    # Convert results back to PyTorch tensors
    #xi_torch = torch.from_numpy(xi)
    pi_torch = torch.from_numpy(pi)
    #token_embeddings_torch = torch.from_numpy(token_embeddings)

    return xi, pi_torch, token_embeddings, beta

def compute_empirical_covariance(xi):
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(xi, torch.Tensor):
        xi = xi.cpu().numpy()
    
    emp_cov_matrix = np.cov(xi, rowvar=False)
    
    # Convert back to PyTorch tensor
    return torch.from_numpy(emp_cov_matrix)

