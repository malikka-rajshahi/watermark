import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

def reduce_dimensionality(embeddings, n_components=100):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings.astype(np.float32)

def norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

def generate_gaussian_samples(token_embeddings, generator, num_samples=1000, beta=1.0, pca_components=10):
    #print('generating samples')
    reduced_embeddings = reduce_dimensionality(token_embeddings, n_components=pca_components)
    
    reduced_embeddings = torch.from_numpy(reduced_embeddings).to(device='cuda:0', dtype=torch.float32)
    distances = F.pdist(reduced_embeddings, p=2)
    
    #print('matrix stuff')
    cov_matrix = torch.exp(-beta * distances).detach().cpu().numpy().astype(np.float32)
    cov_matrix = squareform(cov_matrix)
    # why?
    np.fill_diagonal(cov_matrix, 1.0)
    cov_matrix = torch.from_numpy(cov_matrix).to(device='cuda:0', dtype=torch.float32)

    #print('cholesky')
    L = torch.linalg.cholesky(cov_matrix + 1e-6 * torch.eye(cov_matrix.shape[0], dtype=torch.float32, device='cuda:0'))
    #print('it happened!')

    # Step 1: Generate random samples in PyTorch using the provided generator
    torch_samples = torch.randn((L.shape[1], num_samples), generator=generator, dtype=torch.float32)
    torch_samples = torch_samples.to(device="cuda:0")
    
    # Step 3: Perform matrix multiplication
    gaussian_samples = torch.matmul(L, torch_samples)
    # Apply the CDF using SciPy in NumPy
    #cdf_samples = norm_cdf(gaussian_samples)
    
    #gaussian_samples = np.dot(L, np.random.randn(cov_matrix.shape[0], num_samples).astype(np.float32))
    #L = torch.from_numpy(L).float()  # Convert L to a PyTorch tensor
    # Generate random samples from a standard normal distribution using the provided generator
    #torch_samples = torch.randn((L.shape[1], num_samples), generator=generator, dtype=torch.float32)
    # Perform matrix multiplication using PyTorch
    #gaussian_samples = torch.matmul(L, torch_samples)

    #gaussian_samples = np.dot(L, np.random.randn(cov_matrix.shape[0], num_samples).astype(np.float32))
    
    cdf_samples = norm_cdf(gaussian_samples)
    
    #id_cov = torch.eye(token_embeddings.shape[0])
    #samples = torch.randn((id_cov.shape[1], num_samples), generator=generator, dtype=torch.float32)
    #torch_samples = samples.to(device="cuda:0")
    #cdf_samples = norm_cdf(torch_samples)

    return cdf_samples

def gumbel_mod_key_func(generator, n, vocab_size, token_embeddings, eff_vocab_size=None, beta=1):
    if eff_vocab_size is None:
        eff_vocab_size = vocab_size
    #print(eff_vocab_size)
    #print(f'key.py: {generator}')
    pi = torch.arange(eff_vocab_size)
    #print(pi)

    samples = generate_gaussian_samples(token_embeddings, generator, num_samples=n, beta=beta)
    
    xi = samples.T
    # Convert results back to PyTorch tensors
    #xi = torch.from_numpy(xi)
    #pi = torch.from_numpy(pi)
    #token_embeddings_torch = torch.from_numpy(token_embeddings)

    return xi.cpu(), pi.cpu()

def compute_empirical_covariance(xi):
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(xi, torch.Tensor):
        xi = xi.cpu().numpy()
    
    emp_cov_matrix = np.cov(xi, rowvar=False)
    
    # Convert back to PyTorch tensor
    return torch.from_numpy(emp_cov_matrix)

