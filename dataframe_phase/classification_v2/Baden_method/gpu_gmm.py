"""
GPU-accelerated Gaussian Mixture Model using PyTorch.

This module provides a GPU-accelerated GMM implementation compatible with sklearn's API.
Falls back to CPU if CUDA is not available.
"""

import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Check for PyTorch with CUDA
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.info("PyTorch not available, GPU GMM disabled")


class GaussianMixtureGPU:
    """
    GPU-accelerated Gaussian Mixture Model using PyTorch.
    
    Compatible with sklearn.mixture.GaussianMixture API (subset).
    Only supports diagonal covariance for efficiency.
    
    Note: Results may differ slightly from sklearn due to:
    - Different numerical precision (float32 vs float64 by default)
    - Different initialization (simplified k-means++ vs full k-means)
    - Different random number generators
    
    Set use_float64=True for closer match to sklearn (slower).
    """
    
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = 'diag',
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
        init_params: str = 'kmeans',
        use_float64: bool = False,
    ):
        """
        Initialize GPU GMM.
        
        Args:
            n_components: Number of mixture components.
            covariance_type: Only 'diag' supported for GPU.
            n_init: Number of initializations.
            max_iter: Maximum EM iterations.
            tol: Convergence threshold.
            reg_covar: Regularization for covariance.
            random_state: Random seed.
            init_params: Initialization method ('kmeans' or 'random').
        """
        if covariance_type != 'diag':
            raise ValueError("GPU GMM only supports 'diag' covariance type")
        
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_params = init_params
        self.use_float64 = use_float64
        
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.dtype = torch.float64 if use_float64 else torch.float32
        
        # Model parameters (set after fitting)
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf
        
    def _initialize_parameters(self, X: torch.Tensor, random_state: int):
        """Initialize GMM parameters."""
        n_samples, n_features = X.shape
        
        # Set random seed for both CPU and CUDA for reproducibility
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)  # For multi-GPU
        
        # Initialize weights uniformly
        weights = torch.ones(self.n_components, dtype=self.dtype, device=self.device) / self.n_components
        
        # Initialize means using k-means++ style or random
        if self.init_params == 'kmeans':
            # Simple k-means++ initialization
            indices = [torch.randint(n_samples, (1,), device=self.device).item()]
            for _ in range(1, self.n_components):
                # Compute distances to nearest center
                centers = X[indices]
                dists = torch.cdist(X, centers).min(dim=1)[0]
                probs = dists ** 2
                probs = probs / probs.sum()
                idx = torch.multinomial(probs, 1).item()
                indices.append(idx)
            means = X[indices].clone()
        else:
            # Random initialization
            indices = torch.randperm(n_samples, device=self.device)[:self.n_components]
            means = X[indices].clone()
        
        # Initialize covariances as variance of data
        covariances = torch.var(X, dim=0, keepdim=True).expand(self.n_components, -1).clone()
        covariances = covariances + self.reg_covar
        
        return weights, means, covariances
    
    def _e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """E-step: compute responsibilities."""
        n_samples = X.shape[0]
        
        # Compute log probabilities for each component
        log_probs = torch.zeros(n_samples, self.n_components, device=self.device)
        
        for k in range(self.n_components):
            diff = X - self.means_[k]
            # Log probability for diagonal covariance
            log_det = torch.sum(torch.log(self.covariances_[k]))
            mahal = torch.sum(diff ** 2 / self.covariances_[k], dim=1)
            log_probs[:, k] = -0.5 * (X.shape[1] * np.log(2 * np.pi) + log_det + mahal)
        
        # Add log weights
        log_probs = log_probs + torch.log(self.weights_)
        
        # Log-sum-exp for numerical stability
        log_prob_norm = torch.logsumexp(log_probs, dim=1)
        log_resp = log_probs - log_prob_norm.unsqueeze(1)
        
        return torch.exp(log_resp), log_prob_norm.mean().item()
    
    def _m_step(self, X: torch.Tensor, resp: torch.Tensor):
        """M-step: update parameters."""
        n_samples = X.shape[0]
        
        # Effective number of points per component
        nk = resp.sum(dim=0) + 1e-10
        
        # Update weights
        self.weights_ = nk / n_samples
        
        # Update means
        self.means_ = torch.mm(resp.t(), X) / nk.unsqueeze(1)
        
        # Update covariances (diagonal)
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (resp[:, k].unsqueeze(1) * diff ** 2).sum(dim=0) / nk[k]
            self.covariances_[k] = self.covariances_[k] + self.reg_covar
    
    def _fit_single(self, X: torch.Tensor, random_state: int) -> float:
        """Fit single initialization."""
        self.weights_, self.means_, self.covariances_ = self._initialize_parameters(X, random_state)
        
        lower_bound = -np.inf
        
        for iteration in range(self.max_iter):
            prev_lower_bound = lower_bound
            
            # E-step
            resp, lower_bound = self._e_step(X)
            
            # M-step
            self._m_step(X, resp)
            
            # Check convergence
            change = lower_bound - prev_lower_bound
            if abs(change) < self.tol:
                self.converged_ = True
                break
        
        self.n_iter_ = iteration + 1
        return lower_bound
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureGPU':
        """
        Fit the GMM model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            
        Returns:
            self
        """
        # Enable deterministic operations for reproducibility
        if self.random_state is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        
        best_lower_bound = -np.inf
        best_params = None
        
        base_seed = self.random_state if self.random_state is not None else 0
        
        for init in range(self.n_init):
            seed = base_seed + init
            lower_bound = self._fit_single(X_tensor, seed)
            
            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                best_params = (
                    self.weights_.clone(),
                    self.means_.clone(),
                    self.covariances_.clone(),
                    self.converged_,
                    self.n_iter_,
                )
        
        # Restore best parameters
        self.weights_, self.means_, self.covariances_, self.converged_, self.n_iter_ = best_params
        self.lower_bound_ = best_lower_bound
        
        # Convert to numpy for sklearn compatibility
        self.weights_ = self.weights_.cpu().numpy()
        self.means_ = self.means_.cpu().numpy()
        self.covariances_ = self.covariances_.cpu().numpy()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities."""
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        weights = torch.tensor(self.weights_, dtype=self.dtype, device=self.device)
        means = torch.tensor(self.means_, dtype=self.dtype, device=self.device)
        covariances = torch.tensor(self.covariances_, dtype=self.dtype, device=self.device)
        
        n_samples = X_tensor.shape[0]
        log_probs = torch.zeros(n_samples, self.n_components, device=self.device)
        
        for k in range(self.n_components):
            diff = X_tensor - means[k]
            log_det = torch.sum(torch.log(covariances[k]))
            mahal = torch.sum(diff ** 2 / covariances[k], dim=1)
            log_probs[:, k] = -0.5 * (X.shape[1] * np.log(2 * np.pi) + log_det + mahal)
        
        log_probs = log_probs + torch.log(weights)
        log_prob_norm = torch.logsumexp(log_probs, dim=1, keepdim=True)
        log_resp = log_probs - log_prob_norm
        
        return torch.exp(log_resp).cpu().numpy()
    
    def score(self, X: np.ndarray) -> float:
        """Compute mean log-likelihood."""
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        weights = torch.tensor(self.weights_, dtype=self.dtype, device=self.device)
        means = torch.tensor(self.means_, dtype=self.dtype, device=self.device)
        covariances = torch.tensor(self.covariances_, dtype=self.dtype, device=self.device)
        
        n_samples = X_tensor.shape[0]
        log_probs = torch.zeros(n_samples, self.n_components, device=self.device)
        
        for k in range(self.n_components):
            diff = X_tensor - means[k]
            log_det = torch.sum(torch.log(covariances[k]))
            mahal = torch.sum(diff ** 2 / covariances[k], dim=1)
            log_probs[:, k] = -0.5 * (X.shape[1] * np.log(2 * np.pi) + log_det + mahal)
        
        log_probs = log_probs + torch.log(weights)
        log_prob_norm = torch.logsumexp(log_probs, dim=1)
        
        return log_prob_norm.mean().item()
    
    def bic(self, X: np.ndarray) -> float:
        """Compute Bayesian Information Criterion."""
        n_samples, n_features = X.shape
        
        # Number of parameters for diagonal GMM
        n_params = (
            self.n_components * n_features +  # means
            self.n_components * n_features +  # diagonal covariances
            self.n_components - 1              # weights (minus 1 for constraint)
        )
        
        log_likelihood = self.score(X) * n_samples
        
        return -2 * log_likelihood + n_params * np.log(n_samples)
    
    def aic(self, X: np.ndarray) -> float:
        """Compute Akaike Information Criterion."""
        n_samples, n_features = X.shape
        
        n_params = (
            self.n_components * n_features +
            self.n_components * n_features +
            self.n_components - 1
        )
        
        log_likelihood = self.score(X) * n_samples
        
        return -2 * log_likelihood + 2 * n_params


def get_gmm_class(use_gpu: bool = True):
    """
    Get the appropriate GMM class based on GPU availability.
    
    Args:
        use_gpu: Whether to use GPU if available.
        
    Returns:
        GMM class (GaussianMixtureGPU or sklearn.mixture.GaussianMixture).
    """
    if use_gpu and CUDA_AVAILABLE:
        logger.info("Using GPU-accelerated GMM (PyTorch)")
        return GaussianMixtureGPU
    else:
        from sklearn.mixture import GaussianMixture
        logger.info("Using CPU GMM (sklearn)")
        return GaussianMixture

