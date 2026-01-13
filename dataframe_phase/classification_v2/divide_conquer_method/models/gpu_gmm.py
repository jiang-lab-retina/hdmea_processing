"""
PyTorch GPU-accelerated Gaussian Mixture Model.

Provides a drop-in replacement for sklearn's GaussianMixture that runs on GPU.
Uses diagonal covariance for efficiency (matching the pipeline's sklearn usage).
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class GaussianMixtureGPU:
    """
    GPU-accelerated Gaussian Mixture Model using PyTorch.
    
    Implements EM algorithm with diagonal covariance matrices.
    API is compatible with sklearn's GaussianMixture.
    
    Args:
        n_components: Number of mixture components.
        max_iter: Maximum EM iterations.
        n_init: Number of initializations (best is kept).
        reg_covar: Regularization added to covariance diagonal.
        tol: Convergence threshold for log-likelihood.
        random_state: Random seed for reproducibility.
        device: PyTorch device ('cuda' or 'cpu').
    """
    
    def __init__(
        self,
        n_components: int = 1,
        max_iter: int = 200,
        n_init: int = 10,
        reg_covar: float = 1e-6,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        device: str = 'cuda',
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.tol = tol
        self.random_state = random_state
        
        # Check CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for GMM")
            device = 'cpu'
        self.device = device
        
        # Model parameters (set after fitting)
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf
        
        # Internal tensors
        self._means: Optional[torch.Tensor] = None
        self._vars: Optional[torch.Tensor] = None  # Diagonal variances
        self._weights: Optional[torch.Tensor] = None
    
    def _initialize_parameters(self, X: torch.Tensor) -> None:
        """Initialize GMM parameters using k-means++ style initialization."""
        n_samples, n_features = X.shape
        k = self.n_components
        
        # Initialize means using k-means++ style selection
        indices = torch.zeros(k, dtype=torch.long, device=self.device)
        
        # First center: random sample
        indices[0] = torch.randint(n_samples, (1,), device=self.device)
        
        # Remaining centers: weighted by distance to nearest center
        for i in range(1, k):
            # Compute distances to nearest center
            dists = torch.cdist(X, X[indices[:i]])
            min_dists = dists.min(dim=1).values
            
            # Sample proportional to squared distance
            probs = min_dists ** 2
            probs = probs / probs.sum()
            
            # Sample next center
            idx = torch.multinomial(probs, 1)
            indices[i] = idx
        
        self._means = X[indices].clone()
        
        # Initialize variances as global variance
        global_var = X.var(dim=0) + self.reg_covar
        self._vars = global_var.unsqueeze(0).expand(k, -1).clone()
        
        # Initialize uniform weights
        self._weights = torch.ones(k, device=self.device) / k
    
    def _e_step(self, X: torch.Tensor) -> torch.Tensor:
        """
        E-step: Compute responsibilities (posterior probabilities).
        
        Returns:
            (n_samples, n_components) responsibility matrix.
        """
        n_samples = X.shape[0]
        k = self.n_components
        
        log_probs = torch.zeros(n_samples, k, device=self.device)
        
        for j in range(k):
            diff = X - self._means[j]
            var = self._vars[j]
            
            # Mahalanobis distance for diagonal covariance
            mahal = (diff ** 2 / var).sum(dim=1)
            
            # Log determinant of diagonal covariance
            log_det = var.log().sum()
            
            # Log probability
            n_features = X.shape[1]
            log_probs[:, j] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)
        
        # Add log weights
        log_probs = log_probs + self._weights.log()
        
        # Normalize to get responsibilities
        log_resp = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
        resp = log_resp.exp()
        
        return resp
    
    def _m_step(self, X: torch.Tensor, resp: torch.Tensor) -> None:
        """M-step: Update parameters given responsibilities."""
        n_samples = X.shape[0]
        
        # Effective number of points per component
        nk = resp.sum(dim=0) + 1e-10
        
        # Update weights
        self._weights = nk / n_samples
        
        # Update means
        self._means = (resp.T @ X) / nk.unsqueeze(1)
        
        # Update variances (diagonal)
        for j in range(self.n_components):
            diff = X - self._means[j]
            weighted_sq_diff = resp[:, j].unsqueeze(1) * (diff ** 2)
            self._vars[j] = weighted_sq_diff.sum(dim=0) / nk[j] + self.reg_covar
    
    def _compute_log_likelihood(self, X: torch.Tensor) -> float:
        """Compute log-likelihood of data under current model."""
        n_samples = X.shape[0]
        k = self.n_components
        
        log_probs = torch.zeros(n_samples, k, device=self.device)
        
        for j in range(k):
            diff = X - self._means[j]
            var = self._vars[j]
            mahal = (diff ** 2 / var).sum(dim=1)
            log_det = var.log().sum()
            n_features = X.shape[1]
            log_probs[:, j] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)
        
        log_probs = log_probs + self._weights.log()
        ll = torch.logsumexp(log_probs, dim=1).sum()
        
        return ll.item()
    
    def _fit_single(self, X: torch.Tensor) -> float:
        """Run single EM fitting with current initialization."""
        prev_ll = -np.inf
        
        for iteration in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            
            ll = self._compute_log_likelihood(X)
            
            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                return ll
            
            prev_ll = ll
        
        self.n_iter_ = self.max_iter
        return prev_ll
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureGPU':
        """
        Fit GMM to data using EM algorithm.
        
        Args:
            X: (n_samples, n_features) data array.
        
        Returns:
            self
        """
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        best_ll = -np.inf
        best_params = None
        
        for init in range(self.n_init):
            self._initialize_parameters(X_tensor)
            ll = self._fit_single(X_tensor)
            
            if ll > best_ll:
                best_ll = ll
                best_params = (
                    self._means.clone(),
                    self._vars.clone(),
                    self._weights.clone(),
                    self.converged_,
                    self.n_iter_,
                )
        
        if best_params is not None:
            self._means, self._vars, self._weights, self.converged_, self.n_iter_ = best_params
        
        self.lower_bound_ = best_ll
        
        # Store as numpy arrays for sklearn compatibility
        self.means_ = self._means.cpu().numpy()
        self.covariances_ = self._vars.cpu().numpy()
        self.weights_ = self._weights.cpu().numpy()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for data."""
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        resp = self._e_step(X_tensor)
        return resp.argmax(dim=1).cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities for each component."""
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        resp = self._e_step(X_tensor)
        return resp.cpu().numpy()
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute log-likelihood for each sample."""
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples = X_tensor.shape[0]
        k = self.n_components
        
        log_probs = torch.zeros(n_samples, k, device=self.device)
        
        for j in range(k):
            diff = X_tensor - self._means[j]
            var = self._vars[j]
            mahal = (diff ** 2 / var).sum(dim=1)
            log_det = var.log().sum()
            n_features = X_tensor.shape[1]
            log_probs[:, j] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)
        
        log_probs = log_probs + self._weights.log()
        
        return torch.logsumexp(log_probs, dim=1).cpu().numpy()
    
    def score(self, X: np.ndarray) -> float:
        """Compute mean log-likelihood."""
        return self.score_samples(X).mean()
    
    def bic(self, X: np.ndarray) -> float:
        """
        Compute Bayesian Information Criterion.
        
        BIC = -2 * log_likelihood + n_params * log(n_samples)
        """
        n_samples, n_features = X.shape
        k = self.n_components
        n_params = k * (1 + 2 * n_features) - 1
        
        ll = self.score_samples(X).sum()
        bic = -2 * ll + n_params * np.log(n_samples)
        
        return bic
