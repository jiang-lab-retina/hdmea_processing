"""
Subgroup-Specific Clustering Pipeline.

This module implements multiple approaches to discover sub-clusters
within each RGC subgroup (ipRGC, DSGC, OSGC, Other).

Approaches:
1. Standard Autoencoder + GMM
2. Variational Autoencoder (VAE) + GMM  
3. Deep Embedded Clustering (DEC)
"""

from .config import *
from .data_loader import load_subgroup_data
from .models import ConvAutoencoder, ConvVAE, DECModel
from .clustering import fit_gmm_auto_k, compute_cluster_metrics

