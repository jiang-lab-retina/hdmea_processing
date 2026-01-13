"""
Autoencoder-based RGC Subtype Clustering Pipeline.

This module implements a weakly supervised clustering pipeline that uses
autoencoder-derived 49D latent embeddings with Baden-style GMM clustering.

Key Features:
- Segment-wise autoencoders (10 segments → 49D total)
- Supervised contrastive loss for weak supervision
- Per-group diagonal GMM clustering with BIC selection
- Cross-validation via omitted-label purity
- Bootstrap stability testing

Usage:
    # Run as module
    python -m Autoencoder_method.run_pipeline
    
    # Or import components directly
    from Autoencoder_method.data_loader import load_and_filter_data
    from Autoencoder_method.train import train_autoencoder
"""

__version__ = "0.1.0"
__author__ = "RGC Classification Team"

# No imports here to avoid circular dependencies
# Import submodules directly when needed:
#   from Autoencoder_method import config
#   from Autoencoder_method.data_loader import load_and_filter_data
#   etc.
