"""
CNN model inference for axon signal prediction.

This module provides functions to load the trained CNN model and run
GPU-optimized inference on STA data to predict axon signal probability.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import torch to allow module to be imported without torch installed
_torch = None
_nn = None


def _ensure_torch():
    """Lazy import torch and torch.nn."""
    global _torch, _nn
    if _torch is None:
        try:
            import torch
            import torch.nn as nn

            _torch = torch
            _nn = nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for AP tracking model inference. "
                "Install with: pip install torch"
            )
    return _torch, _nn


class CNN3D_WithVelocity:
    """
    3D CNN model with velocity features for axon signal prediction.

    This class wraps the PyTorch model architecture for compatibility
    with the hdmea package structure.
    """

    def __init__(
        self,
        input_dim: Tuple[int, int, int] = (5, 5, 5),
        aux_features: int = 2,
        num_classes: int = 1,
    ):
        """
        Initialize the CNN model architecture.

        Args:
            input_dim: Input cube dimensions (depth, height, width)
            aux_features: Number of auxiliary features (velocity, direction)
            num_classes: Number of output classes (1 for binary)
        """
        torch, nn = _ensure_torch()

        self.input_dim = input_dim
        self.device = None
        self._model = None

        # Build the model
        self._build_model(nn, input_dim, aux_features, num_classes)

    def _build_model(self, nn, input_dim, aux_features, num_classes):
        """Build the PyTorch model architecture."""
        torch, _ = _ensure_torch()

        class _CNN3D_WithVelocity(nn.Module):
            def __init__(self, input_channels=1, aux_features=2, num_classes=1, input_dim=(5, 5, 5)):
                super().__init__()
                self.input_dim = input_dim

                # CNN Branch - Convolutional Block 1
                self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm3d(16)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

                # CNN Branch - Convolutional Block 2
                self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm3d(32)
                self.relu2 = nn.ReLU()

                # CNN Branch - Convolutional Block 3
                self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm3d(64)
                self.relu3 = nn.ReLU()

                # Calculate flattened size
                self._to_linear = None
                self._get_conv_output_size(input_channels)

                # CNN Branch layers
                self.flatten = nn.Flatten()
                self.dropout = nn.Dropout(0.5)
                self.fc_cnn = nn.Linear(self._to_linear, 64)
                self.relu_fc_cnn = nn.ReLU()

                # Auxiliary Features Branch
                self.fc_aux = nn.Linear(aux_features, 4)
                self.relu_fc_aux = nn.ReLU()

                # Combined Classification Head
                self.fc_combined = nn.Linear(64 + 4, 32)
                self.relu_fc_combined = nn.ReLU()
                self.fc_final = nn.Linear(32, num_classes)
                self.sigmoid = nn.Sigmoid()

            def _get_conv_output_size(self, input_channels):
                with torch.no_grad():
                    dummy_input = torch.randn(1, input_channels, *self.input_dim)
                    x = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
                    x = self.relu2(self.bn2(self.conv2(x)))
                    x = self.relu3(self.bn3(self.conv3(x)))
                    self._to_linear = int(np.prod(x.size()[1:]))

            def forward(self, x, v, n):
                # CNN Branch
                x_cnn = self.pool1(self.relu1(self.bn1(self.conv1(x))))
                x_cnn = self.relu2(self.bn2(self.conv2(x_cnn)))
                x_cnn = self.relu3(self.bn3(self.conv3(x_cnn)))
                x_cnn = self.flatten(x_cnn)
                x_cnn = self.dropout(x_cnn)
                x_cnn = self.relu_fc_cnn(self.fc_cnn(x_cnn))

                # Auxiliary Features Branch
                x_aux = torch.cat([v, n], dim=1)
                x_aux = self.relu_fc_aux(self.fc_aux(x_aux))

                # Combined
                x_combined = torch.cat([x_cnn, x_aux], dim=1)
                x_combined = self.relu_fc_combined(self.fc_combined(x_combined))
                x_combined = self.fc_final(x_combined)
                output = self.sigmoid(x_combined)
                return output

        self._model = _CNN3D_WithVelocity(
            input_channels=1,
            aux_features=aux_features,
            num_classes=num_classes,
            input_dim=input_dim,
        )

    def to(self, device):
        """Move model to specified device."""
        self.device = device
        self._model = self._model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self._model.eval()
        return self

    def load_state_dict(self, state_dict):
        """Load model weights from state dict."""
        self._model.load_state_dict(state_dict)

    def __call__(self, x, v, n):
        """Forward pass."""
        return self._model(x, v, n)


def select_device(force_cpu: bool = False) -> str:
    """
    Select the appropriate device for inference.

    Args:
        force_cpu: Force CPU even if GPU available

    Returns:
        Device string ("cuda", "mps", or "cpu")
    """
    torch, _ = _ensure_torch()

    if force_cpu:
        logger.info("Forced CPU mode")
        return "cpu"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using CUDA: {device_name} ({memory_gb:.1f}GB)")
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon)")
        return "mps"

    logger.info("Using CPU")
    return "cpu"


def load_cnn_model(
    model_path: Path,
    device: str = "auto",
) -> CNN3D_WithVelocity:
    """
    Load trained CNN model for inference.

    Args:
        model_path: Path to .pth model file
        device: Device string ("auto", "cuda", "cpu", "mps")

    Returns:
        Loaded model in eval mode on specified device

    Raises:
        FileNotFoundError: If model file not found
    """
    torch, _ = _ensure_torch()

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Select device
    if device == "auto":
        device = select_device()

    logger.info(f"Loading model from {model_path}")

    # Initialize model architecture
    model = CNN3D_WithVelocity(input_dim=(5, 5, 5), aux_features=2, num_classes=1)

    # Load weights
    state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    logger.info(f"Model loaded on {device}")

    return model


def extract_all_cubes_from_sta(
    sta_data: np.ndarray,
    expected_shape: Tuple[int, int, int] = (5, 5, 5),
) -> Tuple[Optional[np.ndarray], Optional[list], Optional[Tuple[int, int, int]]]:
    """
    Extract all possible cubes from STA data for batch processing.

    Slides a 5x5x5 window over the STA volume and extracts all valid cubes.

    Args:
        sta_data: 3D STA array (time, row, col)
        expected_shape: Expected cube shape (depth, height, width)

    Returns:
        Tuple of (cubes_array, positions, original_shape) or (None, None, None) if failed
    """
    if not isinstance(sta_data, np.ndarray):
        try:
            sta_data = np.array(sta_data)
        except Exception:
            return None, None, None

    if sta_data.ndim != 3:
        return None, None, None

    t_dim, x_dim, y_dim = sta_data.shape
    d_exp, h_exp, w_exp = expected_shape
    pad_t, pad_x, pad_y = d_exp // 2, h_exp // 2, w_exp // 2

    if not (t_dim >= d_exp and x_dim >= h_exp and y_dim >= w_exp):
        return None, None, None

    cubes = []
    positions = []

    for t in range(pad_t, t_dim - pad_t):
        for x in range(pad_x, x_dim - pad_x):
            for y in range(pad_y, y_dim - pad_y):
                cube = sta_data[
                    t - pad_t : t + pad_t + 1,
                    x - pad_x : x + pad_x + 1,
                    y - pad_y : y + pad_y + 1,
                ]
                cubes.append(cube)
                positions.append((t, x, y))

    if not cubes:
        return None, None, None

    cubes_array = np.stack(cubes, axis=0)
    return cubes_array, positions, sta_data.shape


def _get_optimal_batch_size(device: str) -> int:
    """Get optimal batch size based on GPU memory."""
    torch, _ = _ensure_torch()

    if device != "cuda":
        return 64

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if gpu_memory_gb >= 24:
        return 512
    elif gpu_memory_gb >= 16:
        return 256
    elif gpu_memory_gb >= 8:
        return 128
    else:
        return 64


def predict_batch_gpu_optimized(
    cubes_batch: np.ndarray,
    model: CNN3D_WithVelocity,
    device: str,
    batch_size: Optional[int] = None,
) -> list:
    """
    GPU-optimized batch prediction.

    Args:
        cubes_batch: Array of cubes to predict (N, D, H, W)
        model: Loaded CNN model
        device: Device string
        batch_size: Batch size (auto-determined if None)

    Returns:
        List of predictions
    """
    torch, _ = _ensure_torch()

    if cubes_batch is None or len(cubes_batch) == 0:
        return []

    if batch_size is None:
        batch_size = _get_optimal_batch_size(device)

    all_predictions = []
    num_cubes = len(cubes_batch)

    with torch.no_grad():
        for i in range(0, num_cubes, batch_size):
            batch_end = min(i + batch_size, num_cubes)
            batch_cubes = cubes_batch[i:batch_end]

            # Convert to tensor with channel dimension
            batch_tensor = torch.from_numpy(batch_cubes.astype(np.float32))
            batch_tensor = batch_tensor.unsqueeze(1)  # Add channel dim

            # Transfer to device
            if device == "cuda":
                batch_tensor = batch_tensor.pin_memory()
                batch_tensor = batch_tensor.to(device, non_blocking=True)
            else:
                batch_tensor = batch_tensor.to(device)

            # Create dummy auxiliary features
            batch_size_current = batch_tensor.shape[0]
            v_tensor = torch.zeros(batch_size_current, 1, device=device)
            n_tensor = torch.zeros(batch_size_current, 1, device=device)

            # Forward pass
            output = model(batch_tensor, v_tensor, n_tensor)

            # Extract predictions
            predictions = output.squeeze().cpu().numpy()

            if predictions.ndim == 0:
                predictions = [predictions.item()]
            else:
                predictions = predictions.flatten().tolist()

            all_predictions.extend(predictions)

            # Periodic GPU cache cleanup
            if device == "cuda" and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()

    return all_predictions


def run_model_inference(
    sta_data: np.ndarray,
    model: CNN3D_WithVelocity,
    device: str,
    batch_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Run CNN inference on STA data.

    Extracts all possible cubes from STA data, runs batch prediction,
    and reconstructs the prediction map.

    Args:
        sta_data: 3D STA array (time, row, col)
        model: Loaded CNN model
        device: Device string
        batch_size: Batch size (auto if None)

    Returns:
        Prediction array with same shape as sta_data, or None if failed
    """
    # Extract cubes
    cubes_batch, positions, original_shape = extract_all_cubes_from_sta(sta_data)

    if cubes_batch is None:
        logger.warning("Could not extract cubes from STA data")
        return None

    # Run batch prediction
    predictions = predict_batch_gpu_optimized(
        cubes_batch, model, device, batch_size
    )

    if len(predictions) != len(positions):
        logger.warning(
            f"Prediction count ({len(predictions)}) != position count ({len(positions)})"
        )
        return None

    # Reconstruct prediction map
    prediction_map = np.zeros(original_shape, dtype=np.float32)
    for (t, x, y), pred in zip(positions, predictions):
        prediction_map[t, x, y] = pred

    logger.debug(f"Processed {len(predictions)} cubes")
    return prediction_map

