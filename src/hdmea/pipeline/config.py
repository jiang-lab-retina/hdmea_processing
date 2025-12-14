"""
Configuration loading and validation for HD-MEA pipeline.

Uses Pydantic for schema validation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from hdmea.utils.exceptions import ConfigurationError


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================

class StimulusConfig(BaseModel):
    """Configuration for a stimulus type."""
    
    name: str
    display_name: Optional[str] = None
    movie_length_frames: int = Field(gt=0)
    frame_rate_hz: float = Field(gt=0)
    num_repeats: int = Field(ge=1)
    sections: Dict[str, Tuple[int, int]] = Field(default_factory=dict)
    analysis_windows: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    
    @field_validator("sections", mode="before")
    @classmethod
    def convert_sections(cls, v):
        """Convert list values to tuples."""
        if isinstance(v, dict):
            return {k: tuple(val) for k, val in v.items()}
        return v
    
    @field_validator("analysis_windows", mode="before")
    @classmethod
    def convert_windows(cls, v):
        """Convert list values to tuples."""
        if isinstance(v, dict):
            return {k: tuple(val) for k, val in v.items()}
        return v


class StageConfig(BaseModel):
    """Configuration for a pipeline stage."""
    
    enabled: bool = True
    feature_sets: List[str] = Field(default_factory=list)


class FlowConfig(BaseModel):
    """Configuration for a named pipeline flow."""
    
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    stages: Dict[str, StageConfig] = Field(default_factory=dict)
    defaults: Dict[str, Any] = Field(default_factory=dict)
    
    def get_feature_sets(self) -> List[str]:
        """Get list of feature sets to extract."""
        features_stage = self.stages.get("features", StageConfig())
        return features_stage.feature_sets if features_stage.enabled else []


class PipelineConfig(BaseModel):
    """Global pipeline configuration."""
    
    version: str = "0.1.0"
    random_seed: int = 42
    force_recompute: bool = False
    log_level: str = "INFO"


class DefaultsConfig(BaseModel):
    """Default configuration loaded from config/defaults.json."""
    
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    stage1: Dict[str, Any] = Field(default_factory=dict)
    stage2: Dict[str, Any] = Field(default_factory=dict)
    export: Dict[str, Any] = Field(default_factory=dict)
    paths: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Configuration Loading Functions
# =============================================================================

def load_json_config(path: Path) -> Dict[str, Any]:
    """
    Load a JSON configuration file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Parsed JSON as dictionary
    
    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {path}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {path}: {e}")


def load_flow_config(name_or_path: str, config_dir: Optional[Path] = None) -> FlowConfig:
    """
    Load a flow configuration by name or path.
    
    Args:
        name_or_path: Flow name (looks in config/flows/) or direct path
        config_dir: Optional config directory (default: config/)
    
    Returns:
        Validated FlowConfig
    
    Raises:
        ConfigurationError: If config not found or invalid
    """
    if config_dir is None:
        config_dir = Path("config")
    
    # Check if it's a direct path
    path = Path(name_or_path)
    if not path.exists():
        # Try as a name in config/flows/
        path = config_dir / "flows" / f"{name_or_path}.json"
    
    data = load_json_config(path)
    
    try:
        return FlowConfig(**data)
    except Exception as e:
        raise ConfigurationError(f"Invalid flow config in {path}: {e}")


def load_stimulus_config(name_or_path: str, config_dir: Optional[Path] = None) -> StimulusConfig:
    """
    Load a stimulus configuration by name or path.
    
    Args:
        name_or_path: Stimulus name (looks in config/stimuli/) or direct path
        config_dir: Optional config directory (default: config/)
    
    Returns:
        Validated StimulusConfig
    
    Raises:
        ConfigurationError: If config not found or invalid
    """
    if config_dir is None:
        config_dir = Path("config")
    
    # Check if it's a direct path
    path = Path(name_or_path)
    if not path.exists():
        # Try as a name in config/stimuli/
        path = config_dir / "stimuli" / f"{name_or_path}.json"
    
    data = load_json_config(path)
    
    try:
        return StimulusConfig(**data)
    except Exception as e:
        raise ConfigurationError(f"Invalid stimulus config in {path}: {e}")


def load_defaults(config_dir: Optional[Path] = None) -> DefaultsConfig:
    """
    Load default configuration.
    
    Args:
        config_dir: Optional config directory (default: config/)
    
    Returns:
        Validated DefaultsConfig
    """
    if config_dir is None:
        config_dir = Path("config")
    
    path = config_dir / "defaults.json"
    
    if not path.exists():
        logger.warning(f"Defaults file not found: {path}. Using built-in defaults.")
        return DefaultsConfig()
    
    data = load_json_config(path)
    
    try:
        return DefaultsConfig(**data)
    except Exception as e:
        logger.warning(f"Invalid defaults config: {e}. Using built-in defaults.")
        return DefaultsConfig()

