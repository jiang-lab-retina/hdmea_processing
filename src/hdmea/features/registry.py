"""
Feature registry for extensible feature extraction.

The registry pattern allows adding new features without editing existing code.
Features are registered using a decorator.

Per constitution: No eval()-based dispatch. Use decorator-based registration.
"""

from typing import Any, Dict, List, Type
import logging

from hdmea.features.base import FeatureExtractor


logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Registry for feature extractors.
    
    Features are registered using the @register decorator:
    
        @FeatureRegistry.register("my_feature")
        class MyFeatureExtractor(FeatureExtractor):
            ...
    
    Features can then be retrieved by name:
    
        extractor_class = FeatureRegistry.get("my_feature")
        extractor = extractor_class()
    """
    
    _registry: Dict[str, Type[FeatureExtractor]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a feature extractor.
        
        Args:
            name: Unique name for the feature
        
        Returns:
            Decorator function
        
        Raises:
            ValueError: If name is already registered
        
        Example:
            @FeatureRegistry.register("step_up_5s_5i_3x")
            class StepUpExtractor(FeatureExtractor):
                ...
        """
        def decorator(extractor_class: Type[FeatureExtractor]):
            if name in cls._registry:
                raise ValueError(
                    f"Feature '{name}' is already registered by {cls._registry[name].__name__}"
                )
            
            # Set the name attribute on the class
            extractor_class.name = name
            cls._registry[name] = extractor_class
            
            logger.debug(f"Registered feature extractor: {name} -> {extractor_class.__name__}")
            
            return extractor_class
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[FeatureExtractor]:
        """
        Get a registered feature extractor by name.
        
        Args:
            name: Feature name
        
        Returns:
            Feature extractor class
        
        Raises:
            KeyError: If feature is not registered
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise KeyError(
                f"Unknown feature: '{name}'. Available features: {available}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered feature names.
        
        Returns:
            Sorted list of feature names
        """
        return sorted(cls._registry.keys())
    
    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered feature.
        
        Args:
            name: Feature name
        
        Returns:
            Dictionary with feature metadata
        """
        extractor_class = cls.get(name)
        return {
            "name": extractor_class.name,
            "version": extractor_class.version,
            "required_inputs": extractor_class.required_inputs,
            "output_schema": extractor_class.output_schema,
            "runtime_class": extractor_class.runtime_class,
            "description": getattr(extractor_class, "description", ""),
        }
    
    @classmethod
    def get_all_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered features.
        
        Returns:
            Dictionary mapping feature names to their metadata
        """
        return {name: cls.get_metadata(name) for name in cls.list_all()}
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a feature is registered.
        
        Args:
            name: Feature name
        
        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered features.
        
        Warning: This is mainly for testing. Use with caution.
        """
        cls._registry.clear()
        logger.warning("Feature registry cleared")

