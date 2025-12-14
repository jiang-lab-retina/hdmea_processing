"""
Logging utilities for HD-MEA pipeline.

Per constitution: Library code MUST use logging.getLogger(__name__), never print().
"""

import logging
from typing import Optional


def setup_logging(level: str = "INFO", format_style: str = "default") -> None:
    """
    Configure logging for pipeline execution.
    
    This should be called at the application entry point (script/notebook),
    NOT inside library modules.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Format style - "default" for standard, "minimal" for compact
    
    Example:
        >>> from hdmea.utils.logging import setup_logging
        >>> setup_logging(level="DEBUG")
    """
    if format_style == "minimal":
        fmt = "%(levelname)s | %(message)s"
    else:
        fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    This is a convenience wrapper around logging.getLogger().
    Modules should use: logger = get_logger(__name__)
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class that provides a logger property.
    
    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Doing something")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

