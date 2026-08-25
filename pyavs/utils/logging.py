"""
Centralized logging configuration for pyAVS package.

This module provides a unified logging system for all pyAVS components,
allowing for consistent log formatting, levels, and output handling.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
import datetime


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class PyAVSLogger:
    """
    Centralized logger for pyAVS package.
    
    Provides consistent logging across all modules with configurable
    output levels, formats, and destinations.
    """
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def configure(cls, 
                  level: Union[str, int] = 'INFO',
                  console: bool = True,
                  file_path: Optional[Union[str, Path]] = None,
                  format_string: Optional[str] = None,
                  date_format: Optional[str] = None,
                  use_colors: bool = True,
                  max_file_size: int = 10 * 1024 * 1024,  # 10MB
                  backup_count: int = 5) -> None:
        """
        Configure the global logging system for pyAVS.
        
        Parameters
        ----------
        level : str or int, optional
            Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') (default: 'INFO')
        console : bool, optional
            Whether to output to console (default: True)
        file_path : str or Path, optional
            Path to log file. If None, no file logging (default: None)
        format_string : str, optional
            Custom format string for log messages
        date_format : str, optional
            Custom date format for timestamps
        use_colors : bool, optional
            Whether to use colored output in console (default: True)
        max_file_size : int, optional
            Maximum log file size in bytes before rotation (default: 10MB)
        backup_count : int, optional
            Number of backup files to keep during rotation (default: 5)
        """
        if cls._configured:
            cls.get_logger('pyavs').warning("Logging already configured. Reconfiguring...")
        
        # Convert string level to logging constant
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # Default formats
        if format_string is None:
            format_string = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        if date_format is None:
            date_format = '%Y-%m-%d %H:%M:%S'
        
        # Configure root logger for pyavs
        root_logger = logging.getLogger('pyavs')
        root_logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            if use_colors and sys.stdout.isatty():  # Only use colors if terminal supports it
                console_formatter = ColoredFormatter(format_string, date_format)
            else:
                console_formatter = logging.Formatter(format_string, date_format)
            
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if file_path is not None:
            from logging.handlers import RotatingFileHandler
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                file_path, 
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(format_string, date_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        root_logger.propagate = False
        
        cls._configured = True
        
        # Log configuration success
        logger = cls.get_logger('pyavs.logging')
        logger.info(f"Logging configured: level={logging.getLevelName(level)}, "
                   f"console={console}, file={file_path is not None}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.
        
        Parameters
        ----------
        name : str
            Logger name, typically the module name
            
        Returns
        -------
        logging.Logger
            Logger instance
        """
        if not cls._configured:
            # Configure with defaults if not already configured
            cls.configure()
        
        if name not in cls._loggers:
            # Ensure name starts with 'pyavs.'
            if not name.startswith('pyavs'):
                name = f'pyavs.{name}'
            
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, level: Union[str, int], logger_name: Optional[str] = None) -> None:
        """
        Set logging level for specific logger or all loggers.
        
        Parameters
        ----------
        level : str or int
            Logging level
        logger_name : str, optional
            Specific logger name. If None, applies to root pyavs logger
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        if logger_name is None:
            # Apply to root logger and all handlers
            root_logger = logging.getLogger('pyavs')
            root_logger.setLevel(level)
            for handler in root_logger.handlers:
                handler.setLevel(level)
        else:
            logger = cls.get_logger(logger_name)
            logger.setLevel(level)
    
    @classmethod
    def add_file_handler(cls, file_path: Union[str, Path], 
                        level: Union[str, int] = 'INFO') -> None:
        """
        Add an additional file handler to the logging system.
        
        Parameters
        ----------
        file_path : str or Path
            Path to additional log file
        level : str or int, optional
            Logging level for this handler (default: 'INFO')
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        root_logger = logging.getLogger('pyavs')
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# Convenience functions for common logging tasks
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance (convenience function).
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return PyAVSLogger.get_logger(name)


def configure_logging(**kwargs) -> None:
    """
    Configure logging system (convenience function).
    
    Parameters
    ----------
    **kwargs
        Arguments passed to PyAVSLogger.configure()
    """
    PyAVSLogger.configure(**kwargs)


def set_log_level(level: Union[str, int], logger_name: Optional[str] = None) -> None:
    """
    Set logging level (convenience function).
    
    Parameters
    ----------
    level : str or int
        Logging level
    logger_name : str, optional
        Specific logger name
    """
    PyAVSLogger.set_level(level, logger_name)


# Context manager for temporary log level changes
class temporary_log_level:
    """
    Context manager to temporarily change log level.
    
    Usage:
        with temporary_log_level('DEBUG'):
            # Debug logging enabled
            logger.debug("This will be shown")
    """
    
    def __init__(self, level: Union[str, int], logger_name: Optional[str] = None):
        self.new_level = level
        self.logger_name = logger_name
        self.original_level = None
        
    def __enter__(self):
        if isinstance(self.new_level, str):
            self.new_level = getattr(logging, self.new_level.upper())
            
        if self.logger_name is None:
            logger = logging.getLogger('pyavs')
        else:
            logger = PyAVSLogger.get_logger(self.logger_name)
            
        self.original_level = logger.level
        logger.setLevel(self.new_level)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger_name is None:
            logger = logging.getLogger('pyavs')
        else:
            logger = PyAVSLogger.get_logger(self.logger_name)
            
        logger.setLevel(self.original_level)


# Progress logging utilities
def log_processing_start(logger: logging.Logger, operation: str, 
                        details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log the start of a processing operation.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    operation : str
        Description of the operation
    details : dict, optional
        Additional details to log
    """
    if details:
        detail_str = ', '.join([f"{k}={v}" for k, v in details.items()])
        logger.info(f"Starting {operation} ({detail_str})")
    else:
        logger.info(f"Starting {operation}")


def log_processing_end(logger: logging.Logger, operation: str, 
                      success: bool = True, duration: Optional[float] = None,
                      details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log the end of a processing operation.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    operation : str
        Description of the operation
    success : bool, optional
        Whether operation was successful (default: True)
    duration : float, optional
        Duration in seconds
    details : dict, optional
        Additional details to log
    """
    status = "completed" if success else "failed"
    message_parts = [f"{operation} {status}"]
    
    if duration is not None:
        message_parts.append(f"in {duration:.2f}s")
    
    if details:
        detail_str = ', '.join([f"{k}={v}" for k, v in details.items()])
        message_parts.append(f"({detail_str})")
    
    message = ' '.join(message_parts)
    
    if success:
        logger.info(message)
    else:
        logger.error(message)


# Default configuration when module is imported
if not PyAVSLogger._configured:
    # Configure with sensible defaults
    configure_logging(
        level='INFO',
        console=True,
        file_path=None,
        use_colors=True
    )