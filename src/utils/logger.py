import logging
import os
import sys
import atexit
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Union

# Global dictionary to store loggers
_loggers: Dict[str, logging.Logger] = {}

# Global configuration
# Get the project root directory (2 levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_config = {
    'log_level': logging.INFO,
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_date_format': '%Y-%m-%d %H:%M:%S',
    'log_dir': os.path.join(project_root, 'logs'),  # Use project root for logs
    'max_log_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5,
    'console_output': True
}

def configure_logging(log_level: Union[int, str] = logging.INFO,
                     log_format: str = None,
                     log_date_format: str = None,
                     log_dir: str = None,
                     max_log_size: int = None,
                     backup_count: int = None,
                     console_output: bool = None) -> None:
    """
    Configure global logging settings.

    Args:
        log_level (Union[int, str]): Logging level (e.g., logging.INFO or 'INFO')
        log_format (str, optional): Format string for log messages
        log_date_format (str, optional): Format string for log dates
        log_dir (str, optional): Directory for log files
        max_log_size (int, optional): Maximum size of log files in bytes
        backup_count (int, optional): Number of backup log files to keep
        console_output (bool, optional): Whether to output logs to console
    """
    global _config

    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Update configuration
    if log_level is not None:
        _config['log_level'] = log_level
    if log_format is not None:
        _config['log_format'] = log_format
    if log_date_format is not None:
        _config['log_date_format'] = log_date_format
    if log_dir is not None:
        _config['log_dir'] = log_dir
    if max_log_size is not None:
        _config['max_log_size'] = max_log_size
    if backup_count is not None:
        _config['backup_count'] = backup_count
    if console_output is not None:
        _config['console_output'] = console_output

    # Create log directory if it doesn't exist
    if not os.path.exists(_config['log_dir']):
        os.makedirs(_config['log_dir'], exist_ok=True)

    # Update existing loggers with new configuration
    for logger in _loggers.values():
        _configure_logger(logger)

def _configure_logger(logger: logging.Logger) -> None:
    """
    Configure a logger with the current settings.

    Args:
        logger (logging.Logger): The logger to configure
    """
    # Set log level
    logger.setLevel(_config['log_level'])

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        fmt=_config['log_format'],
        datefmt=_config['log_date_format']
    )

    # Add console handler if enabled
    if _config['console_output']:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(_config['log_level'])
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Get log file path
    log_file = os.path.join(_config['log_dir'], f'{logger.name}.log')

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")
        except Exception as e:
            print(f"Error creating log directory: {str(e)}")
            # Fall back to current directory
            log_file = f'{logger.name}.log'
            print(f"Using fallback log file: {log_file}")

    # Add rotating file handler
    try:
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=_config['max_log_size'],
            backupCount=_config['backup_count']
        )
        file_handler.setLevel(_config['log_level'])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file handler: {str(e)}")
        # Continue without file handler, just use console output

def setup_logger(name: str = 'video_annotator', log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name (str): Name of the logger
        log_file (str, optional): Path to the log file. If None, a default log file will be created.

    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(_config['log_dir']):
        try:
            os.makedirs(_config['log_dir'], exist_ok=True)
            print(f"Created log directory: {_config['log_dir']}")
        except Exception as e:
            print(f"Error creating log directory: {str(e)}")
            # Fall back to a directory we know exists
            _config['log_dir'] = os.path.dirname(os.path.abspath(__file__))
            print(f"Using fallback log directory: {_config['log_dir']}")

    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)

    # Store in global dictionary
    _loggers[name] = logger

    # Configure the logger
    _configure_logger(logger)

    # Override log file if specified
    if log_file is not None:
        # Remove existing file handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, (logging.FileHandler, RotatingFileHandler)):
                logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(
            fmt=_config['log_format'],
            datefmt=_config['log_date_format']
        )

        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(_config['log_level'])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Log startup message
    logger.info(f"Logger '{name}' initialized")

    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name (str): Name of the logger

    Returns:
        logging.Logger: The requested logger
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)

def shutdown_logging() -> None:
    """
    Properly shut down all loggers.
    """
    logging.shutdown()

# Register shutdown function
atexit.register(shutdown_logging)

# Initialize logging configuration
configure_logging()
