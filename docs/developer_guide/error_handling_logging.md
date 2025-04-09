# Error Handling and Logging

This document describes the error handling and logging system in the Video Annotator project.

## Error Handling

The Video Annotator project uses a centralized error handling system to ensure consistent error handling throughout the application. The system is implemented in the `src/utils/error_handling.py` module.

### Error Levels

The system defines four error levels:

- `INFO`: Informational messages that don't indicate an error
- `WARNING`: Warning messages that indicate a potential issue
- `ERROR`: Error messages that indicate a problem that prevented an operation from completing
- `CRITICAL`: Critical error messages that indicate a serious problem that may affect the application's stability

### Error Handler

The `ErrorHandler` class is responsible for handling errors. It provides methods for logging errors and displaying error messages to the user.

```python
from src.utils.error_handling import ErrorHandler, ErrorLevel

# Create an error handler with a logger
error_handler = ErrorHandler(logger)

# Handle an error
error_handler.handle_error(
    error=ValueError("Invalid value"),
    level=ErrorLevel.ERROR,
    show_dialog=True,
    parent=None
)
```

### Global Error Handler

The system provides a global error handler that can be used throughout the application. This handler can be accessed using the `handle_error` function.

```python
from src.utils.error_handling import handle_error, ErrorLevel

# Handle an error using the global error handler
handle_error(
    error=ValueError("Invalid value"),
    level=ErrorLevel.ERROR,
    show_dialog=True,
    parent=None
)
```

### Exception Handler Decorator

The system provides an `exception_handler` decorator that can be used to automatically handle exceptions in functions.

```python
from src.utils.error_handling import exception_handler, ErrorLevel

# Basic usage
@exception_handler
def my_function():
    # This function will automatically handle exceptions
    pass

# Advanced usage
@exception_handler(level=ErrorLevel.WARNING, show_dialog=False, return_value=[])
def my_function():
    # This function will handle exceptions with custom settings
    pass
```

## Logging

The Video Annotator project uses a centralized logging system to ensure consistent logging throughout the application. The system is implemented in the `src/utils/logger.py` module.

### Logger Setup

The system provides a `setup_logger` function that creates and configures a logger.

```python
from src.utils.logger import setup_logger

# Create a logger with default settings
logger = setup_logger('my_logger')

# Create a logger with a custom log file
logger = setup_logger('my_logger', log_file='my_log.log')
```

### Global Configuration

The system provides a `configure_logging` function that configures global logging settings.

```python
from src.utils.logger import configure_logging
import logging

# Configure logging with custom settings
configure_logging(
    log_level=logging.DEBUG,
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_date_format='%Y-%m-%d %H:%M:%S',
    log_dir='/path/to/logs',
    max_log_size=10 * 1024 * 1024,  # 10 MB
    backup_count=5,
    console_output=True
)
```

### Getting Loggers

The system provides a `get_logger` function that retrieves an existing logger or creates a new one.

```python
from src.utils.logger import get_logger

# Get a logger
logger = get_logger('my_logger')
```

### Logging Messages

Once you have a logger, you can use it to log messages at different levels.

```python
# Log an informational message
logger.info("This is an informational message")

# Log a warning message
logger.warning("This is a warning message")

# Log an error message
logger.error("This is an error message")

# Log a critical message
logger.critical("This is a critical message")
```

### Log Files

By default, log files are stored in the `src/utils/logs` directory. Each logger creates its own log file named after the logger (e.g., `my_logger.log`).

The system uses rotating log files to prevent log files from growing too large. When a log file reaches the maximum size (default: 10 MB), it is rotated, and a new log file is created. The system keeps a configurable number of backup log files (default: 5).

## Integration with Error Handling

The error handling and logging systems are integrated to provide a seamless experience. When an error is handled using the error handling system, it is automatically logged using the logging system.

```python
from src.utils.error_handling import handle_error, ErrorLevel
from src.utils.logger import setup_logger

# Create a logger
logger = setup_logger('my_logger')

# Set up the error handler to use the logger
from src.utils.error_handling import set_global_error_handler, ErrorHandler
set_global_error_handler(ErrorHandler(logger))

# Handle an error (it will be logged automatically)
handle_error(ValueError("Invalid value"))
```

## Best Practices

### Error Handling

- Use the `exception_handler` decorator for functions that may raise exceptions
- Use appropriate error levels for different types of errors
- Provide meaningful error messages that help users understand and resolve issues
- Only show error dialogs for errors that require user attention

### Logging

- Create a logger for each module using the module name
- Use appropriate log levels for different types of messages
- Include relevant context in log messages
- Don't log sensitive information (e.g., passwords, personal data)

## Example

Here's a complete example of using the error handling and logging systems:

```python
from src.utils.error_handling import exception_handler, ErrorLevel
from src.utils.logger import setup_logger

# Create a logger
logger = setup_logger('my_module')

class MyClass:
    def __init__(self):
        self.logger = logger
    
    @exception_handler
    def my_method(self, value):
        """
        A method that may raise an exception.
        
        Args:
            value: The value to process
            
        Returns:
            The processed value
        """
        self.logger.info(f"Processing value: {value}")
        
        if not isinstance(value, int):
            raise TypeError("Value must be an integer")
        
        if value < 0:
            raise ValueError("Value must be non-negative")
        
        result = value * 2
        self.logger.info(f"Processed value: {result}")
        return result
```

In this example:

1. We create a logger for the module
2. We use the `exception_handler` decorator to automatically handle exceptions in the `my_method` method
3. We log informational messages before and after processing the value
4. We raise appropriate exceptions with meaningful error messages when the value is invalid

If an exception is raised, it will be:

1. Logged at the ERROR level
2. Displayed to the user in an error dialog
3. The method will return None (the default return value for the exception handler)

This ensures that errors are handled consistently and that users are informed of issues in a user-friendly way.
