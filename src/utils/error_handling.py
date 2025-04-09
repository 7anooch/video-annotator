import tkinter as tk
from tkinter import messagebox
import traceback
import sys
import logging
import functools
from enum import Enum, auto
from typing import Callable, Any, Optional, Union, Dict, List, Tuple

def show_error_message(message, title="Error", parent=None):
    """
    Display an error message dialog.

    Args:
        message (str): Error message to display
        title (str, optional): Dialog title. Defaults to "Error".
        parent (tk.Tk, optional): Parent window. Defaults to None.
    """
    messagebox.showerror(title, message, parent=parent)

def show_warning_message(message, title="Warning", parent=None):
    """
    Display a warning message dialog.

    Args:
        message (str): Warning message to display
        title (str, optional): Dialog title. Defaults to "Warning".
        parent (tk.Tk, optional): Parent window. Defaults to None.
    """
    messagebox.showwarning(title, message, parent=parent)

def show_info_message(message, title="Information", parent=None):
    """
    Display an information message dialog.

    Args:
        message (str): Information message to display
        title (str, optional): Dialog title. Defaults to "Information".
        parent (tk.Tk, optional): Parent window. Defaults to None.
    """
    messagebox.showinfo(title, message, parent=parent)

class ErrorLevel(Enum):
    """Enum for error levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class ErrorHandler:
    """Class for handling errors in the application."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the ErrorHandler.

        Args:
            logger (logging.Logger, optional): Logger to use. Defaults to None.
        """
        self.logger = logger

    def handle_error(self, error: Exception, level: ErrorLevel = ErrorLevel.ERROR,
                     show_dialog: bool = True, parent: Optional[tk.Tk] = None) -> None:
        """Handle an error.

        Args:
            error (Exception): The error to handle
            level (ErrorLevel, optional): The error level. Defaults to ErrorLevel.ERROR.
            show_dialog (bool, optional): Whether to show a dialog. Defaults to True.
            parent (tk.Tk, optional): Parent window for the dialog. Defaults to None.
        """
        # Get the error message and traceback
        error_message = str(error)
        error_traceback = traceback.format_exc()

        # Log the error
        if self.logger:
            if level == ErrorLevel.INFO:
                self.logger.info(f"Info: {error_message}")
            elif level == ErrorLevel.WARNING:
                self.logger.warning(f"Warning: {error_message}")
                self.logger.warning(error_traceback)
            elif level == ErrorLevel.ERROR:
                self.logger.error(f"Error: {error_message}")
                self.logger.error(error_traceback)
            elif level == ErrorLevel.CRITICAL:
                self.logger.critical(f"Critical: {error_message}")
                self.logger.critical(error_traceback)
        else:
            print(f"{level.name}: {error_message}")
            if level != ErrorLevel.INFO:
                print(error_traceback)

        # Show a dialog if requested
        if show_dialog:
            if level == ErrorLevel.INFO:
                show_info_message(error_message, parent=parent)
            elif level == ErrorLevel.WARNING:
                show_warning_message(error_message, parent=parent)
            else:  # ERROR or CRITICAL
                show_error_message(error_message, parent=parent)

# Global error handler
_error_handler = ErrorHandler()

def set_global_error_handler(handler: ErrorHandler) -> None:
    """Set the global error handler.

    Args:
        handler (ErrorHandler): The error handler to use
    """
    global _error_handler
    _error_handler = handler

def handle_error(error: Exception, level: ErrorLevel = ErrorLevel.ERROR,
                show_dialog: bool = True, parent: Optional[tk.Tk] = None) -> None:
    """Handle an error using the global error handler.

    Args:
        error (Exception): The error to handle
        level (ErrorLevel, optional): The error level. Defaults to ErrorLevel.ERROR.
        show_dialog (bool, optional): Whether to show a dialog. Defaults to True.
        parent (tk.Tk, optional): Parent window for the dialog. Defaults to None.
    """
    _error_handler.handle_error(error, level, show_dialog, parent)

def exception_handler(func: Callable = None, *, level: ErrorLevel = ErrorLevel.ERROR,
                     show_dialog: bool = True, return_value: Any = None) -> Callable:
    """Decorator for handling exceptions in functions.

    Args:
        func (Callable, optional): The function to decorate. Defaults to None.
        level (ErrorLevel, optional): The error level. Defaults to ErrorLevel.ERROR.
        show_dialog (bool, optional): Whether to show a dialog. Defaults to True.
        return_value (Any, optional): Value to return on error. Defaults to None.

    Returns:
        Callable: The decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the self argument if it exists
                self = args[0] if args and hasattr(args[0], 'logger') else None

                # Use the instance's error handler if available
                if self and hasattr(self, 'error_handler'):
                    self.error_handler.handle_error(e, level, show_dialog)
                # Use the instance's logger if available
                elif self and hasattr(self, 'logger'):
                    handler = ErrorHandler(self.logger)
                    handler.handle_error(e, level, show_dialog)
                # Use the global error handler
                else:
                    handle_error(e, level, show_dialog)

                # Return the specified return value
                return return_value

        return wrapper

    # Handle both @exception_handler and @exception_handler()
    if func is None:
        return decorator
    return decorator(func)
