import unittest
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock, call

class TestErrorHandling(unittest.TestCase):
    """Test cases for the error handling utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid importing before mocks are in place
        from src.utils.error_handling import exception_handler, show_error_message
        self.exception_handler = exception_handler
        self.show_error_message = show_error_message
    
    @patch('src.utils.error_handling.show_error_message')
    def test_exception_handler(self, mock_show_error):
        """Test the exception handler decorator."""
        # Create a test function that raises an exception
        @self.exception_handler
        def test_function():
            raise ValueError("Test error")
        
        # Call the function and check that show_error_message was called
        result = test_function()
        self.assertIsNone(result)
        mock_show_error.assert_called_once_with("An error occurred: Test error")
    
    @patch('src.utils.error_handling.show_error_message')
    def test_exception_handler_with_logger(self, mock_show_error):
        """Test the exception handler decorator with a logger."""
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Create a test class with a logger
        class TestClass:
            def __init__(self):
                self.logger = mock_logger
            
            @self.exception_handler
            def test_method(self):
                raise ValueError("Test error")
        
        # Create an instance and call the method
        test_instance = TestClass()
        result = test_instance.test_method()
        
        # Check that the logger was used
        self.assertIsNone(result)
        mock_logger.error.assert_called()
        mock_show_error.assert_called_once_with("An error occurred: Test error")
    
    @patch('src.utils.error_handling.show_error_message')
    def test_exception_handler_no_exception(self, mock_show_error):
        """Test the exception handler decorator when no exception is raised."""
        # Create a test function that doesn't raise an exception
        @self.exception_handler
        def test_function():
            return "Success"
        
        # Call the function and check that show_error_message was not called
        result = test_function()
        self.assertEqual(result, "Success")
        mock_show_error.assert_not_called()
    
    @patch('tkinter.messagebox.showerror')
    def test_show_error_message(self, mock_showerror):
        """Test the show_error_message function."""
        # Call the function and check that messagebox.showerror was called
        self.show_error_message("Test error", "Test Title", None)
        mock_showerror.assert_called_once_with("Test Title", "Test error", parent=None)

class TestLogger(unittest.TestCase):
    """Test cases for the logger utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here
        from src.utils.logger import setup_logger
        self.setup_logger = setup_logger
        
        # Create a temporary directory for log files
        self.test_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.test_dir, 'test.log')
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary files
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    @patch('logging.FileHandler')
    @patch('logging.StreamHandler')
    def test_setup_logger(self, mock_stream_handler, mock_file_handler):
        """Test setting up a logger."""
        # Setup mock handlers
        mock_stream_handler_instance = MagicMock()
        mock_file_handler_instance = MagicMock()
        mock_stream_handler.return_value = mock_stream_handler_instance
        mock_file_handler.return_value = mock_file_handler_instance
        
        # Create a logger
        logger = self.setup_logger('test_logger', self.log_file)
        
        # Check that the logger was created with the correct name
        self.assertEqual(logger.name, 'test_logger')
        
        # Check that the handlers were added
        mock_stream_handler.assert_called_once()
        mock_file_handler.assert_called_once_with(self.log_file)
        
        # Check that the formatters were set
        mock_stream_handler_instance.setFormatter.assert_called_once()
        mock_file_handler_instance.setFormatter.assert_called_once()
    
    def test_logger_functionality(self):
        """Test the functionality of the logger."""
        # Create a logger with a file handler
        logger = self.setup_logger('test_logger', self.log_file)
        
        # Log some messages
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check that the log file was created
        self.assertTrue(os.path.exists(self.log_file))
        
        # Check the content of the log file
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Info message", log_content)
            self.assertIn("Warning message", log_content)
            self.assertIn("Error message", log_content)

if __name__ == '__main__':
    unittest.main()
