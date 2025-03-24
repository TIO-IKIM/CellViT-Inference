import unittest
import logging
import os
from pathlib import Path
from cellvit.utils.logger import Logger, NullLogger, PrintLogger, ColoredFormatter

class TestLogger(unittest.TestCase):
    def setUp(self):
        """Set up temporary directories and variables for testing."""
        self.log_dir = Path("test_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "logs.log"

    def tearDown(self):
        """Clean up temporary directories and files after tests."""
        if self.log_dir.exists():
            for file in self.log_dir.iterdir():
                file.unlink()
            self.log_dir.rmdir()

    def test_null_logger(self):
        """Test that NullLogger does nothing."""
        logger = NullLogger()
        try:
            logger.info("This should do nothing.")
            logger.warning("This should do nothing.")
            logger.error("This should do nothing.")
            logger.debug("This should do nothing.")
            logger.critical("This should do nothing.")
        except Exception as e:
            self.fail(f"NullLogger raised an exception: {e}")

    def test_print_logger(self):
        """Test that PrintLogger prints messages."""
        logger = PrintLogger()
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.debug("Debug message")
        logger.critical("Critical message")

    def test_logger_creation_with_console_handler(self):
        """Test Logger creation with console handler only."""
        logger_instance = Logger(level="INFO")
        logger = logger_instance.create_logger()
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.DEBUG)  # Logger is set to DEBUG internally

    def test_logger_creation_with_file_handler(self):
        """Test Logger creation with file handler."""
        logger_instance = Logger(level="INFO", log_dir=self.log_dir)
        logger = logger_instance.create_logger()
        self.assertTrue(self.log_file.exists())

    def test_logger_with_timestamp(self):
        """Test Logger creation with timestamped log file."""
        logger_instance = Logger(level="INFO", log_dir=self.log_dir, use_timestamp=True)
        logger_instance.create_logger()
        log_files = list(self.log_dir.glob("*.log"))
        self.assertEqual(len(log_files), 1)
        self.assertIn("_logs.log", log_files[0].name)

    def test_colored_formatter(self):
        """Test ColoredFormatter formats messages with color codes."""
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="Test message", args=(), exc_info=None
        )
        formatted_message = formatter.format(record)
        self.assertIn("Test message", formatted_message)

    def test_file_rollover(self):
        """Test that log file rolls over when it already exists."""
        # Create an initial log file
        self.log_file.touch()
        logger_instance = Logger(level="INFO", log_dir=self.log_dir)
        logger_instance.create_logger()
        self.assertTrue(self.log_file.exists())  # Ensure the file still exists after rollover


if __name__ == "__main__":
    unittest.main()