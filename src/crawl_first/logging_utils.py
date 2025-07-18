"""
Logging utilities for crawl-first.

Handles output capture, logging configuration, and file management.
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO, Tuple, Union

# Global log directory
LOG_DIR = Path(os.getenv("LOG_DIR", "crawl_first/logs"))


class LogCapture:
    """Capture stdout/stderr and redirect to logger."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int = logging.INFO,
        prefix: str = "[STDOUT]",
    ):
        self.logger = logger
        self.level = level
        self.prefix = prefix

    def write(self, data: Union[str, bytes]) -> int:
        """Write data to logger."""
        # Handle str, bytes, and other types that might be passed to write()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        elif isinstance(data, str):
            pass  # Already a string, no conversion needed
        else:
            # Convert other types (int, float, etc.) to string
            data = str(data)

        stripped = data.strip() if data else ""
        if stripped:
            # Remove any trailing newlines and log each line separately
            lines = data.rstrip("\n\r").split("\n")
            for line in lines:
                line_stripped = line.strip()
                if line_stripped:  # Only log non-empty lines
                    self.logger.log(self.level, f"{self.prefix} {line_stripped}")

        return len(data)

    def flush(self) -> None:
        """Flush - required for file-like interface."""
        pass

    def fileno(self) -> int:
        """Return file descriptor - not supported."""
        raise OSError("fileno not supported")

    def isatty(self) -> bool:
        """Return whether this is a tty."""
        return False


class OutputManager:
    """Manage stdout/stderr capture and restoration."""

    def __init__(self) -> None:
        self.original_stdout: Optional[TextIO] = None
        self.original_stderr: Optional[TextIO] = None

    def store_originals(self) -> None:
        """Store original stdout/stderr."""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def restore_originals(self) -> None:
        """Restore original stdout/stderr."""
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
        if self.original_stderr is not None:
            sys.stderr = self.original_stderr


class OutputCapture:
    """Context manager for capturing stdout/stderr output to logger."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.output_manager = OutputManager()
        self.active = False

    def __enter__(self) -> "OutputCapture":
        """Enter context - start capturing output."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - restore original output."""
        self.stop()

    def start(self) -> None:
        """Start capturing output."""
        if not self.active:
            self.output_manager.store_originals()

            # Create log capture objects
            stdout_capture = LogCapture(self.logger, logging.INFO, "[STDOUT]")
            stderr_capture = LogCapture(self.logger, logging.WARNING, "[STDERR]")

            # Replace stdout/stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            self.active = True

    def stop(self) -> None:
        """Stop capturing output and restore originals."""
        if self.active:
            self.output_manager.restore_originals()
            self.active = False


def should_disable_output_capture() -> bool:
    """Check if output capture should be disabled.

    Can be overridden by setting CRAWL_FIRST_FORCE_OUTPUT_CAPTURE=true to enable
    output capture even in pytest environment, or CRAWL_FIRST_DISABLE_OUTPUT_CAPTURE=true
    to disable output capture in any environment.
    """
    # Allow environment variable override
    force_capture = os.getenv("CRAWL_FIRST_FORCE_OUTPUT_CAPTURE", "").lower() == "true"
    if force_capture:
        return False  # Pretend we're not in pytest to enable capture

    disable_capture = (
        os.getenv("CRAWL_FIRST_DISABLE_OUTPUT_CAPTURE", "").lower() == "true"
    )
    if disable_capture:
        return True  # Pretend we're in pytest to disable capture

    # Default behavior: detect pytest environment
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def setup_logging(
    verbose: bool = False, capture_output: bool = True
) -> Tuple[logging.Logger, Optional[OutputCapture]]:
    """Setup logging configuration with file and console output.

    Returns:
        Tuple of (logger, output_capture) where output_capture is None if not enabled
    """
    # Create logs directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Configure root logger to capture all library logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers from root
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create our main logger
    logger = logging.getLogger("crawl_first")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters with UTC timezone
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    detailed_formatter.converter = time.gmtime  # Use UTC for log entries
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler - captures EVERYTHING (our logs + library logs + stdout)
    current_time = datetime.now(timezone.utc)
    log_file = LOG_DIR / f"crawl_first_{current_time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Add file handler to root logger to capture all library logs
    root_logger.addHandler(file_handler)

    # Console handler - only for our main app logs
    console_handler = logging.StreamHandler()
    if verbose:
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(detailed_formatter)
    else:
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)

    logger.addHandler(console_handler)

    # Configure common library loggers to be more verbose in files
    library_loggers = [
        "requests",
        "urllib3",
        "geopy",
        "artl_mcp",
        "nmdc_mcp",
        "ols_mcp",
        "weather_mcp",
        "landuse_mcp",
    ]

    for lib_name in library_loggers:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Capture stdout/stderr if requested and not in testing
    output_capture = None
    if capture_output and not should_disable_output_capture():
        output_capture = OutputCapture(logger)
        output_capture.start()

    # Log the setup
    logger.info(f"Logging initialized. Log file: {log_file}")
    if capture_output:
        logger.info("Output capture enabled - all stdout/stderr will be logged")

    return logger, output_capture
