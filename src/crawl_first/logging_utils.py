"""
Logging utilities for crawl-first.

Handles output capture, logging configuration, file management, performance timing,
and enhanced error logging with contextual information.
"""

import functools
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, TypeVar, Union

import psutil

# Type variables for decorator
F = TypeVar("F", bound=Callable[..., Any])

# Global log directory
LOG_DIR = Path(os.getenv("LOG_DIR", Path(__file__).resolve().parent / "logs"))

# Constants for consistent string handling
CACHE_KEY_SLICE_LENGTH = 50
ARG_STRING_MAX_LENGTH = 100
LARGE_OBJECT_SIZE_THRESHOLD = 1024


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

    def _convert_data_to_str(self, data: Union[str, bytes, Any]) -> str:
        """Convert input data to a string."""
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        elif isinstance(data, str):
            return data  # Already a string, no conversion needed
        else:
            # Convert other types (int, float, etc.) to string
            return str(data)

    def write(self, data: Union[str, bytes]) -> int:
        """Write data to logger."""
        # Convert data to string using the private method
        data = self._convert_data_to_str(data)

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


# Global performance metrics storage
_performance_metrics: Dict[str, list] = defaultdict(list)
_cache_metrics: Dict[str, Dict[str, int]] = defaultdict(
    lambda: {"hits": 0, "misses": 0}
)


class PerformanceTimer:
    """Context manager and decorator for timing operations with detailed logging."""

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        context: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
    ):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger("crawl_first.performance")
        self.context = context or {}
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self) -> "PerformanceTimer":
        """Start timing the operation."""
        self.start_time = time.perf_counter()
        context_str = f" {self.context}" if self.context else ""
        self.logger.log(self.log_level, f"Starting {self.operation_name}{context_str}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and log results."""
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time
        else:
            self.duration = None
            self.logger.warning(
                f"{self.operation_name} timer was not started properly. Duration is undefined."
            )

        # Store metrics for analysis (only if duration is valid)
        if self.duration is not None:
            _performance_metrics[self.operation_name].append(self.duration)

        # Determine log level based on duration and success
        status = "FAILED" if exc_type is not None else "completed"
        context_str = f" {self.context}" if self.context else ""

        # Use different log levels based on duration (slow operations get warnings)
        log_level = self.log_level
        if self.duration is not None:
            if self.duration > 30:  # Very slow
                log_level = logging.WARNING
            elif self.duration > 10:  # Slow
                log_level = logging.INFO

            self.logger.log(
                log_level,
                f"{self.operation_name} {status} in {self.duration:.3f}s{context_str}",
            )
        else:
            self.logger.log(
                logging.WARNING,
                f"{self.operation_name} {status} with invalid timing{context_str}",
            )


def timed_operation(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.INFO,
    include_args: bool = False,
) -> Callable[[F], F]:
    """Decorator to time function execution with enhanced logging.

    Args:
        operation_name: Name of the operation for logging
        logger: Logger instance to use (defaults to performance logger)
        log_level: Logging level for timing messages
        include_args: Whether to include function arguments in context
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract context from function arguments if requested
            context = {}
            if include_args and args:
                # Include first few args and relevant kwargs with safe string representation
                if len(args) > 0:
                    try:
                        # Check type and size before converting to string for performance
                        if (
                            isinstance(args[0], (int, float, str, bool))
                            or sys.getsizeof(args[0]) < LARGE_OBJECT_SIZE_THRESHOLD
                        ):
                            arg_str = str(args[0])
                            # Truncate if too long
                            if len(arg_str) > ARG_STRING_MAX_LENGTH:
                                arg_str = arg_str[: ARG_STRING_MAX_LENGTH - 3] + "..."
                            context["arg1"] = arg_str
                        else:
                            context["arg1"] = "<large_object>"
                    except Exception as e:
                        # Get logger for debugging, fallback to performance logger
                        debug_logger = logger or logging.getLogger(
                            "crawl_first.performance"
                        )
                        debug_logger.debug(
                            f"Failed to process arg1: {type(e).__name__}: {e}",
                            exc_info=True,
                        )
                        context["arg1"] = "<repr_failed>"

                if len(args) > 1:
                    try:
                        # Check type and size before converting to string for performance
                        if (
                            isinstance(args[1], (int, float, str, bool))
                            or sys.getsizeof(args[1]) < LARGE_OBJECT_SIZE_THRESHOLD
                        ):
                            arg_str = str(args[1])
                            if len(arg_str) > ARG_STRING_MAX_LENGTH:
                                arg_str = arg_str[: ARG_STRING_MAX_LENGTH - 3] + "..."
                            context["arg2"] = arg_str
                        else:
                            context["arg2"] = "<large_object>"
                    except Exception as e:
                        # Get logger for debugging, fallback to performance logger
                        debug_logger = logger or logging.getLogger(
                            "crawl_first.performance"
                        )
                        debug_logger.debug(
                            f"Failed to process arg2: {type(e).__name__}: {e}",
                            exc_info=True,
                        )
                        context["arg2"] = "<repr_failed>"

                # Include important kwargs with safe handling
                important_kwargs = ["biosample_id", "doi", "lat", "lon", "email"]
                for key in important_kwargs:
                    if key in kwargs:
                        try:
                            # Apply same size check for kwargs
                            if (
                                isinstance(kwargs[key], (int, float, str, bool))
                                or sys.getsizeof(kwargs[key])
                                < LARGE_OBJECT_SIZE_THRESHOLD
                            ):
                                value_str = str(kwargs[key])
                                if len(value_str) > ARG_STRING_MAX_LENGTH:
                                    value_str = (
                                        value_str[: ARG_STRING_MAX_LENGTH - 3] + "..."
                                    )
                                context[key] = value_str
                            else:
                                context[key] = "<large_object>"
                        except Exception as e:
                            # Get logger for debugging, fallback to performance logger
                            debug_logger = logger or logging.getLogger(
                                "crawl_first.performance"
                            )
                            debug_logger.debug(
                                f"Failed to process key '{key}' in kwargs: {type(e).__name__}: {e}",
                                exc_info=True,
                            )
                            context[key] = "<repr_failed>"

            with PerformanceTimer(operation_name, logger, context, log_level):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def log_cache_operation(
    cache_name: str, operation: str, key: str, logger: Optional[logging.Logger] = None
) -> None:
    """Log cache operations and update metrics.

    Args:
        cache_name: Name of the cache (e.g., 'nmdc_entity', 'full_text')
        operation: 'hit' or 'miss'
        key: Cache key for debugging
        logger: Logger instance
    """
    logger = logger or logging.getLogger("crawl_first.cache")

    # Update metrics
    if operation == "hit":
        _cache_metrics[cache_name]["hits"] += 1
    elif operation == "miss":
        _cache_metrics[cache_name]["misses"] += 1

    # Log cache operation
    total_ops = (
        _cache_metrics[cache_name]["hits"] + _cache_metrics[cache_name]["misses"]
    )
    hit_rate = _cache_metrics[cache_name]["hits"] / max(total_ops, 1) * 100

    # Safe key slicing to avoid IndexError
    key_display = (
        key[:CACHE_KEY_SLICE_LENGTH] if len(key) > CACHE_KEY_SLICE_LENGTH else key
    )
    logger.debug(
        f"Cache {operation}: {cache_name} (key: {key_display}...) "
        f"hit_rate: {hit_rate:.1f}% ({_cache_metrics[cache_name]['hits']}/{total_ops})"
    )


def log_enhanced_error(
    logger: logging.Logger,
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    level: int = logging.ERROR,
) -> None:
    """Log errors with enhanced contextual information.

    Args:
        logger: Logger instance
        error: Exception that occurred
        operation: Description of what operation failed
        context: Dictionary of contextual information (biosample_id, coordinates, etc.)
        level: Logging level to use
    """
    context = context or {}
    context_str = " | ".join([f"{k}: {v}" for k, v in context.items()])
    error_type = type(error).__name__

    logger.log(level, f"{operation} failed: {error_type}: {error}")
    if context_str:
        logger.log(level, f"Context: {context_str}")


def log_data_quality_issue(
    logger: logging.Logger,
    issue_type: str,
    expected: Any,
    actual: Any,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log data quality issues with context.

    Args:
        logger: Logger instance
        issue_type: Type of issue (e.g., 'missing_field', 'invalid_format', 'unexpected_value')
        expected: What was expected
        actual: What was actually found
        context: Contextual information
    """
    context = context or {}
    context_str = " | ".join([f"{k}: {v}" for k, v in context.items()])

    logger.warning(
        f"Data quality issue ({issue_type}): expected {expected}, got {actual}"
    )
    if context_str:
        logger.warning(f"Context: {context_str}")


def log_performance_summary(logger: Optional[logging.Logger] = None) -> None:
    """Log a summary of performance metrics collected during execution."""
    logger = logger or logging.getLogger("crawl_first.performance")

    if not _performance_metrics:
        logger.info("No performance metrics collected")
        return

    logger.info("=== Performance Summary ===")

    # Log timing statistics
    for operation, times in _performance_metrics.items():
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)

            logger.info(
                f"{operation}: {len(times)} calls, "
                f"avg: {avg_time:.3f}s, min: {min_time:.3f}s, max: {max_time:.3f}s, "
                f"total: {total_time:.3f}s"
            )

    # Log cache statistics
    logger.info("=== Cache Performance ===")
    for cache_name, metrics in _cache_metrics.items():
        total = metrics["hits"] + metrics["misses"]
        if total > 0:
            hit_rate = metrics["hits"] / total * 100
            logger.info(
                f"{cache_name}: {hit_rate:.1f}% hit rate "
                f"({metrics['hits']} hits, {metrics['misses']} misses)"
            )


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        "percent": process.memory_percent(),  # Percentage of system memory
    }


def log_memory_usage(logger: logging.Logger, operation: str) -> None:
    """Log current memory usage for debugging memory-intensive operations."""
    try:
        usage = get_memory_usage()
        logger.debug(
            f"Memory usage after {operation}: "
            f"RSS: {usage['rss_mb']:.1f}MB, "
            f"VMS: {usage['vms_mb']:.1f}MB, "
            f"System %: {usage['percent']:.1f}%"
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
        # Catching specific psutil exceptions because memory monitoring is non-critical.
        logger.debug(
            f"Could not measure memory usage after {operation}. Exception: {e}",
            exc_info=True,
        )
    except Exception as e:
        # Catch any other unexpected exceptions
        logger.warning(
            f"Unexpected error during memory monitoring after {operation}: {type(e).__name__}: {e}",
            exc_info=True,
        )


def log_api_call_result(
    logger: logging.Logger,
    api_name: str,
    endpoint: str,
    response_code: Optional[int] = None,
    response_size: Optional[int] = None,
    duration: Optional[float] = None,
    success: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log API call results with detailed information.

    Args:
        logger: Logger instance
        api_name: Name of the API service (e.g., 'NMDC', 'Unpaywall', 'OpenStreetMap')
        endpoint: API endpoint or operation name
        response_code: HTTP response code if applicable
        response_size: Size of response in bytes if applicable
        duration: Duration of the call in seconds
        success: Whether the call was successful
        context: Additional context (coordinates, IDs, etc.)
    """
    context = context or {}
    status = "SUCCESS" if success else "FAILED"

    parts = [f"{api_name} API call {status}: {endpoint}"]

    if response_code is not None:
        parts.append(f"HTTP {response_code}")
    if duration is not None:
        parts.append(f"{duration:.3f}s")
    if response_size is not None:
        parts.append(f"{response_size} bytes")

    context_str = " | ".join([f"{k}: {v}" for k, v in context.items()])
    if context_str:
        parts.append(f"Context: {context_str}")

    level = logging.INFO if success else logging.WARNING
    logger.log(level, " | ".join(parts))
