# This module exports a centralized logger and a @track_node decorator that instruments every node in your graph automatically

from functools import wraps
import os
import time
import logging
from typing import Any, Callable, Dict

os.makedirs("logs", exist_ok=True)

# Configure dual-output enterprise logger
logger = logging.getLogger("CRAG_Engine")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # 1.Rotating File Handler
    file_handler = logging.FileHandler("logs/crag_execution.log", encoding= "utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"        
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # 2.Console Stream Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

def track_node(node_name: str) -> Callable:
    """Decorator to monitor node execution time, state mutations, and exceptions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
            start_time = time.perf_counter()
            logger.info(f"START NODE: [{node_name}] | Keys in state: {list(state.keys())}")

            try:
                result_state_diff = func(state, *args, **kwargs)
                elapsed_time = time.perf_counter() - start_time

                # Log execution speed and updated keys
                logger.info(
                    f"FINISH NODE: [{node_name}] | Elapsed: {elapsed_time:.3f}s | "
                    f"Updated Keys: {list(result_state_diff.keys())}"
                )
                return result_state_diff
            
            except Exception as exc:
                elapsed_time = time.perf_counter() - start_time
                logger.error(
                    f"FAILED NODE: [{node_name}] | Elapsed: {elapsed_time:.3f}s | Error: {str(exc)}",
                    exc_info=True
                )
                raise exc
        return wrapper
    return  decorator
    