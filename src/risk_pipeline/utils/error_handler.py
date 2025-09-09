"""Error handling and recovery utilities."""

import os
import json
import pickle
import traceback
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
import functools


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class RecoverableError(PipelineError):
    """Error that allows pipeline to recover from checkpoint."""
    pass


class CriticalError(PipelineError):
    """Critical error that requires pipeline restart."""
    pass


class ErrorHandler:
    """Handles errors and implements recovery mechanisms."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.error_log = []

    def safe_execute(self, func, *args, **kwargs):
        """Execute function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log_error(e, func.__name__)
            if self.is_recoverable(e):
                raise RecoverableError(f"Recoverable error in {func.__name__}: {str(e)}")
            else:
                raise CriticalError(f"Critical error in {func.__name__}: {str(e)}")

    def is_recoverable(self, error: Exception) -> bool:
        """Determine if error is recoverable."""
        recoverable_types = (
            MemoryError,
            TimeoutError,
            ConnectionError,
            OSError
        )
        return isinstance(error, recoverable_types)

    def log_error(self, error: Exception, context: str):
        """Log error details."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        self.error_log.append(error_entry)

        # Write to error log file
        log_file = self.checkpoint_dir / "error_log.json"
        with open(log_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')

    def retry_on_failure(self, max_retries: int = 3, delay: float = 1.0):
        """Decorator for retrying failed operations."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import time

                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff

                return None
            return wrapper
        return decorator


class CheckpointManager:
    """Manages pipeline checkpoints for recovery."""

    def __init__(self, checkpoint_dir: str = "checkpoints", run_id: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.run_id}.pkl"
        self.metadata_file = self.checkpoint_dir / f"metadata_{self.run_id}.json"

    def save_checkpoint(self, stage: str, data: Dict[str, Any], artifacts: Dict[str, Any]):
        """Save checkpoint for current stage."""
        checkpoint = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'data_keys': list(data.keys()),
            'artifacts': artifacts
        }

        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Save actual data
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ Checkpoint saved: {stage}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint."""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            print(f"✓ Checkpoint loaded: {metadata['stage']} ({metadata['timestamp']})")
            return {
                'metadata': metadata,
                'data': data
            }
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoints(self):
        """Clear all checkpoints for current run."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = []
        for metadata_file in self.checkpoint_dir.glob("metadata_*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                checkpoints.append({
                    'file': metadata_file.name,
                    'stage': metadata['stage'],
                    'timestamp': metadata['timestamp']
                })
            except Exception:
                continue

        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)


def handle_memory_error(func):
    """Decorator to handle memory errors gracefully."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import gc
        import psutil

        try:
            # Check memory before execution
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                print(f"WARNING: High memory usage ({mem.percent}%). Running garbage collection...")
                gc.collect()

            result = func(*args, **kwargs)

            # Clean up after execution
            gc.collect()
            return result

        except MemoryError as e:
            print(f"Memory error in {func.__name__}. Attempting recovery...")

            # Force garbage collection
            gc.collect()

            # Try to free memory
            import sys
            for name in list(sys.modules.keys()):
                if name.startswith('_'):
                    continue
                module = sys.modules[name]
                if hasattr(module, '_cache'):
                    module._cache.clear()

            # Retry once
            try:
                return func(*args, **kwargs)
            except Exception:
                raise CriticalError(f"Memory error in {func.__name__} could not be recovered")

    return wrapper
