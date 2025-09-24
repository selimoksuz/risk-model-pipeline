"""Tests for error handling module."""

import tempfile
import time

import pytest

from risk_pipeline.utils.error_handler import (
    CheckpointManager,
    CriticalError,
    ErrorHandler,
    RecoverableError,
    handle_memory_error,
)


class TestErrorHandler:
    """Test error handling functionality."""

    def test_safe_execute_success(self):
        """Test successful execution with safe_execute."""
        handler = ErrorHandler()

        def good_function(x, y):
            return x + y

        result = handler.safe_execute(good_function, 2, 3)
        assert result == 5

    def test_safe_execute_recoverable_error(self):
        """Test handling of recoverable errors."""
        handler = ErrorHandler()

        def failing_function():
            raise MemoryError("Out of memory")

        with pytest.raises(RecoverableError):
            handler.safe_execute(failing_function)

        # Check error was logged
        assert len(handler.error_log) == 1
        assert handler.error_log[0]['error_type'] == 'MemoryError'

    def test_safe_execute_critical_error(self):
        """Test handling of critical errors."""
        handler = ErrorHandler()

        def failing_function():
            raise ValueError("Invalid value")

        with pytest.raises(CriticalError):
            handler.safe_execute(failing_function)

    def test_is_recoverable(self):
        """Test error classification."""
        handler = ErrorHandler()

        # Recoverable errors
        assert handler.is_recoverable(MemoryError())
        assert handler.is_recoverable(TimeoutError())
        assert handler.is_recoverable(ConnectionError())

        # Non-recoverable errors
        assert not handler.is_recoverable(ValueError())
        assert not handler.is_recoverable(TypeError())
        assert not handler.is_recoverable(KeyError())

    def test_retry_on_failure(self):
        """Test retry decorator."""
        handler = ErrorHandler()

        # Counter to track attempts
        attempts = {'count': 0}

        @handler.retry_on_failure(max_retries=3, delay=0.01)
        def flaky_function():
            attempts['count'] += 1
            if attempts['count'] < 3:
                raise ConnectionError("Network error")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempts['count'] == 3

    def test_retry_on_failure_exhausted(self):
        """Test retry decorator when all retries are exhausted."""
        handler = ErrorHandler()

        @handler.retry_on_failure(max_retries=2, delay=0.01)
        def always_fails():
            raise ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            always_fails()


class TestCheckpointManager:
    """Test checkpoint management functionality."""

    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir, run_id="test_run")

            # Save checkpoint
            data = {'df': [1, 2, 3], 'model': 'test_model'}
            artifacts = {'stage_complete': True}
            manager.save_checkpoint('stage_1', data, artifacts)

            # Load checkpoint
            checkpoint = manager.load_checkpoint()
            assert checkpoint is not None
            assert checkpoint['metadata']['stage'] == 'stage_1'
            assert checkpoint['data'] == data
            assert checkpoint['metadata']['artifacts'] == artifacts

    def test_load_nonexistent_checkpoint(self):
        """Test loading when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            checkpoint = manager.load_checkpoint()
            assert checkpoint is None

    def test_clear_checkpoints(self):
        """Test clearing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir, run_id="test_run")

            # Save checkpoint
            manager.save_checkpoint('stage_1', {'data': 'test'}, {})

            # Verify files exist
            assert manager.checkpoint_file.exists()
            assert manager.metadata_file.exists()

            # Clear checkpoints
            manager.clear_checkpoints()

            # Verify files are removed
            assert not manager.checkpoint_file.exists()
            assert not manager.metadata_file.exists()

    def test_list_checkpoints(self):
        """Test listing available checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple checkpoints
            for i in range(3):
                manager = CheckpointManager(
                    checkpoint_dir=tmpdir,
                    run_id=f"run_{i}"
                )
                manager.save_checkpoint(f'stage_{i}', {'data': i}, {})
                time.sleep(0.01)  # Ensure different timestamps

            # List checkpoints
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            checkpoints = manager.list_checkpoints()

            assert len(checkpoints) == 3
            # Should be sorted by timestamp (newest first)
            assert checkpoints[0]['stage'] == 'stage_2'
            assert checkpoints[1]['stage'] == 'stage_1'
            assert checkpoints[2]['stage'] == 'stage_0'


class TestMemoryErrorHandler:
    """Test memory error handling."""

    def test_handle_memory_error_success(self):
        """Test successful execution with memory monitoring."""
        @handle_memory_error
        def memory_safe_function(x):
            return x * 2

        result = memory_safe_function(5)
        assert result == 10

    def test_handle_memory_error_recovery(self):
        """Test memory error recovery."""
        call_count = {'count': 0}

        @handle_memory_error
        def memory_intensive_function():
            call_count['count'] += 1
            if call_count['count'] == 1:
                raise MemoryError("Out of memory")
            return "recovered"

        # Should retry once after MemoryError
        result = memory_intensive_function()
        assert result == "recovered"
        assert call_count['count'] == 2

    def test_handle_memory_error_failure(self):
        """Test unrecoverable memory error."""
        @handle_memory_error
        def always_fails():
            raise MemoryError("Out of memory")

        with pytest.raises(CriticalError):
            always_fails()


class TestIntegration:
    """Integration tests for error handling."""

    def test_pipeline_with_checkpoints(self):
        """Test simulated pipeline with checkpoint recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ErrorHandler(checkpoint_dir=tmpdir)
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)

            # Simulate pipeline stages
            stages_completed = []

            def stage_1():
                stages_completed.append('stage_1')
                return {'data': 'stage_1_output'}

            def stage_2():
                stages_completed.append('stage_2')
                raise MemoryError("Simulated failure")

            def stage_3():
                stages_completed.append('stage_3')
                return {'data': 'stage_3_output'}

            # Run stage 1
            result1 = handler.safe_execute(stage_1)
            checkpoint_mgr.save_checkpoint('stage_1', result1, {})

            # Stage 2 fails
            with pytest.raises(RecoverableError):
                handler.safe_execute(stage_2)

            # Load checkpoint and continue from stage 3
            checkpoint = checkpoint_mgr.load_checkpoint()
            assert checkpoint['metadata']['stage'] == 'stage_1'

            # Continue with stage 3
            handler.safe_execute(stage_3)

            assert stages_completed == ['stage_1', 'stage_2', 'stage_3']

