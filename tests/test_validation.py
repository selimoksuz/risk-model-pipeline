"""Tests for validation module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from risk_pipeline.utils.validation import (
    InputValidator,
    ValidationError,
    SecurityLogger
)


class TestInputValidator:
    """Test input validation functionality."""

    def test_validate_file_path_valid(self):
        """Test valid file path validation."""
        tf = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        try:
            tf.write(b'col1, col2\n1, 2\n')
            tf.flush()
            tf.close()  # Close before validation to avoid Windows file lock

            path = InputValidator.validate_file_path(tf.name)
            assert path.exists()
            assert path.suffix == '.csv'
        finally:
            try:
                os.unlink(tf.name)
            except (OSError, PermissionError):
                pass  # Ignore file deletion errors on Windows

    def test_validate_file_path_invalid_extension(self):
        """Test rejection of invalid file extensions."""
        tf = tempfile.NamedTemporaryFile(suffix='.exe', delete=False)
        try:
            tf.close()  # Close to avoid Windows file lock
            with pytest.raises(ValidationError, match="Invalid file type"):
                InputValidator.validate_file_path(tf.name)
        finally:
            try:
                os.unlink(tf.name)
            except (OSError, PermissionError):
                pass  # Ignore file deletion errors on Windows

    def test_validate_file_path_too_large(self):
        """Test rejection of files that are too large."""
        tf = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        try:
            # Write large file (simulate)
            tf.write(b'x' * (1001 * 1024 * 1024))  # 1001 MB
            tf.flush()
            tf.close()  # Close to avoid Windows file lock

            # Temporarily reduce max size for testing
            original_max = InputValidator.MAX_FILE_SIZE_MB
            InputValidator.MAX_FILE_SIZE_MB = 1

            try:
                with pytest.raises(ValidationError, match="File too large"):
                    InputValidator.validate_file_path(tf.name)
            finally:
                InputValidator.MAX_FILE_SIZE_MB = original_max
        finally:
            try:
                os.unlink(tf.name)
            except (OSError, PermissionError):
                pass  # Ignore file deletion errors on Windows

    def test_validate_dataframe_empty(self):
        """Test rejection of empty DataFrames."""
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="DataFrame is empty"):
            InputValidator.validate_dataframe(df)

    def test_validate_dataframe_missing_columns(self):
        """Test detection of missing required columns."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'd': [5, 6]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            InputValidator.validate_dataframe(df, required_columns=['a', 'b', 'c'])

    def test_validate_dataframe_invalid_target(self):
        """Test validation of target column values."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 2]  # Invalid: contains 2
        })
        with pytest.raises(ValidationError, match="must contain only 0 and 1"):
            InputValidator.validate_dataframe(df, target_col='target')

    def test_validate_dataframe_imbalanced_target(self, capsys):
        """Test warning for imbalanced target."""
        df = pd.DataFrame({
            'feature1': range(1000),
            'feature2': range(1000, 2000),
            'target': [0] * 995 + [1] * 5  # Highly imbalanced
        })
        InputValidator.validate_dataframe(df, target_col='target')
        captured = capsys.readouterr()
        assert "WARNING: Highly imbalanced target" in captured.out

    def test_validate_dataframe_sql_injection(self):
        """Test detection of SQL injection attempts in column names."""
        df = pd.DataFrame({
            'normal_col': [1, 2],
            'another_col': [5, 6],
            'DROP TABLE users': [3, 4]  # Suspicious column name
        })
        with pytest.raises(ValidationError, match="Suspicious column name"):
            InputValidator.validate_dataframe(df)

    def test_sanitize_string(self):
        """Test string sanitization."""
        # Test removal of dangerous characters
        dangerous = "Hello`$\\\"'\n\r\tWorld"
        sanitized = InputValidator.sanitize_string(dangerous)
        assert sanitized == "HelloWorld"

        # Test truncation
        long_string = "x" * 300
        sanitized = InputValidator.sanitize_string(long_string, max_length=10)
        assert len(sanitized) == 10

        # Test null byte removal
        with_null = "Hello\x00World"
        sanitized = InputValidator.sanitize_string(with_null)
        assert "\x00" not in sanitized

    def test_validate_config_numeric_ranges(self):
        """Test validation of numeric configuration parameters."""
        # Valid config
        config = {
            'oot_window_months': 6,
            'cv_folds': 5,
            'psi_threshold': 0.5
        }
        validated = InputValidator.validate_config(config)
        assert validated == config

        # Invalid range
        config_invalid = {'cv_folds': 15}  # Too high
        with pytest.raises(ValidationError, match="must be between"):
            InputValidator.validate_config(config_invalid)

    def test_validate_config_string_values(self):
        """Test validation of string configuration parameters."""
        # Valid
        config = {'calibration_method': 'isotonic'}
        validated = InputValidator.validate_config(config)
        assert validated == config

        # Invalid
        config_invalid = {'calibration_method': 'invalid_method'}
        with pytest.raises(ValidationError, match="must be one of"):
            InputValidator.validate_config(config_invalid)

    def test_hash_dataframe(self):
        """Test DataFrame hashing for integrity."""
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df3 = pd.DataFrame({'a': [1, 2], 'b': [3, 5]})  # Different

        hash1 = InputValidator.hash_dataframe(df1)
        hash2 = InputValidator.hash_dataframe(df2)
        hash3 = InputValidator.hash_dataframe(df3)

        assert hash1 == hash2  # Same data should have same hash
        assert hash1 != hash3  # Different data should have different hash
        assert len(hash1) == 64  # SHA256 produces 64 character hex string


class TestSecurityLogger:
    """Test security logging functionality."""

    def test_log_event(self):
        """Test logging of security events."""
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        try:
            tf.close()  # Close to avoid Windows file lock
            logger = SecurityLogger(tf.name)
            logger.log_event('test_event', {'key': 'value'})

            # Read and verify log
            with open(tf.name, 'r') as f:
                content = f.read()
                assert 'test_event' in content
                assert 'key' in content
                assert 'value' in content
        finally:
            try:
                os.unlink(tf.name)
            except (OSError, PermissionError):
                pass  # Ignore file deletion errors on Windows

    def test_log_access(self):
        """Test file access logging."""
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        try:
            tf.close()  # Close to avoid Windows file lock
            logger = SecurityLogger(tf.name)
            logger.log_access('/path/to/file.csv', user='test_user')

            with open(tf.name, 'r') as f:
                content = f.read()
                assert 'file_access' in content
                assert '/path/to/file.csv' in content
                assert 'test_user' in content
        finally:
            try:
                os.unlink(tf.name)
            except (OSError, PermissionError):
                pass  # Ignore file deletion errors on Windows

    def test_log_validation_failure(self):
        """Test validation failure logging."""
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        try:
            tf.close()  # Close to avoid Windows file lock
            logger = SecurityLogger(tf.name)
            logger.log_validation_failure('Invalid data', data_hash='abc123')

            with open(tf.name, 'r') as f:
                content = f.read()
                assert 'validation_failure' in content
                assert 'Invalid data' in content
                assert 'abc123' in content
        finally:
            try:
                os.unlink(tf.name)
            except (OSError, PermissionError):
                pass  # Ignore file deletion errors on Windows
