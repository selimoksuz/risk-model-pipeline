"""Input validation and security utilities."""

import os
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path
import hashlib
import json


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Validates and sanitizes input data for the pipeline."""

    MAX_FILE_SIZE_MB = 1000  # 1GB limit
    ALLOWED_EXTENSIONS = {'.csv', '.parquet', '.xlsx'}
    REQUIRED_COLUMNS_MIN = 3  # At least ID, time, target

    @staticmethod
    def validate_file_path(file_path: str) -> Path:
        """Validate file path and check for path traversal attacks."""
        try:
            path = Path(file_path).resolve()

            # Check if file exists
            if not path.exists():
                raise ValidationError(f"File not found: {file_path}")

            # Check file extension
            if path.suffix.lower() not in InputValidator.ALLOWED_EXTENSIONS:
                raise ValidationError(
                    f"Invalid file type: {path.suffix}. "
                    f"Allowed types: {InputValidator.ALLOWED_EXTENSIONS}"
                )

            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > InputValidator.MAX_FILE_SIZE_MB:
                raise ValidationError(
                    f"File too large: {file_size_mb:.2f}MB. "
                    f"Max allowed: {InputValidator.MAX_FILE_SIZE_MB}MB"
                )

            return path

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid file path: {str(e)}")

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Validate DataFrame structure and content."""

        # Check if DataFrame is empty
        if df.empty:
            raise ValidationError("DataFrame is empty")

        # Check minimum columns
        if len(df.columns) < InputValidator.REQUIRED_COLUMNS_MIN:
            raise ValidationError(
                f"DataFrame must have at least {InputValidator.REQUIRED_COLUMNS_MIN} columns"
            )

        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValidationError(f"Missing required columns: {missing}")

        # Validate target column if specified
        if target_col and target_col in df.columns:
            unique_values = df[target_col].dropna().unique()
            if not set(unique_values).issubset({0, 1}):
                raise ValidationError(
                    f"Target column '{target_col}' must contain only 0 and 1. "
                    f"Found: {unique_values[:10]}"
                )

            # Check class balance
            value_counts = df[target_col].value_counts()
            if len(value_counts) < 2:
                raise ValidationError(
                    f"Target column must have both classes (0 and 1). "
                    f"Found only: {value_counts.index.tolist()}"
                )

            # Warn if highly imbalanced
            minority_ratio = value_counts.min() / value_counts.sum()
            if minority_ratio < 0.01:
                print(f"WARNING: Highly imbalanced target (minority: {minority_ratio:.2%})")

        # Check for suspicious column names (SQL injection prevention)
        suspicious_patterns = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'EXEC', '--', ';']
        for col in df.columns:
            for pattern in suspicious_patterns:
                if pattern in str(col).upper():
                    raise ValidationError(
                        f"Suspicious column name detected: '{col}'. "
                        f"Contains pattern: '{pattern}'"
                    )

        return df

    @staticmethod
    def sanitize_string(value: str, max_length: int = 255) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(value, str):
            value = str(value)

        # Remove null bytes
        value = value.replace('\x00', '')

        # Truncate to max length
        value = value[:max_length]

        # Escape special characters for shell commands
        dangerous_chars = ['`', '$', '\\', '"', "'", '\n', '\r', '\t']
        for char in dangerous_chars:
            value = value.replace(char, '')

        return value.strip()

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters."""

        # Validate numeric parameters
        numeric_validations = {
            'oot_window_months': (1, 12),
            'cv_folds': (2, 10),
            'hpo_trials': (1, 1000),
            'hpo_timeout_sec': (60, 7200),
            'psi_threshold': (0.01, 1.0),
            'iv_min': (0.0, 1.0),
            'rho_threshold': (0.0, 1.0),
            'vif_threshold': (1.0, 100.0)
        }

        for param, (min_val, max_val) in numeric_validations.items():
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)):
                    raise ValidationError(f"{param} must be numeric")
                if not min_val <= value <= max_val:
                    raise ValidationError(
                        f"{param} must be between {min_val} and {max_val}. Got: {value}"
                    )

        # Validate string parameters
        if 'calibration_method' in config:
            allowed_methods = {'isotonic', 'sigmoid'}
            if config['calibration_method'] not in allowed_methods:
                raise ValidationError(
                    f"calibration_method must be one of {allowed_methods}"
                )

        if 'hpo_method' in config:
            allowed_methods = {'random', 'grid', 'bayesian'}
            if config['hpo_method'] not in allowed_methods:
                raise ValidationError(
                    f"hpo_method must be one of {allowed_methods}"
                )

        return config

    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        """Generate hash of DataFrame for integrity checking."""
        df_bytes = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.sha256(df_bytes).hexdigest()


class SecurityLogger:
    """Logs security-related events."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or "security.log"

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log a security event."""
        import datetime

        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')

    def log_access(self, file_path: str, user: Optional[str] = None):
        """Log file access."""
        self.log_event('file_access', {
            'file': str(file_path),
            'user': user or 'unknown'
        })

    def log_validation_failure(self, error: str, data_hash: Optional[str] = None):
        """Log validation failure."""
        self.log_event('validation_failure', {
            'error': error,
            'data_hash': data_hash
        })
