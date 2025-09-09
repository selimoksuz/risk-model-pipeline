"""
Pandas 2.x compatibility layer
"""
import pandas as pd
import numpy as np


def safe_string_strip_casefold(series):
    """
    Safely convert series to string and apply strip + casefold
    Compatible with both pandas 1.x and 2.x
    """
    if series is None or series.empty:
        return series

    # Convert to string type explicitly
    if hasattr(series, 'astype'):
        # For pandas 2.x, ensure we use string dtype
        if hasattr(pd, 'StringDtype'):
            str_series = series.astype(pd.StringDtype())
        else:
            str_series = series.astype(str)

        # Apply string methods
        result = str_series.str.strip()
        result = result.str.casefold()
        return result
    return series


def safe_to_numeric_downcast(series, downcast_type="integer"):
    """
    Safe wrapper for pd.to_numeric with downcast parameter
    Compatible with both pandas 1.x and 2.x
    """
    try:
        # Try with downcast (pandas 1.x)
        return pd.to_numeric(series, downcast=downcast_type)
    except (TypeError, ValueError):
        # Fallback for pandas 2.x where downcast is deprecated
        result = pd.to_numeric(series)
        if downcast_type == "integer":
            return result.astype('int64', errors='ignore')
        elif downcast_type == "float":
            return result.astype('float64', errors='ignore')
        return result


def safe_categorical(data, categories=None):
    """
    Safe wrapper for pd.Categorical
    Compatible with both pandas 1.x and 2.x
    """
    if categories is not None:
        # Ensure categories are unique and sorted
        categories = pd.Index(categories).unique()

    return pd.Categorical(data, categories=categories)


def safe_values_to_numpy(data):
    """
    Safe conversion to numpy array
    Compatible with both pandas 1.x and 2.x
    """
    if hasattr(data, 'to_numpy'):
        # pandas 2.x preferred method
        return data.to_numpy()
    else:
        # pandas 1.x fallback
        return data.values


def check_pandas_version():
    """Check pandas version and return major version"""
    import pandas as pd
    version_parts = pd.__version__.split('.')
    return int(version_parts[0])


# Set pandas 2.x compatible options if available
PANDAS_VERSION = check_pandas_version()

if PANDAS_VERSION >= 2:
    # Use nullable dtypes in pandas 2.x if available
    try:
        pd.options.mode.dtype_backend = 'numpy_nullable'
    except (AttributeError, pd._config.config.OptionError):
        # Option not available in this pandas version
        pass
