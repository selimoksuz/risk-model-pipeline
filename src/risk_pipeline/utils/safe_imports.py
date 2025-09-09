"""
Safe import utilities to handle missing dependencies gracefully
"""

import warnings
import sys
from typing import Any, Optional, Callable
from functools import wraps


class OptionalDependency:
    """Wrapper for optional dependencies that might not be installed"""

    def __init__(self, module_name: str, package_name: Optional[str] = None):
        self.module_name = module_name
        self.package_name = package_name or module_name
        self._module = None
        self._available = None

    @property
    def available(self) -> bool:
        """Check if module is available"""
        if self._available is None:
            try:
                import importlib
                self._module = importlib.import_module(self.module_name)
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    @property
    def module(self) -> Any:
        """Get the module if available, otherwise return a dummy object"""
        if self.available:
            return self._module
        else:
            return DummyModule(self.module_name, self.package_name)

    def require(self):
        """Raise error if module is not available"""
        if not self.available:
            raise ImportError(
                f"'{self.module_name}' is required but not installed. "
                f"Install it with: pip install {self.package_name}"
            )
        return self._module


class DummyModule:
    """Dummy module that raises helpful errors when accessed"""

    def __init__(self, module_name: str, package_name: str):
        self.module_name = module_name
        self.package_name = package_name

    def __getattr__(self, name):
        raise ImportError(
            f"Cannot use '{name}' from '{self.module_name}'. "
            f"The module is not installed. Install with: pip install {self.package_name}"
        )

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"Cannot call '{self.module_name}'. "
            f"The module is not installed. Install with: pip install {self.package_name}"
        )


def optional_import(module_name: str, package_name: Optional[str] = None):
    """
    Safely import an optional module

    Usage:
        matplotlib = optional_import('matplotlib')
        if matplotlib.available:
            plt = matplotlib.module.pyplot
        else:
            print("Matplotlib not available, skipping plots")
    """
    return OptionalDependency(module_name, package_name)


def requires(*dependencies: str):
    """
    Decorator to mark functions that require specific dependencies

    Usage:
        @requires('matplotlib', 'seaborn')
        def create_plot():
            import matplotlib.pyplot as plt
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            missing = []
            for dep in dependencies:
                opt_dep = OptionalDependency(dep)
                if not opt_dep.available:
                    missing.append(dep)

            if missing:
                raise ImportError(
                    f"Function '{func.__name__}' requires: {', '.join(missing)}. "
                    f"Install with: pip install {' '.join(missing)}"
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Pre-check common optional dependencies
OPTIONAL_DEPS = {
    'matplotlib': OptionalDependency('matplotlib'),
    'seaborn': OptionalDependency('seaborn'),
    'shap': OptionalDependency('shap'),
    'optuna': OptionalDependency('optuna'),
    'plotly': OptionalDependency('plotly'),
}


def check_dependencies(verbose: bool = False) -> dict:
    """Check which optional dependencies are available"""
    status = {}
    for name, dep in OPTIONAL_DEPS.items():
        status[name] = dep.available
        if verbose:
            if dep.available:
                print(f"✓ {name} is available")
            else:
                print(f"✗ {name} is not installed (optional)")
    return status


def safe_matplotlib_import():
    """Safely import matplotlib with fallback"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        return plt, True
    except ImportError:
        warnings.warn("Matplotlib not available. Plots will be skipped.", UserWarning)
        return None, False
    except Exception as e:
        warnings.warn(f"Matplotlib import failed: {e}. Plots will be skipped.", UserWarning)
        return None, False


def safe_seaborn_import():
    """Safely import seaborn with fallback"""
    try:
        import seaborn as sns
        return sns, True
    except ImportError:
        warnings.warn("Seaborn not available. Advanced plots will be skipped.", UserWarning)
        return None, False
    except Exception as e:
        warnings.warn(f"Seaborn import failed: {e}. Advanced plots will be skipped.", UserWarning)
        return None, False


def safe_shap_import():
    """Safely import SHAP with fallback"""
    try:
        import shap
        return shap, True
    except ImportError:
        warnings.warn("SHAP not available. SHAP analysis will be skipped.", UserWarning)
        return None, False
    except Exception as e:
        warnings.warn(f"SHAP import failed: {e}. SHAP analysis will be skipped.", UserWarning)
        return None, False
