"""
Setup configuration for risk-model-pipeline
Professional PyPI package configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from package
version = {}
with open("src/risk_pipeline/_version.py") as f:
    exec(f.read(), version)

setup(
    name="risk-pipeline",
    version=version["__version__"],
    author="Selim Oksuz",
    author_email="selimoksuz@users.noreply.github.com",
    description="Production-ready risk modeling pipeline with WOE transformation and advanced ML features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/selimoksuz/risk-model-pipeline",
    project_urls={
        "Bug Tracker": "https://github.com/selimoksuz/risk-model-pipeline/issues",
        "Documentation": "https://github.com/selimoksuz/risk-model-pipeline#readme",
        "Source Code": "https://github.com/selimoksuz/risk-model-pipeline",
    },
    
    # Package configuration
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8,<4.0",
    
    # Core dependencies
    install_requires=[
        "pandas>=1.5.0,<2.4.0",
        "numpy>=1.23.0,<1.27.0",
        "scipy>=1.7.0,<1.12.0",
        "scikit-learn>=1.1.0,<1.4.0",
        "joblib>=1.1.0",
        "openpyxl>=3.0.0",
        "xlsxwriter>=3.0.0",
        "matplotlib>=3.6.0,<3.9.0",
        "seaborn>=0.12.0",
        "statsmodels>=0.13.0",
        "typer>=0.9.0",
        "lightgbm>=3.3.0,<4.5.0",
        "catboost>=1.1.0,<2.0.0",
        "xgboost>=1.6.0,<2.0.0",
        "pygam>=0.9.0,<1.0.0",
        "optuna>=3.0.0,<4.0.0",
        "shap>=0.41.0,<0.44.0",
        "imbalanced-learn>=0.9.0,<0.12.0",
        "scikit-learn-extra>=0.2.0,<0.3.0",
        "xbooster>=0.0.12,<1.0.0",
        "pydantic>=1.10,<2.0",
        "nbformat>=5.8.0,<6.0.0",
        "importlib-resources>=5.0; python_version < \"3.9\"",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest==8.4.2",
            "pytest-cov==7.0.0",
            "black==24.8.0",
            "isort==5.13.2",
            "flake8==7.1.0",
            "mypy==1.11.1",
            "pre-commit==3.7.1",
            "twine==5.1.1",
            "wheel==0.43.0",
            "build==1.2.1",
        ],
        "viz": [
            "plotly==5.24.1",
        ],
        "ml": [
            "shap==0.43.0",
        ],
        "notebook": [
            "jupyter==1.1.0",
            "notebook==7.2.1",
            "ipywidgets==8.1.3",
        ],
    },
    
    # CLI entry points
    entry_points={
        "console_scripts": [
            "risk-pipeline=risk_pipeline.cli:main",
        ],
    },
    
    # Package classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "risk-modeling",
        "credit-scoring",
        "machine-learning",
        "woe-transformation",
        "financial-modeling",
        "scikit-learn",
        "data-science",
        "banking",
        "credit-risk",
    ],
)




