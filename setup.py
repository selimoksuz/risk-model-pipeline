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
    name="risk-model-pipeline",
    version=version["__version__"],
    author="Selim Öksüz",
    author_email="your.email@example.com",  # TODO: Update with your email
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
        "pandas>=1.3.0,<2.0.0",
        "numpy>=1.20.0,<1.25.0",
        "scikit-learn>=1.0.0,<1.3.0",
        "joblib>=1.0.0",
        "openpyxl>=3.0.0",
        "xlsxwriter>=3.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.0.0",
            "twine>=4.0.0",
            "wheel>=0.37.0",
            "build>=0.7.0",
        ],
        "viz": [
            "matplotlib>=3.5.0,<3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
        ],
        "ml": [
            "optuna>=3.0.0",
            "shap>=0.41.0",
            "imbalanced-learn>=0.9.0",
            "scikit-learn-extra>=0.2.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    
    # CLI entry points
    entry_points={
        "console_scripts": [
            "risk-pipeline=risk_pipeline.cli:main",
            "risk-pipeline-check=risk_pipeline.utils.environment_check:main",
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