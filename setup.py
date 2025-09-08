"""
Setup configuration for Risk Model Pipeline
Flexible dependency management with extras
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies - MINIMAL for basic functionality
CORE_DEPS = [
    "pandas>=1.3.0,<2.0.0",
    "numpy>=1.20.0,<1.25.0", 
    "scikit-learn>=1.0.0,<1.3.0",
    "joblib>=1.0.0",
    "openpyxl>=3.0.0",
    "xlsxwriter>=3.0.0",
]

# Optional extras for additional features
EXTRAS = {
    # Visualization support
    "viz": [
        "matplotlib>=3.5.0,<3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
    ],
    
    # Advanced ML features
    "ml": [
        "optuna>=3.0.0",
        "shap>=0.41.0",
        "imbalanced-learn>=0.9.0",
        "scikit-learn-extra>=0.2.0",
    ],
    
    # Development tools
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "pre-commit>=2.0.0",
    ],
    
    # Notebook support
    "notebook": [
        "jupyter>=1.0.0",
        "notebook>=6.0.0",
        "ipywidgets>=7.0.0",
    ],
    
    # All optional dependencies
    "all": [],  # Will be filled below
}

# Combine all extras for 'all'
EXTRAS["all"] = list(set(
    dep for extra_deps in EXTRAS.values() 
    for dep in extra_deps if extra_deps != EXTRAS["all"]
))

setup(
    name="risk-model-pipeline",
    version="0.3.0",
    author="Risk Analytics Team",
    description="Production-ready risk model pipeline with WOE transformation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/selimoksuz/risk-model-pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    
    # Only core dependencies are required
    install_requires=CORE_DEPS,
    
    # Optional dependencies
    extras_require=EXTRAS,
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "risk-pipeline=risk_pipeline.cli:main",
            "risk-check=check_environment:main",
        ],
    },
    
    # Include data files
    include_package_data=True,
    package_data={
        "risk_pipeline": ["*.json", "*.yaml", "*.yml"],
    },
)