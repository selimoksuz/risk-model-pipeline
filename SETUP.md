# Setup Instructions

## Quick Fix for numpy.dtype Error

If you encounter the error:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

This is due to version incompatibility between numpy, pandas, and scikit-learn.

## Solution 1: Use Fixed Requirements (Recommended)

```bash
pip install -r requirements_fixed.txt
```

## Solution 2: Automated Setup

### Windows
```bash
setup_environment.bat
```

### Linux/Mac
```bash
bash setup_environment.sh
```

## Solution 3: Manual Installation

```bash
# Uninstall conflicting packages
pip uninstall -y numpy pandas scikit-learn

# Install compatible versions
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install scikit-learn==1.3.0

# Install other requirements
pip install -r requirements.txt
```

## Solution 4: Create Fresh Virtual Environment

### Windows
```bash
# Create environment
python -m venv risk_env

# Activate
risk_env\Scripts\activate

# Install packages
pip install -r requirements_fixed.txt
```

### Linux/Mac
```bash
# Create environment
python3 -m venv risk_env

# Activate
source risk_env/bin/activate

# Install packages
pip install -r requirements_fixed.txt
```

## Testing the Installation

After setup, test with:
```python
import numpy as np
import pandas as pd
import sklearn

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```

Expected versions:
- NumPy: 1.24.3
- Pandas: 1.5.3
- Scikit-learn: 1.3.0

## Running the Notebook

1. Activate your environment
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `notebooks/01_dual_pipeline_example.ipynb`
4. Run cells sequentially

## Troubleshooting

### Import Errors
If you get import errors for the pipeline:
```bash
cd risk-model-pipeline
pip install -e .
```

### Memory Issues
Reduce sample size or model complexity in the notebook configuration.

### Jupyter Not Found
```bash
pip install jupyter notebook
```

## Support

If issues persist:
1. Check Python version (3.8+ required)
2. Try with a clean Python installation
3. Use Anaconda/Miniconda for better package management