# Publishing to PyPI Guide 🚀

## Prerequisites

1. **Create PyPI Account**
   - Register at https://pypi.org/account/register/
   - Register at https://test.pypi.org/account/register/ (for testing)

2. **Generate API Token**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Save it securely (you'll need it for authentication)

## Installation for Publishing

```bash
pip install --upgrade pip setuptools wheel twine build
```

## Building the Package

### 1. Clean Previous Builds
```bash
rm -rf dist/ build/ *.egg-info/
# Windows: rmdir /s dist build
```

### 2. Update Version
Edit `src/risk_pipeline/_version.py`:
```python
__version__ = "0.3.1"  # Increment version
```

### 3. Build Distribution Files
```bash
python -m build
```

This creates:
- `dist/risk-model-pipeline-0.3.1.tar.gz` (source distribution)
- `dist/risk_model_pipeline-0.3.1-py3-none-any.whl` (wheel)

## Testing on TestPyPI (Recommended First)

### 1. Upload to TestPyPI
```bash
python -m twine upload --repository testpypi dist/*
```

Username: `__token__`
Password: `<your-test-pypi-token>`

### 2. Test Installation
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ risk-model-pipeline
```

### 3. Verify Package
```python
import risk_pipeline
print(risk_pipeline.__version__)
from risk_pipeline import Config, DualPipeline
```

## Publishing to PyPI (Production)

### 1. Final Checks
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version number is correct
- [ ] CHANGELOG is updated
- [ ] Git tag created: `git tag v0.3.1`

### 2. Upload to PyPI
```bash
python -m twine upload dist/*
```

Username: `__token__`
Password: `<your-pypi-token>`

### 3. Verify on PyPI
- Check https://pypi.org/project/risk-model-pipeline/
- Test installation: `pip install risk-model-pipeline`

## Automation with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Package Installation Options

After publishing, users can install:

```bash
# Basic installation
pip install risk-model-pipeline

# With visualization support
pip install risk-model-pipeline[viz]

# With ML extras
pip install risk-model-pipeline[ml]

# Everything
pip install risk-model-pipeline[all]

# For development
pip install risk-model-pipeline[dev]
```

## Version Management

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

## Troubleshooting

### "Version already exists"
- Increment version in `_version.py`
- You cannot overwrite existing versions on PyPI

### Authentication Failed
- Use `__token__` as username
- Use your API token (including `pypi-` prefix) as password

### Missing Dependencies
- Ensure all dependencies are in `setup.py`
- Test with fresh virtual environment

## Security Best Practices

1. **Never commit tokens** to git
2. Use **environment variables** for CI/CD
3. Enable **2FA** on PyPI account
4. Use **API tokens** instead of passwords
5. Test on **TestPyPI** first

## Post-Publishing

1. **Announce Release**
   - GitHub release notes
   - Update README with new version

2. **Monitor**
   - Check download statistics
   - Monitor issues on GitHub
   - Respond to user feedback

3. **Document**
   - Update examples for new version
   - Update migration guide if needed