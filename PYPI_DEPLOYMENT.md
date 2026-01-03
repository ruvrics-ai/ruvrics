# PyPI Deployment Guide

This guide explains how to build and publish Ruvrics to PyPI.

## Prerequisites

1. PyPI account (https://pypi.org/account/register/)
2. TestPyPI account (https://test.pypi.org/account/register/) for testing
3. Install build tools:

```bash
pip install build twine
```

## Step 1: Update Version

Edit `pyproject.toml` and update the version number:

```toml
[project]
name = "ruvrics"
version = "0.1.0"  # Update this for new releases
```

## Step 2: Update GitHub URLs

Replace YOUR_GITHUB_USERNAME in `pyproject.toml`:

```toml
[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/ruvrics"
Repository = "https://github.com/YOUR_USERNAME/ruvrics"
Issues = "https://github.com/YOUR_USERNAME/ruvrics/issues"
```

## Step 3: Clean Previous Builds

```bash
rm -rf dist/ build/ *.egg-info
```

## Step 4: Build Distribution

```bash
python -m build
```

This creates:
- `dist/ruvrics-0.1.0.tar.gz` (source distribution)
- `dist/ruvrics-0.1.0-py3-none-any.whl` (wheel distribution)

## Step 5: Test on TestPyPI (Recommended)

Upload to TestPyPI first to verify everything works:

```bash
twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: Your TestPyPI username
- Password: Your TestPyPI password or API token

Test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ ruvrics
```

## Step 6: Upload to PyPI

Once verified on TestPyPI, upload to production PyPI:

```bash
twine upload dist/*
```

You'll be prompted for:
- Username: Your PyPI username (or __token__)
- Password: Your PyPI password or API token

## Step 7: Verify Installation

Test that users can install:

```bash
pip install ruvrics
```

## Using API Tokens (Recommended)

Instead of passwords, use API tokens:

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Create `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
```

Set permissions:

```bash
chmod 600 ~/.pypirc
```

## Automated Uploads

With `~/.pypirc` configured, you can upload without entering credentials:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Version Bumping

Follow semantic versioning:

- MAJOR version (1.0.0): Incompatible API changes
- MINOR version (0.2.0): New functionality, backwards compatible
- PATCH version (0.1.1): Bug fixes, backwards compatible

Update version in:
1. `pyproject.toml`
2. `CHANGELOG.md` (add new section)
3. `ruvrics/cli.py` (if version is shown in --version)

## Release Checklist

Before each release:

- [ ] Run all tests: `pytest tests/ -v`
- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Build distribution: `python -m build`
- [ ] Test on TestPyPI
- [ ] Upload to PyPI
- [ ] Create GitHub release/tag
- [ ] Test installation: `pip install ruvrics`

## Troubleshooting

### "File already exists"

PyPI doesn't allow re-uploading the same version. You must:
1. Increment version number
2. Rebuild: `python -m build`
3. Upload new version

### Import errors after installation

Check `MANIFEST.in` includes all necessary files.

### Missing dependencies

Verify dependencies are listed in `pyproject.toml` under `dependencies`.

## Continuous Deployment (Optional)

For automated releases via GitHub Actions, see:
https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
