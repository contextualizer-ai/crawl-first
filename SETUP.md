# GitHub Setup Instructions

## 1. Create the repository on GitHub

1. Go to https://github.com/contextualizer-ai
2. Click "New repository"
3. Repository name: `crawl-first`
4. Description: "Deterministic biosample enrichment for LLM-ready data preparation"
5. Make it **Public** (or Private if preferred)
6. **Do NOT** initialize with README, .gitignore, or LICENSE (we already have these)
7. Click "Create repository"

## 2. Connect your local repository

From the `crawl-first` directory, run:

```bash
# Add the GitHub remote
git remote add origin https://github.com/contextualizer-ai/crawl-first.git

# Verify the remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## 3. Set up branch protection (optional but recommended)

In the GitHub repository settings:
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable "Require pull request reviews before merging"
4. Enable "Require status checks to pass before merging"

## 4. Development workflow

```bash
# Install dependencies with uv
uv sync --dev

# Test the CLI
uv run crawl-first --help

# Run tests
uv run pytest

# Code formatting and linting
uv run black .
uv run ruff check .
uv run mypy .
uv run deptry .
```

## 5. Release workflow

1. Update version in `pyproject.toml` and `src/crawl_first/__init__.py`
2. Create a tag: `git tag v0.1.0`
3. Push tag: `git push origin v0.1.0`
4. Create a release on GitHub (will trigger PyPI publish via GitHub Actions)

## Repository URL
After setup: https://github.com/contextualizer-ai/crawl-first