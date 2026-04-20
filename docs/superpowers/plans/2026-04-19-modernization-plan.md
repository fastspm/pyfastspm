# pyfastspm Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the `pyfastspm` project with Pixi, Hatch, Ruff, Ty, Prek, and Loguru.

**Architecture:** Transition from `setup.cfg`/`environment.yml` to a unified `pyproject.toml` managed by Pixi, utilizing Hatch as the build backend and Loguru for enhanced logging.

**Tech Stack:** Pixi, Hatch, Hatch-VCS, Ruff, Ty, Prek, Loguru.

---

### Task 1: Initialize Pixi and Migrate Dependencies

**Files:**
- Modify: `pyproject.toml`
- Delete: `environment.yml`, `setup.cfg`, `setup.py`, `requirements.txt`

- [ ] **Step 1: Initialize Pixi with pyproject.toml format**

Run: `pixi init --format pyproject`
Expected: `[tool.pixi]` sections added to `pyproject.toml`.

- [ ] **Step 2: Update `pyproject.toml` with specific configurations**
Update the file to include the Hatch build backend, `hatch-vcs` versioning, dependencies, environments, and tasks as defined in the design.

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "pyfastspm"
dynamic = ["version"]
description = "Python package for loading, processing, and exporting SPM movies acquired with the FAST module."
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Carlo A. Pignedoli", email = "carlo.pignedoli@empa.ch"},
    {name = "Daniele Passerone", email = "daniele.passerone@empa.ch"}
]
dependencies = [
    "h5py",
    "numpy",
    "scipy",
    "scikit-image",
    "pillow",
    "matplotlib",
    "loguru",
    "tqdm",
]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "pyfastspm/_version.py"

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[tool.pixi.dependencies]
ffmpeg = "*"

[tool.pixi.environments]
default = { solve-group = "default" }
dev = { features = ["dev"], solve-group = "default" }

[tool.pixi.feature.dev.dependencies]
ruff = "*"
ty = "*"
pytest = "*"
pytest-cov = "*"
nbmake = "*"

[tool.pixi.tasks]
lint = "ruff check ."
format = "ruff format ."
typecheck = "ty check"
test = "pytest"
```

- [ ] **Step 3: Run `pixi install`**

Run: `pixi install`
Expected: `pixi.lock` created.

- [ ] **Step 4: Remove legacy files**

Run: `rm environment.yml setup.cfg setup.py requirements.txt`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pixi.lock
git rm environment.yml setup.cfg setup.py requirements.txt
git commit -m "refactor: migrate to pixi and hatch"
```

---

### Task 2: Configure Ruff and Ty

**Files:**
- Modify: `pyproject.toml`
- Delete: `.flake8`

- [ ] **Step 1: Add Ruff and Ty configuration to `pyproject.toml`**

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "UP", "B", "SIM", "I"]
ignore = ["E501"]

[tool.ty.environment]
python-version = "3.10"
python-platform = "linux"

[tool.ty.src]
include = ["pyfastspm", "tests"]
```

- [ ] **Step 2: Remove `.flake8`**

Run: `rm .flake8`

- [ ] **Step 3: Run Ruff and Ty to check status**

Run: `pixi run lint && pixi run typecheck`
Expected: Reports errors (to be fixed in later tasks).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git rm .flake8
git commit -m "refactor: configure ruff and ty"
```

---

### Task 3: Replace pre-commit with prek

**Files:**
- Create: `prek.toml`
- Delete: `.pre-commit-config.yaml`

- [ ] **Step 1: Create `prek.toml`**

```toml
[[repos]]
repo = "builtin"
hooks = [
  { id = "trailing-whitespace" },
  { id = "end-of-file-fixer" },
  { id = "check-yaml" },
]

[[repos]]
repo = "local"
hooks = [
  { id = "ruff", name = "ruff", language = "system", entry = "pixi run ruff check --fix", types = ["python"] },
  { id = "ruff-format", name = "ruff format", language = "system", entry = "pixi run ruff format", types = ["python"] },
  { id = "ty-check", name = "ty check", language = "system", entry = "pixi run ty check", types = ["python"], pass_filenames = false },
]
```

- [ ] **Step 2: Remove `.pre-commit-config.yaml`**

Run: `rm .pre-commit-config.yaml`

- [ ] **Step 3: Commit**

```bash
git add prek.toml
git rm .pre-commit-config.yaml
git commit -m "refactor: replace pre-commit with prek"
```

---

### Task 4: Modernize Logging with Loguru

**Files:**
- Modify: `pyfastspm/__init__.py`, `pyfastspm/fast_movie.py`, and other submodules.

- [ ] **Step 1: Update `pyfastspm/__init__.py`**

```python
from loguru import logger
logger.disable("pyfastspm")
```

- [ ] **Step 2: Refactor `pyfastspm/fast_movie.py` to use loguru**
Replace `import logging` and `logger = logging.getLogger(__name__)` with `from loguru import logger`.
Remove `logging.basicConfig` in `_setup_logging`. Use `logger.add()` if needed, but prefer letting the user configure sinks.

- [ ] **Step 3: Convert `print` statements to log messages**
Search for `print(` in the codebase and replace with `logger.debug()` or `logger.info()`.

- [ ] **Step 4: Run tests to ensure no regressions**

Run: `pixi run test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyfastspm/
git commit -m "refactor: modernize logging with loguru"
```

---

### Task 5: Simplify CI/CD

**Files:**
- Modify: `.gitlab-ci.yml`
- Delete: `tests/Dockerfile`, `tests/environment-dev.yml`

- [ ] **Step 1: Update `.gitlab-ci.yml`**

```yaml
default:
  image: ghcr.io/prefix-dev/pixi:latest
  before_script:
    - pixi install

stages:
  - test
  - build
  - release

test_job:
  stage: test
  script:
    - pixi run lint
    - pixi run typecheck
    - pixi run test

build_job:
  stage: build
  script:
    - pixi run python -m build
  artifacts:
    paths:
      - dist/

pages:
  stage: release
  only:
    - tags
  script:
    - mkdir public
    - cd docs/
    - pixi run sphinx-apidoc --private --force -d 0 --no-headings --module-first --no-toc -o API/ ../pyfastspm
    - pixi run sphinx-build -b html -d _build/doctrees . _build/html
    - mv -v _build/html/* ../public/
  artifacts:
    paths:
      - public
```

- [ ] **Step 2: Remove Docker files**

Run: `rm tests/Dockerfile tests/environment-dev.yml`

- [ ] **Step 3: Commit**

```bash
git add .gitlab-ci.yml
git rm tests/Dockerfile tests/environment-dev.yml
git commit -m "refactor: simplify ci/cd with pixi"
```
