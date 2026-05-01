# Design Doc: pyfastspm Modernization

Modernizing the development workflow and technical infrastructure of `pyfastspm` using modern Python tooling.

## Goals
- Migrate package management to `pixi`.
- Replace `black`, `isort`, and `flake8` with `ruff`.
- Introduce `ty` for static type checking.
- Replace `pre-commit` with `prek`.
- Modernize logging using `loguru` and clean up `print` statements.

## Architecture & Tooling

### 1. Package Management & Build: Pixi & Hatch
- **Format:** `pyproject.toml` (native integration).
- **Build Backend:** `hatchling.build`.
- **Versioning:** `hatch-vcs` (replaces `setuptools_scm`).
- **Channels:** `conda-forge`.
- **Environments:**
    - `default`: Core library dependencies (`numpy`, `scipy`, `h5py`, `matplotlib`, `loguru`, `scikit-image`, `pillow`, `tqdm`, `ffmpeg`).
    - `dev`: `default` + development tools (`ruff`, `ty`, `pytest`, `nbmake`).
- **Tasks:**
    - `lint`: `ruff check .`
    - `format`: `ruff format .`
    - `typecheck`: `ty check`
    - `test`: `pytest`

### 2. Linting & Formatting: Ruff
- Consolidate rules in `pyproject.toml`.
- Enable rule sets: `E`, `F`, `UP`, `B`, `SIM`, `I`.
- Replace existing `black`, `isort`, and `flake8` configs.

### 3. Type Checking: Ty
- Configure in `pyproject.toml`.
- Target Python 3.10 and Linux platform.

### 4. Git Hooks: Prek
- Config file: `prek.toml`.
- Hooks: `ruff`, `ruff-format`, `ty check`, and basic file hygiene (whitespace, end-of-file).
- Note: `prek` is assumed to be installed system-wide.

### 5. Logging: Loguru
- Replace standard `logging` with `loguru`.
- Follow "library" pattern: `logger.disable("pyfastspm")` in `__init__.py`.
- Refactor `FastMovie` class to use `loguru`.
- Identify and convert `print()` statements to `logger.debug()` or `logger.info()` as appropriate.

### 6. CI/CD Simplification
- Remove `tests/Dockerfile` and the `build_docker_image` stage.
- Switch `.gitlab-ci.yml` to use `ghcr.io/prefix-dev/pixi` as the base image.
- Leverage `pixi run` for all stages (lint, typecheck, test, build, docs).
- Use Pixi caching to speed up CI runs.

## Data Flow
1. **Developer Setup:** `pixi install`.
2. **Local Work:** `pixi run <task>`.
3. **Commit:** `prek run` (manual or via git hook).

## Validation Plan
1. **Infrastructure:** Verify `pixi` can solve and install both environments.
2. **Functional:** Ensure `pytest` passes after refactoring logging.
3. **Quality:** Run `ruff` and `ty` to ensure the codebase meets the new standards.
4. **Manual Check:** Verify `loguru` behaves correctly when enabled/disabled.
