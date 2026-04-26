# Design: Hatch Migration for PyPI Publishing

## Goal
Unify the build and release workflow by transitioning from `python -m build` and `twine` to `hatch`. This leverages the existing `hatchling` backend and simplifies the development environment.

## Context
- The project already uses `hatchling` and `hatch-vcs` for metadata and versioning.
- Pixi is used as the environment manager and task runner.
- Publishing currently uses `twine` in a manual or planned CI step.
- GitLab CI variables will be renamed from `TWINE_*` to `HATCH_INDEX_*` for clean integration.

## Proposed Changes

### 1. Dependency Management
- **Remove** `twine` from `[tool.pixi.feature.dev.dependencies]` in `pyproject.toml`.
- **Keep** `python-build` as requested for other use cases.
- **Ensure** `hatch` remains in the dev dependencies.

### 2. Task Configuration (`pyproject.toml`)
- Update `build_pypi` to use `hatch build`.
- Update `publish_pypi` to use `hatch publish`.
- Remove manual environment variable mapping from the task command, as authentication will be handled by CI environment variables.

### 3. GitLab CI Integration (`.gitlab-ci.yml`)
- Add a `publish_pypi` job to the `release` stage.
- Configure the job to run only on tags.
- The job will execute `pixi run publish_pypi`.
- Authentication will rely on GitLab CI variables `HATCH_INDEX_USER` (set to `__token__`) and `HATCH_INDEX_AUTH` (set to the PyPI token).

## Architecture & Data Flow
1. **Developer** tags a release in Git.
2. **GitLab CI** triggers the `release` stage.
3. **Build Job** generates artifacts in `dist/`.
4. **Publish Job** picks up artifacts and runs `hatch publish`.
5. **Hatch** reads `HATCH_INDEX_USER` and `HATCH_INDEX_AUTH` from the environment and uploads to PyPI.

## Verification Plan
- **Local:** Run `pixi run build_pypi` and verify artifacts in `dist/`.
- **CI Simulation:** Verify the `.gitlab-ci.yml` syntax and job logic.
- **Dry Run:** Note that `hatch publish` does not have a formal dry-run flag that prevents network activity entirely in all versions, but `hatch build` serves as the primary local validation.
