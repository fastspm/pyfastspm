# Design: Pixi Environment Caching in GitLab CI

## Goal
Implement robust and efficient caching for `pixi` environments in GitLab CI to reduce pipeline execution time and minimize network usage.

## Context
- The project uses `pixi` for multi-platform environment management.
- Pipelines currently install the environment from scratch on every run.
- Caching the `.pixi` directory (installed env) and the package cache (`.pixi-cache`) will significantly speed up jobs.

## Proposed Changes

### 1. Global CI Variables
- **`PIXI_CACHE_DIR`**: Set to `$CI_PROJECT_DIR/.pixi-cache`. This ensures the downloaded package archives are stored within the project root, allowing GitLab CI to cache them.
- **`PIXI_FROZEN`**: Set to `true`. This ensures that `pixi` uses the exact versions specified in `pixi.lock` and fails if the lockfile is out of sync, which is essential for reproducible CI.

### 2. Cache Configuration
A global `cache` block will be added with:
- **`key:files`**: Using `pixi.lock` as the key. This ensures the cache is unique to the dependency state.
- **`paths`**:
  - `.pixi/`: The directory where `pixi` creates the project environment.
  - `.pixi-cache/`: The directory where `pixi` stores downloaded package archives.

### 3. Pipeline Integration
The cache will be defined at the top level of `.gitlab-ci.yml`, making it available to all jobs. Each job will attempt to restore the cache before running its `before_script`.

## Architecture & Data Flow
1. **Runner starts**: GitLab runner checks for a cache matching the hash of `pixi.lock`.
2. **Cache Restore**: If found, `.pixi/` and `.pixi-cache/` are extracted into the workspace.
3. **`before_script`**: `pixi global install git` is run.
4. **Job Script**: `pixi run ...` executes. If the cache was hit, `pixi` will find the environment already present and proceed immediately.
5. **Runner finishes**: If the cache was modified or is new, it is compressed and uploaded for future runs.

## Verification Plan
- **Pipeline Monitoring**: Observe the "Restoring cache" and "Saving cache" logs in GitLab CI.
- **Execution Time**: Compare the duration of the `test_job` and `build_job` with and without caching.
- **Lockfile Check**: Verify that the pipeline correctly fails if `pyproject.toml` is updated but `pixi.lock` is not.
