# Design: GitLab CI Modernization (Parallel DAG)

## Goal
Modernize the `.gitlab-ci.yml` configuration to use contemporary GitLab CI features (`rules`, `needs`, `workflow`) and optimize pipeline execution speed using a Directed Acyclic Graph (DAG) approach.

## Context
- The project uses `pixi` for environment management.
- Current pipeline uses legacy `only` syntax and strict sequential stages.
- The pipeline handles testing, PyPI building/releasing, and documentation (GitLab Pages).

## Proposed Changes

### 1. Workflow & Global Logic
- **`workflow:rules`**: Prevent duplicate pipelines for Merge Requests and branch pushes.
- **`default` configuration**: Centralize the `pixi` image and `before_script` to ensure consistency across all jobs.

### 2. Modernized Syntax
- **`rules` instead of `only`**: Use the standard `rules` keyword for defining job triggers.
- **`needs` (DAG)**: Decouple jobs from strict stage boundaries.
  - `build_job` and `test_job` will run in parallel immediately.
  - `release_pypi` will start as soon as both `build_job` and `test_job` succeed.
  - `pages` will start as soon as `build_job` succeeds (independent of `test_job`).

### 3. Job Structure

| Job | Trigger | Stage | Needs |
| :--- | :--- | :--- | :--- |
| `test_job` | MRs, Default Branch | `test` | None |
| `build_job` | MRs, Default Branch, Tags | `build` | None |
| `release_pypi` | Tags Only | `release` | `test_job`, `build_job` |
| `pages` | Tags Only | `release` | `build_job` |

## Verification Plan
- **CI Linting**: Use the GitLab CI Lint tool (if accessible) or verify syntax manually.
- **Logic Check**: Confirm that `release_pypi` and `pages` are correctly restricted to tags.
- **DAG Check**: Ensure `needs` accurately reflects the data flow (e.g., `pages` requires build artifacts but not necessarily test results).
