# Hatch Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transition the PyPI build and publishing workflow from `build`/`twine` to `hatch` to unify the toolset and simplify CI/CD.

**Architecture:** We will replace the current `pixi` tasks for building and publishing with `hatch` commands and update the GitLab CI configuration to use these tasks with the new `HATCH_INDEX_*` environment variables.

**Tech Stack:** `hatch`, `pixi`, GitLab CI.

---

### Task 1: Update `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Remove `twine` from dev dependencies**

Remove `twine = ">=6.1.0"` from `[tool.pixi.feature.dev.dependencies]`.

- [ ] **Step 2: Update `build_pypi` and `publish_pypi` tasks**

Update the commands to use `hatch`.

```toml
build_pypi = { cmd = "hatch build", default-environment = "dev" }
publish_pypi = { cmd = "hatch publish", default-environment = "dev", depends-on = ["build_pypi"] }
```

- [ ] **Step 3: Verify pixi environment updates**

Run: `pixi install`
Expected: Success, `twine` is removed, `hatch` is available.

- [ ] **Step 4: Commit changes**

```bash
git add pyproject.toml
git commit -m "build: migrate pypi tasks to hatch and remove twine"
```

---

### Task 2: Update `.gitlab-ci.yml`

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Update `build_job` to use the new task**

Change the script to use `pixi run build_pypi`.

```yaml
build_job:
  stage: build
  script:
    - pixi run build_pypi
  artifacts:
    paths:
      - dist/
```

- [ ] **Step 2: Add `release_job` for PyPI publishing**

Add a new job that runs on tags and uses `pixi run publish_pypi`.

```yaml
release_pypi:
  stage: release
  only:
    - tags
  script:
    - pixi run publish_pypi
```

- [ ] **Step 3: Commit changes**

```bash
git add .gitlab-ci.yml
git commit -m "ci: update pypi build and add release job using hatch"
```

---

### Task 3: Local Verification

- [ ] **Step 1: Run local build**

Run: `pixi run build_pypi`
Expected: `hatch` builds the project and produces `.tar.gz` and `.whl` files in `dist/`.

- [ ] **Step 2: Verify artifacts**

Run: `ls -l dist/`
Expected: Check for presence of sdist and wheel.

- [ ] **Step 3: Final commit and cleanup**

```bash
git add dist/  # If needed, though usually gitignored
# No action needed if dist is gitignored
```
