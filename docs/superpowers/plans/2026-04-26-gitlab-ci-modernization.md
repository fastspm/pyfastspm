# GitLab CI Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize `.gitlab-ci.yml` using `workflow:rules`, `rules`, and `needs` (DAG) for faster, more reliable pipelines.

**Architecture:** We will replace legacy `only` syntax with `rules`, implement `workflow:rules` to avoid duplicate pipelines, and use `needs` to allow parallel execution of jobs across stages.

**Tech Stack:** GitLab CI, Pixi.

---

### Task 1: Implement Workflow and Default Configuration

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Add `workflow:rules`**

Add top-level `workflow` to handle MRs, default branch, and tags.

```yaml
workflow:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_COMMIT_TAG
```

- [ ] **Step 2: Update `default` configuration**

Ensure `default` includes the image and `before_script`.

```yaml
default:
  image: ghcr.io/prefix-dev/pixi:latest
  before_script:
    - pixi install
```

- [ ] **Step 3: Commit changes**

```bash
git add .gitlab-ci.yml
git commit -m "ci: add workflow rules and ensure default config"
```

---

### Task 2: Modernize `test_job` and `build_job`

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Update `test_job` rules**

Replace implicit logic with explicit rules.

```yaml
test_job:
  stage: test
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
  script:
    - pixi run lint
    - pixi run test
    - pixi run test_notebook
```

- [ ] **Step 2: Update `build_job` rules**

Allow build on MRs, default branch, and tags.

```yaml
build_job:
  stage: build
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_COMMIT_TAG
  script:
    - pixi run build_pypi
  artifacts:
    paths:
      - dist/
```

- [ ] **Step 3: Commit changes**

```bash
git add .gitlab-ci.yml
git commit -m "ci: modernize test and build job rules"
```

---

### Task 3: Implement Parallel DAG for Release Jobs

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Modernize `release_pypi` with `needs`**

Replace `only: - tags` with `rules` and add `needs`.

```yaml
release_pypi:
  stage: release
  rules:
    - if: $CI_COMMIT_TAG
  needs:
    - job: test_job
    - job: build_job
  script:
    - pixi run publish_pypi
```

- [ ] **Step 2: Modernize `pages` with `needs`**

Replace `only: - tags` with `rules` and add `needs` on `build_job`.

```yaml
pages:
  stage: release
  rules:
    - if: $CI_COMMIT_TAG
  needs:
    - job: build_job
  script:
    - mkdir public
    - cd docs/
    - pixi run sphinx-apidoc --private --force -d 0 --no-headings --module-first
      --no-toc -o API/ ../pyfastspm
    - pixi run sphinx-build -b html -d _build/doctrees . _build/html
    - mv -v _build/html/* ../public/
  artifacts:
    paths:
      - public
```

- [ ] **Step 3: Commit changes**

```bash
git add .gitlab-ci.yml
git commit -m "ci: implement parallel DAG using needs and modernize release rules"
```

---

### Task 4: Final Validation

- [ ] **Step 1: Check YAML syntax**

Run: `cat .gitlab-ci.yml` and visually inspect for indentation or logical errors.

- [ ] **Step 2: Verify rule consistency**

Ensure that no job is missing a `rules` block that might cause it to run unexpectedly.
