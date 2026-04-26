# Pixi Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GitLab CI caching for `pixi` environments and enable lockfile tracking for reproducible builds.

**Architecture:** We will configure GitLab CI to use a local cache directory for `pixi` packages, key the cache on the `pixi.lock` file, and ensure the lockfile is tracked in Git while the cache directory is ignored.

**Tech Stack:** GitLab CI, Pixi, Git.

---

### Task 1: Update `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add `.pixi-cache/` and remove `pixi.lock`**

Update the `# Pixi` section to ignore the cache but allow the lockfile.

```gitignore
# Pixi
.pixi/
.pixi-cache/
```

- [ ] **Step 2: Verify git status**

Run: `git status`
Expected: `pixi.lock` should now show as an untracked file (if it exists).

- [ ] **Step 3: Commit changes**

```bash
git add .gitignore
git commit -m "build: ignore pixi cache and allow tracking of pixi.lock"
```

---

### Task 2: Update `.gitlab-ci.yml`

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Add global variables**

Add `PIXI_CACHE_DIR` and `PIXI_FROZEN` to the top-level `variables` block.

```yaml
variables:
  PIXI_CACHE_DIR: "$CI_PROJECT_DIR/.pixi-cache"
  PIXI_FROZEN: "true"
```

- [ ] **Step 2: Add global cache configuration**

Add a `cache` block at the top level.

```yaml
cache:
  key:
    files:
      - pixi.lock
  paths:
    - .pixi/
    - .pixi-cache/
```

- [ ] **Step 3: Commit changes**

```bash
git add .gitlab-ci.yml
git commit -m "ci: implement pixi environment caching"
```

---

### Task 3: Track `pixi.lock`

- [ ] **Step 1: Ensure `pixi.lock` is up to date**

Run: `pixi update`
Expected: `pixi.lock` is generated/updated.

- [ ] **Step 2: Add and commit `pixi.lock`**

```bash
git add pixi.lock
git commit -m "build: track pixi.lock for reproducible CI"
```

---

### Task 4: Local Verification

- [ ] **Step 1: Verify YAML structure**

Run: `cat .gitlab-ci.yml` and check for correct indentation of `variables` and `cache`.

- [ ] **Step 2: Verify `.gitignore`**

Run: `git check-ignore .pixi-cache/`
Expected: Output showing `.pixi-cache/` is ignored.
