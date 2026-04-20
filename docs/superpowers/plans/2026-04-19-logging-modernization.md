# Logging Modernization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `logging` and `print()` with `loguru.logger` across the entire `pyfastspm` package.

**Architecture:** Use `from loguru import logger` globally. Rename all existing `log` variables to `logger`. Convert `print()` calls to appropriate `logger` levels. Fix `FastMovie` handler management for `loguru`.

**Tech Stack:** `loguru`, `pytest`.

---

### Task 1: Update `pyfastspm/fast_movie.py`

**Files:**
- Modify: `pyfastspm/fast_movie.py`

- [ ] **Step 1: Update imports and rename `log` to `logger`**
    - Replace all `log.` with `logger.`.
    - `logger` is already imported from `loguru`.

- [ ] **Step 2: Fix `__init__` to store handler ID**
    - Store the result of `logger.add()` in `self._handler_id`.

```python
        if log_processing:
            self._handler_id = logger.add(
                self._log_file,
                mode="w",
                format="{level:1.1s}[{time:YYYY-MM-DD HH:mm:ss} - {module}.{function}]: {message}",
                filter=lambda record: record["extra"].get("file_name") == file_name,
            )
        else:
            self._handler_id = None
```

- [ ] **Step 3: Fix `close()` method**
    - Remove the handler using `logger.remove(self._handler_id)`.

```python
    def close(self):
        """Deinitializes the logging handler for this movie instance"""
        if self._handler_id is not None:
            logger.remove(self._handler_id)
            self._handler_id = None
        logger.info("Logging handler successfully closed.")
```

- [ ] **Step 4: Commit**
```bash
git add pyfastspm/fast_movie.py
git commit -m "refactor: update fast_movie.py to use loguru correctly"
```

### Task 2: Update `pyfastspm/tools/`

**Files:**
- Modify: `pyfastspm/tools/exporter.py`
- Modify: `pyfastspm/tools/frame_artists.py`
- Modify: `pyfastspm/tools/file_handling_tools.py`
- Modify: `pyfastspm/tools/error_catcher.py`

- [ ] **Step 1: Update `exporter.py`**
    - Replace `import logging` with `from loguru import logger`.
    - Remove `log = logging.getLogger(__name__)`.
    - Replace `log.` with `logger.`.

- [ ] **Step 2: Update `frame_artists.py`**
    - Replace `import logging` with `from loguru import logger`.
    - Remove `log = logging.getLogger(__name__)`.
    - Replace `log.` with `logger.`.

- [ ] **Step 3: Update `file_handling_tools.py`**
    - Replace `import logging` with `from loguru import logger`.
    - Remove `log = logging.getLogger(__name__)`.
    - Replace `log.` with `logger.`.
    - Note: Do NOT replace `print()` in docstring examples.

- [ ] **Step 4: Update `error_catcher.py`**
    - Ensure `from loguru import logger` is present (add if missing).
    - Replace all `print(...)` with `logger.error(...)`.

- [ ] **Step 5: Commit**
```bash
git add pyfastspm/tools/*.py
git commit -m "refactor: update tools to use loguru"
```

### Task 3: Update `pyfastspm/artefact_removal/`

**Files:**
- Modify: `pyfastspm/artefact_removal/filters.py`
- Modify: `pyfastspm/artefact_removal/interpolate.py`
- Modify: `pyfastspm/artefact_removal/fft.py`
- Modify: `pyfastspm/artefact_removal/creep.py`
- Modify: `pyfastspm/artefact_removal/drift.py`

- [ ] **Step 1: Update `filters.py`, `interpolate.py`, `fft.py`, `creep.py`**
    - Replace `import logging` with `from loguru import logger`.
    - Remove `log = logging.getLogger(__name__)`.
    - Replace `log.` with `logger.`.

- [ ] **Step 2: Convert `print()` in `fft.py`, `creep.py`, `drift.py`**
    - `fft.py`: `print` -> `logger.info` (except import failure which should be `logger.warning`).
    - `creep.py`: `print` -> `logger.info` (except exceptions which should be `logger.error`).
    - `drift.py`: `print` -> `logger.info`.

- [ ] **Step 3: Commit**
```bash
git add pyfastspm/artefact_removal/*.py
git commit -m "refactor: update artefact_removal to use loguru"
```

### Task 4: Update `pyfastspm/tracking/pixel_trace.py`

**Files:**
- Modify: `pyfastspm/tracking/pixel_trace.py`

- [ ] **Step 1: Update `pixel_trace.py`**
    - Replace `import logging` with `from loguru import logger`.
    - Remove `log = logging.getLogger(__name__)`.
    - Replace `log.` with `logger.`.
    - Replace `print(points)` with `logger.debug(points)`.

- [ ] **Step 2: Commit**
```bash
git add pyfastspm/tracking/pixel_trace.py
git commit -m "refactor: update tracking to use loguru"
```

### Task 5: Final Verification

- [ ] **Step 1: Run tests**
Run: `pytest`
Expected: All tests pass.

- [ ] **Step 2: Check for remaining `print(` or `import logging`**
Run: `grep -r "import logging" pyfastspm/`
Run: `grep -r "print(" pyfastspm/` (verify only docstrings/comments remain)

- [ ] **Step 3: Final Commit**
```bash
git commit --allow-empty -m "chore: logging modernization complete"
```
