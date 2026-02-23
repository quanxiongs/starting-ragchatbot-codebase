# Frontend Changes

## Code Quality Tooling Setup

### What was added

This change adds `black` as the project's Python code formatter, installs it as a dev dependency, and provides a developer-facing script for running quality checks.

---

### Files changed

| File | Change |
|------|--------|
| `pyproject.toml` | Added `black>=24.0.0` to `[dependency-groups] dev`. Added `[tool.black]` configuration block (line-length 88, target Python 3.13, excludes `chroma_db`/`.git`/`__pycache__`). |
| `format.sh` | New executable script. Run `./format.sh` to auto-format; `./format.sh --check` to validate without modifying files (suitable for CI). |
| `backend/*.py` (14 files) | Auto-formatted by black for consistent style throughout the codebase. |

---

### How to use

```bash
# Auto-format all Python source files
./format.sh

# Check formatting without modifying files (CI / pre-commit)
./format.sh --check
```

### Files reformatted by black

- `backend/ai_generator.py`
- `backend/app.py`
- `backend/config.py`
- `backend/document_processor.py`
- `backend/models.py`
- `backend/rag_system.py`
- `backend/search_tools.py`
- `backend/session_manager.py`
- `backend/vector_store.py`
- `backend/tests/conftest.py`
- `backend/tests/helpers.py`
- `backend/tests/test_ai_generator.py`
- `backend/tests/test_rag_system.py`
- `backend/tests/test_search_tool.py`
