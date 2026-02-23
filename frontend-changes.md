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

---

## Dark / Light Mode Toggle Button

### Feature Overview
Added a fixed, circular icon-based toggle button in the top-right corner that switches the UI between dark mode (default) and light mode.

---

### Files Modified

#### `frontend/index.html`
- Added a `<button id="themeToggle" class="theme-toggle">` element positioned **before** the main `.container` div so it floats above all content.
- The button contains two inline SVG icons:
  - **Moon** (`icon-moon`) — visible in dark mode.
  - **Sun** (`icon-sun`) — visible in light mode.
- `aria-label` is set to `"Switch to light mode"` by default and updated dynamically via JS.
- `title="Toggle dark/light mode"` provides a native tooltip.
- Updated stylesheet and script cache-buster query strings from `v=9` → `v=10`.

#### `frontend/style.css`
- **Light-mode CSS variables** (`body.light-mode { … }`) — overrides all colour tokens for a clean light theme:
  - Background: `#f8fafc`, Surface: `#ffffff`, text, borders, and shadows adjusted accordingly.
- **Global smooth transition** — a `*, *::before, *::after` rule applies `transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease` so every element animates during theme changes.
- **`.theme-toggle` button styles**:
  - `position: fixed; top: 1rem; right: 1rem; z-index: 1000` — always visible top-right.
  - 42 × 42 px circle, border matches `--border-color`, background matches `--surface`.
  - Hover: lifts slightly (`translateY(-1px)`), primary-colour border and icon tint, soft blue glow.
  - Focus: `box-shadow: 0 0 0 3px var(--focus-ring)` — keyboard-accessible outline.
  - Active: drops back to original position.
- **Icon visibility rules** — `.icon-sun` hidden by default; `.icon-moon` visible. Inverted for `body.light-mode`.
- **Spin-in animation** (`@keyframes iconSpin`) — each icon animates in with a subtle rotate + fade when it becomes visible.

#### `frontend/script.js`
- **`initTheme()`** — called on `DOMContentLoaded`; reads `localStorage.getItem('theme')` and applies `body.light-mode` + updates `aria-label` if the saved preference is `"light"`.
- **`toggleTheme()`** — toggles `body.classList.toggle('light-mode')`, persists the new preference to `localStorage`, and updates `aria-label` accordingly.
- **`setupEventListeners()`** — wires `themeToggle.addEventListener('click', toggleTheme)`.
- Added `themeToggle` to the module-level DOM element declarations.

---

### Accessibility
- Button has a descriptive `aria-label` that updates to reflect the current action ("Switch to light mode" / "Switch to dark mode").
- Focus ring matches the existing design system (`--focus-ring`).
- Icons marked `aria-hidden="true"` so screen readers rely on the button label.
- Fully keyboard-navigable (Tab to focus, Enter/Space to activate).

### Persistence
User preference is stored in `localStorage` under the key `"theme"` (`"light"` or `"dark"`) and restored on every page load.
