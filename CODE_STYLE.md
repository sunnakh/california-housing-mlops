# Code Style Guide — Ruff + Pylance Zero-Error Rules

Follow every rule in this file.  
Every rule has a ✅ correct example and a ❌ wrong example.  
Breaking any rule = Ruff or Pylance error.

---

## 1. Import Order — always this exact sequence

Every file must follow this order. No exceptions.

```python
# STEP 1 — always the very first line if you use forward references
from __future__ import annotations

# STEP 2 — Python standard library (alphabetical)
import logging
import sys
from pathlib import Path
from typing import Any

# STEP 3 — third-party packages (alphabetical)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# STEP 4 — your own local modules (alphabetical)
from common.logger import get_logger
from src.data.loader import FEATURE_COLS, TARGET_COL
```

### Rules
- Blank line between each group (stdlib / third-party / local)
- Alphabetical inside each group
- `from __future__ import annotations` is always line 1 if used
- **Never** put any code between import groups
- **Never** use `sys.path.insert` — use editable install instead (`uv pip install -e .`)

### Wrong
```python
# ❌ local import before sys.path.insert → Ruff E402
from common.logger import get_logger
import sys
sys.path.insert(0, "../../")

# ❌ mixed groups — no blank line between stdlib and third-party
import sys
import numpy as np
from pathlib import Path
```

---

## 2. No sys.path.insert — Ever

```python
# ❌ NEVER do this — causes E402 on every import after it
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.logger import get_logger
```

### Fix — install project as editable package once

```bash
uv pip install -e .
```

After this, Python finds `src` and `common` automatically.  
Delete every `sys.path.insert` line from every file.

---

## 3. Type Annotations — every function, every argument

Every function must have:
- Type on every parameter
- Return type on every function (even `-> None`)

```python
# ✅ Correct
def add_log_features(
    df: pd.DataFrame,
    cols: list[str],
    suffix: str = "_log",
) -> pd.DataFrame:
    ...

def validate_data(df: pd.DataFrame) -> None:
    ...

def get_best_model(metric: str = "rmse") -> str:
    ...
```

```python
# ❌ Wrong — Pylance reportMissingTypeArgument / reportUnknownParameterType
def add_log_features(df, cols, suffix="_log"):
    ...

def fit(self, X=np.ndarray):   # = is default value, not type annotation
    ...
```

### Type cheatsheet

| Situation | Write this |
|-----------|-----------|
| Optional value | `value: int \| None = None` |
| List of strings | `cols: list[str]` |
| Dict | `params: dict[str, Any]` |
| Union | `x: int \| float` |
| Callable | `fn: Callable[[int], str]` |
| Return nothing | `-> None` |
| Return self | `-> "MyClass"` or `-> Self` (Python 3.11+) |

### Never use old-style typing imports

```python
# ❌ Old — Ruff UP006 / UP035
from typing import List, Dict, Optional, Tuple
def fn(x: Optional[int]) -> List[str]: ...

# ✅ Modern — Python 3.10+
def fn(x: int | None) -> list[str]: ...
```

---

## 4. Constants — UPPER_CASE with type annotation

```python
# ✅ Correct
LOG_COLS: list[str] = ["MedInc", "AveRooms", "Population", "AveOccup"]
DEFAULT_THRESHOLD: float = 0.01
TARGET_COL: str = "MedHouseVal"

# ❌ Wrong — Ruff N816 / Pylance doesn't know the type
log_cols = ["MedInc", "AveRooms"]
defaultThreshold = 0.01
```

---

## 5. Logger — use %s style, not f-strings

```python
logger = get_logger(__name__)   # always module-level, never inside a function

# ✅ Correct — lazy evaluation, faster, no Ruff G004
logger.info("Fitted on %d samples, %d features.", X.shape[0], X.shape[1])
logger.warning("Column '%s' not found — skipping.", col)
logger.debug("Means: %s", means.tolist())

# ❌ Wrong — Ruff G004
logger.info(f"Fitted on {X.shape[0]} samples")
```

---

## 6. Classes — full OOP structure

```python
class MyClass:
    """One-line summary.

    Longer description if needed.

    Example:
        >>> obj = MyClass()
        >>> obj.do_something()
    """

    # ── class-level constants (if any) ────────────────────────────────────
    MAX_RETRIES: int = 3

    def __init__(self, name: str, value: int = 0) -> None:
        # typed instance attributes
        self._name: str = name
        self._value: int = value
        self._is_ready: bool = False

    # ── public methods first ───────────────────────────────────────────────

    def process(self, data: np.ndarray) -> np.ndarray:
        """What this does, one line.

        Args:
            data: Input array of shape (n_samples, n_features).

        Returns:
            Processed array of same shape.

        Raises:
            RuntimeError: If called before fit().
        """
        self._check_is_ready()
        return data * self._value

    # ── private helpers last ───────────────────────────────────────────────

    def _check_is_ready(self) -> None:
        if not self._is_ready:
            raise RuntimeError(
                f"{type(self).__name__} is not ready. Call fit() first."
            )
```

### Class rules
- `__init__` always has `-> None`
- All instance attributes declared and typed inside `__init__`
- Public methods before private methods
- Private methods/attributes start with `_`
- One class per file (for large classes)
- Never use bare `self.x = dict[str, Any]` — that assigns the TYPE, not a value

```python
# ❌ Wrong — assigns the type object itself, not an empty dict
self._results = dict[str, Any]

# ✅ Correct — assigns an actual empty dict with type annotation
self._results: dict[str, Any] = {}
```

---

## 7. Mutable Default Arguments — never use list/dict as default

```python
# ❌ Wrong — Ruff B006 — same list shared across ALL calls
def select_features(cols: list[str] = []) -> list[str]:
    ...

# ✅ Correct — None as sentinel, new list created each call
def select_features(cols: list[str] | None = None) -> list[str]:
    if cols is None:
        cols = []
    ...
```

---

## 8. Nullable Attribute Narrowing for Pylance

When an attribute can be `None` initially (like sklearn's `mean_` before fitting),  
Pylance will complain unless you narrow it first.

```python
# ❌ Wrong — Pylance: "mean_" is possibly None
return self._scaler.mean_.tolist()

# ✅ Correct — assert narrows the type, Pylance is happy
assert self._scaler.mean_ is not None and self._scaler.scale_ is not None
return {
    "mean": self._scaler.mean_.tolist(),
    "std":  self._scaler.scale_.tolist(),
}
```

---

## 9. Variable Naming Conventions

| What | Rule | Example |
|------|------|---------|
| Variables | `snake_case` | `feature_cols`, `best_model` |
| Functions | `snake_case` | `load_data()`, `get_best_model()` |
| Classes | `PascalCase` | `ModelComparison`, `RegressionPreprocessor` |
| Constants | `UPPER_SNAKE_CASE` | `LOG_COLS`, `TARGET_COL` |
| Private | `_prefix` | `self._results`, `_check_is_fitted()` |
| Type params | `PascalCase` | `T`, `ModelT` |

```python
# ❌ Wrong — Ruff N803 (argument), N806 (variable in function)
def fit(self, X: np.ndarray) -> None:   # X is fine for ML convention — add noqa if needed
    MyVar = 10                           # should be my_var

# ✅ Correct
def fit(self, X: np.ndarray) -> None:  # noqa: N803  ← suppress when ML convention requires it
    my_var = 10
```

---

## 10. Exception Handling — always specific, never bare

```python
# ❌ Wrong — Ruff BLE001 — hides real errors
try:
    result = model.predict(X)
except:
    pass

# ❌ Wrong — still too broad
try:
    result = model.predict(X)
except Exception:
    pass

# ✅ Correct — catch exactly what you expect
try:
    result = model.predict(X)
except ValueError as e:
    logger.error("Prediction failed — bad input: %s", e)
    raise
```

---

## 11. Docstrings — every public function and class

```python
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute regression metrics between true and predicted values.

    Args:
        y_true:  Ground truth target values. Shape (n_samples,).
        y_pred:  Model predictions. Shape (n_samples,).
        prefix:  Optional string prefix for metric keys.

    Returns:
        Dictionary with keys: ``rmse``, ``mae``, ``r2``, ``mape``.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths.

    Example:
        >>> metrics = compute_metrics(y_true, y_pred, prefix="test_")
        >>> print(metrics["test_rmse"])
    """
```

- Private methods (`_name`) don't need docstrings
- One-liners are fine for very simple functions

---

## 12. f-strings vs % formatting

```python
# Logger → always % style
logger.info("Processing %d rows.", len(df))

# Regular strings → always f-string
raise ValueError(f"Column '{col}' not found in DataFrame.")
print(f"Best model: {best_model}, RMSE: {rmse:.4f}")
```

---

## 13. Full file template — copy this for every new file

```python
from __future__ import annotations

# stdlib
from pathlib import Path
from typing import Any

# third-party
import numpy as np
import pandas as pd

# local
from common.logger import get_logger

logger = get_logger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

MY_CONSTANT: str = "value"


# ── functions ─────────────────────────────────────────────────────────────────


def my_function(
    df: pd.DataFrame,
    threshold: float = 0.01,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """One-line summary.

    Args:
        df:         Input DataFrame.
        threshold:  Minimum value threshold.
        cols:       Columns to process. Defaults to all columns.

    Returns:
        Processed DataFrame.

    Raises:
        ValueError: If DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if cols is None:
        cols = df.columns.tolist()

    logger.info("Processing %d columns.", len(cols))
    return df


# ── classes ───────────────────────────────────────────────────────────────────


class MyProcessor:
    """Short description of what this class does."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = config or {}
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray) -> MyProcessor:
        """Fit on data. Returns self."""
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transformation."""
        self._check_is_fitted()
        return X

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call fit() first."
            )
```

---

## Quick checklist before committing any file

- [ ] `from __future__ import annotations` at the top (if needed)
- [ ] Imports in order: stdlib → third-party → local, blank line between groups
- [ ] No `sys.path.insert` anywhere
- [ ] Every function has full type annotations and `-> ReturnType`
- [ ] Constants are `UPPER_CASE` with type annotation
- [ ] Logger uses `%s` style not f-strings
- [ ] No mutable default arguments (`[]` or `{}` as defaults)
- [ ] Nullable attributes narrowed with `assert` before use
- [ ] No bare `except:` or `except Exception: pass`
- [ ] Every public function has a docstring
- [ ] Instance attributes typed inside `__init__`, not assigned as types
