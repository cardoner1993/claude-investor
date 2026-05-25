"""Test bootstrap.

1. Load `.env` so `@pytest.mark.network` smoke tests can hit APIs that
   need credentials (e.g. FRED).
2. Redirect `cache._DB` to a session-scoped tmp file BEFORE any test
   imports `gpt_investor.storage.cache`, so no test (or buggy fixture, or stray
   smoke-test snippet) can pollute the real `analyses.db`.
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_TMP_DB = Path(tempfile.gettempdir()) / "claude-investor-test.db"
os.environ["ANALYSES_DB"] = str(_TMP_DB)

# Eagerly override the module-level constant in case `cache` was imported
# transitively before conftest ran.
try:
    from gpt_investor.storage import cache as _cache
    _cache._DB = str(_TMP_DB)
except ImportError:
    pass


def pytest_sessionfinish(session, exitstatus):
    """Wipe the tmp DB at session end — leaving it can confuse the next run
    if the schema changes."""
    try:
        _TMP_DB.unlink(missing_ok=True)
    except OSError:
        pass
