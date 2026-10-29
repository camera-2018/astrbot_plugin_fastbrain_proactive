"""Keep SDK import-time configuration away from the real AstrBot data."""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("ASTRBOT_ROOT", tempfile.mkdtemp(prefix="fastbrain-tests-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
