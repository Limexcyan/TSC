"""Small shared I/O helpers for TaSC paper scripts."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Optional


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None
