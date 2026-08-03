"""Load BIST VİOP equity symbol list."""

from __future__ import annotations

from datetime import date

import pandas as pd

from config import VIOP_CACHE, ensure_dirs


VIOP_FALLBACK_SYMBOLS = [
    "AEFES", "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "DOAS",
    "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR",
    "KCHOL", "KOZAA", "KOZAL", "KRDMD", "MGROS", "ODAS", "OYAKC", "PETKM",
    "PGSUS", "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO", "TKFEN",
    "TOASO", "TSKB", "TTKOM", "TUPRS", "VAKBN", "YKBNK",
]


def _save_cache(symbols: list[str]) -> None:
    ensure_dirs()
    pd.DataFrame({"symbol": symbols, "updated": date.today().isoformat()}).to_csv(
        VIOP_CACHE, index=False
    )


def _load_cache() -> list[str]:
    if not VIOP_CACHE.exists():
        return []
    try:
        df = pd.read_csv(VIOP_CACHE)
        col = "symbol" if "symbol" in df.columns else ("Symbol" if "Symbol" in df.columns else df.columns[0])
        return sorted(df[col].dropna().astype(str).str.strip().str.upper().unique().tolist())
    except Exception:
        return []


def load_viop_symbols(refresh: bool = False) -> list[str]:
    """Return the cached VİOP equity list or the built-in fallback list."""
    if not refresh:
        cached = _load_cache()
        if cached:
            return cached

    symbols = sorted(set(VIOP_FALLBACK_SYMBOLS))
    _save_cache(symbols)
    return symbols


def viop_symbol_set(refresh: bool = False) -> frozenset[str]:
    return frozenset(load_viop_symbols(refresh=refresh))
