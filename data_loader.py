"""Price loading for BIST breakout scanner (1H / 1D / 1W / 1M)."""

from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import (
    BIST50_CACHE,
    CACHE_DAILY,
    CACHE_HOURLY,
    DEFAULT_WATCHLIST,
    HOURLY_PERIOD,
    LOOKBACK_DAYS,
    MONTHLY_LOOKBACK_DAYS,
    STOCKDNA_UNIVERSE,
    UNIVERSE_CACHE,
    UNIVERSE_CHOICES,
    UNIVERSE_BIST10,
    UNIVERSE_BIST50,
    UNIVERSE_BIST_ALL,
    UNIVERSE_VIOP,
    YAHOO_TICKER_MAP,
    YFINANCE_SUFFIX,
    ensure_dirs,
)
from viop_loader import load_viop_symbols

_YF_LOCK = threading.Lock()
_MARKET_TZ = ZoneInfo("Europe/Istanbul")


def _previous_weekday(value: date) -> date:
    while value.weekday() >= 5:
        value -= timedelta(days=1)
    return value


def expected_latest_daily_date(now: datetime | None = None) -> date:
    """Latest BIST session that should be available from an EOD data feed."""
    local_now = now.astimezone(_MARKET_TZ) if now and now.tzinfo else (now or datetime.now(_MARKET_TZ))
    candidate = local_now.date()
    if candidate.weekday() >= 5:
        return _previous_weekday(candidate)
    # Before the evening data update, the latest completed session is yesterday.
    if local_now.time() < time(18, 30):
        return _previous_weekday(candidate - timedelta(days=1))
    return candidate


def fetch_tradingview_daily_snapshots(
    symbols: list[str],
    now: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Return the latest completed daily OHLCV bar from TradingView.

    This is a lightweight fallback for the newest bar when Yahoo throttles the
    Streamlit server. It is deliberately disabled during the live BIST session
    so an incomplete intraday candle is never treated as a completed day.
    """
    wanted = {str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()}
    if not wanted:
        return {}

    local_now = now.astimezone(_MARKET_TZ) if now and now.tzinfo else (now or datetime.now(_MARKET_TZ))
    if local_now.weekday() < 5 and time(10, 0) <= local_now.time() < time(18, 30):
        return {}

    try:
        from tradingview_screener import Query

        _, raw = (
            Query()
            .set_markets("turkey")
            .select("name", "exchange", "open", "high", "low", "close", "volume")
            .limit(1000)
            .get_scanner_data()
        )
    except Exception:
        return {}

    if raw is None or raw.empty:
        return {}

    trade_date = pd.Timestamp(expected_latest_daily_date(local_now))
    snapshots: dict[str, pd.DataFrame] = {}
    for _, row in raw.iterrows():
        symbol = str(row.get("name", "")).upper().strip()
        if symbol not in wanted or str(row.get("exchange", "")).upper() != "BIST":
            continue
        values = {
            key: pd.to_numeric(row.get(key), errors="coerce")
            for key in ("open", "high", "low", "close", "volume")
        }
        if pd.isna(values["close"]):
            continue
        frame = pd.DataFrame([values], index=pd.DatetimeIndex([trade_date], name="date"))
        snapshots[symbol] = _normalize_ohlcv(frame)
        snapshots[symbol].index.name = "date"
    return snapshots


def merge_daily_snapshot(
    symbol: str,
    daily: pd.DataFrame,
    snapshot: pd.DataFrame | None,
    *,
    persist: bool = True,
) -> pd.DataFrame:
    """Merge a TradingView completed bar into daily history and its cache."""
    if snapshot is None or snapshot.empty:
        return daily
    merged = pd.concat([daily, snapshot]) if not daily.empty else snapshot.copy()
    merged = _normalize_ohlcv(merged)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.index.name = "date"
    if persist and not merged.empty:
        path = CACHE_DAILY / f"{symbol.upper().strip()}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path)
    return merged


def yahoo_ticker(symbol: str) -> str:
    sym = symbol.upper().strip()
    return YAHOO_TICKER_MAP.get(sym, f"{sym}{YFINANCE_SUFFIX}")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df.loc[:, ~df.columns.duplicated()]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df[keep].copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index().dropna(subset=["close"])


def load_universe_symbols() -> list[str]:
    """Load all BIST symbols from cache or TradingView."""
    ensure_dirs()
    for path in (UNIVERSE_CACHE, STOCKDNA_UNIVERSE):
        if path.is_file():
            try:
                col = "symbol" if "symbol" in pd.read_csv(path, nrows=0).columns else None
                df = pd.read_csv(path)
                key = col or ("Symbol" if "Symbol" in df.columns else df.columns[0])
                symbols = df[key].dropna().astype(str).str.upper().str.strip().tolist()
                if symbols:
                    if path != UNIVERSE_CACHE:
                        pd.DataFrame({"symbol": symbols}).to_csv(UNIVERSE_CACHE, index=False)
                    return symbols
            except Exception:
                pass

    try:
        from tradingview_screener import get_all_symbols

        raw_symbols = get_all_symbols(market="turkey")
        symbols = []
        for raw in raw_symbols:
            text = str(raw).upper().strip()
            if not text.startswith("BIST:"):
                continue
            symbol = text.split(":", 1)[1]
            if symbol.startswith(("XU", "XB", "XT", "XS", "XY")):
                continue
            symbols.append(symbol)
        symbols = sorted(set(symbols))
        if symbols:
            pd.DataFrame({"symbol": symbols}).to_csv(UNIVERSE_CACHE, index=False)
            return symbols
    except Exception:
        pass
    return DEFAULT_WATCHLIST.copy()


def _read_symbol_column(path: Path) -> list[str]:
    df = pd.read_csv(path)
    col = "symbol" if "symbol" in df.columns else ("Symbol" if "Symbol" in df.columns else df.columns[0])
    return df[col].dropna().astype(str).str.upper().str.strip().tolist()


def load_bist50_symbols() -> list[str]:
    """Load the BIST 50 list from cache or the built-in fallback list."""
    ensure_dirs()
    if BIST50_CACHE.is_file():
        try:
            symbols = _read_symbol_column(BIST50_CACHE)
            if symbols:
                return sorted(set(symbols))
        except Exception:
            pass

    symbols = [
        "AEFES", "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BERA", "BIMAS", "BRSAN", "CCOLA",
        "CIMSA", "DOAS", "EKGYO", "ENKAI", "EREGL", "EUPWR", "FROTO", "GARAN", "GUBRF", "HEKTS",
        "ISCTR", "KCHOL", "KONTR", "KOZAA", "KOZAL", "KRDMD", "MAVI", "MGROS", "MIATK", "ODAS",
        "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "SOKM", "TABGD", "TAVHL", "TCELL",
        "THYAO", "TKFEN", "TOASO", "TSKB", "TTKOM", "TUPRS", "ULKER", "VAKBN", "YEOTK", "YKBNK",
    ]
    pd.DataFrame({"symbol": symbols}).to_csv(BIST50_CACHE, index=False)
    return symbols


def resolve_universe_symbols(
    choice: str,
    bist_all: list[str] | None = None,
    *,
    max_symbols: int | None = None,
) -> tuple[list[str], str, int]:
    """
    Resolve sidebar universe choice to a symbol list.

    Returns (symbols, sample_mode, universe_total) where sample_mode is
    'bist10', 'bist50', 'viop', 'full', or 'even'.
    """
    choice = choice or UNIVERSE_BIST10
    if choice == UNIVERSE_BIST10:
        top_10 = ["AKBNK", "ASELS", "BIMAS", "EREGL", "GARAN", "ISCTR", "KCHOL", "THYAO", "TUPRS", "YKBNK"]
        return top_10, "bist10", 10

    if choice == UNIVERSE_BIST50:
        symbols = load_bist50_symbols()
        return symbols, "bist50", len(symbols)

    if choice == UNIVERSE_VIOP:
        symbols = load_viop_symbols()
        return symbols, "viop", len(symbols)

    pool = bist_all if bist_all is not None else load_universe_symbols()
    total = len(pool)
    cap = max_symbols if max_symbols is not None else total
    symbols = select_scan_universe(pool, cap)
    mode = "full" if cap >= total else "even"
    return symbols, mode, total


def select_scan_universe(symbols: list[str], max_symbols: int) -> list[str]:
    """
    Choose symbols for a scan. When max_symbols < universe size, pick evenly
    across the sorted BIST list.
    """
    unique = sorted({s.upper().strip() for s in symbols if s and str(s).strip()})
    n = len(unique)
    if n == 0:
        return []
    if max_symbols >= n:
        return unique
    step = n / max_symbols
    return [unique[min(int(i * step), n - 1)] for i in range(max_symbols)]


def _read_csv_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return _normalize_ohlcv(df)


def _sibling_daily_cache(symbol: str) -> pd.DataFrame:
    return pd.DataFrame()


def fetch_daily(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    ticker = yahoo_ticker(symbol)
    end = datetime.now(_MARKET_TZ).date() + timedelta(days=1)
    start = end - timedelta(days=int(days * 1.6))
    try:
        with _YF_LOCK:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        return _normalize_ohlcv(df)
    except Exception:
        return pd.DataFrame()


def fetch_hourly(symbol: str, period: str = HOURLY_PERIOD) -> pd.DataFrame:
    ticker = yahoo_ticker(symbol)
    try:
        with _YF_LOCK:
            df = yf.download(
                ticker,
                period=period,
                interval="1h",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        out = _normalize_ohlcv(df)
        # Keep regular session bars only (drop zero-volume stale rows)
        if "volume" in out.columns:
            out = out[out["volume"].fillna(0) >= 0]
        return out
    except Exception:
        return pd.DataFrame()


def fetch_daily_range(symbol: str, start: date, end: date) -> pd.DataFrame:
    ticker = yahoo_ticker(symbol)
    try:
        with _YF_LOCK:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        return _normalize_ohlcv(df)
    except Exception:
        return pd.DataFrame()


def fetch_hourly_range(symbol: str, start: date, end: date) -> pd.DataFrame:
    ticker = yahoo_ticker(symbol)
    try:
        with _YF_LOCK:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1h",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        out = _normalize_ohlcv(df)
        if "volume" in out.columns:
            out = out[out["volume"].fillna(0) >= 0]
        return out
    except Exception:
        return pd.DataFrame()


_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    grouped = daily.sort_index().groupby(daily.sort_index().index.to_period("W-FRI"))
    weekly = grouped.agg(_OHLCV_AGG).dropna(subset=["close"])
    weekly.index = pd.DatetimeIndex(grouped.apply(lambda frame: frame.index[-1]).values)
    weekly.index.name = daily.index.name
    return weekly


def resample_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    daily = daily.sort_index()
    grouped = daily.groupby(daily.index.to_period("M"))
    monthly = grouped.agg(_OHLCV_AGG).dropna(subset=["close"])
    monthly.index = pd.DatetimeIndex(grouped.apply(lambda frame: frame.index[-1]).values)
    monthly.index.name = daily.index.name
    # Extrapolate the in-progress month's volume to a full-month estimate so
    # the volume-surge filter is comparable with completed months (a month has
    # ~21 trading sessions; without this, a mid-month scan can never fire).
    if len(monthly) >= 2 and "volume" in monthly.columns:
        last_start = monthly.index[-1].to_period("M").start_time
        elapsed = int((daily.index >= last_start).sum())
        if 0 < elapsed < 18:
            monthly["volume"] = monthly["volume"].astype(float)
            vol_col = monthly.columns.get_loc("volume")
            monthly.iloc[-1, vol_col] = monthly.iloc[-1, vol_col] * 21.0 / elapsed
    return monthly


def load_daily(
    symbol: str,
    days: int = LOOKBACK_DAYS,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    sym = symbol.upper()
    path = CACHE_DAILY / f"{sym}.csv"
    min_rows = min(60, days // 3) if days <= 500 else int(days * 0.45)

    df_cached = pd.DataFrame()
    if (use_cache or refresh) and path.is_file():
        try:
            df_cached = _read_csv_cache(path)
            fresh = not df_cached.empty and df_cached.index[-1].date() >= expected_latest_daily_date()
            if not refresh and fresh and len(df_cached) >= min_rows:
                return df_cached
        except Exception:
            pass

    # Incremental update logic
    if not df_cached.empty:
        try:
            last_cached_date = df_cached.index[-1].date()
            expected_date = expected_latest_daily_date()
            if refresh or last_cached_date < expected_date:
                start_fetch = last_cached_date - timedelta(days=5) # 5 days overlap for safety
                end_fetch = datetime.now(_MARKET_TZ).date() + timedelta(days=1)
                df_new = fetch_daily_range(sym, start_fetch, end_fetch)
                if not df_new.empty:
                    df_merged = pd.concat([df_cached, df_new])
                    df_merged = df_merged[~df_merged.index.duplicated(keep="last")].sort_index()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df_merged.to_csv(path)
                    return df_merged
        except Exception:
            pass

    # Sibling daily cache or full download fallback
    df = _sibling_daily_cache(sym)
    if df.empty or len(df) < min_rows:
        df = fetch_daily(sym, days=days)
    if not df.empty and (use_cache or refresh):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    return df


def load_hourly(symbol: str, use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    sym = symbol.upper()
    path = CACHE_HOURLY / f"{sym}.csv"
    min_rows = 40

    df_cached = pd.DataFrame()
    if (use_cache or refresh) and path.is_file():
        try:
            df_cached = _read_csv_cache(path)
            fresh = not df_cached.empty and df_cached.index[-1].date() >= date.today() - timedelta(days=2)
            if not refresh and fresh and len(df_cached) >= min_rows:
                return df_cached
        except Exception:
            pass

    # Incremental update logic
    if not df_cached.empty:
        try:
            last_cached_date = df_cached.index[-1].date()
            if refresh or last_cached_date < datetime.now(_MARKET_TZ).date():
                start_fetch = last_cached_date - timedelta(days=3)
                end_fetch = datetime.now(_MARKET_TZ).date() + timedelta(days=1)
                df_new = fetch_hourly_range(sym, start_fetch, end_fetch)
                if not df_new.empty:
                    df_merged = pd.concat([df_cached, df_new])
                    df_merged = df_merged[~df_merged.index.duplicated(keep="last")].sort_index()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df_merged.to_csv(path)
                    return df_merged
        except Exception:
            pass

    df = fetch_hourly(sym)
    if not df.empty and (use_cache or refresh):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    return df


def load_bars(
    symbol: str,
    timeframe: str,
    *,
    use_cache: bool = True,
    refresh: bool = False,
    days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Return OHLCV for the requested timeframe key: 1H, 1D, 1W, 1M."""
    tf = timeframe.upper()
    if tf == "1H":
        return load_hourly(symbol, use_cache=use_cache, refresh=refresh)
    if tf == "1M":
        daily = load_daily(
            symbol,
            days=max(days, MONTHLY_LOOKBACK_DAYS),
            use_cache=use_cache,
            refresh=refresh,
        )
        return resample_monthly(daily)
    daily = load_daily(symbol, days=days, use_cache=use_cache, refresh=refresh)
    if tf == "1W":
        return resample_weekly(daily)
    return daily
