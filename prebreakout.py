"""Pre-breakout consolidation scanner for BIST symbols.

This scanner does not assign a score.  A row is returned only when every
configured condition is satisfied.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import pandas as pd

from config import LOOKBACK_DAYS, MONTHLY_LOOKBACK_DAYS, sort_timeframes
from data_loader import (
    expected_latest_daily_date,
    fetch_tradingview_daily_snapshots,
    load_daily,
    load_hourly,
    merge_daily_snapshot,
    resample_monthly,
    resample_weekly,
)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.where(avg_loss.ne(0.0), 100.0)


def detect_prebreakout(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
    *,
    sma_period: int = 44,
    ema_fast: int = 10,
    ema_slow: int = 34,
    rsi_period: int = 14,
    rsi_min: float = 50.0,
    rsi_max: float = 65.0,
    resistance_lookback: int = 20,
    max_distance_pct: float = 5.0,
    consolidation_bars: int = 10,
    max_range_pct: float = 8.0,
    recent_volume_bars: int = 5,
    baseline_volume_bars: int = 20,
    min_recent_volume_ratio: float = 1.0,
) -> Optional[dict]:
    """Return a candidate only if all pre-breakout conditions pass."""
    required = {"high", "low", "close", "volume"}
    if bars is None or bars.empty or not required.issubset(bars.columns):
        return None

    df = bars.copy().sort_index()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    min_bars = max(
        sma_period,
        ema_slow,
        rsi_period + 1,
        resistance_lookback + 1,
        consolidation_bars,
        recent_volume_bars + baseline_volume_bars,
    )
    if len(df) < min_bars:
        return None

    close_series = df["close"]
    sma44 = close_series.rolling(sma_period).mean().iloc[-1]
    ema10 = close_series.ewm(span=ema_fast, adjust=False).mean().iloc[-1]
    ema34 = close_series.ewm(span=ema_slow, adjust=False).mean().iloc[-1]
    rsi14 = _rsi(close_series, rsi_period).iloc[-1]
    resistance = df["high"].shift(1).rolling(resistance_lookback).max().iloc[-1]

    latest = df.iloc[-1]
    close = float(latest["close"])
    if pd.isna(resistance) or resistance <= 0:
        return None
    distance_pct = (float(resistance) - close) / float(resistance) * 100.0

    recent_range = df.tail(consolidation_bars)
    range_pct = (
        (float(recent_range["high"].max()) - float(recent_range["low"].min()))
        / close
        * 100.0
    )

    recent_volume = df["volume"].tail(recent_volume_bars).mean()
    baseline_end = len(df) - recent_volume_bars
    baseline_start = max(0, baseline_end - baseline_volume_bars)
    baseline_volume = df["volume"].iloc[baseline_start:baseline_end].mean()
    volume_ratio = (
        float(recent_volume / baseline_volume)
        if pd.notna(baseline_volume) and baseline_volume > 0
        else 0.0
    )

    values = (sma44, ema10, ema34, rsi14, range_pct, volume_ratio)
    if any(pd.isna(value) for value in values):
        return None

    conditions = (
        close > float(sma44)  # The main price filter is SMA44, not EMA44.
        and float(ema10) > float(ema34)
        and rsi_min <= float(rsi14) <= rsi_max
        and 0.0 <= distance_pct <= max_distance_pct
        and range_pct <= max_range_pct
        and volume_ratio >= min_recent_volume_ratio
    )
    if not conditions:
        return None

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "bar_time": df.index[-1],
        "close": round(close, 4),
        "sma44": round(float(sma44), 4),
        "ema10": round(float(ema10), 4),
        "ema34": round(float(ema34), 4),
        "rsi14": round(float(rsi14), 2),
        "resistance": round(float(resistance), 4),
        "distance_to_resistance_pct": round(distance_pct, 2),
        "consolidation_range_pct": round(range_pct, 2),
        "recent_volume_ratio": round(volume_ratio, 2),
    }


def scan_prebreakout_universe(
    symbols: list[str],
    timeframes: list[str],
    *,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    use_cache: bool = True,
    refresh_prices: bool = False,
    max_workers: int = 8,
    **detector_settings,
) -> pd.DataFrame:
    """Scan BIST symbols for pre-breakout consolidation candidates."""
    symbols = [str(symbol).upper().strip() for symbol in symbols]
    timeframes = sort_timeframes(timeframes)
    snapshots = (
        fetch_tradingview_daily_snapshots(symbols)
        if refresh_prices and any(tf in {"1D", "1W", "1M"} for tf in timeframes)
        else {}
    )

    def _load_symbol(symbol: str) -> tuple[str, dict[str, pd.DataFrame]]:
        frames: dict[str, pd.DataFrame] = {}
        daily_tfs = [tf for tf in timeframes if tf in {"1D", "1W", "1M"}]
        if daily_tfs:
            days = MONTHLY_LOOKBACK_DAYS if "1M" in daily_tfs else LOOKBACK_DAYS
            daily = load_daily(symbol, days=days, use_cache=use_cache, refresh=refresh_prices)
            daily = merge_daily_snapshot(symbol, daily, snapshots.get(symbol))
            if refresh_prices and (
                daily.empty or daily.index[-1].date() < expected_latest_daily_date()
            ):
                daily = pd.DataFrame(columns=daily.columns)
            if "1D" in daily_tfs:
                frames["1D"] = daily
            if "1W" in daily_tfs:
                frames["1W"] = resample_weekly(daily)
            if "1M" in daily_tfs:
                frames["1M"] = resample_monthly(daily)
        if "1H" in timeframes:
            frames["1H"] = load_hourly(
                symbol,
                use_cache=use_cache,
                refresh=refresh_prices,
            )
        return symbol, frames

    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    total = len(symbols)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_load_symbol, symbol) for symbol in symbols]
        for future in as_completed(futures):
            symbol, frames = future.result()
            for timeframe, frame in frames.items():
                loaded[(symbol, timeframe)] = frame
            done += 1
            if progress_callback:
                progress_callback(done, total, symbol)

    rows: list[dict] = []
    for symbol in symbols:
        for timeframe in timeframes:
            row = detect_prebreakout(
                loaded.get((symbol, timeframe), pd.DataFrame()),
                symbol,
                timeframe,
                **detector_settings,
            )
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    tf_rank = {"1H": 0, "1D": 1, "1W": 2, "1M": 3}
    result["_tf"] = result["timeframe"].map(tf_rank).fillna(9)
    result = result.sort_values(
        ["_tf", "distance_to_resistance_pct", "recent_volume_ratio"],
        ascending=[True, True, False],
    )
    return result.drop(columns="_tf").reset_index(drop=True)
