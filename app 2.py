"""BIST multi-timeframe breakout scanner — Streamlit UI."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from config import (
    STRICT_ATR_MULT,
    STRICT_VOL_MULT,
    TIMEFRAMES,
    TIMEFRAME_ORDER,
    UNIVERSE_CHOICES,
    UNIVERSE_BIST10,
    UNIVERSE_BIST_ALL,
    UNIVERSE_VIOP,
    DATA_DIR,
    YFINANCE_SUFFIX,
    ensure_dirs,
    sort_timeframes,
)
from data_loader import load_bars, load_daily, load_universe_symbols, resolve_universe_symbols
from viop_loader import viop_symbol_set, load_viop_symbols
from results_store import (
    cached_scan_available,
    format_scanned_at,
    load_scan_results,
    save_scan_results,
)
from scanner import filter_results, scan_universe

DISCLAIMER_URL = "#disclaimer"

CAMARILLA_CACHE = DATA_DIR / "camarilla_daily"
CAMARILLA_LEVEL_PRIORITY = (
    "10Y R5", "10Y R4", "10Y P",
    "5Y R5", "5Y R4", "5Y P",
    "1Y R5", "1Y R4", "1Y P",
)


def _cam_p(high: float, low: float, close: float) -> float:
    return (high + low + close) / 3.0


def _cam_r4(high: float, low: float, close: float) -> float:
    return close + 1.1 * (high - low) / 2.0


def _cam_r5(high: float, low: float, close: float) -> float:
    return (high / low) * close if low else float("nan")


def _cam_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns=lambda value: str(value).lower())
    out = out.loc[:, ~out.columns.duplicated()]
    keep = [name for name in ("open", "high", "low", "close", "volume") if name in out]
    out = out[keep]
    for name in keep:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["high", "low", "close"])


def _cam_ticker(symbol: str) -> str:
    return f"{symbol.upper().strip()}{YFINANCE_SUFFIX}"


def _cam_required_start(selected_levels: list[str], today: date) -> date:
    starts = [today.year - 1]
    if any(level.startswith("5Y") for level in selected_levels):
        starts.append((today.year // 5) * 5 - 5)
    if any(level.startswith("10Y") for level in selected_levels):
        starts.append((today.year // 10) * 10 - 10)
    return date(min(starts), 1, 1)


def _cam_read_cache(symbol: str) -> pd.DataFrame:
    path = CAMARILLA_CACHE / f"{symbol.upper()}.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return _cam_normalize(pd.read_csv(path, parse_dates=["date"], index_col="date"))
    except Exception:
        return pd.DataFrame()


def _cam_write_cache(symbol: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    CAMARILLA_CACHE.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out.index.name = "date"
    out.to_csv(CAMARILLA_CACHE / f"{symbol.upper()}.csv")


def _cam_extract_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        return _cam_normalize(raw)
    level0 = raw.columns.get_level_values(0)
    level1 = raw.columns.get_level_values(1)
    try:
        if ticker in level0:
            return _cam_normalize(raw[ticker])
        if ticker in level1:
            return _cam_normalize(raw.xs(ticker, axis=1, level=1))
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _cam_download_batch(symbols: list[str], start: date) -> dict[str, pd.DataFrame]:
    tickers = [_cam_ticker(symbol) for symbol in symbols]
    try:
        raw = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=(date.today() + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return {}
    return {symbol: _cam_extract_ticker(raw, _cam_ticker(symbol)) for symbol in symbols}


def _cam_live_snapshots(symbols: list[str]) -> dict[str, pd.DataFrame]:
    wanted = {symbol.upper().strip() for symbol in symbols}
    payload = {
        "markets": ["turkey"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "options": {"lang": "en"},
        "columns": ["name", "exchange", "open", "high", "low", "close", "volume"],
        "range": [0, 1000],
    }
    try:
        request = Request(
            "https://scanner.tradingview.com/turkey/scan",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}
    snapshots: dict[str, pd.DataFrame] = {}
    columns = payload["columns"]
    for item in body.get("data", []):
        values = item.get("d", [])
        if len(values) != len(columns):
            continue
        row = dict(zip(columns, values))
        symbol = str(row.get("name", "")).upper().strip()
        if symbol not in wanted or str(row.get("exchange", "")).upper() != "BIST":
            continue
        frame = pd.DataFrame(
            [{key: row.get(key) for key in ("open", "high", "low", "close", "volume")}],
            index=pd.DatetimeIndex([pd.Timestamp.today().normalize()], name="date"),
        )
        snapshots[symbol] = _cam_normalize(frame)
    return snapshots


def _cam_calculate_levels(daily: pd.DataFrame, today: date | None = None) -> dict[str, float]:
    if daily is None or daily.empty:
        return {}
    today = today or date.today()
    frame = daily.sort_index()
    annual = frame.groupby(frame.index.year).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).dropna(subset=["high", "low", "close"])
    blocks = {
        "1Y": (today.year - 1, today.year - 1),
        "5Y": ((today.year // 5) * 5 - 5, (today.year // 5) * 5 - 1),
        "10Y": ((today.year // 10) * 10 - 10, (today.year // 10) * 10 - 1),
    }
    levels: dict[str, float] = {}
    for label, (start_year, end_year) in blocks.items():
        block = annual.loc[(annual.index >= start_year) & (annual.index <= end_year)]
        if block.empty:
            continue
        high = float(block["high"].max())
        low = float(block["low"].min())
        close = float(block.iloc[-1]["close"])
        levels[f"{label} P"] = _cam_p(high, low, close)
        levels[f"{label} R4"] = _cam_r4(high, low, close)
        levels[f"{label} R5"] = _cam_r5(high, low, close)
    return levels


def _cam_detect(
    daily: pd.DataFrame,
    symbol: str,
    selected_levels: list[str],
    *,
    use_close: bool = False,
) -> Optional[dict]:
    if daily is None or len(daily) < 2:
        return None
    frame = daily.sort_index()
    levels = _cam_calculate_levels(frame)
    current, previous = frame.iloc[-1], frame.iloc[-2]
    source = "close" if use_close else "high"
    current_source, previous_source = float(current[source]), float(previous[source])
    current_close = float(current["close"])
    fresh, earlier = [], []
    for name in selected_levels:
        level = levels.get(name)
        if level is None or pd.isna(level):
            continue
        if current_source > level and previous_source <= level:
            fresh.append(name)
        elif current_close > level:
            earlier.append(name)
    matched = fresh or earlier
    if not matched:
        return None
    strongest = next(name for name in CAMARILLA_LEVEL_PRIORITY if name in matched)
    level_price = float(levels[strongest])
    return {
        "symbol": symbol.upper(),
        "status": "Bugün geçen" if fresh else "Daha önce geçen",
        "level": strongest,
        "level_price": round(level_price, 4),
        "close": round(current_close, 4),
        "high": round(float(current["high"]), 4),
        "cross_pct": round((current_source / level_price - 1.0) * 100.0, 2),
        "bar_time": frame.index[-1],
        "method": "Kapanış" if use_close else "Mum içi yüksek",
    }


def scan_camarilla_universe(
    symbols: list[str],
    selected_levels: list[str],
    *,
    use_close: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = 6,
) -> pd.DataFrame:
    symbols = [str(symbol).upper().strip() for symbol in symbols]
    CAMARILLA_CACHE.mkdir(parents=True, exist_ok=True)
    required_start = _cam_required_start(selected_levels, date.today())
    histories = {symbol: _cam_read_cache(symbol) for symbol in symbols}
    missing = [
        symbol for symbol, frame in histories.items()
        if frame.empty or frame.index[0].date() > required_start
    ]
    chunks = [missing[index:index + 25] for index in range(0, len(missing), 25)]
    downloaded: dict[str, pd.DataFrame] = {}
    if chunks:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as pool:
            futures = [pool.submit(_cam_download_batch, chunk, required_start) for chunk in chunks]
            for future in as_completed(futures):
                downloaded.update(future.result())
    snapshots = _cam_live_snapshots(symbols)
    rows: list[dict] = []
    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        frame = downloaded.get(symbol, histories.get(symbol, pd.DataFrame()))
        cached = histories.get(symbol, pd.DataFrame())
        if not cached.empty and not frame.empty:
            frame = _cam_normalize(pd.concat([cached, frame]))
        snapshot = snapshots.get(symbol)
        if snapshot is not None and not snapshot.empty:
            frame = _cam_normalize(pd.concat([frame, snapshot]))
        if not frame.empty:
            _cam_write_cache(symbol, frame)
            row = _cam_detect(frame, symbol, selected_levels, use_close=use_close)
            if row:
                rows.append(row)
        if progress_callback:
            progress_callback(index, total, symbol)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    status_rank = {"Bugün geçen": 0, "Daha önce geçen": 1}
    level_rank = {name: index for index, name in enumerate(CAMARILLA_LEVEL_PRIORITY)}
    result["_status"] = result["status"].map(status_rank).fillna(9)
    result["_level"] = result["level"].map(level_rank).fillna(99)
    result = result.sort_values(["_status", "_level", "cross_pct"], ascending=[True, True, False])
    return result.drop(columns=["_status", "_level"]).reset_index(drop=True)

st.set_page_config(
    page_title="Özlem BIST Breakout Scanner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.breakout-card {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    min-height: 130px;
}
.breakout-card.bullish {
    background: linear-gradient(145deg, #0d2818 0%, #1b4332 45%, #2d6a4f 100%);
    border-left: 5px solid #52b788;
}
.breakout-card.bearish {
    background: linear-gradient(145deg, #3b0a0a 0%, #6b1515 45%, #9b2226 100%);
    border-left: 5px solid #f4845f;
}
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
    margin-bottom: 0.55rem;
    flex-wrap: wrap;
}
.card-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    justify-content: flex-end;
}
.card-symbol {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.02em;
}
.card-badges span {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    color: #f8fafc;
}
.card-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    color: #f8fafc;
    margin-left: 0.25rem;
}
.card-badges .high52,
.card-pill.high52 {
    background: rgba(251, 191, 36, 0.25);
    color: #fde68a;
}
.card-stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.75rem;
    margin-bottom: 0.45rem;
}
.card-stat {
    font-size: 0.82rem;
    color: #e2e8f0;
}
.card-stat b {
    color: #ffffff;
    font-weight: 700;
}
.card-foot {
    font-size: 0.74rem;
    color: #cbd5e1;
    opacity: 0.92;
}
.summary-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.summary-metric {
    flex: 1 1 120px;
    min-width: 110px;
    padding: 0.5rem 0.65rem;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}
.summary-metric .sm-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.15rem;
    opacity: 0.9;
}
.summary-metric .sm-value {
    font-size: 0.95rem;
    font-weight: 700;
    line-height: 1.25;
    word-break: break-word;
}
.summary-metric.breakouts {
    background: linear-gradient(145deg, #1e1b4b 0%, #312e81 100%);
    border-left: 3px solid #a78bfa;
}
.summary-metric.breakouts .sm-label { color: #c4b5fd; }
.summary-metric.breakouts .sm-value { color: #ede9fe; }
.summary-metric.bullish {
    background: linear-gradient(145deg, #052e16 0%, #14532d 100%);
    border-left: 3px solid #4ade80;
}
.summary-metric.bullish .sm-label { color: #86efac; }
.summary-metric.bullish .sm-value { color: #dcfce7; }
.summary-metric.bearish {
    background: linear-gradient(145deg, #450a0a 0%, #7f1d1d 100%);
    border-left: 3px solid #f87171;
}
.summary-metric.bearish .sm-label { color: #fca5a5; }
.summary-metric.bearish .sm-value { color: #fee2e2; }
.summary-metric.symbols {
    background: linear-gradient(145deg, #0c4a6e 0%, #075985 100%);
    border-left: 3px solid #38bdf8;
}
.summary-metric.symbols .sm-label { color: #7dd3fc; }
.summary-metric.symbols .sm-value { color: #e0f2fe; }
.summary-metric.scanned {
    background: linear-gradient(145deg, #451a03 0%, #78350f 100%);
    border-left: 3px solid #fbbf24;
}
.summary-metric.scanned .sm-label { color: #fcd34d; }
.summary-metric.scanned .sm-value { color: #fef3c7; font-size: 0.82rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

_DIR_STYLE = {
    "bullish": ("🟢 Bullish", "#16a34a"),
    "bearish": ("🔴 Bearish", "#dc2626"),
}



_INDEX_SYMBOLS = frozenset({"XU100", "XU050", "XU030"})

_DISCLAIMER_SIDEBAR = """
**Not financial advice.** For informational, educational, and research use only.

- Scan results are **algorithmic outputs** — not buy/sell recommendations  
- Trading securities involves **substantial risk of loss**  
- Market data may be **delayed, incomplete, or inaccurate**  
- Past breakouts do **not** guarantee future performance  
- Authors are **not** SPK-licensed investment advisers  
- Consult a **qualified financial adviser** before trading  

[Full disclaimer]({url})
""".format(
    url=DISCLAIMER_URL
)


def _render_disclaimer_banner() -> None:
    st.markdown(
        f"""
<div style="background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.35);
border-radius:10px;padding:0.75rem 1rem;margin-bottom:1rem;">
<p style="color:#fde68a;margin:0;font-size:0.88rem;line-height:1.5;">
<strong>⚠️ Disclaimer:</strong> This app is for <strong>research and education only</strong> — not investment advice.
Breakout signals are not recommendations to trade. You may lose capital. Data from third-party sources
may be delayed or wrong.
<a href="{DISCLAIMER_URL}" target="_blank" rel="noopener" style="color:#fcd34d;">Read full disclaimer</a>
</p></div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1800)
def get_git_version_info() -> dict:
    import subprocess
    from datetime import datetime
    from pathlib import Path

    # 1. Try local git command
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        date_str = subprocess.check_output(["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M:%S"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        if sha and date_str:
            return {"version": sha, "updated_at": date_str}
    except Exception:
        pass

    # 2. Fallback to file modified time
    try:
        app_file = Path(__file__).resolve()
        mtime = datetime.fromtimestamp(app_file.stat().st_mtime)
        return {"version": "v1.2.0 (local)", "updated_at": mtime.strftime("%Y-%m-%d %H:%M:%S")}
    except Exception:
        pass

    return {"version": "v1.2.0", "updated_at": "Unknown"}


def _render_disclaimer_sidebar() -> None:
    st.divider()
    st.markdown("#### ⚖️ Legal disclaimer")
    st.markdown(_DISCLAIMER_SIDEBAR)
    
    # Render version information
    info = get_git_version_info()
    st.divider()
    st.caption(
        f"**App Version**: `{info['version']}`  \n"
        f"**Last Updated**: {info['updated_at']}"
    )


def _render_summary_metrics(
    *,
    breakouts: int,
    bullish: int,
    bearish: int,
    symbols_scanned: int | str,
    scanned_label: str,
) -> None:
    html = f"""
<div class="summary-metrics">
  <div class="summary-metric breakouts">
    <div class="sm-label">Breakouts</div>
    <div class="sm-value">{breakouts}</div>
  </div>
  <div class="summary-metric bullish">
    <div class="sm-label">Bullish</div>
    <div class="sm-value">{bullish}</div>
  </div>
  <div class="summary-metric bearish">
    <div class="sm-label">Bearish</div>
    <div class="sm-value">{bearish}</div>
  </div>
  <div class="summary-metric symbols">
    <div class="sm-label">Symbols Scanned</div>
    <div class="sm-value">{symbols_scanned}</div>
  </div>
  <div class="summary-metric scanned">
    <div class="sm-label">Last Scanned</div>
    <div class="sm-value">{scanned_label}</div>
  </div>
</div>
"""
    _render_card_html(html)



def _render_last_scan_panel(meta: dict, results: pd.DataFrame | None = None) -> None:
    scanned_at = meta.get("scanned_at_display") or format_scanned_at(meta.get("scanned_at") or meta.get("saved_at"))
    if not scanned_at or scanned_at == "—":
        return

    bo_count_val = meta.get("breakout_count")
    if bo_count_val is None or pd.isna(bo_count_val):
        n_breakouts = len(results) if results is not None else 0
    else:
        try:
            n_breakouts = int(bo_count_val)
        except (ValueError, TypeError):
            n_breakouts = len(results) if results is not None else 0
    n_sym = meta.get("symbols_scanned", meta.get("symbols", "—"))
    timeframes = meta.get("timeframes", "")
    if isinstance(timeframes, list):
        timeframes = ", ".join(timeframes)
    mode = meta.get("mode", meta.get("breakout_mode", "—"))
    universe_label = meta.get("universe_choice", "")
    sample = meta.get("universe_sample", "")
    total = meta.get("universe_total")
    if sample == "even" and total:
        universe_txt = f"{n_sym} of {total} BIST symbols (even sample)"
    elif universe_label:
        universe_txt = f"{n_sym} symbols · {universe_label}"
    else:
        universe_txt = f"{n_sym} symbols"

    st.markdown(
        f"""
<div style="background:rgba(59,130,246,0.10);border:1px solid rgba(59,130,246,0.35);
border-radius:10px;padding:0.85rem 1.1rem;margin-bottom:0.85rem;">
<p style="color:#93c5fd;margin:0 0 0.35rem;font-size:0.95rem;font-weight:700;">
🕒 Last scanned: {scanned_at}
</p>
<p style="color:#cbd5e1;margin:0;font-size:0.84rem;line-height:1.55;">
<strong>{n_breakouts}</strong> breakouts · <strong>{universe_txt}</strong> ·
<strong>{timeframes or "—"}</strong> · <strong>{mode}</strong>
</p>
<p style="color:#94a3b8;margin:0.35rem 0 0;font-size:0.76rem;">
Cached locally in <code>data_cache/scan_results.csv</code> and <code>data_cache/scan_info.csv</code>
</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_disclaimer_footer() -> None:
    info = get_git_version_info()
    st.markdown(
        f"""
---
<p id="disclaimer" style="font-size:0.75rem;color:#94a3b8;line-height:1.55;margin-top:1.5rem;">
<strong>Disclaimer:</strong> The BIST Breakout Scanner is provided for informational and educational
purposes only. It does not constitute investment, trading, tax, or legal advice and does not create an
adviser–client relationship. Trading in securities involves substantial risk of loss. Algorithmic scan
outputs are based on historical data and may be inaccurate. Past performance is not indicative of future
results. Use at your own risk.
<a href="{DISCLAIMER_URL}" target="_blank" rel="noopener" style="color:#94a3b8;">Full disclaimer</a>
·
<span style="color:#64748b;">Version: <code>{info['version']}</code> (Updated: {info['updated_at']})</span>
</p>
""",
        unsafe_allow_html=True,
    )


def _breakout_card_html(row: pd.Series) -> str:
    direction = str(row.get("direction", "")).lower()
    cls = "bullish" if direction == "bullish" else "bearish"
    dir_label = _DIR_STYLE.get(direction, ("—", ""))[0]
    tf = row.get("timeframe", "")
    tf_label = TIMEFRAMES[tf].label if tf in TIMEFRAMES else tf
    high52 = row.get("is_52w_high", False)
    if not isinstance(high52, bool):
        high52 = str(high52).strip().lower() in {"1", "true", "yes", "t"}
    break_pct = float(row.get("breakout_pct", 0))
    sign = "+" if break_pct >= 0 else ""
    vol = row.get("volume_ratio")
    vol_txt = f"{float(vol):.2f}×" if vol is not None and pd.notna(vol) else "—"
    tr_atr = row.get("tr_atr_ratio")
    bar_time = row.get("bar_time", "")
    is_strict = str(row.get("mode", "")).lower() == "strict"
    try:
        lookback_val = int(float(row.get("lookback", 20)))
    except (ValueError, TypeError):
        lookback_val = 20

    badges = [
        f'<span class="card-pill">{tf_label}</span>',
        f'<span class="card-pill">{dir_label}</span>',
    ]
    if high52:
        badges.append('<span class="card-pill high52">52W High</span>')
    if is_strict:
        badges.append('<span class="card-pill high52">Strict</span>')
    
    ml_conf = row.get("ml_confidence")
    if ml_conf is not None and pd.notna(ml_conf):
        badges.append(f'<span class="card-pill ml-conf" style="background: rgba(37,99,235,0.25); color: #bfdbfe; font-weight: bold;">ML Conf: {float(ml_conf):.0f}%</span>')

    stats_row1 = [
        f'<span class="card-stat">Close <b>₺{float(row["close"]):,.2f}</b></span>',
        f'<span class="card-stat">Break <b>{sign}{break_pct:.2f}%</b></span>',
        f'<span class="card-stat">Vol <b>{vol_txt}</b></span>',
    ]
    if tr_atr is not None and pd.notna(tr_atr):
        stats_row1.append(f'<span class="card-stat">TR/ATR <b>{float(tr_atr):.2f}×</b></span>')

    stats_row2 = [
        f'<span class="card-stat">Level <b>₺{float(row["level"]):,.2f}</b></span>',
        (
            f'<span class="card-stat">Prior H/L '
            f'<b>₺{float(row["prior_high"]):,.0f}</b> / <b>₺{float(row["prior_low"]):,.0f}</b></span>'
        ),
    ]

    return (
        f'<div class="breakout-card {cls}">'
        f'<div class="card-top">'
        f'<span class="card-symbol">{row["symbol"]}</span>'
        f'<span class="card-badges">{"".join(badges)}</span>'
        f"</div>"
        f'<div class="card-stat-row">{"".join(stats_row1)}</div>'
        f'<div class="card-stat-row">{"".join(stats_row2)}</div>'
        f'<span class="card-foot">Bar {bar_time} · Lookback {lookback_val} bars</span>'
        f"</div>"
    )


def _render_card_html(html: str) -> None:
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_breakout_cards(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No breakouts in this view.")
        return

    cols_per_row = 3
    for i in range(0, len(df), cols_per_row):
        chunk = df.iloc[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for c_idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[c_idx]:
                _render_card_html(_breakout_card_html(row))


def _style_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["direction"] = out["direction"].map(lambda d: _DIR_STYLE.get(d, ("—", ""))[0])
    out["timeframe"] = out["timeframe"].map(lambda t: TIMEFRAMES[t].label if t in TIMEFRAMES else t)
    out["is_52w_high"] = out["is_52w_high"].map(lambda x: "Yes" if x else "No")
    if "ml_confidence" in out.columns:
        out["ml_confidence"] = out["ml_confidence"].map(
            lambda x: f"{float(x):.1f}%" if x is not None and pd.notna(x) else "—"
        )
    if "mode" in out.columns:
        out["mode"] = out["mode"].map(lambda m: "Strict (ATR)" if m == "strict" else "Standard")
    rename = {
        "symbol": "Symbol",
        "timeframe": "Timeframe",
        "direction": "Direction",
        "mode": "Mode",
        "close": "Close",
        "level": "Break Level",
        "breakout_pct": "Break %",
        "volume_ratio": "Vol Ratio",
        "tr_atr_ratio": "TR/ATR",
        "true_range": "True Range",
        "atr": "ATR(14)",
        "prior_high": "Prior High",
        "prior_low": "Prior Low",
        "bar_time": "Bar Date",
        "lookback": "Lookback",
        "is_52w_high": "52W High?",
        "ml_confidence": "ML Confidence",
    }
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


def _chart(symbol: str, timeframe: str, level: float) -> go.Figure:
    df = load_bars(symbol, timeframe, use_cache=True)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{symbol} — no data")
        return fig

    tail_n = {"1H": 120, "1D": 120, "1W": 52, "1M": 36}.get(timeframe.upper(), 100)
    tail = df.tail(tail_n)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)

    fig.add_trace(
        go.Candlestick(
            x=tail.index,
            open=tail["open"],
            high=tail["high"],
            low=tail["low"],
            close=tail["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=level, line_dash="dash", line_color="#f59e0b", annotation_text="Break level", row=1, col=1)

    if "volume" in tail.columns:
        colors = ["#16a34a" if c >= o else "#dc2626" for c, o in zip(tail["close"], tail["open"])]
        fig.add_trace(
            go.Bar(x=tail.index, y=tail["volume"], marker_color=colors, name="Volume", showlegend=False),
            row=2,
            col=1,
        )

    tf_label = TIMEFRAMES.get(timeframe.upper())
    title = f"{symbol} — {tf_label.label if tf_label else timeframe} breakout"
    fig.update_layout(
        title=title,
        height=480,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d24",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig





def _confluence_card_html(symbol: str, direction: str, group_df: pd.DataFrame) -> str:
    cls = "bullish" if direction.lower() == "bullish" else "bearish"
    dir_label = "Bullish" if direction.lower() == "bullish" else "Bearish"
    
    tf_keys = list(TIMEFRAMES.keys())
    group_df = group_df.copy()
    group_df["_tf_order"] = group_df["timeframe"].map(lambda t: tf_keys.index(t) if t in tf_keys else 99)
    group_df = group_df.sort_values("_tf_order")
    
    badges = [f'<span class="card-pill">{dir_label}</span>']
    
    timeframe_rows_html = []
    for _, row in group_df.iterrows():
        tf = row["timeframe"]
        tf_label = TIMEFRAMES[tf].label if tf in TIMEFRAMES else tf
        close_val = float(row["close"])
        break_pct = float(row.get("breakout_pct", 0))
        sign = "+" if break_pct >= 0 else ""
        vol = row.get("volume_ratio")
        vol_txt = f"{float(vol):.2f}×" if vol is not None and pd.notna(vol) else "—"
        ml_conf = row.get("ml_confidence")
        ml_txt = f"{float(ml_conf):.0f}%" if ml_conf is not None and pd.notna(ml_conf) else "—"
        
        tf_badge_class = "high52" if tf in ("1W", "1M") else ""
        
        row_html = (
            f'<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px; margin-top: 6px; font-size: 0.82rem; color: #cbd5e1;">'
            f'  <span class="card-pill {tf_badge_class}" style="margin: 0; padding: 2px 6px; font-size: 0.72rem;">{tf_label}</span>'
            f'  <span>₺{close_val:,.2f} ({sign}{break_pct:.1f}%)</span>'
            f'  <span>Vol: <b>{vol_txt}</b></span>'
            f'  <span style="color: #93c5fd;">ML: <b>{ml_txt}</b></span>'
            f'</div>'
        )
        timeframe_rows_html.append(row_html)
        
    return (
        f'<div class="breakout-card {cls}" style="padding: 1.1rem; border-radius: 12px; margin-bottom: 12px; background: rgba(30,41,59,0.45); border: 1px solid rgba(255,255,255,0.08);">'
        f'<div class="card-top" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
        f'<span class="card-symbol" style="font-size: 1.25rem; font-weight: 800; color: white;">{symbol}</span>'
        f'<span class="card-badges">{"".join(badges)}</span>'
        f"</div>"
        f'{"".join(timeframe_rows_html)}'
        f"</div>"
    )


def _style_confluence_results(confluence_symbols) -> pd.DataFrame:
    rows = []
    for symbol, direction, group in confluence_symbols:
        tf_keys = list(TIMEFRAMES.keys())
        tfs = sorted(group["timeframe"].unique(), key=lambda t: tf_keys.index(t) if t in tf_keys else 99)
        tfs_str = ", ".join([TIMEFRAMES[t].label if t in TIMEFRAMES else t for t in tfs])
        
        row = {
            "Symbol": symbol,
            "Direction": "Bullish" if direction.lower() == "bullish" else "Bearish",
            "Timeframes": tfs_str,
        }
        
        ml_confs = [float(val) for val in group["ml_confidence"].dropna() if pd.notna(val)]
        row["Max ML Confidence"] = f"{max(ml_confs):.0f}%" if ml_confs else "—"
        
        for tf in ("1H", "1D", "1W", "1M"):
            tf_row = group[group["timeframe"] == tf]
            if not tf_row.empty:
                bp = float(tf_row.iloc[0].get("breakout_pct", 0))
                close_val = float(tf_row.iloc[0]["close"])
                row[f"{tf} Breakout"] = f"₺{close_val:,.1f} ({bp:+.1f}%)"
            else:
                row[f"{tf} Breakout"] = "—"
                
        rows.append(row)
        
    return pd.DataFrame(rows)


def render_confluence_view(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No breakouts found to analyze confluence.")
        return

    st.markdown("#### Filter Confluence Timeframes")
    col1, col2 = st.columns([2, 1])
    with col1:
        req_tfs = st.multiselect(
            "Require breakout in timeframes",
            options=["1H", "1D", "1W", "1M"],
            default=["1H", "1D", "1W", "1M"],
            key="conf_tfs_req",
        )
    with col2:
        min_tfs_count = st.slider(
            "Min matching timeframes",
            1,
            4,
            len(req_tfs) if req_tfs else 2,
            key="conf_tfs_min",
        )

    if not req_tfs:
        st.warning("Please select at least one timeframe to find confluence.")
        return

    groups = df.groupby(["symbol", "direction"])
    confluence_symbols = []
    for (symbol, direction), group in groups:
        tfs_in_group = set(group["timeframe"].unique())
        matching_tfs = tfs_in_group.intersection(req_tfs)
        if len(matching_tfs) >= min_tfs_count:
            confluence_symbols.append((symbol, direction, group))

    if not confluence_symbols:
        st.info("No confluence breakouts found matching the selected timeframe criteria.")
        return

    st.markdown(f"Found **{len(confluence_symbols)}** stocks with multi-timeframe confluence.")

    view = st.radio(
        "View mode",
        ["Cards", "Table"],
        horizontal=True,
        key="confluence_view_mode",
        label_visibility="collapsed",
    )

    if view == "Cards":
        cols_per_row = 3
        for i in range(0, len(confluence_symbols), cols_per_row):
            chunk = confluence_symbols[i : i + cols_per_row]
            cols = st.columns(cols_per_row)
            for c_idx, (symbol, direction, group) in enumerate(chunk):
                with cols[c_idx]:
                    _render_card_html(_confluence_card_html(symbol, direction, group))
    else:
        styled = _style_confluence_results(confluence_symbols)
        st.dataframe(styled, use_container_width=True, hide_index=True, key="df_confluence_table")


def render_breakout_tab(
    scan_symbols: list[str],
    *,
    universe_choice: str,
    universe_total: int,
    universe_sample: str,
) -> None:
    symbols = scan_symbols
    dir_filter = None if direction == "Both" else direction.lower()

    cached_df, cached_meta = load_scan_results()

    force_refresh = st.button(
        "Force Refresh Scan",
        type="primary",
        help="Re-run scan and overwrite local CSV cache.",
        key="breakout_force_refresh",
    )

    if not force_refresh and cached_df is not None and "breakout_results" not in st.session_state:
        st.session_state["breakout_results"] = cached_df
        st.session_state["breakout_scan_meta"] = cached_meta

    if force_refresh:
        progress = st.progress(0.0, text="Loading prices…")
        status = st.empty()

        def on_progress(done: int, total: int, label: str) -> None:
            progress.progress(done / max(total, 1), text=f"Loading {label} ({done}/{total})…")
            status.caption(label)

        with st.spinner("Scanning breakouts…"):
            raw = scan_universe(
                symbols,
                selected_tfs or ["1D"],
                mode=breakout_mode,
                progress_callback=on_progress,
                use_cache=use_cache,
                refresh_prices=True,
                vol_mult=vol_mult,
                lookback=lookback,
                atr_mult=atr_mult if breakout_mode == "strict" else None,
                direction_filter=dir_filter,
            )
            filtered = filter_results(
                raw,
                timeframes=selected_tfs,
                directions=[dir_filter] if dir_filter else None,
                min_vol_ratio=vol_mult,
                only_52w=only_52w,
            )

        progress.progress(1.0, text="Done")
        status.empty()

        scan_meta = {
            "symbols": len(symbols),
            "universe_total": universe_total,
            "universe_sample": universe_sample,
            "universe_choice": universe_choice,
            "timeframes": sort_timeframes(selected_tfs or ["1D"]),
            "mode": breakout_mode_label,
            "breakout_mode": breakout_mode,
            "direction": direction,
            "vol_mult": vol_mult,
            "lookback": lookback,
            "atr_mult": atr_mult if breakout_mode == "strict" else None,
            "only_52w": only_52w,
            "max_symbols": max_symbols,
        }
        save_scan_results(filtered, scan_meta)
        _, saved_meta = load_scan_results()
        st.session_state["breakout_results"] = filtered
        st.session_state["breakout_scan_meta"] = saved_meta or scan_meta
        scanned_display = format_scanned_at(
            st.session_state.get("breakout_scan_meta", {}).get("scanned_at")
        )
        st.success(
            f"Scan complete — {len(filtered)} breakouts saved at **{scanned_display}** "
            f"to `data_cache/scan_results.csv`."
        )

    panel_meta = st.session_state.get("breakout_scan_meta") or cached_meta
    panel_results = st.session_state.get("breakout_results", cached_df)
    if panel_meta and panel_results is not None:
        _render_last_scan_panel(panel_meta, panel_results)

    if "breakout_results" in st.session_state:
        results = st.session_state["breakout_results"]
        meta = st.session_state.get("breakout_scan_meta", {})
        n_sym = meta.get("symbols_scanned", meta.get("symbols", len(symbols)))

        cached_tfs_raw = meta.get("timeframes") or results["timeframe"].unique().tolist()
        if isinstance(cached_tfs_raw, str):
            cached_tfs = sort_timeframes([t.strip() for t in cached_tfs_raw.split(",") if t.strip()])
        else:
            cached_tfs = sort_timeframes(cached_tfs_raw)
        display_tfs = [t for t in cached_tfs if t in TIMEFRAMES]
        if selected_tfs and set(selected_tfs) != set(cached_tfs):
            st.info("Sidebar filters changed. Results below are from the last saved scan until you force refresh.")

        if results.empty:
            st.warning(f"No breakouts found across {n_sym} symbols on selected filters.")
        else:
            bull = int((results["direction"] == "bullish").sum())
            bear = int((results["direction"] == "bearish").sum())
            scanned_label = meta.get("scanned_at_display") or format_scanned_at(
                meta.get("scanned_at") or meta.get("saved_at"),
                short=True,
            )
            _render_summary_metrics(
                breakouts=len(results),
                bullish=bull,
                bearish=bear,
                symbols_scanned=n_sym,
                scanned_label=scanned_label,
            )
            if meta.get("mode"):
                st.caption(f"Mode: **{meta['mode']}** · cached results — use **Force Refresh Scan** to update")

            tf_tabs = st.tabs(["All", "Confluence 🌀"] + [TIMEFRAMES[t].label for t in display_tfs])

            def _show(df: pd.DataFrame, key: str) -> None:
                if not df.empty and "bar_time" in df.columns:
                    df = df.sort_values("bar_time", ascending=False)
                view = st.radio(
                    "View",
                    ["Cards", "Table"],
                    horizontal=True,
                    key=f"view_{key}",
                    label_visibility="collapsed",
                )

                if view == "Cards":
                    render_breakout_cards(df)
                else:
                    styled = _style_results(df)
                    st.dataframe(styled, use_container_width=True, hide_index=True, key=f"df_{key}")

                if not df.empty:
                    pick = st.selectbox(
                        "Chart symbol",
                        df["symbol"].tolist(),
                        key=f"chart_pick_{key}",
                    )
                    row = df[df["symbol"] == pick].iloc[0]
                    st.plotly_chart(
                        _chart(pick, row["timeframe"], float(row["level"])),
                        use_container_width=True,
                        key=f"plotly_{key}_{pick}",
                    )
                    st.download_button(
                        "Download CSV",
                        df.to_csv(index=False),
                        file_name=f"breakouts_{key}.csv",
                        mime="text/csv",
                        key=f"dl_{key}",
                    )

            with tf_tabs[0]:
                _show(results, "all")
            with tf_tabs[1]:
                render_confluence_view(results)
            for i, tf in enumerate(display_tfs, start=2):
                with tf_tabs[i]:
                    _show(results[results["timeframe"] == tf], tf.lower())

    elif not cached_scan_available():
        st.info("No cached scan yet. Configure settings and click **Force Refresh Scan**.")


def render_camarilla_tab(scan_symbols: list[str]) -> None:
    st.markdown("### Camarilla P–R4–R5 Taraması")
    st.caption(
        "Seçtiğiniz 1Y, 5Y ve 10Y Camarilla seviyelerini tarar. "
        "Aynı mumda birden fazla seviye geçilirse Pine kodundaki en güçlü seviye gösterilir."
    )
    if not camarilla_levels:
        st.warning("Taramak için en az bir Camarilla seviyesi seçin.")
        return

    run_scan = st.button(
        "Camarilla Taramasını Başlat",
        type="primary",
        key="camarilla_run",
    )
    if run_scan:
        progress = st.progress(0.0, text="Uzun dönem fiyat geçmişi yükleniyor…")
        status = st.empty()

        def on_progress(done: int, total: int, label: str) -> None:
            progress.progress(done / max(total, 1), text=f"{label} taranıyor ({done}/{total})…")
            status.caption(label)

        with st.spinner("Camarilla seviyeleri hesaplanıyor…"):
            results = scan_camarilla_universe(
                scan_symbols,
                camarilla_levels,
                use_close=camarilla_use_close,
                progress_callback=on_progress,
            )
        progress.progress(1.0, text="Tamamlandı")
        status.empty()
        st.session_state["camarilla_results"] = results
        st.session_state["camarilla_symbols_scanned"] = len(scan_symbols)

    if "camarilla_results" not in st.session_state:
        st.info("Seviyeleri seçip **Camarilla Taramasını Başlat** düğmesine basın.")
        return

    results = st.session_state["camarilla_results"]
    scanned = st.session_state.get("camarilla_symbols_scanned", len(scan_symbols))
    if results.empty:
        st.warning(f"{scanned} hissede seçilen seviyelere uygun sonuç bulunamadı.")
        return

    today_results = results[results["status"] == "Bugün geçen"]
    earlier_results = results[results["status"] == "Daha önce geçen"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Bugün geçen", len(today_results))
    col2.metric("Daha önce geçen", len(earlier_results))
    col3.metric("Taranan", scanned)

    def _show(frame: pd.DataFrame, key: str) -> None:
        if frame.empty:
            st.info("Bu bölümde sonuç bulunamadı.")
            return
        display = frame.rename(
            columns={
                "symbol": "Hisse",
                "status": "Durum",
                "level": "Seviye",
                "level_price": "Seviye fiyatı",
                "close": "Kapanış",
                "high": "Yüksek",
                "cross_pct": "Seviye üstü %",
                "bar_time": "Mum tarihi",
                "method": "Yöntem",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "CSV İndir",
            frame.to_csv(index=False),
            file_name=f"camarilla_{key}.csv",
            mime="text/csv",
            key=f"camarilla_download_{key}",
        )

    today_tab, earlier_tab, all_tab = st.tabs(["Bugün geçenler", "Daha önce geçenler", "Tümü"])
    with today_tab:
        _show(today_results, "bugun_gecenler")
    with earlier_tab:
        _show(earlier_results, "daha_once_gecenler")
    with all_tab:
        _show(results, "tum_sonuclar")


ensure_dirs()

st.markdown(
    """
<div style="background:linear-gradient(135deg,#1e3a5f 0%,#7c3aed 55%,#db2777 100%);
padding:1.2rem 1.5rem;border-radius:12px;margin-bottom:1rem;">
<h2 style="color:white;margin:0;">🚀 Özlem BIST Breakout Scanner</h2>
<p style="color:#e2e8f0;margin:0.4rem 0 0;">
Donchian breakouts with volume confirmation on <strong>1 Hour</strong>, <strong>1 Day</strong>,
<strong>1 Week</strong> and <strong>1 Month</strong> timeframes.
<br><br>
BIST hisselerini Donchian kanal kırılımı ve hacim teyidiyle; <strong>1 Saat</strong>,
<strong>1 Gün</strong>, <strong>1 Hafta</strong> ve <strong>1 Ay</strong> dönemlerinde tarar.
Araştırma ve eğitim amaçlıdır, yatırım tavsiyesi değildir.
</p></div>
""",
    unsafe_allow_html=True,
)

_render_disclaimer_banner()

universe = load_universe_symbols()

with st.sidebar:
    st.header("Market Regime")
    if "market_regime" not in st.session_state:
        with st.spinner("Classifying market regime..."):
            try:
                from ml_engine import get_market_regime_classification
                st.session_state["market_regime"] = get_market_regime_classification()
            except Exception:
                st.session_state["market_regime"] = {
                    "name": "Unknown (ML Offline)",
                    "desc": "Regime classifier is offline (missing dependencies or initialization error).",
                    "color": "#9aa0a6"
                }
            
    regime = st.session_state["market_regime"]
    st.markdown(
        f'<div style="background-color: {regime["color"]}18; padding: 12px; border-left: 4px solid {regime["color"]}; border-radius: 4px; margin-bottom: 15px;">'
        f'<span style="font-size: 0.78rem; text-transform: uppercase; color: #9aa0a6; font-weight: bold;">XU100 Durumu</span><br>'
        f'<span style="font-size: 1.1rem; font-weight: bold; color: {regime["color"]};">{regime["name"]}</span><br>'
        f'<span style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.35; display: inline-block; margin-top: 4px;">{regime["desc"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    if st.button("Refresh Market Regime", use_container_width=True, key="refresh_regime_btn"):
        try:
            from ml_engine import get_market_regime_classification
            st.session_state["market_regime"] = get_market_regime_classification()
        except Exception:
            st.session_state["market_regime"] = {
                "name": "Unknown (ML Offline)",
                "desc": "Regime classifier is offline.",
                "color": "#9aa0a6"
            }
        st.rerun()

    st.divider()
    st.header("Universe")
    universe_choice = st.selectbox(
        "Symbol universe",
        list(UNIVERSE_CHOICES),
        index=list(UNIVERSE_CHOICES).index(UNIVERSE_BIST10),
        help="Select the universe of stocks to scan.",
    )
    max_symbols = st.slider(
        "Max symbols (Tüm BIST only)",
        10,
        len(universe),
        len(universe),
        10,
        disabled=universe_choice != UNIVERSE_BIST_ALL,
        help="Evenly sample across BIST when less than the full market.",
    )
    if universe_choice == UNIVERSE_VIOP:
        if "viop_symbols" not in st.session_state:
            st.session_state.viop_symbols = viop_symbol_set()
        viop_count = len(st.session_state.viop_symbols)
        if st.button("Refresh VİOP list", use_container_width=True):
            st.session_state.viop_symbols = viop_symbol_set(refresh=True)
            st.rerun()
        st.caption(f"**{viop_count}** VİOP equity symbols (indices excluded).")
    elif universe_choice == UNIVERSE_BIST_ALL and max_symbols < len(universe):
        st.caption(
            f"Scanning **{max_symbols}** of **{len(universe)}** BIST symbols "
            "(evenly spaced across the index)."
        )

    scan_symbols, universe_sample, universe_total = resolve_universe_symbols(
        universe_choice,
        universe,
        max_symbols=max_symbols if universe_choice == UNIVERSE_BIST_ALL else None,
    )
    st.metric("Symbols to scan", len(scan_symbols))

    st.divider()
    scan_type = st.radio(
        "Tarama türü",
        ["Donchian Kırılımı", "Camarilla P–R4–R5"],
        horizontal=False,
        help="Donchian kırılımı veya uzun dönem Camarilla seviyeleri.",
    )

    if scan_type == "Donchian Kırılımı":
        st.header("Breakout Scan Settings")
        breakout_mode_label = st.selectbox(
            "Breakout mode",
            ["Standard", "Strict (ATR)"],
            index=0,
            help=(
                "Standard: Donchian + volume + strong close. "
                "Strict: adds true range > ATR multiplier × ATR(14); default 1.5× volume on 1D."
            ),
        )
        breakout_mode = "strict" if breakout_mode_label == "Strict (ATR)" else "standard"
        selected_tfs = st.multiselect(
            "Timeframes",
            options=list(TIMEFRAME_ORDER),
            default=["1H", "1D", "1W", "1M"],
            format_func=lambda k: TIMEFRAMES[k].label,
        )
        selected_tfs = sort_timeframes(selected_tfs)
        direction = st.selectbox("Direction", ["Both", "Bullish", "Bearish"], index=0)
        vol_default = STRICT_VOL_MULT if breakout_mode == "strict" else 1.25
        vol_mult = st.slider("Min volume ratio", 1.0, 3.0, vol_default, 0.05)
        lookback = st.slider("Donchian lookback (bars)", 5, 60, 20, 1)
        atr_mult = st.slider(
            "Min TR / ATR(14) ratio",
            0.8,
            2.0,
            STRICT_ATR_MULT,
            0.05,
            disabled=breakout_mode != "strict",
            help="Breakout bar true range must exceed this multiple of 14-bar ATR.",
        )
        only_52w = st.checkbox("52-week high breakouts only (1D/1W/1M)", value=False)
        use_cache = st.checkbox(
            "Use price cache",
            value=True,
            help="Speeds up later scans. Force Refresh still checks Yahoo for the newest bars.",
        )
        if breakout_mode == "strict":
            st.caption(
                "Strict: close > prior N-bar high/low · volume > threshold × 20-bar avg · "
                "true range > ATR mult × ATR(14) · strong close."
            )
        else:
            st.caption(
                "Standard: close > prior N-bar high/low + volume surge + strong close. "
                "Weekly (Fri close) and monthly bars resampled from daily data."
            )

        st.header("ML Confidence Models")
        if st.button("Train ML Model", use_container_width=True, help="Re-trains the Random Forest breakout classifier on stock histories."):
            from ml_engine import train_confidence_model
            with st.spinner("Training Breakout ML Model..."):
                m_breakout = train_confidence_model(use_cache=False)
                if m_breakout is not None:
                    st.success("Breakout model trained successfully!")
                else:
                    st.error("Failed to train model.")
    else:
        st.header("Camarilla P–R4–R5")
        st.info(
            "Önceki tamamlanmış **1, 5 ve 10 yıllık** dönemlerin Camarilla "
            "seviyelerini tarar."
        )
        camarilla_levels = st.multiselect(
            "Taranacak seviyeler",
            options=[
                "1Y P", "1Y R4", "1Y R5",
                "5Y P", "5Y R4", "5Y R5",
                "10Y P", "10Y R4", "10Y R5",
            ],
            default=["1Y R4", "1Y R5", "5Y R4", "5Y R5", "10Y R4", "10Y R5"],
            help="P seviyeleri daha erken sinyal verebilir; başlangıçta yalnız R4 ve R5 seçilidir.",
            key="camarilla_levels",
        )
        camarilla_method = st.radio(
            "Geçiş yöntemi",
            ["Mum içi yüksek", "Kapanış"],
            index=0,
            help="Mum içi yüksek Pine kodundaki varsayılan alarm davranışıdır.",
            key="camarilla_method",
        )
        camarilla_use_close = camarilla_method == "Kapanış"
        st.caption(
            "P = merkez pivot · R4 = güçlü direnç · R5 = üst hedef. "
            "İlk Tüm BIST taraması uzun tarih verisini bir kez indirip önbelleğe alır."
        )

    _render_disclaimer_sidebar()

if scan_type == "Donchian Kırılımı":
    render_breakout_tab(
        scan_symbols,
        universe_choice=universe_choice,
        universe_total=universe_total,
        universe_sample=universe_sample,
    )
else:
    render_camarilla_tab(scan_symbols)

_render_disclaimer_footer()
