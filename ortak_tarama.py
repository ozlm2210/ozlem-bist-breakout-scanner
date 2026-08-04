"""Independent volume, DMI and EMA scans for the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from data_loader import load_bars, load_daily


@dataclass(frozen=True)
class OrtakTaramaResult:
    hacim: pd.DataFrame
    dmi: pd.DataFrame
    ema: pd.DataFrame
    failed: pd.DataFrame


def _last(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else float("nan")


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    denominator = volume.rolling(length).sum().replace(0, np.nan)
    return (close * volume).rolling(length).sum() / denominator


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    level = _ema(close, 12) - _ema(close, 26)
    return level, _ema(level, 9)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _dmi_adx(frame: pd.DataFrame, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )
    atr = _true_range(frame).ewm(alpha=1 / length, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    return plus_di, minus_di, dx.ewm(alpha=1 / length, adjust=False).mean()


def _cross_up_last(left: pd.Series, right: pd.Series) -> bool:
    common = left.dropna().index.intersection(right.dropna().index)
    if len(common) < 2:
        return False
    a = left.loc[common]
    b = right.loc[common]
    return bool(a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1])


def _last_bar_date(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    return pd.Timestamp(frame.index[-1]).strftime("%d.%m.%Y %H:%M")


def _resample_four_hour(hourly: pd.DataFrame) -> pd.DataFrame:
    """Combine each trading session's hourly candles into sequential 4-bar candles."""
    if hourly is None or hourly.empty:
        return pd.DataFrame()
    frame = hourly.sort_index().copy()
    frame["_session"] = pd.to_datetime(frame.index).date
    frame["_block"] = frame.groupby("_session").cumcount() // 4
    grouped = frame.groupby(["_session", "_block"], sort=True)
    result = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    result.index = pd.DatetimeIndex(grouped.apply(lambda part: part.index[-1]).values)
    return result.sort_index().dropna(subset=["close"])


def _today_vwap(hourly: pd.DataFrame) -> float:
    if hourly is None or hourly.empty:
        return float("nan")
    dates = pd.Series(pd.to_datetime(hourly.index).date, index=hourly.index)
    day = hourly.loc[dates == dates.iloc[-1]]
    if day.empty or day["volume"].sum() == 0:
        return float("nan")
    typical = (day["high"] + day["low"] + day["close"]) / 3
    return float((typical * day["volume"]).sum() / day["volume"].sum())


def _previous_year_camarilla_p(daily: pd.DataFrame) -> float:
    if daily is None or daily.empty:
        return float("nan")
    years = pd.Series(pd.to_datetime(daily.index).year, index=daily.index)
    previous = daily.loc[years == int(years.iloc[-1]) - 1]
    if previous.empty:
        return float("nan")
    return float((previous["high"].max() + previous["low"].min() + previous["close"].iloc[-1]) / 3)


def _scan_symbol(symbol: str, refresh: bool) -> tuple[dict | None, dict | None, dict | None]:
    daily = load_daily(symbol, days=650, use_cache=True, refresh=refresh)
    hourly = load_bars(symbol, "1H", use_cache=True, refresh=refresh)
    four_hour = _resample_four_hour(hourly)

    hacim_row = None
    if len(daily) >= 30 and len(hourly) >= 30 and len(four_hour) >= 35:
        price = _last(daily["close"])
        vwma20 = _last(_vwma(hourly["close"], hourly["volume"], 20))
        vwap = _today_vwap(hourly)
        volume_now = _last(daily["volume"])
        volume_previous = _last(daily["volume"].shift(1))
        volume_average = _last(daily["volume"].rolling(20).mean())
        volume_change = ((volume_now / volume_previous) - 1) * 100 if volume_previous > 0 else np.nan
        relative_volume = volume_now / volume_average if volume_average > 0 else np.nan
        macd_level, macd_signal = _macd(four_hour["close"])
        macd_level_last = _last(macd_level)
        macd_signal_last = _last(macd_signal)
        passed = bool(
            price > vwma20
            and price > vwap
            and relative_volume > 1.30
            and macd_level_last > macd_signal_last
            and volume_change > 10
        )
        if passed:
            hacim_row = {
                "Hisse": symbol,
                "Fiyat": round(price, 4),
                "Relatif hacim": round(relative_volume, 2),
                "Hacim değişimi %": round(volume_change, 2),
                "1S VWMA20": round(vwma20, 4),
                "Gün içi VWAP": round(vwap, 4),
                "4S MACD": round(macd_level_last, 4),
                "4S MACD sinyal": round(macd_signal_last, 4),
                "Son günlük mum": _last_bar_date(daily),
            }

    dmi_row = None
    if len(daily) >= 260 and len(hourly) >= 80:
        price = _last(daily["close"])
        _, minus_di, adx = _dmi_adx(hourly, 14)
        rsi14 = _rsi(hourly["close"], 14)
        camarilla_p = _previous_year_camarilla_p(daily)
        passed = bool(
            _cross_up_last(adx, minus_di)
            and _last(rsi14) > 45
            and not np.isnan(camarilla_p)
            and price > camarilla_p
        )
        if passed:
            dmi_row = {
                "Hisse": symbol,
                "Fiyat": round(price, 4),
                "1S ADX14": round(_last(adx), 2),
                "1S -DI14": round(_last(minus_di), 2),
                "1S RSI14": round(_last(rsi14), 2),
                "Önceki yıl Camarilla P": round(camarilla_p, 4),
                "Kesişim mumu": _last_bar_date(hourly),
            }

    ema_row = None
    if len(daily) >= 80 and len(four_hour) >= 40:
        daily_vwma20 = _vwma(daily["close"], daily["volume"], 20)
        daily_ema34 = _ema(daily["close"], 34)
        daily_ema10 = _ema(daily["close"], 10)
        daily_ema40 = _ema(daily["close"], 40)
        four_ema5 = _ema(four_hour["close"], 5)
        four_ema10 = _ema(four_hour["close"], 10)
        four_ema34 = _ema(four_hour["close"], 34)
        daily_passed = _cross_up_last(daily_vwma20, daily_ema34) and _cross_up_last(daily_ema10, daily_ema40)
        four_passed = _cross_up_last(four_ema5, four_ema34) and _cross_up_last(four_ema10, four_ema34)
        if daily_passed or four_passed:
            passed_parts = []
            if daily_passed:
                passed_parts.append("Günlük EMA")
            if four_passed:
                passed_parts.append("4 Saat EMA")
            ema_row = {
                "Hisse": symbol,
                "Geçtiği EMA taraması": ", ".join(passed_parts),
                "Günlük fiyat": round(_last(daily["close"]), 4),
                "Günlük VWMA20": round(_last(daily_vwma20), 4),
                "Günlük EMA34": round(_last(daily_ema34), 4),
                "Günlük EMA10": round(_last(daily_ema10), 4),
                "Günlük EMA40": round(_last(daily_ema40), 4),
                "4S EMA5": round(_last(four_ema5), 4),
                "4S EMA10": round(_last(four_ema10), 4),
                "4S EMA34": round(_last(four_ema34), 4),
                "Son günlük mum": _last_bar_date(daily),
                "Son 4S mum": _last_bar_date(four_hour),
            }

    return hacim_row, dmi_row, ema_row


def run_ortak_tarama(symbols: list[str], *, refresh: bool = True) -> OrtakTaramaResult:
    hacim_rows: list[dict] = []
    dmi_rows: list[dict] = []
    ema_rows: list[dict] = []
    failed_rows: list[dict] = []
    progress = st.progress(0, text="Tarama hazırlanıyor…")

    for position, symbol in enumerate(symbols, start=1):
        try:
            hacim_row, dmi_row, ema_row = _scan_symbol(symbol, refresh)
            if hacim_row:
                hacim_rows.append(hacim_row)
            if dmi_row:
                dmi_rows.append(dmi_row)
            if ema_row:
                ema_rows.append(ema_row)
        except Exception as exc:
            failed_rows.append({"Hisse": symbol, "Hata": str(exc)})
        progress.progress(position / max(len(symbols), 1), text=f"{symbol} yükleniyor ({position}/{len(symbols)})")

    progress.empty()
    return OrtakTaramaResult(
        hacim=pd.DataFrame(hacim_rows),
        dmi=pd.DataFrame(dmi_rows),
        ema=pd.DataFrame(ema_rows),
        failed=pd.DataFrame(failed_rows),
    )


def _download_excel(result: OrtakTaramaResult) -> bytes | None:
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            (result.hacim if not result.hacim.empty else pd.DataFrame({"Bilgi": ["Son mumda eşleşme yok."]})).to_excel(
                writer, sheet_name="Hacim", index=False
            )
            (result.dmi if not result.dmi.empty else pd.DataFrame({"Bilgi": ["Son mumda eşleşme yok."]})).to_excel(
                writer, sheet_name="DMI", index=False
            )
            (result.ema if not result.ema.empty else pd.DataFrame({"Bilgi": ["Son mumda eşleşme yok."]})).to_excel(
                writer, sheet_name="EMA", index=False
            )
        return output.getvalue()
    except Exception:
        return None


def _render_result_table(frame: pd.DataFrame, empty_text: str, key: str) -> None:
    if frame.empty:
        st.info(empty_text)
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "CSV indir",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


def render_ortak_tarama(symbols: list[str]) -> None:
    st.header("Hacim – DMI – EMA Taraması")
    st.caption(
        "Üç tarama bağımsız çalışır; bir hissenin diğer filtreleri de geçmesi gerekmez. "
        "Kesişim koşulları yalnızca en son mumda aranır."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        start = st.button("Taramayı başlat", type="primary", use_container_width=True)
    with col2:
        refresh = st.checkbox(
            "Veriyi güncelle",
            value=True,
            help="Açıkken önbellek güncel mumlarla tamamlanır; ilk Tüm BIST taraması daha uzun sürebilir.",
        )

    if start:
        with st.spinner("Hacim, DMI ve EMA koşulları taranıyor…"):
            st.session_state["ortak_tarama_result"] = run_ortak_tarama(symbols, refresh=refresh)

    result = st.session_state.get("ortak_tarama_result")
    if result is None:
        st.info("Evreni seçin ve **Taramayı başlat** düğmesine basın.")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Hacim", len(result.hacim))
    metric_columns[1].metric("DMI", len(result.dmi))
    metric_columns[2].metric("EMA", len(result.ema))

    hacim_tab, dmi_tab, ema_tab = st.tabs(["Hacim", "DMI", "EMA"])
    with hacim_tab:
        st.caption("Fiyat > 1S VWMA20 ve VWAP · RelVol > 1,30 · 4S MACD > sinyal · hacim değişimi > %10")
        _render_result_table(result.hacim, "Son mumda Hacim koşullarını geçen hisse yok.", "hacim_sonuclari")
    with dmi_tab:
        st.caption("1S ADX14, -DI14'ü son mumda yukarı keser · RSI14 > 45 · fiyat > önceki yıl Camarilla P")
        _render_result_table(result.dmi, "Son mumda DMI koşullarını geçen hisse yok.", "dmi_sonuclari")
    with ema_tab:
        st.caption("Günlük ve 4 saatlik EMA kesişimleri ayrı değerlendirilir; ikisinden birini geçen gösterilir.")
        _render_result_table(result.ema, "Son mumda EMA koşullarını geçen hisse yok.", "ema_sonuclari")

    excel = _download_excel(result)
    if excel:
        st.download_button(
            "Üç sonucu Excel indir",
            excel,
            file_name="ozlem_hacim_dmi_ema.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if not result.failed.empty:
        st.caption(f"Verisi alınamayan/işlenemeyen sembol: {len(result.failed)}")
