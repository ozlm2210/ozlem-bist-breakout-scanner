"""Özlem BIST ortak tarama — gönderilen Colab kodunun Streamlit uyarlaması.

Dört ana filtre birbirinden bağımsız hesaplanır:
1) Hacim
2) DMI
3) Günlük EMA
4) 4 Saat EMA

Ekranda ve Excel'de yalnız Kesisim_Ozeti gösterilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import time

import numpy as np
import pandas as pd
import streamlit as st
from tvDatafeed import Interval, TvDatafeed


# ============================================================
# AYARLAR — GÖNDERİLEN KOD İLE AYNI
# ============================================================

MAX_SYMBOL = 700
EXCHANGE = "BIST"

N_1H = 500
N_4H = 500
N_DAILY = 650

SLEEP_SEC = 0.10
RETRY = 2

DMI_CROSS_LOOKBACK = 1
GUNLUK_EMA_CROSS_LOOKBACK = 1
EMA_4SA_CROSS_LOOKBACK = 1


@dataclass(frozen=True)
class OrtakTaramaResult:
    ozet: pd.DataFrame
    hacim_sayisi: int
    dmi_sayisi: int
    gunluk_ema_sayisi: int
    ema_4sa_sayisi: int
    failed: pd.DataFrame


@st.cache_resource(show_spinner=False)
def _tv_client() -> TvDatafeed:
    return TvDatafeed()


# ============================================================
# VERİ ÇEKME
# ============================================================

def get_hist(symbol: str, interval: Interval, n_bars: int) -> pd.DataFrame | None:
    tv = _tv_client()

    for _ in range(RETRY):
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange=EXCHANGE,
                interval=interval,
                n_bars=n_bars,
            )

            if df is None or df.empty:
                time.sleep(SLEEP_SEC)
                continue

            df = df.copy()
            df.columns = [str(c).lower() for c in df.columns]

            need = ["open", "high", "low", "close", "volume"]
            for column in need:
                if column not in df.columns:
                    return None

            df = df[need].dropna()
            if len(df) < 50:
                return None

            return df

        except Exception:
            time.sleep(SLEEP_SEC)

    return None


# ============================================================
# İNDİKATÖRLER
# ============================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    denominator = volume.rolling(length).sum().replace(0, np.nan)
    return (close * volume).rolling(length).sum() / denominator


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    level = ema(close, fast) - ema(close, slow)
    sig = ema(level, signal)
    hist = level - sig
    return level, sig, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def dmi_adx(
    df: pd.DataFrame,
    length: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean().replace(0, np.nan)

    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1 / length, adjust=False)
        .mean()
        / atr
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1 / length, adjust=False)
        .mean()
        / atr
    )

    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()

    return plus_di, minus_di, adx


def cross_up_recent(a: pd.Series, b: pd.Series, lookback: int = 1) -> bool:
    a = a.dropna()
    b = b.dropna()
    common = a.index.intersection(b.index)

    if len(common) < lookback + 2:
        return False

    a = a.loc[common]
    b = b.loc[common]

    for i in range(1, lookback + 1):
        if a.iloc[-i - 1] <= b.iloc[-i - 1] and a.iloc[-i] > b.iloc[-i]:
            return True

    return False


def last_cross_date(a: pd.Series, b: pd.Series, lookback: int = 1) -> str:
    a = a.dropna()
    b = b.dropna()
    common = a.index.intersection(b.index)

    if len(common) < lookback + 2:
        return ""

    a = a.loc[common]
    b = b.loc[common]

    for i in range(1, lookback + 1):
        if a.iloc[-i - 1] <= b.iloc[-i - 1] and a.iloc[-i] > b.iloc[-i]:
            return str(a.index[-i])

    return ""


def last(series: pd.Series) -> float:
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan


def calc_vwap_today(df_1h: pd.DataFrame) -> float:
    try:
        frame = df_1h.copy()
        frame["date"] = frame.index.date
        last_date = frame["date"].iloc[-1]
        day = frame[frame["date"] == last_date].copy()

        if day.empty:
            return np.nan

        typical = (day["high"] + day["low"] + day["close"]) / 3
        denominator = day["volume"].cumsum().replace(0, np.nan)
        vwap_series = (typical * day["volume"]).cumsum() / denominator

        return float(vwap_series.iloc[-1])
    except Exception:
        return np.nan


def previous_year_camarilla_p(df_daily: pd.DataFrame) -> float:
    try:
        frame = df_daily.copy()
        frame["year"] = frame.index.year

        current_year = int(frame["year"].iloc[-1])
        prev_year = current_year - 1
        previous = frame[frame["year"] == prev_year].copy()

        if previous.empty:
            return np.nan

        high = previous["high"].max()
        low = previous["low"].min()
        close = previous["close"].iloc[-1]

        return float((high + low + close) / 3)
    except Exception:
        return np.nan


# ============================================================
# 1) HACİM EKRANI
# VWMA20 1sa < Fiyat
# VWAP < Fiyat
# Rel Vol > 1.3
# MACD 4sa Level > Signal
# Vol chg > 10 ZORUNLU
# ============================================================

def scan_hacim(
    symbol: str,
    df_d: pd.DataFrame | None,
    df_1h: pd.DataFrame | None,
    df_4h: pd.DataFrame | None,
) -> dict | None:
    if df_d is None or df_1h is None or df_4h is None:
        return None

    if len(df_d) < 30 or len(df_1h) < 30 or len(df_4h) < 60:
        return None

    price = last(df_d["close"])
    vwma20_1h_last = last(vwma(df_1h["close"], df_1h["volume"], 20))
    vwap_today = calc_vwap_today(df_1h)

    vol_today = last(df_d["volume"])
    vol_prev = last(df_d["volume"].shift(1))
    vol_avg20 = last(df_d["volume"].rolling(20).mean())

    vol_chg = ((vol_today / vol_prev) - 1) * 100 if vol_prev and vol_prev != 0 else np.nan
    rel_vol = vol_today / vol_avg20 if vol_avg20 and vol_avg20 != 0 else np.nan

    macd_level, macd_signal, _ = macd(df_4h["close"], 12, 26, 9)
    macd_level_last = last(macd_level)
    macd_signal_last = last(macd_signal)

    c1 = price > vwma20_1h_last
    c2 = price > vwap_today
    c3 = rel_vol > 1.30
    c4 = macd_level_last > macd_signal_last
    c5 = vol_chg > 10

    gecer = bool(c1 and c2 and c3 and c4 and c5)

    return {
        "Hisse": symbol,
        "Hacim": "EVET" if gecer else "HAYIR",
        "Hacim_Fiyat": round(price, 4),
        "Hacim_RelVol": round(rel_vol, 2),
        "Hacim_Vol_Chg_%": round(vol_chg, 2),
        "Hacim_MACD_Level": round(macd_level_last, 4),
        "Hacim_MACD_Signal": round(macd_signal_last, 4),
        "Hacim_Gecti": gecer,
    }


# ============================================================
# 2) DMI EKRANI
# ADX(14) 1H, -DI çizgisini SON MUMDA yukarı kesmiş
# RSI(14) 1H > 45
# Camarilla P < Fiyat
# ============================================================

def scan_dmi(
    symbol: str,
    df_d: pd.DataFrame | None,
    df_1h: pd.DataFrame | None,
) -> dict | None:
    if df_d is None or df_1h is None:
        return None

    if len(df_d) < 260 or len(df_1h) < 80:
        return None

    price = last(df_d["close"])

    _, minus_di, adx14 = dmi_adx(df_1h, 14)
    rsi14 = rsi(df_1h["close"], 14)
    cam_p = previous_year_camarilla_p(df_d)

    adx_last = last(adx14)
    minus_last = last(minus_di)
    rsi_last = last(rsi14)

    c1 = cross_up_recent(adx14, minus_di, lookback=DMI_CROSS_LOOKBACK)
    c2 = rsi_last > 45
    c3 = price > cam_p if not np.isnan(cam_p) else False

    cross_date = last_cross_date(adx14, minus_di, lookback=DMI_CROSS_LOOKBACK)
    gecer = bool(c1 and c2 and c3)

    return {
        "Hisse": symbol,
        "DMI": "EVET" if gecer else "HAYIR",
        "DMI_Fiyat": round(price, 4),
        "ADX14_1sa": round(adx_last, 2),
        "MinusDI14_1sa": round(minus_last, 2),
        "RSI14_1sa": round(rsi_last, 2),
        "Camarilla_P": round(cam_p, 4) if not np.isnan(cam_p) else np.nan,
        "ADX_MinusDI_Kesisim_Tarihi": cross_date,
        "DMI_Gecti": gecer,
    }


# ============================================================
# 3) GÜNLÜK EMA EKRANI
# VWMA20 günlük EMA34'ü SON MUMDA yukarı kesmiş
# EMA10 günlük EMA40'ı SON MUMDA yukarı kesmiş
# ============================================================

def scan_gunluk_ema(symbol: str, df_d: pd.DataFrame | None) -> dict | None:
    if df_d is None or len(df_d) < 80:
        return None

    price = last(df_d["close"])

    vwma20_d = vwma(df_d["close"], df_d["volume"], 20)
    ema34_d = ema(df_d["close"], 34)
    ema10_d = ema(df_d["close"], 10)
    ema40_d = ema(df_d["close"], 40)

    c1 = cross_up_recent(vwma20_d, ema34_d, lookback=GUNLUK_EMA_CROSS_LOOKBACK)
    c2 = cross_up_recent(ema10_d, ema40_d, lookback=GUNLUK_EMA_CROSS_LOOKBACK)

    d1 = last_cross_date(vwma20_d, ema34_d, lookback=GUNLUK_EMA_CROSS_LOOKBACK)
    d2 = last_cross_date(ema10_d, ema40_d, lookback=GUNLUK_EMA_CROSS_LOOKBACK)

    gecer = bool(c1 and c2)

    return {
        "Hisse": symbol,
        "Gunluk_EMA": "EVET" if gecer else "HAYIR",
        "Gunluk_Fiyat": round(price, 4),
        "VWMA20_Gunluk": round(last(vwma20_d), 4),
        "EMA34_Gunluk": round(last(ema34_d), 4),
        "EMA10_Gunluk": round(last(ema10_d), 4),
        "EMA40_Gunluk": round(last(ema40_d), 4),
        "VWMA20_EMA34_Kesisim_Tarihi": d1,
        "EMA10_EMA40_Kesisim_Tarihi": d2,
        "Gunluk_EMA_Gecti": gecer,
    }


# ============================================================
# 4) 4 SAAT EMA EKRANI
# EMA5 4sa EMA34'ü SON MUMDA yukarı kesmiş
# EMA10 4sa EMA34'ü SON MUMDA yukarı kesmiş
# ============================================================

def scan_4s_ema(symbol: str, df_4h: pd.DataFrame | None) -> dict | None:
    if df_4h is None or len(df_4h) < 80:
        return None

    price = last(df_4h["close"])

    ema5_4h = ema(df_4h["close"], 5)
    ema10_4h = ema(df_4h["close"], 10)
    ema34_4h = ema(df_4h["close"], 34)

    c1 = cross_up_recent(ema5_4h, ema34_4h, lookback=EMA_4SA_CROSS_LOOKBACK)
    c2 = cross_up_recent(ema10_4h, ema34_4h, lookback=EMA_4SA_CROSS_LOOKBACK)

    d1 = last_cross_date(ema5_4h, ema34_4h, lookback=EMA_4SA_CROSS_LOOKBACK)
    d2 = last_cross_date(ema10_4h, ema34_4h, lookback=EMA_4SA_CROSS_LOOKBACK)

    gecer = bool(c1 and c2)

    return {
        "Hisse": symbol,
        "EMA_4sa": "EVET" if gecer else "HAYIR",
        "EMA4sa_Fiyat": round(price, 4),
        "EMA5_4sa": round(last(ema5_4h), 4),
        "EMA10_4sa": round(last(ema10_4h), 4),
        "EMA34_4sa": round(last(ema34_4h), 4),
        "EMA5_EMA34_Kesisim_Tarihi": d1,
        "EMA10_EMA34_Kesisim_Tarihi": d2,
        "EMA_4sa_Gecti": gecer,
    }


# ============================================================
# ANA TARAMA + ORTAK KESİŞİM
# ============================================================

def run_ortak_tarama(symbols: list[str]) -> OrtakTaramaResult:
    symbols = [str(symbol).upper().strip() for symbol in symbols[:MAX_SYMBOL]]

    rows_hacim: list[dict] = []
    rows_dmi: list[dict] = []
    rows_gunluk: list[dict] = []
    rows_4s: list[dict] = []
    failed: list[dict] = []

    progress = st.progress(0, text="Tarama hazırlanıyor…")

    for i, symbol in enumerate(symbols, start=1):
        try:
            progress.progress(
                i / max(len(symbols), 1),
                text=f"{symbol} taranıyor ({i}/{len(symbols)})",
            )

            df_d = get_hist(symbol, Interval.in_daily, N_DAILY)
            time.sleep(SLEEP_SEC)

            df_1h = get_hist(symbol, Interval.in_1_hour, N_1H)
            time.sleep(SLEEP_SEC)

            df_4h = get_hist(symbol, Interval.in_4_hour, N_4H)
            time.sleep(SLEEP_SEC)

            if df_d is None and df_1h is None and df_4h is None:
                failed.append({"Hisse": symbol, "Sebep": "Veri alınamadı"})
                continue

            r1 = scan_hacim(symbol, df_d, df_1h, df_4h)
            r2 = scan_dmi(symbol, df_d, df_1h)
            r3 = scan_gunluk_ema(symbol, df_d)
            r4 = scan_4s_ema(symbol, df_4h)

            if r1 is not None:
                rows_hacim.append(r1)
            if r2 is not None:
                rows_dmi.append(r2)
            if r3 is not None:
                rows_gunluk.append(r3)
            if r4 is not None:
                rows_4s.append(r4)

        except Exception as exc:
            failed.append({"Hisse": symbol, "Sebep": str(exc)})

    progress.empty()

    df_hacim_all = pd.DataFrame(rows_hacim)
    df_dmi_all = pd.DataFrame(rows_dmi)
    df_gunluk_all = pd.DataFrame(rows_gunluk)
    df_4s_all = pd.DataFrame(rows_4s)
    df_failed = pd.DataFrame(failed)

    df_hacim = (
        df_hacim_all[df_hacim_all["Hacim_Gecti"] == True].copy()
        if not df_hacim_all.empty
        else pd.DataFrame()
    )
    df_dmi = (
        df_dmi_all[df_dmi_all["DMI_Gecti"] == True].copy()
        if not df_dmi_all.empty
        else pd.DataFrame()
    )
    df_gunluk = (
        df_gunluk_all[df_gunluk_all["Gunluk_EMA_Gecti"] == True].copy()
        if not df_gunluk_all.empty
        else pd.DataFrame()
    )
    df_4s = (
        df_4s_all[df_4s_all["EMA_4sa_Gecti"] == True].copy()
        if not df_4s_all.empty
        else pd.DataFrame()
    )

    set_hacim = set(df_hacim["Hisse"]) if not df_hacim.empty else set()
    set_dmi = set(df_dmi["Hisse"]) if not df_dmi.empty else set()
    set_gunluk = set(df_gunluk["Hisse"]) if not df_gunluk.empty else set()
    set_4s = set(df_4s["Hisse"]) if not df_4s.empty else set()

    all_passed = sorted(set_hacim | set_dmi | set_gunluk | set_4s)
    summary_rows: list[dict] = []

    for symbol in all_passed:
        gecen: list[str] = []

        if symbol in set_hacim:
            gecen.append("Hacim")
        if symbol in set_dmi:
            gecen.append("DMI")
        if symbol in set_gunluk:
            gecen.append("Gunluk_EMA")
        if symbol in set_4s:
            gecen.append("EMA_4sa")

        summary_rows.append(
            {
                "Hisse": symbol,
                "Gecen_Filtre_Sayisi": len(gecen),
                "Gectigi_Filtreler": ", ".join(gecen),
                "Hacim": "EVET" if symbol in set_hacim else "HAYIR",
                "DMI": "EVET" if symbol in set_dmi else "HAYIR",
                "Gunluk_EMA": "EVET" if symbol in set_gunluk else "HAYIR",
                "EMA_4sa": "EVET" if symbol in set_4s else "HAYIR",
            }
        )

    df_ozet = pd.DataFrame(summary_rows)

    if not df_ozet.empty:
        df_ozet = df_ozet.sort_values(
            ["Gecen_Filtre_Sayisi", "Hisse"],
            ascending=[False, True],
        ).reset_index(drop=True)
    else:
        df_ozet = pd.DataFrame({"Bilgi": ["Ortak geçen yok."]})

    return OrtakTaramaResult(
        ozet=df_ozet,
        hacim_sayisi=len(df_hacim),
        dmi_sayisi=len(df_dmi),
        gunluk_ema_sayisi=len(df_gunluk),
        ema_4sa_sayisi=len(df_4s),
        failed=df_failed,
    )


# ============================================================
# EXCEL — SADECE Kesisim_Ozeti
# ============================================================

def _download_excel(result: OrtakTaramaResult) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.ozet.to_excel(writer, sheet_name="Kesisim_Ozeti", index=False)
    return output.getvalue()


# ============================================================
# STREAMLIT EKRANI — SADECE İLK SAYFA
# ============================================================

def render_ortak_tarama(symbols: list[str]) -> None:
    st.header("BIST Ortak Tarama")
    st.caption(
        "Hacim, DMI, Günlük EMA ve 4 Saat EMA ayrı hesaplanır. "
        "Sonuçta yalnız Kesişim Özeti gösterilir; en çok filtreyi geçenler üsttedir."
    )

    with st.expander("Tarama koşulları", expanded=False):
        st.markdown(
            """
**Hacim:** Fiyat > 1S VWMA20 · Fiyat > gün içi VWAP · RelVol > 1,30 · 4S MACD > sinyal · hacim değişimi > %10  
**DMI:** 1S ADX14, -DI14'ü son mumda yukarı keser · 1S RSI14 > 45 · fiyat > önceki yıl Camarilla P  
**Günlük EMA:** VWMA20, EMA34'ü son mumda yukarı keser **ve** EMA10, EMA40'ı son mumda yukarı keser  
**4 Saat EMA:** EMA5, EMA34'ü son mumda yukarı keser **ve** EMA10, EMA34'ü son mumda yukarı keser
"""
        )

    if st.button("Taramayı başlat", type="primary", use_container_width=True):
        with st.spinner("Dört tarama çalışıyor…"):
            st.session_state["ortak_tarama_result"] = run_ortak_tarama(symbols)

    result = st.session_state.get("ortak_tarama_result")
    if result is None:
        st.info("Evreni seçin ve **Taramayı başlat** düğmesine basın.")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Hacim", result.hacim_sayisi)
    metric_columns[1].metric("DMI", result.dmi_sayisi)
    metric_columns[2].metric("Günlük EMA", result.gunluk_ema_sayisi)
    metric_columns[3].metric("4 Saat EMA", result.ema_4sa_sayisi)

    st.subheader("Kesişim Özeti")
    st.dataframe(result.ozet, use_container_width=True, hide_index=True)

    st.download_button(
        "Kesişim özetini Excel indir",
        data=_download_excel(result),
        file_name="ozlem_ortak_tarama_sadece_ilk_sayfa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    if not result.failed.empty:
        st.caption(f"Verisi alınamayan/işlenemeyen sembol: {len(result.failed)}")
