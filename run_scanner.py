#!/usr/bin/env python3
"""CLI for BIST breakout scanner."""

from __future__ import annotations

import argparse
import sys

from config import (
    STRICT_VOL_MULT,
    TIMEFRAMES,
    TIMEFRAME_ORDER,
    UNIVERSE_CHOICES,
    UNIVERSE_BIST50,
    UNIVERSE_BIST_ALL,
    UNIVERSE_VIOP,
    sort_timeframes,
)
from data_loader import load_universe_symbols, resolve_universe_symbols
from results_store import save_scan_results
from scanner import filter_results, scan_universe


def main() -> int:
    parser = argparse.ArgumentParser(description="BIST multi-timeframe breakout scanner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan universe for breakouts")
    scan.add_argument(
        "--timeframe",
        "-t",
        action="append",
        choices=list(TIMEFRAMES.keys()),
        help="Timeframe(s): 1H, 1D, 1W, 1M (repeatable)",
    )
    scan.add_argument("--max", type=int, default=500, help="Max symbols for BIST (even sample if < size)")
    scan.add_argument(
        "--universe",
        "-u",
        choices=["bist50", "viop", "bist"],
        default="bist",
        help="Symbol universe: bist50, viop, or bist",
    )
    scan.add_argument("--vol-mult", type=float, default=None, help="Min volume ratio vs average (strict default 1.5)")
    scan.add_argument("--lookback", type=int, default=20, help="Donchian lookback bars")
    scan.add_argument(
        "--mode",
        choices=["standard", "strict"],
        default="standard",
        help="standard = Donchian+vol+strong close; strict = adds TR > ATR expansion",
    )
    scan.add_argument("--atr-mult", type=float, default=1.2, help="Strict mode: min TR/ATR ratio")
    scan.add_argument("--direction", choices=["both", "bullish", "bearish"], default="both")
    scan.add_argument("--only-52w", action="store_true", help="52-week high breakouts only")
    scan.add_argument("--no-cache", action="store_true", help="Bypass local price cache")
    scan.add_argument("--csv", type=str, default="", help="Write results to CSV path (also saves app cache)")

    app = sub.add_parser("app", help="Launch Streamlit UI")
    app.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()

    if args.cmd == "app":
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(args.port)],
            check=False,
        )
        return 0

    timeframes = sort_timeframes(args.timeframe or list(TIMEFRAME_ORDER))
    bist_all = load_universe_symbols()
    universe_map = {
        "bist50": UNIVERSE_BIST50,
        "viop": UNIVERSE_VIOP,
        "bist": UNIVERSE_BIST_ALL,
    }
    universe_choice = universe_map[args.universe]
    symbols, universe_sample, universe_total = resolve_universe_symbols(
        universe_choice,
        bist_all,
        max_symbols=args.max if args.universe == "bist" else None,
    )
    dir_filter = None if args.direction == "both" else args.direction
    vol_mult = args.vol_mult
    if vol_mult is None:
        vol_mult = STRICT_VOL_MULT if args.mode == "strict" else 1.25

    print(f"Scanning {len(symbols)} symbols ({universe_choice}) on {', '.join(timeframes)} ({args.mode})…")

    def progress(done: int, total: int, label: str) -> None:
        if done % 25 == 0 or done == total:
            print(f"  [{done}/{total}] {label}")

    df = scan_universe(
        symbols,
        timeframes,
        mode=args.mode,
        progress_callback=progress,
        use_cache=not args.no_cache,
        vol_mult=vol_mult,
        lookback=args.lookback,
        atr_mult=args.atr_mult if args.mode == "strict" else None,
        direction_filter=dir_filter,
    )
    df = filter_results(
        df,
        min_vol_ratio=vol_mult,
        only_52w=args.only_52w,
    )

    if df.empty:
        print("No breakouts found.")
        return 0

    print(df.to_string(index=False))
    print(f"\nTotal: {len(df)} breakouts")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved to {args.csv}")

    save_scan_results(
        df,
        {
            "symbols": len(symbols),
            "universe_total": universe_total,
            "universe_sample": universe_sample,
            "universe_choice": universe_choice,
            "timeframes": timeframes,
            "mode": "Strict (ATR)" if args.mode == "strict" else "Standard",
            "breakout_mode": args.mode,
            "direction": args.direction.title() if args.direction != "both" else "Both",
            "vol_mult": vol_mult,
            "lookback": args.lookback,
            "atr_mult": args.atr_mult if args.mode == "strict" else None,
            "only_52w": args.only_52w,
            "max_symbols": args.max,
        },
    )
    print(f"App cache updated: data_cache/scan_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
