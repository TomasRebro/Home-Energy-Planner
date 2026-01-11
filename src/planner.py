#!/usr/bin/env python3
"""
Schedule cheapest 15-minute charging slots for a home battery using:
- OTE 15-min day-ahead prices (EUR/MWh, XLSX)
- CNB EUR→CZK exchange rate (XLSX)
- Home Assistant forecast for tomorrow PV production (kWh)
- Battery params (capacity, current energy, charge power)
- Max acceptable price (CZK/MWh)

Default goal:
If tomorrow PV forecast can fill the battery from current energy to full, do NOT charge from grid.
Otherwise, charge only the *deficit* at the cheapest eligible 15-min slots (by default overnight).

Example:
python planner.py \
  --ha-base-url http://homeassistant.local:8123 \
  --ha-token "YOUR_LONG_LIVED_TOKEN" \
  --current-kwh 6.2 \
  --charger-switch switch.battery_charger
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests


# -----------------------------
# Helpers
# -----------------------------

def prague_tz() -> dt.tzinfo:
    # Python 3.9+ supports zoneinfo
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Prague")
    except Exception:
        # Fallback: treat as local naive time
        return dt.timezone(dt.timedelta(hours=1))


def parse_hhmm(s: str) -> dt.time:
    hh, mm = s.split(":")
    return dt.time(int(hh), int(mm))


def daterange_window(date_local: dt.date, start: dt.time, end: dt.time, tz: dt.tzinfo) -> Tuple[dt.datetime, dt.datetime]:
    """
    Returns [start_dt, end_dt) in local tz.
    If end < start (crosses midnight), end is on next day.
    """
    start_dt = dt.datetime.combine(date_local, start, tzinfo=tz)
    end_dt = dt.datetime.combine(date_local, end, tzinfo=tz)
    if end_dt <= start_dt:
        end_dt = end_dt + dt.timedelta(days=1)
    return start_dt, end_dt


def build_ote_url(for_date: dt.date) -> str:
    # Matches your example pattern:
    # https://www.ote-cr.cz/pubweb/attachments/01/2026/month01/day05/DT_15MIN_05_01_2026_CZ.xlsx
    return (
        f"https://www.ote-cr.cz/pubweb/attachments/"
        f"{for_date:%m}/{for_date:%Y}/month{for_date:%m}/day{for_date:%d}/"
        f"DT_15MIN_{for_date:%d_%m_%Y}_CZ.xlsx"
    )


def build_cnb_url(for_date: dt.date) -> str:
    # Matches your example pattern:
    # https://www.ote-cr.cz/pubweb/attachments/01/2026/Kurzovni_listek_CNB_2026.xlsx
    return f"https://www.ote-cr.cz/pubweb/attachments/{for_date:%m}/{for_date:%Y}/Kurzovni_listek_CNB_{for_date:%Y}.xlsx"


def http_get_bytes(url: str, timeout_s: int = 30) -> bytes:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def read_excel_bytes(xlsx_bytes: bytes, **kwargs) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_excel(BytesIO(xlsx_bytes), **kwargs)


def to_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


# -----------------------------
# Data extraction
# -----------------------------

def load_ote_15min_prices_eur_per_mwh(ote_url: str, tz: dt.tzinfo) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      - start (timezone-aware datetime Europe/Prague)
      - price_eur_per_mwh (float)
    """
    xlsx = http_get_bytes(ote_url)
    df = read_excel_bytes(xlsx)

    # Heuristics: find a datetime/timestamp column + a numeric price column
    # Common OTE patterns: columns like 'Datum', 'Čas', 'Cena (EUR/MWh)' etc.
    cols = [c for c in df.columns]

    # Find price column: first numeric-ish column that isn't obviously an index or hour
    candidate_price_cols = []
    for c in cols:
        s = str(c).lower()
        if "eur" in s and ("mwh" in s or "mw" in s or "m" in s) and ("cen" in s or "price" in s):
            candidate_price_cols.append(c)
    if not candidate_price_cols:
        # fallback: pick the last numeric column
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise RuntimeError("Could not find a numeric price column in OTE XLSX.")
        price_col = numeric_cols[-1]
    else:
        price_col = candidate_price_cols[0]

    # Find time columns
    # Try: a single datetime column
    datetime_col = None
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            datetime_col = c
            break

    if datetime_col is not None:
        starts = pd.to_datetime(df[datetime_col], errors="coerce")
    else:
        # Try separate date + time columns
        date_col = None
        time_col = None
        for c in cols:
            s = str(c).lower()
            if date_col is None and ("datum" in s or "date" in s):
                date_col = c
            if time_col is None and ("čas" in s or "cas" in s or "time" in s or "hod" in s):
                time_col = c

        if date_col is None or time_col is None:
            # Fallback: maybe first two columns are date/time
            if len(cols) >= 2:
                date_col, time_col = cols[0], cols[1]
            else:
                raise RuntimeError("Could not find date/time columns in OTE XLSX.")

        dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
        # time might be datetime, timedelta, string, or number
        time_raw = df[time_col]

        def parse_time(v) -> Optional[dt.time]:
            if pd.isna(v):
                return None
            if isinstance(v, dt.time):
                return v
            if isinstance(v, dt.datetime):
                return v.time()
            if isinstance(v, (pd.Timestamp,)):
                return v.to_pydatetime().time()
            if isinstance(v, dt.timedelta):
                secs = int(v.total_seconds())
                return (dt.datetime.min + dt.timedelta(seconds=secs)).time()
            # Excel time can come as float fraction of day
            fv = to_float(v)
            if fv is not None and 0 <= fv < 1.1:
                secs = int(round(fv * 24 * 3600))
                return (dt.datetime.min + dt.timedelta(seconds=secs)).time()
            # string "HH:MM"
            try:
                s = str(v).strip()
                if ":" in s:
                    hh, mm = s.split(":")[:2]
                    return dt.time(int(hh), int(mm))
            except Exception:
                return None
            return None

        times = time_raw.map(parse_time)
        starts = [
            dt.datetime.combine(d, t, tzinfo=tz) if (d is not None and t is not None) else pd.NaT
            for d, t in zip(dates, times)
        ]
        starts = pd.to_datetime(starts, errors="coerce")

    prices = pd.to_numeric(df[price_col], errors="coerce")

    out = pd.DataFrame({"start": starts, "price_eur_per_mwh": prices}).dropna()
    out = out.sort_values("start").reset_index(drop=True)

    # Keep only exact 15-min aligned entries (best-effort)
    out = out[out["start"].dt.minute.isin([0, 15, 30, 45])]

    if out.empty:
        raise RuntimeError("OTE price table parsed, but produced 0 usable rows (check XLSX layout/headers).")

    return out


def load_cnb_eur_czk_rate(cnb_url: str, for_date: dt.date) -> float:
    """
    CNB sheet layout (per screenshot):
      columns: Den | Kurz | Měna
      Den formatted like dd/mm/yyyy
      Kurz is numeric (CZK per 1 EUR)
      Měna is 'EUR'

    Returns EUR→CZK for the given date, or nearest previous date if missing.
    """
    xlsx = http_get_bytes(cnb_url)

    # Read without assuming header row; we'll find the header by content
    df = read_excel_bytes(xlsx, header=None)

    # Find the header row containing "Den" and "Kurz" and "Měna"
    header_row_idx = None
    for i in range(min(50, len(df))):
        row_vals = [str(v).strip().lower() for v in df.iloc[i].tolist() if not pd.isna(v)]
        if ("den" in row_vals) and ("kurz" in row_vals) and (("měna" in row_vals) or ("mena" in row_vals)):
            header_row_idx = i
            break

    if header_row_idx is None:
        raise RuntimeError("Could not find CNB header row with columns Den/Kurz/Měna.")

    # Re-read using that row as header
    df2 = read_excel_bytes(xlsx, header=header_row_idx)

    # Normalize column names (handle diacritics variants)
    cols = {str(c).strip().lower(): c for c in df2.columns}
    den_col = cols.get("den")
    kurz_col = cols.get("kurz")
    mena_col = cols.get("měna") or cols.get("mena")

    if not den_col or not kurz_col or not mena_col:
        raise RuntimeError(f"Expected columns Den/Kurz/Měna, got: {list(df2.columns)}")

    # Parse
    out = df2[[den_col, kurz_col, mena_col]].copy()
    out = out.rename(columns={den_col: "date", kurz_col: "rate", mena_col: "ccy"})

    # Den is dd/mm/yyyy
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce").dt.date
    out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
    out["ccy"] = out["ccy"].astype(str).str.strip().str.upper()

    out = out.dropna(subset=["date", "rate"])
    out = out[out["ccy"] == "EUR"].sort_values("date")

    if out.empty:
        raise RuntimeError("CNB sheet parsed but no EUR rows found.")

    # Exact date first; otherwise nearest previous date
    exact = out[out["date"] == for_date]
    if not exact.empty:
        return float(exact.iloc[0]["rate"])

    prev = out[out["date"] <= for_date]
    if prev.empty:
        raise RuntimeError(f"No CNB EUR rate found on or before {for_date}.")

    return float(prev.iloc[-1]["rate"])


def load_ha_tomorrow_pv_kwh(ha_base_url: str, ha_token: str, entity_id: str) -> float:
    url = ha_base_url.rstrip("/") + f"/api/states/{entity_id}"
    headers = {"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    state = data.get("state")
    pv = float(state)
    return pv


# -----------------------------
# Planning
# -----------------------------

@dataclass(frozen=True)
class Slot:
    start: dt.datetime
    end: dt.datetime
    price_czk_per_mwh: float

    @property
    def price_czk_per_kwh(self) -> float:
        return self.price_czk_per_mwh / 1000.0


def pick_cheapest_slots(
    slots: List[Slot],
    needed_kwh: float,
    charge_power_kw: float,
    slot_minutes: int = 15,
) -> List[Slot]:
    if needed_kwh <= 0:
        return []

    kwh_per_slot = charge_power_kw * (slot_minutes / 60.0)
    needed_slots = int(math.ceil(needed_kwh / kwh_per_slot))

    # sort by price then time
    slots_sorted = sorted(slots, key=lambda s: (s.price_czk_per_mwh, s.start))
    chosen = slots_sorted[:needed_slots]
    chosen = sorted(chosen, key=lambda s: s.start)
    return chosen


def merge_contiguous_slots(chosen: List[Slot]) -> List[Slot]:
    if not chosen:
        return []
    merged: List[Slot] = []
    cur = chosen[0]
    for s in chosen[1:]:
        if s.start == cur.end and math.isclose(s.price_czk_per_mwh, cur.price_czk_per_mwh, rel_tol=0, abs_tol=1e-9):
            # same price and contiguous -> merge
            cur = Slot(start=cur.start, end=s.end, price_czk_per_mwh=cur.price_czk_per_mwh)
        elif s.start == cur.end:
            # contiguous but different price: still merge time-wise is OK for a real charger,
            # but keep prices separate for transparency -> do NOT merge.
            merged.append(cur)
            cur = s
        else:
            merged.append(cur)
            cur = s
    merged.append(cur)
    return merged


def main() -> int:
    tz = prague_tz()

    ap = argparse.ArgumentParser(description="Plan overnight battery charging from OTE 15-min spot prices.")
    ap.add_argument("--date", help="Date to plan for (YYYY-MM-DD) in Europe/Prague. Default: tomorrow.", default=None)

    ap.add_argument("--max-price-czk-per-mwh", type=float, default=2708.0, help="Do not charge above this price.")
    ap.add_argument("--charge-power-kw", type=float, default=6.0)
    ap.add_argument("--battery-capacity-kwh", type=float, default=14.2)
    ap.add_argument("--current-kwh", type=float, required=True, help="Current energy in battery (kWh).")

    ap.add_argument("--pv-entity", default="sensor.energy_production_tomorrow",
                    help="HA entity id that returns tomorrow PV production (kWh).")
    ap.add_argument("--ha-base-url", default=os.getenv("HA_BASE_URL", ""), help="Home Assistant base URL.")
    ap.add_argument("--ha-token", default=os.getenv("HA_TOKEN", ""), help="Home Assistant long-lived token.")

    ap.add_argument("--window-start", default="00:00", help="Eligible charging window start (HH:MM) local time.")
    ap.add_argument("--window-end", default="06:00", help="Eligible charging window end (HH:MM) local time.")
    ap.add_argument("--prefer-window", action="store_true",
                    help="If set, ONLY charge inside window. Otherwise you can spill outside window if needed.")

    ap.add_argument("--ote-url", default=None, help="Override OTE URL. If omitted, generated from --date.")
    ap.add_argument("--cnb-url", default=None, help="Override CNB URL. If omitted, generated from --date.")

    ap.add_argument("--charger-switch", default=None,
                    help="If provided, print Home Assistant service calls for this switch entity_id.")
    args = ap.parse_args()

    now = dt.datetime.now(tz)
    if args.date:
        plan_date = dt.date.fromisoformat(args.date)
    else:
        plan_date = (now + dt.timedelta(days=1)).date()

    ote_url = args.ote_url or build_ote_url(plan_date)
    cnb_url = args.cnb_url or build_cnb_url(plan_date)

    # if not args.ha_base_url or not args.ha_token:
    #     print("ERROR: Provide --ha-base-url and --ha-token (or env HA_BASE_URL / HA_TOKEN).", file=sys.stderr)
    #     return 2

    # Load inputs
    # pv_kwh = load_ha_tomorrow_pv_kwh(args.ha_base_url, args.ha_token, args.pv_entity)
    pv_kwh = 0.0
    eur_czk = load_cnb_eur_czk_rate(cnb_url, plan_date)
    print(f"EUR→CZK rate used: {eur_czk:.4f}")
    ote_df = load_ote_15min_prices_eur_per_mwh(ote_url, tz=tz)

    # Convert OTE prices to CZK/MWh
    ote_df["price_czk_per_mwh"] = ote_df["price_eur_per_mwh"] * eur_czk

    # Build slot list
    slots_all: List[Slot] = []
    for _, row in ote_df.iterrows():
        start = row["start"].to_pydatetime()
        end = start + dt.timedelta(minutes=15)
        p = float(row["price_czk_per_mwh"])
        slots_all.append(Slot(start=start, end=end, price_czk_per_mwh=p))

    # Determine how much grid energy is needed (default goal: end of tomorrow = full battery)
    capacity = args.battery_capacity_kwh
    current = args.current_kwh
    if current < 0 or current > capacity * 1.05:
        print(f"ERROR: --current-kwh looks invalid vs capacity ({current} vs {capacity}).", file=sys.stderr)
        return 2

    needed_to_full = max(0.0, capacity - current)
    deficit_after_pv = max(0.0, needed_to_full - pv_kwh)

    # Eligible slots by price cap
    eligible_by_price = [s for s in slots_all if s.price_czk_per_mwh <= args.max_price_czk_per_mwh]

    # Window filter
    wstart = parse_hhmm(args.window_start)
    wend = parse_hhmm(args.window_end)
    win_start_dt, win_end_dt = daterange_window(plan_date, wstart, wend, tz)

    in_window = [s for s in eligible_by_price if (s.start >= win_start_dt and s.end <= win_end_dt)]
    out_window = [s for s in eligible_by_price if s not in in_window]

    # Choose charging slots
    if deficit_after_pv <= 0:
        chosen: List[Slot] = []
    else:
        if args.prefer_window:
            pool = in_window
        else:
            # try window first; if not enough slots, allow spill to any eligible slot
            pool = in_window

        chosen = pick_cheapest_slots(pool, deficit_after_pv, args.charge_power_kw)

        if not args.prefer_window:
            # If we couldn't get enough inside window, spill outside window
            kwh_per_slot = args.charge_power_kw * 0.25
            have_kwh = len(chosen) * kwh_per_slot
            remaining = max(0.0, deficit_after_pv - have_kwh)
            if remaining > 1e-9:
                extra = pick_cheapest_slots(out_window, remaining, args.charge_power_kw)
                chosen = sorted(chosen + extra, key=lambda s: s.start)

    chosen_merged = merge_contiguous_slots(chosen)

    # Report
    print("\n=== Inputs ===")
    print(f"Plan date (local):           {plan_date.isoformat()}")
    print(f"OTE URL:                     {ote_url}")
    print(f"CNB URL:                     {cnb_url}")
    print(f"EUR→CZK rate used:           {eur_czk:.4f}")
    print(f"Battery capacity:            {capacity:.2f} kWh")
    print(f"Current battery energy:      {current:.2f} kWh")
    print(f"Forecast PV tomorrow (HA):   {pv_kwh:.2f} kWh")
    print(f"Charge power:                {args.charge_power_kw:.2f} kW")
    print(f"Max price:                   {args.max_price_czk_per_mwh:.2f} CZK/MWh ({args.max_price_czk_per_mwh/1000:.4f} CZK/kWh)")
    print(f"Eligible window:             {win_start_dt:%Y-%m-%d %H:%M} → {win_end_dt:%Y-%m-%d %H:%M} ({'ONLY' if args.prefer_window else 'preferred'})")

    print("\n=== Decision ===")
    print(f"Needed to full now:          {needed_to_full:.2f} kWh")
    print(f"Deficit after PV forecast:   {deficit_after_pv:.2f} kWh")

    if not chosen:
        print("\n✅ No grid charging needed (PV forecast should cover filling the battery).")
        return 0

    kwh_per_slot = args.charge_power_kw * 0.25
    planned_kwh = len(chosen) * kwh_per_slot

    # weighted average price
    avg_price_czk_per_mwh = sum(s.price_czk_per_mwh for s in chosen) / max(1, len(chosen))

    print("\n⚡ Planned grid charging slots (15 min):")
    for s in chosen:
        print(f"- {s.start:%Y-%m-%d %H:%M} → {s.end:%H:%M}  |  {s.price_czk_per_mwh:8.2f} CZK/MWh  ({s.price_czk_per_kwh:.4f} CZK/kWh)")

    print("\n📌 Merged intervals (easier to execute):")
    for s in chosen_merged:
        print(f"- {s.start:%Y-%m-%d %H:%M} → {s.end:%Y-%m-%d %H:%M}  |  {s.price_czk_per_mwh:8.2f} CZK/MWh")

    print("\n=== Summary ===")
    print(f"Slots selected:              {len(chosen)} (≈ {planned_kwh:.2f} kWh at {kwh_per_slot:.2f} kWh/slot)")
    print(f"Average selected price:      {avg_price_czk_per_mwh:.2f} CZK/MWh ({avg_price_czk_per_mwh/1000:.4f} CZK/kWh)")

    # Optional: print HA service calls (you still need an automation/scheduler to run them at times)
    if args.charger_switch:
        print("\n=== Home Assistant service call snippets ===")
        print("# Turn ON at each interval start, OFF at each interval end.")
        print("# You can paste these into HA scripts or use an external scheduler (cron) calling HA REST API.")
        print("# curl examples:")
        for s in chosen_merged:
            on_time = s.start.strftime("%Y-%m-%d %H:%M:%S")
            off_time = s.end.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n# Interval {on_time} → {off_time}")
            print(
                f"curl -s -X POST '{args.ha_base_url.rstrip('/')}/api/services/switch/turn_on' "
                f"-H 'Authorization: Bearer {args.ha_token}' -H 'Content-Type: application/json' "
                f"-d '{{\"entity_id\":\"{args.charger_switch}\"}}'"
            )
            print(
                f"curl -s -X POST '{args.ha_base_url.rstrip('/')}/api/services/switch/turn_off' "
                f"-H 'Authorization: Bearer {args.ha_token}' -H 'Content-Type: application/json' "
                f"-d '{{\"entity_id\":\"{args.charger_switch}\"}}'"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())