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
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def daterange_window(
    date_local: dt.date, start: dt.time, end: dt.time, tz: dt.tzinfo
) -> Tuple[dt.datetime, dt.datetime]:
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
    Parse OTE 15-minute day-ahead XLSX in the format of DT_15MIN_.._CZ.xlsx.

    Observed layout:
      - A header row containing columns like:
          Perioda | Časový interval | 15 min cena (EUR/MWh) | ...
      - Data rows with time intervals like "00:00-00:15".

    Returns DataFrame with columns:
      - start (timezone-aware datetime in `tz`)
      - price_eur_per_mwh (float)

    Notes:
      - The date is inferred from the URL/filename (preferred) or from the sheet title.
      - Rows without a Perioda are ignored.
    """
    xlsx = http_get_bytes(ote_url)

    # Read without headers first so we can locate the table header row reliably.
    df0 = read_excel_bytes(xlsx, header=None)

    # Find the header row that contains both "Časový interval" and "15 min cena".
    header_row_idx = None
    for i in range(min(100, len(df0))):
        row_vals = [
            str(v).strip().lower() for v in df0.iloc[i].tolist() if not pd.isna(v)
        ]
        has_interval = any(
            ("časový interval" in v)
            or ("casový interval" in v)
            or ("casovy interval" in v)
            or ("time interval" in v)
            for v in row_vals
        )
        has_15min_price = any(
            ("15 min cena" in v)
            or ("15min cena" in v)
            or (("15 min" in v) and ("cena" in v))
            or (("15min" in v) and ("cena" in v))
            for v in row_vals
        )
        if has_interval and has_15min_price:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise RuntimeError(
            "Could not find OTE table header row (expected columns like 'Časový interval' and '15 min cena')."
        )

    # Re-read using that row as the header.
    df = read_excel_bytes(xlsx, header=header_row_idx)

    # Normalize/locate required columns.
    cols = {str(c).strip().lower(): c for c in df.columns}

    # Period column: 'Perioda' (sometimes 'Period')
    period_col = None
    for k, orig in cols.items():
        if k == "perioda" or "period" in k:
            period_col = orig
            break

    # Time interval column: 'Časový interval'
    time_col = None
    for k, orig in cols.items():
        if (
            "časový interval" in k
            or "casový interval" in k
            or "casovy interval" in k
            or "time interval" in k
        ):
            time_col = orig
            break

    # 15-min price column: typically "15 min cena\n(EUR/MWh)"
    price_col = None
    for k, orig in cols.items():
        if ("15 min" in k) and ("cena" in k) and ("eur/mwh" in k):
            price_col = orig
            break

    if period_col is None or time_col is None or price_col is None:
        raise RuntimeError(
            f"OTE XLSX columns not recognized. Found columns: {list(df.columns)}"
        )

    # Infer the market day date from URL (DT_15MIN_dd_mm_yyyy) or from the title cell.
    def infer_date_from_url(url: str) -> Optional[dt.date]:
        m = re.search(r"DT_15MIN_(\d{2})_(\d{2})_(\d{4})", url)
        if m:
            dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
            return dt.date(int(yyyy), int(mm), int(dd))
        m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", url)
        if m:
            dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
            return dt.date(int(yyyy), int(mm), int(dd))
        return None

    def infer_date_from_sheet() -> Optional[dt.date]:
        # Look for something like "SPOT MARKET INDEX - 05.01.2026" in the first ~30 rows / first few cols
        for i in range(min(30, len(df0))):
            for j in range(min(6, df0.shape[1])):
                v = df0.iat[i, j]
                if pd.isna(v):
                    continue
                s = str(v)
                m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", s)
                if m:
                    dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
                    return dt.date(int(yyyy), int(mm), int(dd))
        return None

    market_date = infer_date_from_url(ote_url) or infer_date_from_sheet()
    if market_date is None:
        raise RuntimeError(
            "Could not infer market date from OTE URL/filename or sheet title."
        )

    # Keep only rows with a valid Perioda.
    df = df.dropna(subset=[period_col]).copy()

    # Parse time interval like "00:00-00:15" -> start time
    def parse_interval_start(v) -> Optional[dt.time]:
        if pd.isna(v):
            return None
        s = str(v).strip()
        m = re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", s)
        if not m:
            return None
        hh, mm = int(m.group(1)), int(m.group(2))
        return dt.time(hh, mm)

    starts_t = df[time_col].map(parse_interval_start)
    starts_dt = [
        dt.datetime.combine(market_date, t, tzinfo=tz) if t is not None else pd.NaT
        for t in starts_t
    ]

    prices = pd.to_numeric(df[price_col], errors="coerce")

    out = pd.DataFrame(
        {
            "start": pd.to_datetime(starts_dt, errors="coerce"),
            "price_eur_per_mwh": prices,
        }
    ).dropna()

    out = out.sort_values("start").reset_index(drop=True)

    if out.empty:
        raise RuntimeError(
            "OTE price table parsed, but produced 0 usable rows (check XLSX layout/headers)."
        )

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
        row_vals = [
            str(v).strip().lower() for v in df.iloc[i].tolist() if not pd.isna(v)
        ]
        if (
            ("den" in row_vals)
            and ("kurz" in row_vals)
            and (("měna" in row_vals) or ("mena" in row_vals))
        ):
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

    print(f"Using CNB EUR rate for date: {prev.iloc[-1]['date']}")
    return float(prev.iloc[-1]["rate"])


def load_ha_entity_state(ha_base_url: str, ha_token: str, entity_id: str) -> str:
    url = ha_base_url.rstrip("/") + f"/api/states/{entity_id}"
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    state = data.get("state")
    return state


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
        if s.start == cur.end and math.isclose(
            s.price_czk_per_mwh, cur.price_czk_per_mwh, rel_tol=0, abs_tol=1e-9
        ):
            # same price and contiguous -> merge
            cur = Slot(
                start=cur.start, end=s.end, price_czk_per_mwh=cur.price_czk_per_mwh
            )
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

    ap = argparse.ArgumentParser(
        description="Plan overnight battery charging from OTE 15-min spot prices."
    )
    ap.add_argument(
        "--date",
        help="Date to plan for (YYYY-MM-DD) in Europe/Prague. Default: tomorrow.",
        default=None,
    )

    ap.add_argument(
        "--max-price-czk-per-mwh",
        type=float,
        default=2708.0,
        help="Do not charge above this price.",
    )
    ap.add_argument("--charge-power-kw", type=float, default=3.0)
    ap.add_argument("--battery-capacity-kwh", type=float, default=14.2)
    ap.add_argument(
        "--current-battery-state-entity",
        default="sensor.battery_state_of_charge_capacity",
        help="HA entity id that returns current battery state of charge (kWh).",
    )

    ap.add_argument(
        "--pv-entity",
        default="sensor.energy_production_tomorrow",
        help="HA entity id that returns tomorrow PV production (kWh).",
    )
    ap.add_argument(
        "--ha-base-url",
        default=os.getenv("HA_BASE_URL", ""),
        help="Home Assistant base URL.",
    )
    ap.add_argument(
        "--ha-token",
        default=os.getenv("HA_TOKEN", ""),
        help="Home Assistant long-lived token.",
    )

    ap.add_argument(
        "--window-start",
        default="00:00",
        help="Eligible charging window start (HH:MM) local time.",
    )
    ap.add_argument(
        "--window-end",
        default="06:00",
        help="Eligible charging window end (HH:MM) local time.",
    )
    ap.add_argument(
        "--prefer-window",
        action="store_true",
        help="If set, ONLY charge inside window. Otherwise you can spill outside window if needed.",
    )

    ap.add_argument(
        "--ote-url",
        default=None,
        help="Override OTE URL. If omitted, generated from --date.",
    )
    ap.add_argument(
        "--cnb-url",
        default=None,
        help="Override CNB URL. If omitted, generated from --date.",
    )

    ap.add_argument(
        "--charger-switch",
        default=None,
        help="If provided, print Home Assistant service calls for this switch entity_id.",
    )
    args = ap.parse_args()

    now = dt.datetime.now(tz)
    if args.date:
        plan_date = dt.date.fromisoformat(args.date)
    else:
        plan_date = (now + dt.timedelta(days=1)).date()

    ote_url = args.ote_url or build_ote_url(plan_date)
    cnb_url = args.cnb_url or build_cnb_url(
        now.date()
    )  # Use today's rate for planning, because CNB rates are updated daily

    if not args.ha_base_url or not args.ha_token:
        print(
            "ERROR: Provide --ha-base-url and --ha-token (or env HA_BASE_URL / HA_TOKEN).",
            file=sys.stderr,
        )
        return 2

    # Load inputs
    pv_kwh = float(
        load_ha_entity_state(args.ha_base_url, args.ha_token, args.pv_entity)
    )
    print(f"PV forecast tomorrow (HA): {pv_kwh:.2f} kWh")
    current_battery_state = float(
        load_ha_entity_state(
            args.ha_base_url, args.ha_token, args.current_battery_state_entity
        )
    )
    print(f"Current battery state (HA): {current_battery_state:.2f} kWh")
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

    # --- Morning reserve logic ---
    # We want to ensure there is enough energy to cover consumption until 09:00 and still have
    # at least 10% of battery left as a reserve.
    reserve_fraction = 0.10
    reserve_kwh = capacity * reserve_fraction

    # Average consumption: 300 Wh/h + 10% (dishwasher factor)
    avg_consumption_kwh_per_h = 0.300
    dishwasher_factor = 1.10

    now_local = dt.datetime.now(tz)

    # Next occurrence of 09:00 local time
    nine_am_today = dt.datetime.combine(now_local.date(), dt.time(9, 0), tzinfo=tz)
    nine_am = (
        nine_am_today
        if now_local < nine_am_today
        else nine_am_today + dt.timedelta(days=1)
    )

    hours_until_9 = max(0.0, (nine_am - now_local).total_seconds() / 3600.0)
    expected_consumption_until_9_kwh = (
        hours_until_9 * avg_consumption_kwh_per_h * dishwasher_factor
    )

    # Minimum energy we should have NOW to still have reserve at 09:00
    required_now_for_morning_kwh = expected_consumption_until_9_kwh + reserve_kwh
    deficit_for_morning_reserve_kwh = max(
        0.0, required_now_for_morning_kwh - current_battery_state
    )

    # --- Daytime load vs PV forecast ---
    # PV forecast is not fully available to charge the battery, because a big part of it will be
    # consumed by household load during the day (roughly 09:00–21:00).
    daytime_load_kwh = 7.0

    # Only PV beyond daytime load can be assumed to charge the battery (simplified model).
    pv_available_for_battery_kwh = max(0.0, pv_kwh - daytime_load_kwh)

    # Original goal: end of tomorrow = full battery (unless PV covers it)
    needed_to_full = max(0.0, capacity - current_battery_state)
    deficit_after_pv_for_full_kwh = max(
        0.0, needed_to_full - pv_available_for_battery_kwh
    )

    # Final grid charging requirement: at least cover morning reserve deficit, and (optionally)
    # also cover deficit-to-full after PV forecast.
    deficit_after_pv = max(
        deficit_for_morning_reserve_kwh, deficit_after_pv_for_full_kwh
    )

    print(
        f"Morning reserve target: {reserve_fraction * 100:.0f}% ({reserve_kwh:.2f} kWh). "
        f"Hours until 09:00: {hours_until_9:.2f}h. "
        f"Expected consumption until 09:00: {expected_consumption_until_9_kwh:.2f} kWh. "
        f"Deficit for morning reserve: {deficit_for_morning_reserve_kwh:.2f} kWh."
    )

    # Eligible slots by price cap
    eligible_by_price = [
        s for s in slots_all if s.price_czk_per_mwh <= args.max_price_czk_per_mwh
    ]

    # Window filter
    wstart = parse_hhmm(args.window_start)
    wend = parse_hhmm(args.window_end)
    win_start_dt, win_end_dt = daterange_window(plan_date, wstart, wend, tz)

    in_window = [
        s
        for s in eligible_by_price
        if (s.start >= win_start_dt and s.end <= win_end_dt)
    ]
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
    print(f"Current battery energy:      {current_battery_state:.2f} kWh")
    print(f"Forecast PV tomorrow (HA):   {pv_kwh:.2f} kWh")
    print(f"Charge power:                {args.charge_power_kw:.2f} kW")
    print(
        f"Max price:                   {args.max_price_czk_per_mwh:.2f} CZK/MWh ({args.max_price_czk_per_mwh / 1000:.4f} CZK/kWh)"
    )
    print(
        f"Eligible window:             {win_start_dt:%Y-%m-%d %H:%M} → {win_end_dt:%Y-%m-%d %H:%M} ({'ONLY' if args.prefer_window else 'preferred'})"
    )

    print("\n=== Decision ===")
    print(f"Needed to full now:          {needed_to_full:.2f} kWh")
    print(f"Deficit after PV forecast (full): {deficit_after_pv_for_full_kwh:.2f} kWh")
    print(f"Grid charge needed (incl. morning reserve): {deficit_after_pv:.2f} kWh")

    if not chosen:
        print(
            "\n✅ No grid charging needed (PV forecast should cover filling the battery)."
        )
        return 0

    kwh_per_slot = args.charge_power_kw * 0.25
    planned_kwh = len(chosen) * kwh_per_slot

    # weighted average price
    avg_price_czk_per_mwh = sum(s.price_czk_per_mwh for s in chosen) / max(
        1, len(chosen)
    )

    print("\n⚡ Planned grid charging slots (15 min):")
    for s in chosen:
        print(
            f"- {s.start:%Y-%m-%d %H:%M} → {s.end:%H:%M}  |  {s.price_czk_per_mwh:8.2f} CZK/MWh  ({s.price_czk_per_kwh:.4f} CZK/kWh)"
        )

    print("\n📌 Merged intervals (easier to execute):")
    for s in chosen_merged:
        print(
            f"- {s.start:%Y-%m-%d %H:%M} → {s.end:%Y-%m-%d %H:%M}  |  {s.price_czk_per_mwh:8.2f} CZK/MWh"
        )

    print("\n=== Summary ===")
    print(
        f"Slots selected:              {len(chosen)} (≈ {planned_kwh:.2f} kWh at {kwh_per_slot:.2f} kWh/slot)"
    )
    print(
        f"Average selected price:      {avg_price_czk_per_mwh:.2f} CZK/MWh ({avg_price_czk_per_mwh / 1000:.4f} CZK/kWh)"
    )

    # Optional: print HA service calls (you still need an automation/scheduler to run them at times)
    if args.charger_switch:
        print("\n=== Home Assistant service call snippets ===")
        print("# Turn ON at each interval start, OFF at each interval end.")
        print(
            "# You can paste these into HA scripts or use an external scheduler (cron) calling HA REST API."
        )
        print("# curl examples:")
        for s in chosen_merged:
            on_time = s.start.strftime("%Y-%m-%d %H:%M:%S")
            off_time = s.end.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n# Interval {on_time} → {off_time}")
            print(
                f"curl -s -X POST '{args.ha_base_url.rstrip('/')}/api/services/switch/turn_on' "
                f"-H 'Authorization: Bearer {args.ha_token}' -H 'Content-Type: application/json' "
                f'-d \'{{"entity_id":"{args.charger_switch}"}}\''
            )
            print(
                f"curl -s -X POST '{args.ha_base_url.rstrip('/')}/api/services/switch/turn_off' "
                f"-H 'Authorization: Bearer {args.ha_token}' -H 'Content-Type: application/json' "
                f'-d \'{{"entity_id":"{args.charger_switch}"}}\''
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
