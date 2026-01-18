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

import datetime as dt
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def get_bytes(url_or_path: str, timeout_s: int = 30) -> bytes:
    """Read bytes from either an http(s) URL or a local file path.

    Supports:
      - https://... / http://...
      - file:///absolute/path
      - relative/or/absolute filesystem paths
    """
    s = (url_or_path or "").strip()
    if not s:
        raise ValueError("Empty url_or_path")

    if s.startswith("file://"):
        path = s[len("file://") :]
        with open(path, "rb") as f:
            return f.read()

    if os.path.exists(s):
        with open(s, "rb") as f:
            return f.read()

    r = requests.get(s, timeout=timeout_s)
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
    Parse OTE 15-minute day-ahead XLSX in the format of DT_15MIN_.._CZ.xlsx (URL or local file path).

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
    xlsx = get_bytes(ote_url)

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
    xlsx = get_bytes(cnb_url)

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
# Testable planning inputs / outputs
# -----------------------------


@dataclass(frozen=True)
class PlanningInputs:
    plan_date: dt.date
    eur_czk: float
    pv_kwh: float
    current_battery_state_kwh: float
    battery_capacity_kwh: float
    charge_power_kw: float
    max_price_czk_per_mwh: float
    window_start: dt.time
    window_end: dt.time
    prefer_window: bool

    # Policy knobs (same defaults as before)
    reserve_fraction: float = 0.10
    avg_consumption_kwh_per_h: float = 0.300
    dishwasher_consumption_kwh: float = 0.7
    daytime_load_kwh: float = 7.0


@dataclass(frozen=True)
class PlanningDecision:
    needed_to_full_kwh: float
    deficit_after_pv_for_full_kwh: float
    deficit_for_morning_reserve_kwh: float
    grid_charge_needed_kwh: float
    hours_until_9: float
    expected_consumption_until_9_kwh: float


@dataclass(frozen=True)
class PlanningResult:
    inputs: PlanningInputs
    decision: PlanningDecision
    chosen_slots: List["Slot"]
    merged_slots: List["Slot"]


def compute_decision(
    inputs: PlanningInputs, tz: dt.tzinfo, now_local: Optional[dt.datetime] = None
) -> PlanningDecision:
    """Compute how much energy must be charged from grid.

    Pure/testable: pass a fixed `now_local` in tests for deterministic results.
    """
    now_local = now_local or dt.datetime.now(tz)

    capacity = inputs.battery_capacity_kwh
    reserve_kwh = capacity * inputs.reserve_fraction

    nine_am_today = dt.datetime.combine(now_local.date(), dt.time(9, 0), tzinfo=tz)
    nine_am = (
        nine_am_today
        if now_local < nine_am_today
        else nine_am_today + dt.timedelta(days=1)
    )

    hours_until_9 = max(0.0, (nine_am - now_local).total_seconds() / 3600.0)
    expected_consumption_until_9_kwh = (
        hours_until_9 * inputs.avg_consumption_kwh_per_h
        + inputs.dishwasher_consumption_kwh
    )

    required_now_for_morning_kwh = expected_consumption_until_9_kwh + reserve_kwh
    deficit_for_morning_reserve_kwh = max(
        0.0, required_now_for_morning_kwh - inputs.current_battery_state_kwh
    )

    pv_available_for_battery_kwh = max(0.0, inputs.pv_kwh - inputs.daytime_load_kwh)

    needed_to_full_kwh = max(0.0, capacity - inputs.current_battery_state_kwh)
    deficit_after_pv_for_full_kwh = max(
        0.0, needed_to_full_kwh - pv_available_for_battery_kwh
    )

    grid_charge_needed_kwh = max(
        deficit_for_morning_reserve_kwh, deficit_after_pv_for_full_kwh
    )

    return PlanningDecision(
        needed_to_full_kwh=needed_to_full_kwh,
        deficit_after_pv_for_full_kwh=deficit_after_pv_for_full_kwh,
        deficit_for_morning_reserve_kwh=deficit_for_morning_reserve_kwh,
        grid_charge_needed_kwh=grid_charge_needed_kwh,
        hours_until_9=hours_until_9,
        expected_consumption_until_9_kwh=expected_consumption_until_9_kwh,
    )


def build_slots_from_ote_df(
    ote_df: pd.DataFrame, tz: dt.tzinfo, eur_czk: float
) -> List["Slot"]:
    """Convert OTE dataframe (start, price_eur_per_mwh) into Slot list in CZK/MWh."""
    df = ote_df.copy()
    df["price_czk_per_mwh"] = df["price_eur_per_mwh"] * eur_czk

    slots: List[Slot] = []
    for _, row in df.iterrows():
        start = row["start"].to_pydatetime()
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz)
        end = start + dt.timedelta(minutes=15)
        p = float(row["price_czk_per_mwh"])
        slots.append(Slot(start=start, end=end, price_czk_per_mwh=p))
    return slots


def plan_charging(
    ote_df: pd.DataFrame,
    inputs: PlanningInputs,
    tz: dt.tzinfo,
    now_local: Optional[dt.datetime] = None,
) -> PlanningResult:
    """Pure planning function: prices + inputs -> selected slots."""
    decision = compute_decision(inputs, tz=tz, now_local=now_local)

    slots_all = build_slots_from_ote_df(ote_df, tz=tz, eur_czk=inputs.eur_czk)

    eligible_by_price = [
        s for s in slots_all if s.price_czk_per_mwh <= inputs.max_price_czk_per_mwh
    ]

    win_start_dt, win_end_dt = daterange_window(
        inputs.plan_date, inputs.window_start, inputs.window_end, tz
    )

    in_window = [
        s
        for s in eligible_by_price
        if (s.start >= win_start_dt and s.end <= win_end_dt)
    ]
    out_window = [s for s in eligible_by_price if s not in in_window]

    if decision.grid_charge_needed_kwh <= 0:
        chosen: List[Slot] = []
    else:
        chosen = pick_cheapest_slots(
            in_window, decision.grid_charge_needed_kwh, inputs.charge_power_kw
        )

        if not inputs.prefer_window:
            kwh_per_slot = inputs.charge_power_kw * 0.25
            have_kwh = len(chosen) * kwh_per_slot
            remaining = max(0.0, decision.grid_charge_needed_kwh - have_kwh)
            if remaining > 1e-9:
                extra = pick_cheapest_slots(
                    out_window, remaining, inputs.charge_power_kw
                )
                chosen = sorted(chosen + extra, key=lambda s: s.start)

    merged = merge_contiguous_slots(chosen)
    return PlanningResult(
        inputs=inputs, decision=decision, chosen_slots=chosen, merged_slots=merged
    )


def load_inputs_file(path: str, default_plan_date: dt.date) -> Dict[str, Any]:
    """Load a local JSON inputs file for test runs."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("inputs file must contain a JSON object")

    if "plan_date" in data and isinstance(data["plan_date"], str):
        data["plan_date"] = dt.date.fromisoformat(data["plan_date"])
    else:
        data.setdefault("plan_date", default_plan_date)

    if "window_start" in data and isinstance(data["window_start"], str):
        data["window_start"] = parse_hhmm(data["window_start"])
    if "window_end" in data and isinstance(data["window_end"], str):
        data["window_end"] = parse_hhmm(data["window_end"])

    return data


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
