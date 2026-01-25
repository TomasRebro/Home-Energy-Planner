#!/usr/bin/env python3
"""CLI entrypoint for home-energy-planner.

This module is intentionally thin glue code around `planner.py` so unit tests can
focus on pure planning functions (compute_decision / plan_charging) without
exercising argparse, network, or Home Assistant integrations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import requests

from planner import (
    PlanningInputs,
    build_cnb_url,
    build_ote_url,
    daterange_window,
    load_cnb_eur_czk_rate,
    load_ha_entity_state,
    load_inputs_file,
    load_ote_15min_prices_eur_per_mwh,
    parse_hhmm,
    plan_charging,
    prague_tz,
)


def send_pushover_notification(
    message: str, title: str = "Home Energy Planner"
) -> None:
    """Send a notification via Pushover API."""
    api_key = os.getenv("PUSHOVER_API_KEY")
    user_key = os.getenv("PUSHOVER_USER_KEY")

    if not api_key or not user_key:
        print(
            "⚠️  Pushover credentials not available; skipping notification",
            file=sys.stderr,
        )
        return

    try:
        url = "https://api.pushover.net/1/messages.json"
        payload = {
            "token": api_key,
            "user": user_key,
            "title": title,
            "message": message,
        }
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(
            f"⚠️  Failed to send Pushover notification: {e}",
            file=sys.stderr,
        )


def main() -> int:
    try:
        return _main()
    except Exception as e:
        error_msg = f"Planning failed: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        send_pushover_notification(
            message=error_msg, title="Home Energy Planner - Error"
        )
        return 1


def _main() -> int:
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
        "--inputs-file",
        default=None,
        help=(
            "Path to local JSON inputs file to run planning without Home Assistant. "
            "If provided, it overrides HA-derived battery/PV values (and can optionally provide eur_czk, plan_date, etc.)."
        ),
    )
    ap.add_argument(
        "--current-battery-kwh",
        type=float,
        default=None,
        help="Override current battery energy (kWh). Useful with --inputs-file.",
    )
    ap.add_argument(
        "--pv-kwh",
        type=float,
        default=None,
        help="Override PV forecast for tomorrow (kWh). Useful with --inputs-file.",
    )
    ap.add_argument(
        "--eur-czk",
        type=float,
        default=None,
        help="Override EUR→CZK exchange rate. Useful for tests.",
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
        help="Override OTE URL (or local path). If omitted, generated from --date.",
    )
    ap.add_argument(
        "--cnb-url",
        default=None,
        help="Override CNB URL (or local path). If omitted, generated from --date.",
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

    # URLs (or local paths)
    ote_url = args.ote_url or build_ote_url(plan_date)
    cnb_url = args.cnb_url or build_cnb_url(
        now.date()
    )  # Use today's rate for planning, because CNB rates are updated daily

    use_local_inputs = bool(args.inputs_file)

    if not use_local_inputs:
        if not args.ha_base_url or not args.ha_token:
            print(
                "ERROR: Provide --ha-base-url and --ha-token (or env HA_BASE_URL / HA_TOKEN), OR run with --inputs-file.",
                file=sys.stderr,
            )
            return 2

    # Always load prices first
    ote_df = load_ote_15min_prices_eur_per_mwh(ote_url, tz=tz)

    if use_local_inputs:
        data = load_inputs_file(args.inputs_file, default_plan_date=plan_date)

        pv_kwh = float(
            args.pv_kwh if args.pv_kwh is not None else data.get("pv_kwh", 0.0)
        )
        current_battery_state = float(
            args.current_battery_kwh
            if args.current_battery_kwh is not None
            else data.get(
                "current_battery_state_kwh", data.get("current_battery_kwh", 0.0)
            )
        )

        if args.eur_czk is not None:
            eur_czk = float(args.eur_czk)
        elif "eur_czk" in data:
            eur_czk = float(data["eur_czk"])
        else:
            eur_czk = load_cnb_eur_czk_rate(cnb_url, plan_date)

        # Optional plan_date override inside inputs file
        if isinstance(data.get("plan_date"), dt.date):
            plan_date = data["plan_date"]
            ote_url = args.ote_url or build_ote_url(plan_date)
            ote_df = load_ote_15min_prices_eur_per_mwh(ote_url, tz=tz)

        print(f"PV forecast tomorrow (inputs): {pv_kwh:.2f} kWh")
        print(f"Current battery state (inputs): {current_battery_state:.2f} kWh")
        print(f"EUR→CZK rate used: {eur_czk:.4f}")

    else:
        pv_kwh = float(
            load_ha_entity_state(args.ha_base_url, args.ha_token, args.pv_entity)
        )
        print(f"PV forecast tomorrow (HA): {pv_kwh:.2f} kWh")
        current_battery_state = float(
            load_ha_entity_state(
                args.ha_base_url,
                args.ha_token,
                args.current_battery_state_entity,
            )
        )
        print(f"Current battery state (HA): {current_battery_state:.2f} kWh")
        eur_czk = load_cnb_eur_czk_rate(cnb_url, plan_date)
        print(f"EUR→CZK rate used: {eur_czk:.4f}")

    # Run the pure planner
    wstart = parse_hhmm(args.window_start)
    wend = parse_hhmm(args.window_end)

    inputs = PlanningInputs(
        plan_date=plan_date,
        eur_czk=eur_czk,
        pv_kwh=pv_kwh,
        current_battery_state_kwh=current_battery_state,
        battery_capacity_kwh=args.battery_capacity_kwh,
        charge_power_kw=args.charge_power_kw,
        max_price_czk_per_mwh=args.max_price_czk_per_mwh,
        window_start=wstart,
        window_end=wend,
        prefer_window=bool(args.prefer_window),
    )

    result = plan_charging(ote_df=ote_df, inputs=inputs, tz=tz)

    win_start_dt, win_end_dt = daterange_window(plan_date, wstart, wend, tz)

    print(
        f"Morning reserve target: {inputs.reserve_fraction * 100:.0f}% ({inputs.battery_capacity_kwh * inputs.reserve_fraction:.2f} kWh). "
        f"Hours until 09:00: {result.decision.hours_until_9:.2f}h. "
        f"Expected consumption until 09:00: {result.decision.expected_consumption_until_9_kwh:.2f} kWh. "
        f"Deficit for morning reserve: {result.decision.deficit_for_morning_reserve_kwh:.2f} kWh."
    )

    # Report
    print("\n=== Inputs ===")
    print(f"Plan date (local):           {plan_date.isoformat()}")
    print(f"OTE URL/path:                {ote_url}")
    print(f"CNB URL/path:                {cnb_url}")
    print(f"EUR→CZK rate used:           {eur_czk:.4f}")
    print(f"Battery capacity:            {inputs.battery_capacity_kwh:.2f} kWh")
    print(f"Current battery energy:      {inputs.current_battery_state_kwh:.2f} kWh")
    print(f"Forecast PV tomorrow:        {inputs.pv_kwh:.2f} kWh")
    print(f"Charge power:                {inputs.charge_power_kw:.2f} kW")
    print(
        f"Max price:                   {inputs.max_price_czk_per_mwh:.2f} CZK/MWh ({inputs.max_price_czk_per_mwh / 1000:.4f} CZK/kWh)"
    )
    print(
        f"Eligible window:             {win_start_dt:%Y-%m-%d %H:%M} → {win_end_dt:%Y-%m-%d %H:%M} ({'ONLY' if inputs.prefer_window else 'preferred'})"
    )

    print("\n=== Decision ===")
    print(f"Needed to full now:          {result.decision.needed_to_full_kwh:.2f} kWh")
    print(
        f"Deficit after PV forecast (full): {result.decision.deficit_after_pv_for_full_kwh:.2f} kWh"
    )
    print(
        f"Grid charge needed (incl. morning reserve): {result.decision.grid_charge_needed_kwh:.2f} kWh"
    )

    if not result.chosen_slots:
        print(
            "\n✅ No grid charging needed (PV forecast should cover filling the battery)."
        )
        # Send notification for no charging needed
        notification_msg = f"✅ No grid charging needed for {plan_date.isoformat()}\n(PV forecast should cover filling the battery)."
        send_pushover_notification(
            message=notification_msg, title="Home Energy Planner"
        )
        return 0

    kwh_per_slot = inputs.charge_power_kw * 0.25
    planned_kwh = len(result.chosen_slots) * kwh_per_slot

    avg_price_czk_per_mwh = sum(s.price_czk_per_mwh for s in result.chosen_slots) / max(
        1, len(result.chosen_slots)
    )

    print("\n⚡ Planned grid charging slots (15 min):")
    for s in result.chosen_slots:
        print(
            f"- {s.start:%Y-%m-%d %H:%M} → {s.end:%H:%M}  |  {s.price_czk_per_mwh:8.2f} CZK/MWh  ({s.price_czk_per_kwh:.4f} CZK/kWh)"
        )

    print("\n📌 Merged intervals (easier to execute):")
    for s in result.merged_slots:
        print(
            f"- {s.start:%Y-%m-%d %H:%M} → {s.end:%Y-%m-%d %H:%M}  |  {s.price_czk_per_mwh:8.2f} CZK/MWh"
        )

    ha_text_config = ",".join(
        [
            f"{s.start.hour:02d}{s.start.minute:02d}-{s.end.hour:02d}{s.end.minute:02d}"
            for s in result.merged_slots
        ]
    )
    print("HA text config: ", ha_text_config)

    # Set the HA helper variable
    if args.ha_base_url and args.ha_token:
        try:
            url = args.ha_base_url.rstrip("/") + "/api/services/input_text/set_value"
            headers = {
                "Authorization": f"Bearer {args.ha_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "entity_id": "input_text.charge_windows",
                "value": ha_text_config,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=15)
            r.raise_for_status()
            print("✅ Updated input_text.charge_windows in Home Assistant")
        except Exception as e:
            print(
                f"⚠️  Failed to update input_text.charge_windows: {e}",
                file=sys.stderr,
            )
    elif not use_local_inputs:
        print(
            "⚠️  HA credentials not available; skipping input_text.charge_windows update",
            file=sys.stderr,
        )

    print("\n=== Summary ===")
    print(
        f"Slots selected:              {len(result.chosen_slots)} (≈ {planned_kwh:.2f} kWh at {kwh_per_slot:.2f} kWh/slot)"
    )
    print(
        f"Average selected price:      {avg_price_czk_per_mwh:.2f} CZK/MWh ({avg_price_czk_per_mwh / 1000:.4f} CZK/kWh)"
    )

    if args.charger_switch:
        if not args.ha_base_url or not args.ha_token:
            print(
                "\nNOTE: --charger-switch was provided but HA credentials are missing; skipping curl snippet output.",
                file=sys.stderr,
            )
            return 0

        print("\n=== Home Assistant service call snippets ===")
        print("# Turn ON at each interval start, OFF at each interval end.")
        print(
            "# You can paste these into HA scripts or use an external scheduler (cron) calling HA REST API."
        )
        print("# curl examples:")
        for s in result.merged_slots:
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

    # Send Pushover notification with results
    # Format merged slots
    slots_text = "\n".join(
        [
            f"• {s.start:%H:%M} - {s.end:%H:%M} ({s.price_czk_per_mwh:.2f} CZK/MWh)"
            for s in result.merged_slots
        ]
    )

    # Calculate total charging time
    total_minutes = sum(
        int((s.end - s.start).total_seconds() / 60) for s in result.merged_slots
    )
    total_hours = total_minutes // 60
    remaining_minutes = total_minutes % 60
    if total_hours > 0:
        total_time_str = (
            f"{total_hours}h {remaining_minutes}min"
            if remaining_minutes > 0
            else f"{total_hours}h"
        )
    else:
        total_time_str = f"{remaining_minutes}min"

    notification_msg = (
        f"Average price: {avg_price_czk_per_mwh:.2f} CZK/MWh\n"
        f"Total charging time: {total_time_str}\n\n"
        f"Time slots:\n{slots_text}"
    )

    send_pushover_notification(
        message=notification_msg, title=f"⚡ Charging plan for {plan_date.isoformat()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
