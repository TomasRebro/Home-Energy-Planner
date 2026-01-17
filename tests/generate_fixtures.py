#!/usr/bin/env python3
"""Generate test fixture XLSX files for OTE and CNB data."""

import datetime as dt
from pathlib import Path

import pandas as pd

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_ote_fixture(output_path: Path):
    """Generate a sample OTE XLSX file."""
    _market_date = dt.date(2026, 1, 5)

    # Generate 96 15-minute slots for the day
    slots = []
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            start_time = dt.time(hour, minute)
            end_minute = (minute + 15) % 60
            end_hour = hour if minute < 45 else (hour + 1) % 24
            end_time = dt.time(end_hour, end_minute)
            interval = f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}"

            # Generate realistic price data (EUR/MWh)
            # Lower prices overnight (00:00-06:00), higher during day
            if 0 <= hour < 6:
                price = 45.0 + (hour * 2.0) + (minute / 15.0 * 0.5)  # 45-60 EUR/MWh
            elif 6 <= hour < 12:
                price = 60.0 + (hour - 6) * 5.0 + (minute / 15.0 * 1.0)  # 60-90 EUR/MWh
            elif 12 <= hour < 18:
                price = (
                    90.0 + (hour - 12) * 3.0 + (minute / 15.0 * 0.8)
                )  # 90-108 EUR/MWh
            else:
                price = (
                    108.0 - (hour - 18) * 4.0 - (minute / 15.0 * 0.5)
                )  # 108-60 EUR/MWh

            slots.append(
                {
                    "Perioda": f"{len(slots) + 1}",
                    "Časový interval": interval,
                    "15 min cena (EUR/MWh)": round(price, 2),
                }
            )

    # Create DataFrame with proper structure
    df = pd.DataFrame(slots)

    # Create XLSX file with header row
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Write a title row (as OTE files often have)
        title_df = pd.DataFrame([["SPOT MARKET INDEX - 05.01.2026"]])
        title_df.to_excel(
            writer, sheet_name="Sheet1", index=False, header=False, startrow=0
        )

        # Write the data with header
        df.to_excel(writer, sheet_name="Sheet1", index=False, header=True, startrow=1)

    print(f"Generated OTE fixture: {output_path}")


def generate_cnb_fixture(output_path: Path):
    """Generate a sample CNB XLSX file."""
    year = 2026

    # Generate exchange rates for January 2026
    dates = []
    rates = []
    currencies = []

    # Generate data for first 10 days of January
    for day in range(1, 11):
        date = dt.date(year, 1, day)
        # Realistic EUR/CZK rate around 25.0
        rate = 24.8 + (day % 3) * 0.1  # Slight variation
        dates.append(date.strftime("%d/%m/%Y"))
        rates.append(round(rate, 4))
        currencies.append("EUR")

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Den": dates,
            "Kurz": rates,
            "Měna": currencies,
        }
    )

    # Create XLSX file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False, header=True)

    print(f"Generated CNB fixture: {output_path}")


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "fixtures"

    ote_path = fixtures_dir / "DT_15MIN_05_01_2026_CZ.xlsx"
    cnb_path = fixtures_dir / "Kurzovni_listek_CNB_2026.xlsx"

    generate_ote_fixture(ote_path)
    generate_cnb_fixture(cnb_path)

    print("\n✅ All fixtures generated successfully!")
