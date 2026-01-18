"""Pytest configuration and fixtures for planner tests."""

import datetime as dt
from pathlib import Path

import pytest

# Add src to path so we can import planner
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def fixtures_dir():
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def cheap_night_expensive_day_ote_xlsx_path(fixtures_dir):
    """Return path to OTE XLSX file."""
    return str(fixtures_dir / "cheap_night_expensive_day_DT_15MIN_16_01_2026_CZ.xlsx")


@pytest.fixture
def cheap_night_expensive_day_cnb_xlsx_path(fixtures_dir):
    """Return path to CNB XLSX file."""
    return str(
        fixtures_dir
        / "cheap_night_expensive_day_Kurzovni_listek_CNB_2026_16_01_2026.xlsx"
    )


@pytest.fixture
def prague_tz():
    """Return Prague timezone."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Europe/Prague")
    except Exception:
        return dt.timezone(dt.timedelta(hours=1))
