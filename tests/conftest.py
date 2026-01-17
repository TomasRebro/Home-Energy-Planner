"""Pytest configuration and fixtures for planner tests."""

import datetime as dt
import importlib.util
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
def sample_ote_xlsx_path(fixtures_dir):
    """Return path to sample OTE XLSX file (creates it if missing)."""
    xlsx_path = fixtures_dir / "DT_15MIN_05_01_2026_CZ.xlsx"

    # If file doesn't exist, generate it
    if not xlsx_path.exists():
        # Load generate_fixtures module
        spec = importlib.util.spec_from_file_location(
            "generate_fixtures", Path(__file__).parent / "generate_fixtures.py"
        )
        gen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_mod)
        gen_mod.generate_ote_fixture(xlsx_path)

    return str(xlsx_path)


@pytest.fixture
def sample_cnb_xlsx_path(fixtures_dir):
    """Return path to sample CNB XLSX file (creates it if missing)."""
    xlsx_path = fixtures_dir / "Kurzovni_listek_CNB_2026.xlsx"

    # If file doesn't exist, generate it
    if not xlsx_path.exists():
        # Load generate_fixtures module
        spec = importlib.util.spec_from_file_location(
            "generate_fixtures", Path(__file__).parent / "generate_fixtures.py"
        )
        gen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_mod)
        gen_mod.generate_cnb_fixture(xlsx_path)

    return str(xlsx_path)


@pytest.fixture
def prague_tz():
    """Return Prague timezone."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Europe/Prague")
    except Exception:
        return dt.timezone(dt.timedelta(hours=1))
