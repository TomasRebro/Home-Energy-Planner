"""Tests for planner functionality using local XLSX files."""

import datetime as dt

from planner import (
    PlanningInputs,
    compute_decision,
    load_cnb_eur_czk_rate,
    load_ote_15min_prices_eur_per_mwh,
    plan_charging,
)


class TestOTELoading:
    """Test loading OTE 15-minute price data from XLSX files."""

    def test_load_ote_prices_from_local_file(
        self, cheap_night_expensive_day_ote_xlsx_path, prague_tz
    ):
        """Test loading OTE prices from a local XLSX file."""
        df = load_ote_15min_prices_eur_per_mwh(
            cheap_night_expensive_day_ote_xlsx_path, tz=prague_tz
        )

        assert not df.empty, "Should load price data"
        assert "start" in df.columns, "Should have 'start' column"
        assert "price_eur_per_mwh" in df.columns, (
            "Should have 'price_eur_per_mwh' column"
        )

        # Should have 96 slots (24 hours * 4 slots per hour)
        assert len(df) == 96, f"Expected 96 slots, got {len(df)}"

        # Check that start times are timezone-aware
        assert df["start"].iloc[0].tzinfo is not None, (
            "Start times should be timezone-aware"
        )

        # Check that prices are numeric
        assert df["price_eur_per_mwh"].dtype in ["float64", "float32"], (
            "Prices should be numeric"
        )

        # Check that prices are reasonable (EUR/MWh)
        assert df["price_eur_per_mwh"].max() < 1000, "Prices should be reasonable"

    def test_ote_prices_sorted_by_time(
        self, cheap_night_expensive_day_ote_xlsx_path, prague_tz
    ):
        """Test that OTE prices are sorted by start time."""
        df = load_ote_15min_prices_eur_per_mwh(
            cheap_night_expensive_day_ote_xlsx_path, tz=prague_tz
        )

        starts = df["start"].tolist()
        assert starts == sorted(starts), "Prices should be sorted by start time"


class TestCNBLoading:
    """Test loading CNB exchange rate data from XLSX files."""

    def test_load_cnb_rate_from_local_file(
        self, cheap_night_expensive_day_cnb_xlsx_path
    ):
        """Test loading CNB EUR/CZK rate from a local XLSX file."""
        target_date = dt.date(2026, 1, 15)
        rate = load_cnb_eur_czk_rate(
            cheap_night_expensive_day_cnb_xlsx_path, target_date
        )

        assert isinstance(rate, float), "Rate should be a float"
        assert rate == 24.275

    def test_load_cnb_rate_from_local_file_past_date(
        self, cheap_night_expensive_day_cnb_xlsx_path
    ):
        """Test loading CNB EUR/CZK rate from a local XLSX file."""
        target_date = dt.date(2026, 1, 14)
        rate = load_cnb_eur_czk_rate(
            cheap_night_expensive_day_cnb_xlsx_path, target_date
        )

        assert isinstance(rate, float), "Rate should be a float"
        assert rate == 24.27

    def test_cnb_rate_fallback_to_previous_date(
        self, cheap_night_expensive_day_cnb_xlsx_path
    ):
        """Test that CNB rate falls back to previous date if exact date not found."""
        # Request date that doesn't exist in fixture (Jan 15)
        target_date = dt.date(2026, 1, 16)
        rate = load_cnb_eur_czk_rate(
            cheap_night_expensive_day_cnb_xlsx_path, target_date
        )

        assert isinstance(rate, float), (
            "Should return a rate from nearest previous date"
        )
        assert rate == 24.275


class TestDecisionComputation:
    """Test the decision computation logic."""

    def test_compute_decision_no_charging_needed(self, prague_tz):
        """Test decision when PV forecast covers battery needs."""
        inputs = PlanningInputs(
            plan_date=dt.date(2026, 1, 5),
            eur_czk=25.0,
            pv_kwh=20.0,  # High PV forecast
            current_battery_state_kwh=10.0,
            battery_capacity_kwh=14.2,
            charge_power_kw=3.0,
            max_price_czk_per_mwh=2600.0,
            window_start=dt.time(0, 0),
            window_end=dt.time(6, 0),
            prefer_window=False,
        )

        # Set now to early morning so we have time until 9am
        now = dt.datetime(2026, 1, 5, 2, 0, tzinfo=prague_tz)
        decision = compute_decision(inputs, tz=prague_tz, now_local=now)

        assert decision.needed_to_full_kwh > 0, "Should need energy to fill battery"
        # If PV is high enough, deficit_after_pv_for_full should be low or zero
        assert decision.deficit_after_pv_for_full_kwh >= 0, (
            "Deficit should be non-negative"
        )

    def test_compute_decision_charging_needed(self, prague_tz):
        """Test decision when grid charging is needed."""
        inputs = PlanningInputs(
            plan_date=dt.date(2026, 1, 5),
            eur_czk=25.0,
            pv_kwh=2.0,  # Low PV forecast
            current_battery_state_kwh=5.0,  # Low battery state
            battery_capacity_kwh=14.2,
            charge_power_kw=3.0,
            max_price_czk_per_mwh=2600.0,
            window_start=dt.time(0, 0),
            window_end=dt.time(6, 0),
            prefer_window=False,
        )

        now = dt.datetime(2026, 1, 5, 2, 0, tzinfo=prague_tz)
        decision = compute_decision(inputs, tz=prague_tz, now_local=now)

        assert decision.grid_charge_needed_kwh > 0, "Should need grid charging"
        assert decision.needed_to_full_kwh > 0, "Should need energy to fill battery"


class TestPlanning:
    """Test the full planning functionality."""

    def test_plan_charging_cheap_night_expensive_day_slow_charging(
        self,
        cheap_night_expensive_day_ote_xlsx_path,
        cheap_night_expensive_day_cnb_xlsx_path,
        prague_tz,
    ):
        """Test full planning workflow with local XLSX files."""
        # Load data
        ote_df = load_ote_15min_prices_eur_per_mwh(
            cheap_night_expensive_day_ote_xlsx_path, tz=prague_tz
        )
        eur_czk = load_cnb_eur_czk_rate(
            cheap_night_expensive_day_cnb_xlsx_path, dt.date(2026, 1, 16)
        )

        # Create inputs
        inputs = PlanningInputs(
            plan_date=dt.date(2026, 1, 5),
            eur_czk=eur_czk,
            pv_kwh=5.0,  # Moderate PV
            current_battery_state_kwh=8.0,
            battery_capacity_kwh=14.2,
            charge_power_kw=3.0,
            max_price_czk_per_mwh=2600.0,
            window_start=dt.time(0, 0),
            window_end=dt.time(6, 0),
            prefer_window=False,
        )

        # Run planning
        now = dt.datetime(2026, 1, 15, 20, 0, tzinfo=prague_tz)
        result = plan_charging(
            ote_df=ote_df, inputs=inputs, tz=prague_tz, now_local=now
        )

        # Verify result structure
        assert result.inputs == inputs, "Result should contain inputs"
        assert result.decision is not None, "Result should contain decision"
        assert isinstance(result.chosen_slots, list), (
            "Result should contain chosen slots"
        )
        assert isinstance(result.merged_slots, list), (
            "Result should contain merged slots"
        )

        # If charging is needed, verify slots
        if result.decision.grid_charge_needed_kwh > 0:
            assert len(result.chosen_slots) > 0, (
                "Should have chosen slots if charging needed"
            )
            assert len(result.merged_slots) > 0, (
                "Should have merged slots if charging needed"
            )

            # Verify slot prices are within max price
            for slot in result.chosen_slots:
                assert slot.price_czk_per_mwh <= inputs.max_price_czk_per_mwh, (
                    f"Slot price {slot.price_czk_per_mwh} exceeds max {inputs.max_price_czk_per_mwh}"
                )

    def test_plan_charging_no_charging_needed(
        self,
        cheap_night_expensive_day_ote_xlsx_path,
        cheap_night_expensive_day_cnb_xlsx_path,
        prague_tz,
    ):
        """Test planning when no grid charging is needed."""
        ote_df = load_ote_15min_prices_eur_per_mwh(
            cheap_night_expensive_day_ote_xlsx_path, tz=prague_tz
        )
        eur_czk = load_cnb_eur_czk_rate(
            cheap_night_expensive_day_cnb_xlsx_path, dt.date(2026, 1, 5)
        )

        inputs = PlanningInputs(
            plan_date=dt.date(2026, 1, 5),
            eur_czk=eur_czk,
            pv_kwh=20.0,  # High PV - should cover needs
            current_battery_state_kwh=12.0,  # Already fairly full
            battery_capacity_kwh=14.2,
            charge_power_kw=3.0,
            max_price_czk_per_mwh=2708.0,
            window_start=dt.time(0, 0),
            window_end=dt.time(6, 0),
            prefer_window=False,
        )

        now = dt.datetime(2026, 1, 5, 1, 0, tzinfo=prague_tz)
        result = plan_charging(
            ote_df=ote_df, inputs=inputs, tz=prague_tz, now_local=now
        )

        # If no charging needed, slots should be empty
        if result.decision.grid_charge_needed_kwh <= 0:
            assert len(result.chosen_slots) == 0, (
                "Should have no slots if no charging needed"
            )
            assert len(result.merged_slots) == 0, (
                "Should have no merged slots if no charging needed"
            )

    def test_plan_charging_window_preference(
        self,
        cheap_night_expensive_day_ote_xlsx_path,
        cheap_night_expensive_day_cnb_xlsx_path,
        prague_tz,
    ):
        """Test planning with window preference settings."""
        ote_df = load_ote_15min_prices_eur_per_mwh(
            cheap_night_expensive_day_ote_xlsx_path, tz=prague_tz
        )
        eur_czk = load_cnb_eur_czk_rate(
            cheap_night_expensive_day_cnb_xlsx_path, dt.date(2026, 1, 5)
        )

        inputs = PlanningInputs(
            plan_date=dt.date(2026, 1, 5),
            eur_czk=eur_czk,
            pv_kwh=3.0,
            current_battery_state_kwh=6.0,
            battery_capacity_kwh=14.2,
            charge_power_kw=3.0,
            max_price_czk_per_mwh=2708.0,
            window_start=dt.time(0, 0),
            window_end=dt.time(6, 0),
            prefer_window=True,  # Only charge in window
        )

        now = dt.datetime(2026, 1, 5, 1, 0, tzinfo=prague_tz)
        result = plan_charging(
            ote_df=ote_df, inputs=inputs, tz=prague_tz, now_local=now
        )

        if result.decision.grid_charge_needed_kwh > 0 and len(result.chosen_slots) > 0:
            # All slots should be within the window
            window_start_dt = dt.datetime.combine(
                inputs.plan_date, inputs.window_start, tzinfo=prague_tz
            )
            window_end_dt = dt.datetime.combine(
                inputs.plan_date, inputs.window_end, tzinfo=prague_tz
            )
            if window_end_dt <= window_start_dt:
                window_end_dt += dt.timedelta(days=1)

            for slot in result.chosen_slots:
                assert window_start_dt <= slot.start < window_end_dt, (
                    f"Slot {slot.start} should be within window when prefer_window=True"
                )
