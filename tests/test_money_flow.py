from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from economy_simulator.domain import ESSENTIAL_SECTOR_KEYS, Entrepreneur, SECTOR_BY_KEY, SECTOR_SPECS, SimulationConfig
from economy_simulator.engine import EconomySimulation
from economy_simulator.policies import scenario_policy_presets
from economy_simulator.reporting import core_history_frame, history_frame


def _initial_liquid_money(sim: EconomySimulation) -> float:
    return (
        sum(household.savings for household in sim.households if household.alive)
        + sum(owner.wealth + owner.vault_cash for owner in sim.entrepreneurs)
        + sum(firm.cash for firm in sim.firms if firm.active)
        + sim.government.treasury_cash
    )


class MoneyFlowTests(unittest.TestCase):
    def test_policy_presets_include_all_country_profiles(self) -> None:
        presets = scenario_policy_presets()

        self.assertEqual(
            set(presets),
            {
                "Guatemala (mas liberal)",
                "Estados Unidos (mixto)",
                "Noruega (economia del bienestar)",
                "Estado social intensivo (benchmark)",
            },
        )

    def test_run_stops_early_once_population_is_extinct(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=5, households=60, firms_per_sector=1, seed=17)
        )

        with patch.object(sim, "step", side_effect=[SimpleNamespace(population=0)]) as step_mock:
            result = sim.run()

        self.assertEqual(step_mock.call_count, 1)
        self.assertIs(result.history, sim.history)

    def test_firm_history_is_disabled_by_default(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=2, households=80, firms_per_sector=2, seed=41)
        )

        result = sim.run()

        self.assertEqual(result.firm_history, [])

    def test_history_frame_exposes_core_macro_columns(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=3, households=120, firms_per_sector=2, seed=42)
        )

        sim.run()
        frame = history_frame(
            sim.history,
            periods_per_year=sim.config.periods_per_year,
            target_unemployment=sim.config.target_unemployment,
        )

        for column in (
            "cpi",
            "average_wage",
            "real_average_wage",
            "employment_count",
            "potential_gdp_nominal",
            "potential_real_gdp",
            "output_gap_share",
            "recession_flag",
            "recession_intensity",
            "government_countercyclical_spending",
            "household_final_consumption_share_gdp",
            "government_deficit_share_gdp",
        ):
            self.assertIn(column, frame.columns)
        self.assertIn("essential_production_units", frame.columns)

    def test_core_history_frame_keeps_only_macro_subset(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=3, households=120, firms_per_sector=2, seed=43)
        )

        sim.run()
        frame = core_history_frame(
            sim.history,
            periods_per_year=sim.config.periods_per_year,
            target_unemployment=sim.config.target_unemployment,
        )

        self.assertIn("gdp_nominal", frame.columns)
        self.assertIn("bank_equity", frame.columns)
        self.assertIn("school_enrollment_share", frame.columns)
        self.assertIn("recession_intensity", frame.columns)
        self.assertNotIn("capitalist_bank_deposits", frame.columns)

    def test_government_recession_signal_activates_countercyclical_mode(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=1, seed=52)
        )
        sim.history = [
            SimpleNamespace(unemployment_rate=0.24, gdp_nominal=100.0, price_index=1.0, employment_rate=0.76),
            SimpleNamespace(unemployment_rate=0.27, gdp_nominal=98.0, price_index=1.0, employment_rate=0.73),
            SimpleNamespace(unemployment_rate=0.29, gdp_nominal=96.0, price_index=1.0, employment_rate=0.71),
        ]

        recession_flag, recession_intensity = sim._government_recession_signal()

        self.assertTrue(recession_flag)
        self.assertGreater(recession_intensity, 0.0)
        self.assertGreater(
            sim._government_countercyclical_support_multiplier(recession_intensity),
            1.0,
        )
        self.assertGreater(
            sim._government_countercyclical_procurement_multiplier(recession_intensity),
            1.0,
        )

    def test_essential_array_backend_matches_household_formula(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=18)
        )
        sim._refresh_period_household_caches()
        sim._refresh_period_sector_caches()

        household = next(household for household in sim.households if household.alive)
        sim._ensure_household_demand_shares(household)
        age_years = sim._household_age_years(household)
        if age_years < sim.config.entry_age_years:
            progress = max(0.0, min(1.0, age_years / max(1.0, sim.config.entry_age_years)))
            consumption_multiplier = max(
                sim.config.child_consumption_multiplier,
                min(
                    1.0,
                    sim.config.child_consumption_multiplier
                    + (1.0 - sim.config.child_consumption_multiplier) * progress,
                ),
            )
        elif age_years < sim.config.senior_age_years:
            consumption_multiplier = 1.0
        elif age_years >= sim.config.max_age_years:
            consumption_multiplier = sim.config.senior_consumption_multiplier
        else:
            senior_span = max(1.0, sim.config.max_age_years - sim.config.senior_age_years)
            progress = max(0.0, min(1.0, (age_years - sim.config.senior_age_years) / senior_span))
            consumption_multiplier = max(
                0.65,
                min(1.0, sim.config.senior_consumption_multiplier - 0.10 * progress),
            )

        base_units = household.need_scale * consumption_multiplier
        expected_units = {
            sector_key: base_units * household.essential_shares[sector_key]
            for sector_key in ESSENTIAL_SECTOR_KEYS
        }
        discretionary_scale = (
            sim.config.nonessential_demand_multiplier
            * sum(SECTOR_BY_KEY[key].household_demand_share for key in ("manufactured", "leisure", "school", "university"))
        )
        expected_units["manufactured"] = base_units * discretionary_scale * household.discretionary_shares["manufactured"]
        expected_units["leisure"] = base_units * discretionary_scale * household.discretionary_shares["leisure"]

        for sector_key, expected in expected_units.items():
            self.assertAlmostEqual(
                sim._household_sector_desired_units(household, sector_key),
                expected,
                places=9,
            )

        expected_budget = sum(
            expected_units[sector_key] * sim._average_sector_price(sector_key)
            for sector_key in ESSENTIAL_SECTOR_KEYS
        )
        self.assertAlmostEqual(sim._essential_budget(household), expected_budget, places=9)

    def test_school_demand_uses_family_resources_not_child_cash(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=160, firms_per_sector=2, seed=44)
        )
        sim._refresh_period_household_caches()
        sim._refresh_period_family_cache()

        child = next(
            household
            for household in sim.households
            if household.alive and sim._is_school_age(household)
        )
        guardian = next(
            household
            for household in sim._family_groups()[sim._family_root_for_child(child)]
            if sim._household_age_years(household) >= sim.config.entry_age_years
        )

        child.savings = 0.0
        child.last_available_cash = 0.0
        guardian.savings = max(guardian.savings, sim._essential_budget(guardian) * 3.0)
        guardian.last_available_cash = guardian.savings
        sim._period_family_resource_coverage_cache = {}

        self.assertGreater(sim._school_service_target_units(child), 0.0)

    def test_public_school_spending_survives_without_private_school_firms(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=240, firms_per_sector=2, seed=45)
        )
        for firm in sim.firms:
            if firm.sector == "school":
                firm.active = False
                firm.inventory = 0.0
                firm.workers = []

        sim.government.treasury_cash = 50_000.0
        sim._reset_period_counters()
        sim._refresh_period_household_caches()
        sim._refresh_period_sector_caches()
        sim._refresh_family_links()
        sim._refresh_period_family_cache()
        sim._consume_households()

        self.assertGreater(sim._period_government_school_spending, 0.0)
        self.assertTrue(
            any(
                household.last_consumption.get("school", 0.0) > 0.0
                for household in sim.households
                if household.alive and sim._is_school_age(household)
            )
        )

    def test_essential_marginal_utility_is_strongly_decreasing(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=31)
        )

        utility_low = sim._essential_marginal_utility(0.25)
        utility_target = sim._essential_marginal_utility(1.0)
        utility_high = sim._essential_marginal_utility(2.0)

        self.assertGreater(utility_low, utility_target)
        self.assertGreater(utility_target, utility_high)
        self.assertGreater(utility_low, utility_high * 10.0)

    def test_extra_essential_budget_share_drops_after_coverage(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=37)
        )

        share_unmet = sim._essential_extra_budget_share(0.6)
        share_covered = sim._essential_extra_budget_share(1.2)
        share_overcovered = sim._essential_extra_budget_share(1.8)

        self.assertGreater(share_unmet, share_covered)
        self.assertGreater(share_covered, share_overcovered)

    def test_affordability_pressure_relaxes_when_shortage_is_physical(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=2, households=120, firms_per_sector=2, startup_grace_periods=1, seed=38)
        )
        sim.step()

        abundant_snapshot = dataclasses.replace(
            sim.history[-1],
            family_income_to_basket_ratio=0.80,
            essential_fulfillment_rate=1.00,
        )
        scarce_snapshot = dataclasses.replace(
            sim.history[-1],
            family_income_to_basket_ratio=0.80,
            essential_fulfillment_rate=0.30,
        )

        sim.history[-1] = abundant_snapshot
        pressure_with_abundant_supply = sim._essential_affordability_pressure()
        sim.history[-1] = scarce_snapshot
        pressure_with_scarcity = sim._essential_affordability_pressure()

        self.assertGreater(pressure_with_abundant_supply, pressure_with_scarcity)
        self.assertGreater(pressure_with_abundant_supply, 0.0)

    def test_startup_essential_candidate_prices_respect_cost_floor(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=39)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        firm.price = 5.0
        firm.last_unit_cost = 6.2

        with patch.object(sim, "_firm_max_price_hike_ratio", return_value=1.03):
            candidate_prices = sim._startup_essential_candidate_prices(firm, variable_unit_cost=6.0)

        self.assertTrue(candidate_prices)
        self.assertGreaterEqual(min(candidate_prices), 6.0 * 1.01 - 1e-9)

    def test_essential_price_search_respects_cost_floor_even_if_current_price_is_too_low(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, startup_grace_periods=0, seed=40)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "housing")
        firm.price = 2.0
        firm.inventory = 5.0
        firm.target_inventory = 4.0
        firm.cash = 200.0
        firm.last_wage_bill = 40.0
        firm.fixed_overhead = 5.0
        firm.capital = 20.0
        firm.price_aggressiveness = 1.0
        firm.cash_conservatism = 1.0

        candidates = sim._price_search_candidates(
            firm,
            SECTOR_BY_KEY["housing"],
            variable_unit_cost=4.0,
            target_price=4.5,
        )

        self.assertTrue(candidates)
        self.assertGreaterEqual(min(candidates), 4.0 * 1.01 - 1e-9)

    def test_leisure_utility_weight_rises_once_basic_needs_are_covered(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=39)
        )

        stressed_leisure = sim._discretionary_sector_utility_weight(
            "leisure",
            1.0,
            essential_coverage=0.75,
            family_remaining_cash=0.0,
            family_basic_basket_cost=100.0,
        )
        comfortable_leisure = sim._discretionary_sector_utility_weight(
            "leisure",
            1.0,
            essential_coverage=1.25,
            family_remaining_cash=140.0,
            family_basic_basket_cost=100.0,
        )
        comfortable_manufactured = sim._discretionary_sector_utility_weight(
            "manufactured",
            1.0,
            essential_coverage=1.25,
            family_remaining_cash=140.0,
            family_basic_basket_cost=100.0,
        )
        comfortable_school = sim._discretionary_sector_utility_weight(
            "school",
            1.0,
            essential_coverage=1.25,
            family_remaining_cash=140.0,
            family_basic_basket_cost=100.0,
        )
        comfortable_university = sim._discretionary_sector_utility_weight(
            "university",
            1.0,
            essential_coverage=1.25,
            family_remaining_cash=140.0,
            family_basic_basket_cost=100.0,
        )

        self.assertGreater(comfortable_leisure, stressed_leisure)
        self.assertGreater(comfortable_school, comfortable_leisure)
        self.assertGreater(comfortable_university, comfortable_school)
        self.assertGreater(comfortable_leisure, comfortable_manufactured)

    def test_school_market_factor_starts_before_full_affluence(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=40)
        )

        school_age = next(h for h in sim.households if h.alive and sim._is_school_age(h))
        school_age.savings = sim._essential_budget(school_age) * 0.35
        sim._period_family_resource_coverage_cache = {school_age.id: 0.35}

        market_factor = sim._household_education_market_factor(school_age, advanced=False)

        self.assertGreater(market_factor, 0.0)

    def test_private_school_startup_price_comes_from_service_cost(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=2, seed=92)
        )

        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "school")
        spec = SECTOR_BY_KEY["school"]

        self.assertGreater(firm.education_level_span, 0.0)
        self.assertGreater(sim._education_firm_capacity(firm), 0.0)
        self.assertAlmostEqual(firm.price, firm.last_unit_cost * (1.0 + spec.markup), places=6)

    def test_school_service_capacity_limits_production_and_resets_inventory(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=220, firms_per_sector=2, seed=93)
        )

        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "school")
        for worker_id in list(firm.workers):
            sim._release_household_from_employment(sim.households[worker_id])
        firm.workers.clear()

        replacement_workers = [
            household
            for household in sim.households
            if household.alive and sim._household_age_years(household) >= sim.config.entry_age_years
        ][:8]
        for household in replacement_workers:
            sim._release_household_from_employment(household)
            household.employed_by = firm.id
            firm.workers.append(household.id)

        firm.cash = max(firm.cash, 500.0)
        firm.capital = sim.config.school_classroom_capital_cost
        firm.education_level_span = sim.config.school_years_required
        firm.inventory = 999.0
        capacity = sim._education_firm_capacity(firm)

        sim._produce_and_pay_wages()

        self.assertLessEqual(firm.last_production, capacity + 1e-9)
        self.assertAlmostEqual(firm.inventory, firm.last_production, places=6)

    def test_extra_essential_gap_units_caps_top_up_near_target(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, extra_essential_coverage_cap=1.10, seed=41)
        )

        self.assertAlmostEqual(sim._extra_essential_gap_units(10.0, 6.0), 5.0)
        self.assertAlmostEqual(sim._extra_essential_gap_units(10.0, 11.0), 0.0)
        self.assertAlmostEqual(sim._extra_essential_gap_units(0.0, 5.0), 0.0)

    def test_leisure_sector_has_structural_wage_premium_and_hiring_bonus(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=40)
        )

        leisure_spec = SECTOR_BY_KEY["leisure"]
        manufactured_spec = SECTOR_BY_KEY["manufactured"]
        bonus = sim._sector_wage_pressure_bonus(
            "leisure",
            vacancy_ratio=0.5,
            labor_tightness=0.05,
            living_wage_gap=0.15,
            wage_room=0.20,
        )

        self.assertGreater(leisure_spec.base_wage, manufactured_spec.base_wage)
        self.assertGreater(bonus, 0.0)
        self.assertEqual(
            sim._sector_wage_pressure_bonus(
                "manufactured",
                vacancy_ratio=0.5,
                labor_tightness=0.05,
                living_wage_gap=0.15,
                wage_room=0.20,
            ),
            0.0,
        )

    def test_family_savings_rate_tracks_buffer_trust_and_present_bias(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=37)
        )
        members = [
            candidate
            for candidate in sim.households
            if sim._household_age_years(candidate) >= sim.config.entry_age_years
        ][:2]
        self.assertEqual(len(members), 2)

        for member in members:
            member.deprivation_streak = 0
            member.employed_by = 1
            member.price_sensitivity = 1.0
            member.saving_propensity = 0.25
            member.money_trust = 0.85
            member.consumption_impatience = 0.15

        low_buffer_rate = sim._family_savings_rate(
            members,
            family_cash=80.0,
            family_basic_basket_cost=100.0,
            family_price_sensitivity=1.0,
            family_saving_propensity=0.25,
            family_consumption_multiplier=1.0,
            inflation_pressure=0.0,
        )
        near_cushion_rate = sim._family_savings_rate(
            members,
            family_cash=150.0,
            family_basic_basket_cost=100.0,
            family_price_sensitivity=1.0,
            family_saving_propensity=0.25,
            family_consumption_multiplier=1.0,
            inflation_pressure=0.0,
        )
        high_buffer_rate = sim._family_savings_rate(
            members,
            family_cash=260.0,
            family_basic_basket_cost=100.0,
            family_price_sensitivity=1.0,
            family_saving_propensity=0.25,
            family_consumption_multiplier=1.0,
            inflation_pressure=0.0,
        )

        for member in members:
            member.money_trust = 0.15
            member.consumption_impatience = 0.15
        low_trust_rate = sim._family_savings_rate(
            members,
            family_cash=150.0,
            family_basic_basket_cost=100.0,
            family_price_sensitivity=1.0,
            family_saving_propensity=0.25,
            family_consumption_multiplier=1.0,
            inflation_pressure=0.0,
        )

        for member in members:
            member.money_trust = 0.85
            member.consumption_impatience = 0.90
        high_impatience_rate = sim._family_savings_rate(
            members,
            family_cash=150.0,
            family_basic_basket_cost=100.0,
            family_price_sensitivity=1.0,
            family_saving_propensity=0.25,
            family_consumption_multiplier=1.0,
            inflation_pressure=0.0,
        )

        self.assertGreater(low_buffer_rate, 0.0)
        self.assertGreater(near_cushion_rate, 0.0)
        self.assertLess(high_buffer_rate, low_buffer_rate)
        self.assertGreater(low_trust_rate, near_cushion_rate)
        self.assertLess(high_impatience_rate, near_cushion_rate)

    def test_worker_savings_rate_is_bounded_in_history(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=4, households=150, firms_per_sector=3, seed=41)
        )

        for _ in range(4):
            sim.step()

        frame = history_frame(sim.history)
        self.assertIn("worker_savings_rate", frame.columns)
        self.assertIn("worker_involuntary_retention_rate", frame.columns)
        self.assertTrue((frame["worker_cash_saved"] <= frame["worker_cash_available"] + 1e-9).all())
        self.assertTrue((frame["worker_voluntary_saved"] <= frame["worker_cash_saved"] + 1e-9).all())
        self.assertTrue((frame["worker_involuntary_retained"] <= frame["worker_cash_saved"] + 1e-9).all())
        self.assertTrue(
            (
                (frame["worker_voluntary_saved"] + frame["worker_involuntary_retained"] - frame["worker_cash_saved"])
                .abs()
                <= 1e-9
            ).all()
        )
        bounded = frame["worker_savings_rate"].dropna().between(0.0, 1.0)
        self.assertTrue(bounded.all())
        rationed_bounded = frame["worker_involuntary_retention_rate"].dropna().between(0.0, 1.0)
        self.assertTrue(rationed_bounded.all())

    def test_history_frame_exposes_average_perceived_utility(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=4, households=150, firms_per_sector=3, seed=43)
        )

        for _ in range(4):
            sim.step()

        frame = history_frame(sim.history)

        self.assertIn("average_perceived_utility", frame.columns)
        self.assertIn("perceived_utility_growth", frame.columns)
        self.assertTrue((frame["average_perceived_utility"] >= 0.0).all())

    def test_education_demand_respects_age_and_university_track(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=2, seed=44)
        )

        school_age = next(h for h in sim.households if sim._is_school_age(h))
        young_adult = next(
            h
            for h in sim.households
            if sim._is_university_age(h)
            and sim._household_age_years(h) >= sim.config.entry_age_years
            and not sim._household_has_university_credential(h)
        )
        older_adult = next(h for h in sim.households if sim._household_age_years(h) > sim.config.university_age_max_years)

        school_age.school_years_completed = 0.0
        young_adult.school_years_completed = sim.config.school_years_required
        young_adult.university_years_completed = 0.0
        young_adult.higher_education_affinity = 0.99
        older_adult.school_years_completed = sim.config.school_years_required
        older_adult.university_years_completed = 0.0
        older_adult.higher_education_affinity = 0.99
        young_adult.savings = 500.0
        older_adult.savings = 500.0
        sim._period_family_resource_coverage_cache.clear()

        self.assertGreater(sim._household_sector_desired_units(school_age, "school"), 0.0)
        self.assertEqual(sim._household_sector_desired_units(school_age, "university"), 0.0)
        self.assertGreater(sim._household_sector_desired_units(young_adult, "university"), 0.0)
        self.assertEqual(sim._household_sector_desired_units(older_adult, "university"), 0.0)

    def test_education_credentials_raise_skilled_sector_productivity(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=2, seed=46)
        )

        worker = next(
            h for h in sim.households if sim._household_age_years(h) >= sim.config.entry_age_years
        )
        worker.school_years_completed = sim.config.school_years_required
        worker.university_years_completed = sim.config.university_years_required

        skilled_multiplier = sim._household_skill_multiplier(worker, "leisure")
        university_multiplier = sim._household_skill_multiplier(worker, "university")
        essential_multiplier = sim._household_skill_multiplier(worker, "food")

        self.assertGreater(skilled_multiplier, 1.0)
        self.assertGreater(university_multiplier, skilled_multiplier)
        self.assertEqual(essential_multiplier, 1.0)

    def test_history_frame_exposes_sna_like_expenditure_components(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=4, households=180, firms_per_sector=3, seed=45)
        )

        for _ in range(4):
            sim.step()

        frame = history_frame(sim.history)

        self.assertIn("household_final_consumption", frame.columns)
        self.assertIn("government_final_consumption", frame.columns)
        self.assertIn("gross_fixed_capital_formation", frame.columns)
        self.assertIn("change_in_inventories", frame.columns)
        self.assertIn("gross_capital_formation", frame.columns)
        self.assertIn("net_exports", frame.columns)
        self.assertIn("gdp_expenditure_gap", frame.columns)
        self.assertIn("household_final_consumption_share_gdp", frame.columns)
        self.assertIn("gross_capital_formation_share_gdp", frame.columns)
        self.assertTrue(
            (
                frame["gross_capital_formation"]
                - frame["gross_fixed_capital_formation"]
                - frame["change_in_inventories"]
                - frame["valuables_acquisition"]
            ).abs().le(1e-9).all()
        )
        self.assertTrue((frame["net_exports"] == frame["exports"] - frame["imports"]).all())

    def test_history_frame_exposes_education_metrics(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=4, households=180, firms_per_sector=2, seed=47)
        )

        for _ in range(4):
            sim.step()

        frame = history_frame(sim.history)

        self.assertIn("school_enrollment_share", frame.columns)
        self.assertIn("university_enrollment_share", frame.columns)
        self.assertIn("school_labor_share", frame.columns)
        self.assertIn("low_resource_school_enrollment_share", frame.columns)
        self.assertIn("low_resource_university_enrollment_share", frame.columns)
        self.assertIn("university_income_premium", frame.columns)
        self.assertIn("poverty_rate_with_university", frame.columns)
        self.assertIn("low_resource_origin_upward_mobility_share", frame.columns)
        self.assertIn("skilled_job_fill_rate", frame.columns)
        self.assertIn("skilled_labor_supply_to_demand_ratio", frame.columns)
        self.assertIn("education_poverty_gap", frame.columns)
        self.assertIn("government_education_spending_share_gdp", frame.columns)
        self.assertIn("active_school_firms", frame.columns)
        self.assertIn("active_university_firms", frame.columns)

    def test_snapshot_tracks_low_resource_access_and_origin_mobility_metrics(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=40,
                firms_per_sector=1,
                seed=53,
                annual_birth_rate=0.0,
                annual_base_death_rate=0.0,
                annual_senior_death_rate=0.0,
            )
        )
        zero_consumption = {spec.key: 0.0 for spec in SECTOR_SPECS}
        for household in sim.households:
            household.alive = False
            household.partner_id = None
            household.guardian_id = None
            household.mother_id = None
            household.father_id = None
            household.children_count = 0
            household.dependent_children = 0
            household.employed_by = None
            household.last_income = 0.0
            household.wage_income = 0.0
            household.savings = 0.0
            household.last_available_cash = 0.0
            household.last_consumption = zero_consumption.copy()
            household.school_years_completed = 0.0
            household.university_years_completed = 0.0
            household.origin_record_period = -1
            household.low_resource_origin = False

        adult_uni = sim.households[0]
        adult_nonuni = sim.households[1]
        school_child = sim.households[2]
        poor_uni_candidate = sim.households[3]
        rich_uni_student = sim.households[4]

        for adult in (adult_uni, adult_nonuni):
            adult.alive = True
            adult.age_periods = int(28 * sim.config.periods_per_year)
            adult.school_years_completed = sim.config.school_years_required
            adult.origin_record_period = 1
            adult.low_resource_origin = True

        adult_uni.university_years_completed = sim.config.university_years_required
        adult_uni.last_income = 120.0
        adult_uni.last_available_cash = 140.0
        adult_uni.savings = 140.0

        adult_nonuni.last_income = 20.0
        adult_nonuni.last_available_cash = 5.0

        school_child.alive = True
        school_child.age_periods = int(10 * sim.config.periods_per_year)
        school_child.last_available_cash = 0.0

        poor_uni_candidate.alive = True
        poor_uni_candidate.age_periods = int(20 * sim.config.periods_per_year)
        poor_uni_candidate.school_years_completed = sim.config.school_years_required
        poor_uni_candidate.university_years_completed = 0.0
        poor_uni_candidate.higher_education_affinity = 0.99
        poor_uni_candidate.last_available_cash = 0.0

        rich_uni_student.alive = True
        rich_uni_student.age_periods = int(20 * sim.config.periods_per_year)
        rich_uni_student.school_years_completed = sim.config.school_years_required
        rich_uni_student.university_years_completed = 0.0
        rich_uni_student.higher_education_affinity = 0.99
        rich_uni_student.last_available_cash = 90.0
        rich_uni_student.savings = 90.0
        rich_uni_student.last_income = 40.0

        sim._period_active_households_cache = [household for household in sim.households if household.alive]
        sim._period_family_groups_cache = None
        sim._period_household_age_years_cache = {}
        sim._period_household_desired_units_cache = {}
        sim._period_essential_budget_cache = {}
        school_child.savings = sim._essential_budget(school_child) * 0.35

        school_target = sim._household_sector_desired_units(school_child, "school")
        rich_uni_target = sim._household_sector_desired_units(rich_uni_student, "university")
        poor_uni_target = sim._household_sector_desired_units(poor_uni_candidate, "university")

        school_child.last_consumption["school"] = school_target
        rich_uni_student.last_consumption["university"] = rich_uni_target
        poor_uni_candidate.last_consumption["university"] = poor_uni_target

        snapshot = sim._build_snapshot()

        self.assertAlmostEqual(snapshot.low_resource_school_enrollment_share, 1.0)
        self.assertAlmostEqual(snapshot.low_resource_university_enrollment_share, 0.0)
        self.assertAlmostEqual(snapshot.low_resource_university_student_share, 0.0)
        self.assertEqual(snapshot.tracked_origin_adults, 2)
        self.assertEqual(snapshot.low_resource_origin_adults, 2)
        self.assertAlmostEqual(snapshot.low_resource_origin_upward_mobility_share, 0.5)
        self.assertAlmostEqual(snapshot.low_resource_origin_university_completion_share, 0.5)
        self.assertAlmostEqual(snapshot.low_resource_origin_university_upward_mobility_share, 1.0)
        self.assertAlmostEqual(snapshot.low_resource_origin_nonuniversity_upward_mobility_share, 0.0)
        self.assertLess(snapshot.poverty_rate_with_university, snapshot.poverty_rate_without_university)
        self.assertGreater(snapshot.university_income_premium, 1.0)

    def test_snapshot_tracks_skilled_labor_supply_against_demand(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=40,
                firms_per_sector=1,
                seed=54,
                annual_birth_rate=0.0,
                annual_base_death_rate=0.0,
                annual_senior_death_rate=0.0,
            )
        )
        zero_consumption = {spec.key: 0.0 for spec in SECTOR_SPECS}
        for household in sim.households:
            household.alive = False
            household.partner_id = None
            household.guardian_id = None
            household.mother_id = None
            household.father_id = None
            household.employed_by = None
            household.last_income = 0.0
            household.wage_income = 0.0
            household.last_available_cash = 50.0
            household.last_consumption = zero_consumption.copy()
            household.school_years_completed = 0.0
            household.university_years_completed = 0.0

        university_workers = [sim.households[0], sim.households[1]]
        school_worker = sim.households[2]
        basic_worker = sim.households[3]
        for household in university_workers + [school_worker, basic_worker]:
            household.alive = True
            household.age_periods = int(30 * sim.config.periods_per_year)
            household.school_years_completed = sim.config.school_years_required
        for household in university_workers:
            household.university_years_completed = sim.config.university_years_required
        basic_worker.school_years_completed = 0.0

        for firm in sim.firms:
            firm.active = False
            firm.workers.clear()
            firm.desired_workers = 0

        manufactured_firm = next(firm for firm in sim.firms if firm.sector == "manufactured")
        school_firm = next(firm for firm in sim.firms if firm.sector == "school")
        food_firm = next(firm for firm in sim.firms if firm.sector == "food")

        manufactured_firm.active = True
        manufactured_firm.desired_workers = 6
        manufactured_firm.workers = [university_workers[0].id, university_workers[1].id]
        for household in university_workers:
            household.employed_by = manufactured_firm.id

        school_firm.active = True
        school_firm.desired_workers = 4
        school_firm.workers = [school_worker.id]
        school_worker.employed_by = school_firm.id

        food_firm.active = True
        food_firm.desired_workers = 10
        food_firm.workers = [basic_worker.id]
        basic_worker.employed_by = food_firm.id

        sim._period_active_households_cache = [household for household in sim.households if household.alive]
        sim._period_active_firms_by_sector_cache = None
        sim._period_family_groups_cache = None
        sim._period_household_age_years_cache = {}
        sim._period_household_desired_units_cache = {}
        sim._period_essential_budget_cache = {}
        sim._refresh_period_sector_caches()

        snapshot = sim._build_snapshot()

        self.assertAlmostEqual(snapshot.school_labor_share, 0.75)
        self.assertAlmostEqual(snapshot.skilled_labor_share, 0.50)
        self.assertAlmostEqual(snapshot.skilled_job_demand_share, 0.50)
        self.assertAlmostEqual(snapshot.skilled_job_fill_rate, 0.30)
        self.assertAlmostEqual(snapshot.skilled_labor_supply_to_demand_ratio, 0.20)

    def test_endogenous_entry_can_append_new_firm_slots(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=220,
                firms_per_sector=2,
                initial_private_school_firms=2,
                initial_private_university_firms=1,
                essential_protection_periods=0,
                seed=48,
            )
        )

        owner = sim.entrepreneurs[0]
        owner.wealth = 50000.0
        before_count = len(sim.firms_by_sector["university"])
        before_active = sum(1 for firm in sim.firms_by_sector["university"] if firm.active)
        original_baseline = sim._baseline_demand

        with patch.object(
            sim,
            "_baseline_demand",
            side_effect=lambda sector_key, use_current_period=False: (
                180.0 if sector_key == "university" else original_baseline(sector_key, use_current_period=use_current_period)
            ),
        ), patch.object(sim, "_select_entry_owner", return_value=owner):
            for firm in sim.firms_by_sector["university"]:
                firm.last_expected_sales = 0.5
                firm.last_sales = 0.5
            sim._attempt_endogenous_sector_entry()

        self.assertGreater(len(sim.firms_by_sector["university"]), before_count)
        self.assertGreater(sum(1 for firm in sim.firms_by_sector["university"] if firm.active), before_active)

    def test_essential_protection_blocks_nonessential_entry_until_basics_stabilize(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=2,
                households=220,
                firms_per_sector=2,
                initial_private_school_firms=2,
                initial_private_university_firms=1,
                essential_protection_periods=24,
                seed=148,
            )
        )

        owner = sim.entrepreneurs[0]
        owner.wealth = 50000.0
        sim.period = 1
        sim.history.append(
            sim.history[-1] if sim.history else sim._build_snapshot()
        )
        sim.history[-1] = dataclasses.replace(sim.history[-1], essential_fulfillment_rate=0.70)
        before_count = len(sim.firms_by_sector["university"])
        before_active = sum(1 for firm in sim.firms_by_sector["university"] if firm.active)

        with patch.object(sim, "_baseline_demand", return_value=180.0), patch.object(
            sim, "_select_entry_owner", return_value=owner
        ):
            for firm in sim.firms_by_sector["university"]:
                firm.last_expected_sales = 0.5
                firm.last_sales = 0.5
            sim._attempt_endogenous_sector_entry()

        self.assertEqual(len(sim.firms_by_sector["university"]), before_count)
        self.assertEqual(sum(1 for firm in sim.firms_by_sector["university"] if firm.active), before_active)

    def test_age_zero_firm_snapshots_do_not_report_current_period_output(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=220,
                firms_per_sector=2,
                initial_private_school_firms=2,
                initial_private_university_firms=1,
                essential_protection_periods=24,
                seed=149,
            )
        )

        sim.step()
        age_zero_firm = next(firm for firm in sim.firms if firm.active)
        age_zero_firm.age = 0
        age_zero_firm.last_production = 25.0
        age_zero_firm.last_sales = 20.0
        age_zero_firm.last_revenue = 100.0
        age_zero_firm.last_profit = 15.0

        age_zero_snapshots = [snapshot for snapshot in sim._build_firm_period_snapshots() if snapshot.age == 0]
        self.assertTrue(age_zero_snapshots)
        self.assertTrue(all(snapshot.production == 0.0 for snapshot in age_zero_snapshots))
        self.assertTrue(all(snapshot.sales == 0.0 for snapshot in age_zero_snapshots))

    def test_family_consumption_order_is_not_fixed_by_household_id(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=80,
                firms_per_sector=2,
                initial_private_school_firms=2,
                initial_private_university_firms=1,
                seed=150,
            )
        )

        family_groups = sim._family_groups()
        family_signatures = [tuple(sorted(member.id for member in members)) for members in family_groups.values()]
        consumption_signatures = [
            tuple(sorted(member.id for member in members))
            for members in sim._family_groups_consumption_order()
        ]

        self.assertCountEqual(consumption_signatures, family_signatures)
        self.assertNotEqual(consumption_signatures, family_signatures)

    def test_snapshot_tracks_people_with_full_essential_coverage(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=2,
                households=120,
                firms_per_sector=2,
                initial_private_school_firms=2,
                initial_private_university_firms=1,
                seed=151,
            )
        )

        sim.step()
        snapshot = sim.history[-1]
        direct_count = sum(
            1
            for household in sim._active_households()
            if all(sim._household_sector_coverage(household, sector_key) >= 1.0 for sector_key in ESSENTIAL_SECTOR_KEYS)
        )

        self.assertEqual(snapshot.people_full_essential_coverage, direct_count)
        self.assertAlmostEqual(
            snapshot.full_essential_coverage_share,
            direct_count / max(1, snapshot.population),
        )

    def test_food_hunger_dominates_nonfood_fragility_in_death_probability(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=91)
        )
        households = [
            candidate
            for candidate in sim.households
            if sim._household_age_years(candidate) >= sim.config.entry_age_years
        ][:2]
        self.assertEqual(len(households), 2)
        hungry, nonfood_fragile = households

        hungry.health_fragility = 0.4
        hungry.deprivation_streak = 2
        hungry.severe_hunger_streak = 2
        hungry.housing_deprivation_streak = 0
        hungry.clothing_deprivation_streak = 0

        nonfood_fragile.health_fragility = 1.8
        nonfood_fragile.deprivation_streak = 0
        nonfood_fragile.severe_hunger_streak = 0
        nonfood_fragile.housing_deprivation_streak = 3
        nonfood_fragile.clothing_deprivation_streak = 3

        hungry_probability = sim._household_death_probability(hungry, unemployment_rate=0.1, average_savings=10.0)
        nonfood_probability = sim._household_death_probability(
            nonfood_fragile,
            unemployment_rate=0.1,
            average_savings=10.0,
        )

        self.assertGreater(hungry_probability, nonfood_probability)

    def test_snapshot_exposes_food_condition_metrics(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=2, households=180, firms_per_sector=3, seed=95)
        )

        snapshot = sim.step()
        frame = history_frame([snapshot])

        self.assertIn("average_food_meals_per_person", frame.columns)
        self.assertIn("food_sufficient_share", frame.columns)
        self.assertIn("food_subsistence_share", frame.columns)
        self.assertIn("food_acute_hunger_share", frame.columns)
        self.assertIn("food_severe_hunger_share", frame.columns)
        self.assertIn("average_health_fragility", frame.columns)
        self.assertIn("average_perceived_utility", frame.columns)
        self.assertGreaterEqual(snapshot.average_food_meals_per_person, 0.0)
        self.assertTrue(0.0 <= snapshot.food_sufficient_share <= 1.0)
        self.assertTrue(0.0 <= snapshot.food_severe_hunger_share <= 1.0)
        self.assertGreaterEqual(snapshot.average_perceived_utility, 0.0)

    def test_entry_owner_selection_uses_heterogeneous_market_evaluation(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=97)
        )
        conservative_rich = Entrepreneur(
            id=0,
            wealth=420.0,
            entry_appetite=0.75,
            market_research_skill=0.70,
            entry_optimism=-0.10,
        )
        prepared_smaller = Entrepreneur(
            id=1,
            wealth=220.0,
            entry_appetite=1.35,
            market_research_skill=1.40,
            entry_optimism=0.10,
        )
        sim.entrepreneurs = [conservative_rich, prepared_smaller]

        with patch.object(sim, "_sector_entry_opportunity_signal", return_value=1.0):
            with patch.object(sim.rng, "gauss", side_effect=[-0.25, 0.05]):
                selected = sim._select_entry_owner(
                    "leisure",
                    demand_signal=120.0,
                    entry_gap=35.0,
                    base_restart_cost=80.0,
                )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.id, prepared_smaller.id)

    def test_survival_essential_metrics_ignore_extra_essential_purchases(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=2, seed=53)
        )

        household = next(
            candidate
            for candidate in sim.households
            if sim._household_age_years(candidate) >= sim.config.entry_age_years
        )
        household.id = 0
        household.savings = 1000.0
        household.wage_income = 0.0
        household.saving_propensity = 0.0
        household.price_sensitivity = 0.0
        household.need_scale = 1.0
        household.partner_id = None
        household.guardian_id = None
        sim.households = [household]
        sim._next_household_id = 1

        for firm in sim.firms:
            if not firm.active:
                continue
            firm.inventory = 1000.0
            if firm.sector in ESSENTIAL_SECTOR_KEYS:
                firm.price = 1.0

        sim._reset_period_counters()
        sim._refresh_period_household_caches()
        sim._refresh_period_family_cache()
        sim._refresh_period_sector_caches()
        sim._consume_households()

        basic_target_units = sum(
            sim._household_sector_desired_units(household, sector_key)
            for sector_key in ESSENTIAL_SECTOR_KEYS
        )
        essential_market_demand = sum(
            sim._period_sector_budget_demand_units[sector_key]
            for sector_key in ESSENTIAL_SECTOR_KEYS
        )

        self.assertAlmostEqual(sim._period_essential_demand_units, basic_target_units, places=6)
        self.assertLessEqual(sim._period_essential_sales_units, sim._period_essential_demand_units + 1e-6)
        self.assertGreater(essential_market_demand, sim._period_essential_demand_units)

    def test_firms_have_heterogeneous_forecast_caution(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=7)
        )
        caution_levels = {round(firm.forecast_caution, 3) for firm in sim.firms if firm.active}
        self.assertGreater(len(caution_levels), 1)

    def test_period_one_wages_respect_subsistence_anchor(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=400, firms_per_sector=4, seed=7)
        )

        sim.step()
        living_wage_anchor = sim._living_wage_anchor()

        self.assertGreater(living_wage_anchor, 0.0)
        active_firms = [firm for firm in sim.firms if firm.active and firm.age > 0]
        self.assertTrue(active_firms)
        self.assertTrue(
            all(
                firm.wage_offer
                >= living_wage_anchor
                * (sim.config.reservation_wage_floor_share + sim._sector_wage_floor_premium(firm.sector))
                - 1e-9
                for firm in active_firms
            )
        )

    def test_update_firm_policies_can_reduce_wage_offer_toward_floor(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=220, firms_per_sector=2, seed=152)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        firm.wage_offer = 20.0
        firm.price = 5.0
        firm.cash = 5.0
        firm.inventory = 50.0
        firm.target_inventory = 10.0
        firm.last_profit = -100.0
        firm.last_revenue = 100.0
        firm.last_worker_count = 20
        firm.last_expected_sales = 8.0
        firm.last_sales = 8.0
        firm.last_production = 8.0
        firm.last_wage_bill = 400.0
        original_wage = firm.wage_offer

        with patch.object(sim, "_living_wage_anchor", return_value=8.0), patch.object(
            sim, "_baseline_demand", return_value=5.0
        ), patch.object(sim, "_firm_effective_productivity", return_value=10.0), patch.object(
            sim, "_firm_learning_maturity", return_value=1.0
        ), patch.object(sim, "_sales_anchor", return_value=5.0), patch.object(
            sim, "_sector_wage_pressure_bonus", return_value=0.0
        ), patch.object(sim, "_target_price_for_firm", return_value=5.0), patch.object(
            sim, "_price_search_candidates", return_value=[5.0]
        ), patch.object(sim, "_inventory_clearance_discount", return_value=0.0), patch.object(
            sim, "_expected_demand_for_price", return_value=5.0
        ), patch.object(sim, "_candidate_market_retention", return_value=(1.0, 0.0)), patch.object(
            sim, "_conservative_expected_sales", return_value=5.0
        ), patch.object(sim, "_candidate_price_objective", return_value=0.0):
            sim._update_firm_policies(last_unemployment=0.20)

        sector_floor = 8.0 * (
            sim.config.reservation_wage_floor_share + sim._sector_wage_floor_premium(firm.sector)
        )
        self.assertLess(firm.wage_offer, original_wage)
        self.assertGreaterEqual(firm.wage_offer, sector_floor - 1e-9)

    def test_startups_begin_with_positive_cash_buffer(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=400, firms_per_sector=4, seed=7)
        )

        for firm in sim.firms:
            if not firm.active:
                continue
            self.assertGreater(firm.cash, 0.0)
            self.assertGreater(
                firm.cash + firm.capital,
                firm.fixed_overhead,
            )

    def test_food_sector_keeps_at_least_one_free_input_firm(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=200, firms_per_sector=5, seed=19)
        )

        food_firms = [firm for firm in sim.firms if firm.sector == "food"]
        exempt_food_firms = [firm for firm in food_firms if firm.input_cost_exempt]

        self.assertTrue(exempt_food_firms)
        self.assertTrue(any(firm.active for firm in exempt_food_firms))
        self.assertTrue(all(firm.input_cost_per_unit == 0.0 for firm in exempt_food_firms))
        self.assertFalse(any(firm.input_cost_exempt for firm in sim.firms if firm.sector == "manufactured"))

    def test_business_cost_recycling_channels_match_total(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=200, firms_per_sector=4, seed=1)
        )

        snapshot = sim.step()

        self.assertGreater(snapshot.business_cost_recycled, 0.0)
        self.assertAlmostEqual(
            snapshot.business_cost_recycled,
            snapshot.business_cost_to_firms
            + snapshot.business_cost_to_households
            + snapshot.business_cost_to_owners,
            places=6,
        )

    def test_commercial_banks_start_diversified_and_reserve_compliant(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=13)
        )

        self.assertEqual(len(sim.banks), 3)
        household_bank_ids = {household.bank_id for household in sim.households if household.alive}
        firm_bank_ids = {firm.bank_id for firm in sim.firms if firm.active}

        self.assertGreater(len(household_bank_ids), 1)
        self.assertGreater(len(firm_bank_ids), 1)
        for bank in sim.banks:
            self.assertGreaterEqual(bank.reserves + 1e-9, bank.reserve_ratio * bank.deposits)

    def test_bank_credit_expands_after_central_bank_issue(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=300,
                firms_per_sector=4,
                seed=59,
                central_bank_enabled=True,
                government_enabled=False,
                central_bank_rule="fisher",
                central_bank_target_velocity=0.12,
                central_bank_max_issue_share=0.05,
            )
        )

        snapshot = sim.step()

        self.assertGreater(snapshot.central_bank_issuance, 0.0)
        self.assertTrue(any(bank.loans_households + bank.loans_firms > 0.0 for bank in sim.banks))
        for bank in sim.banks:
            self.assertGreaterEqual(bank.reserves + 1e-9, bank.reserve_ratio * bank.deposits)

    def test_snapshot_exposes_policy_and_bank_rates(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=67)
        )

        snapshot = sim.step()

        self.assertGreaterEqual(snapshot.central_bank_policy_rate, 0.0)
        self.assertGreaterEqual(snapshot.average_bank_deposit_rate, 0.0)
        self.assertGreater(snapshot.average_bank_loan_rate, snapshot.average_bank_deposit_rate)
        self.assertAlmostEqual(
            snapshot.capitalist_liquid_share + snapshot.worker_liquid_share,
            1.0,
            places=6,
        )

    def test_nonemployed_adult_in_covered_family_can_stay_out_of_labor_force(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=77)
        )
        adults = [
            household
            for household in sim.households
            if sim.config.entry_age_years <= sim._household_age_years(household) < sim.config.senior_age_years
        ][:2]
        self.assertEqual(len(adults), 2)
        employed_adult, covered_spouse = adults
        employed_adult.partner_id = covered_spouse.id
        covered_spouse.partner_id = employed_adult.id
        employed_adult.employed_by = sim.firms[0].id
        covered_spouse.employed_by = None
        employed_adult.last_income = 5000.0
        employed_adult.savings = 5000.0
        covered_spouse.last_income = 0.0
        covered_spouse.savings = 5000.0

        sim._refresh_period_household_caches()
        sim._refresh_family_links()
        sim._refresh_period_family_cache()
        for member in sim._family_groups().get(min(employed_adult.id, covered_spouse.id), []):
            if member.id not in {employed_adult.id, covered_spouse.id}:
                member.alive = False

        sim._refresh_period_household_caches()
        sim._refresh_family_links()
        sim._refresh_period_family_cache()
        participant_ids = sim._labor_force_participant_ids()
        self.assertIn(employed_adult.id, participant_ids)
        self.assertNotIn(covered_spouse.id, participant_ids)

    def test_nonemployed_adult_in_income_stressed_family_counts_in_labor_force(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=79)
        )
        adults = [
            household
            for household in sim.households
            if sim.config.entry_age_years <= sim._household_age_years(household) < sim.config.senior_age_years
        ][:2]
        self.assertEqual(len(adults), 2)
        employed_adult, stressed_spouse = adults
        employed_adult.partner_id = stressed_spouse.id
        stressed_spouse.partner_id = employed_adult.id
        employed_adult.employed_by = sim.firms[0].id
        stressed_spouse.employed_by = None
        employed_adult.last_income = 5.0
        employed_adult.savings = 0.0
        stressed_spouse.last_income = 0.0
        stressed_spouse.savings = 0.0

        sim._refresh_period_household_caches()
        sim._refresh_family_links()
        sim._refresh_period_family_cache()
        for member in sim._family_groups().get(min(employed_adult.id, stressed_spouse.id), []):
            if member.id not in {employed_adult.id, stressed_spouse.id}:
                member.alive = False

        sim._refresh_period_household_caches()
        sim._refresh_family_links()
        sim._refresh_period_family_cache()
        participant_ids = sim._labor_force_participant_ids()
        self.assertIn(employed_adult.id, participant_ids)
        self.assertIn(stressed_spouse.id, participant_ids)

    def test_refresh_family_links_matches_each_female_once_in_greedy_order(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=80,
                firms_per_sector=2,
                seed=81,
                partnership_base_match_probability=1.0,
                partnership_age_gap_hard_penalty=1.0,
                partnership_age_gap_soft_penalty=0.0,
            )
        )
        males = [
            household
            for household in sim.households
            if household.sex == "M"
            and sim.config.entry_age_years <= sim._household_age_years(household) < sim.config.senior_age_years
        ][:2]
        females = [
            household
            for household in sim.households
            if household.sex == "F"
            and sim.config.entry_age_years <= sim._household_age_years(household) < sim.config.senior_age_years
        ][:2]
        self.assertEqual(len(males), 2)
        self.assertEqual(len(females), 2)
        male_1, male_2 = males
        female_1, female_2 = females

        selected_ids = {male_1.id, male_2.id, female_1.id, female_2.id}
        for household in sim.households:
            household.alive = household.id in selected_ids
            household.partner_id = None
            if household.alive:
                household.last_birth_period = -999

        periods_per_year = sim.config.periods_per_year
        male_1.age_periods = 30 * periods_per_year
        male_2.age_periods = 30 * periods_per_year
        female_1.age_periods = 30 * periods_per_year
        female_2.age_periods = 31 * periods_per_year

        for household in (male_1, male_2, female_1, female_2):
            household.savings = 100.0
            household.desired_children = 2
            household.partnership_affinity_code = 7
            household.next_partnership_attempt_period = 0

        sim._refresh_period_household_caches()
        sim._refresh_family_links()

        self.assertEqual(male_1.partner_id, female_1.id)
        self.assertEqual(female_1.partner_id, male_1.id)
        self.assertEqual(male_2.partner_id, female_2.id)
        self.assertEqual(female_2.partner_id, male_2.id)

    def test_match_labor_is_not_limited_by_firm_employment_share_cap(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=1, seed=81, max_firm_employment_share=0.02)
        )
        target_firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        for firm in sim.firms:
            if firm.id != target_firm.id:
                firm.active = False
                firm.workers = []

        eligible_workers = [
            household
            for household in sim.households
            if sim.config.entry_age_years <= sim._household_age_years(household) < sim.config.senior_age_years
        ][:12]
        self.assertEqual(len(eligible_workers), 12)

        for household in eligible_workers:
            household.employed_by = None
            household.reservation_wage = 0.0

        target_firm.desired_workers = 8
        target_firm.wage_offer = max(target_firm.wage_offer, 50.0)
        target_firm.workers = []

        sim._eligible_households = lambda: list(eligible_workers)  # type: ignore[method-assign]
        sim._match_labor()

        self.assertEqual(len(target_firm.workers), 8)

    def test_snapshot_exposes_bank_balance_and_credit_creation_metrics(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=2, households=320, firms_per_sector=4, seed=71)
        )

        snapshot = sim.step()
        frame = history_frame([snapshot])

        self.assertIn("total_bank_assets", frame.columns)
        self.assertIn("total_bank_liabilities", frame.columns)
        self.assertIn("bank_equity", frame.columns)
        self.assertIn("bank_recapitalization", frame.columns)
        self.assertIn("bank_resolution_events", frame.columns)
        self.assertIn("bank_undercapitalized_share", frame.columns)
        self.assertIn("bank_writeoffs", frame.columns)
        self.assertIn("bank_nonperforming_loan_share", frame.columns)
        self.assertIn("gdp_expenditure_sna", frame.columns)
        self.assertIn("gdp_expenditure_gap", frame.columns)
        self.assertIn("commercial_bank_credit_creation", frame.columns)
        self.assertIn("worker_bank_deposits", frame.columns)
        self.assertIn("capitalist_productive_capital", frame.columns)
        self.assertGreaterEqual(snapshot.total_bank_assets, 0.0)
        self.assertGreaterEqual(snapshot.total_bank_liabilities, 0.0)
        self.assertGreaterEqual(snapshot.total_bank_reserves, 0.0)
        self.assertTrue(0.0 <= snapshot.bank_insolvent_share <= 1.0)
        self.assertAlmostEqual(
            snapshot.commercial_bank_credit_creation,
            snapshot.household_credit_creation + snapshot.firm_credit_creation,
            places=6,
        )

    def test_bank_discount_window_borrowing_replaces_free_reserve_fill(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=2, seed=83)
        )
        bank = sim.banks[0]
        bank.deposits = 100.0
        bank.reserves = 0.0
        bank.bond_holdings = 0.0
        bank.central_bank_borrowing = 0.0

        sim._reconcile_bank_reserves()

        self.assertGreater(bank.central_bank_borrowing, 0.0)
        self.assertGreaterEqual(bank.reserves + 1e-9, bank.reserve_ratio * bank.deposits)

    def test_bank_effective_lending_capacity_respects_capital_ratio(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=2, seed=85, bank_min_capital_ratio=0.10)
        )
        bank = sim.banks[0]
        bank.deposits = 1000.0
        bank.reserves = 1200.0
        bank.loans_households = 0.0
        bank.loans_firms = 0.0
        bank.bond_holdings = 0.0
        bank.central_bank_borrowing = 0.0

        self.assertAlmostEqual(sim._bank_lending_capacity(bank), 5000.0, places=6)
        self.assertAlmostEqual(sim._bank_capital_lending_capacity(bank), 2000.0, places=6)
        self.assertAlmostEqual(sim._bank_effective_lending_capacity(bank), 2000.0, places=6)

    def test_bank_prudential_lending_multiplier_turns_restrictive_before_insolvency(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=2, seed=86, bank_min_capital_ratio=0.08)
        )
        bank = sim.banks[0]
        bank.deposits = 1000.0
        bank.central_bank_borrowing = 0.0
        bank.bond_holdings = 0.0
        bank.loans_households = 1000.0
        bank.loans_firms = 0.0

        bank.reserves = 300.0
        healthy_multiplier = sim._bank_prudential_lending_multiplier(bank)

        bank.reserves = 110.0
        weak_multiplier = sim._bank_prudential_lending_multiplier(bank)

        self.assertGreater(healthy_multiplier, weak_multiplier)
        self.assertGreaterEqual(weak_multiplier, 0.0)
        self.assertLess(weak_multiplier, 0.5)

    def test_undercapitalized_bank_restores_private_capital_before_insolvency(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=87, bank_min_capital_ratio=0.08)
        )
        bank = sim.banks[0]

        for household in sim.households:
            household.savings = 0.0
            household.wage_income = 0.0
            household.loan_balance = 0.0
            household.bank_id = bank.id
        for owner in sim.entrepreneurs:
            owner.wealth = 0.0
            owner.vault_cash = 0.0
            owner.bank_id = bank.id
        for firm in sim.firms:
            firm.cash = 0.0
            firm.loan_balance = 0.0
            firm.bank_id = bank.id

        sim.households[0].savings = 1000.0
        sim.households[1].loan_balance = 1000.0
        sim.entrepreneurs[0].vault_cash = 500.0
        bank.reserves = 100.0
        bank.bond_holdings = 0.0
        bank.central_bank_borrowing = 0.0

        sim._refresh_bank_balance_sheets()
        initial_ratio = sim._bank_capital_ratio(bank)
        sim._stabilize_bank_capital_positions()
        restored_ratio = sim._bank_capital_ratio(bank)

        self.assertLess(initial_ratio, sim._bank_warning_capital_ratio())
        self.assertGreater(sim._period_bank_recapitalization, 0.0)
        self.assertGreaterEqual(restored_ratio, sim._bank_warning_capital_ratio() - 1e-9)
        self.assertGreater(sim._period_bank_undercapitalized_share_signal, 0.0)

    def test_negative_bank_equity_triggers_recapitalization(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=88,
                central_bank_enabled=False,
                bank_min_capital_ratio=0.10,
            )
        )
        bank = sim.banks[0]
        household = sim.households[0]
        owner = sim.entrepreneurs[0]
        household.bank_id = bank.id
        household.savings = 100.0
        household.loan_balance = 0.0
        owner.bank_id = bank.id
        owner.wealth = 0.0
        owner.vault_cash = 1500.0

        sim._refresh_bank_balance_sheets()
        bank.reserves = 0.0
        bank.bond_holdings = 0.0
        bank.loans_households = 0.0
        bank.loans_firms = 0.0
        bank.central_bank_borrowing = 150.0

        sim._resolve_bank_insolvency()

        self.assertGreater(sim._period_bank_recapitalization, 0.0)
        self.assertGreaterEqual(sim._period_bank_resolution_events, 1)
        self.assertGreaterEqual(sim._bank_equity(bank), 0.0)
        self.assertGreater(sim._period_bank_insolvent_share_signal, 0.0)

    def test_uncreditworthy_household_request_is_denied(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=87)
        )
        borrower = sim.households[0]
        borrower.savings = 0.0
        borrower.loan_balance = 500.0
        borrower.last_income = 0.0
        borrower.employed_by = None
        bank = sim.banks[sim._bank_id_for_household(borrower)]
        bank.loan_rate = 0.05

        self.assertFalse(sim._household_creditworthy(borrower, 100.0, bank))

    def test_household_default_writes_off_loan_and_blocks_new_credit(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=89,
                household_loan_restructure_delinquency=3,
                household_loan_default_delinquency=2,
            )
        )
        borrower = next(
            household
            for household in sim.households
            if household.alive and sim._household_age_years(household) >= sim.config.entry_age_years
        )
        borrower.partner_id = None
        borrower.guardian_id = None
        borrower.mother_id = None
        borrower.father_id = None
        borrower.savings = 0.0
        borrower.wage_income = 0.0
        borrower.loan_balance = 120.0
        borrower.loan_restructure_count = 1
        borrower.loan_delinquency_periods = 1
        borrower.last_income = 0.0
        borrower.employed_by = None

        sim._refresh_period_household_caches()
        sim._refresh_family_links()
        sim._refresh_period_family_cache()
        sim._service_household_loans()

        self.assertEqual(borrower.loan_balance, 0.0)
        self.assertEqual(sim._period_household_loan_defaults, 1)
        self.assertGreater(sim._period_bank_writeoffs, 0.0)
        self.assertGreater(borrower.credit_exclusion_periods, 0)
        bank = sim.banks[sim._bank_id_for_household(borrower)]
        self.assertFalse(sim._household_creditworthy(borrower, 10.0, bank))

    def test_nonessential_speculative_firm_credit_is_denied_after_repeated_missed_sales(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=88)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "manufactured")
        bank = sim.banks[sim._bank_id_for_firm(firm)]
        bank.loan_rate = 0.04

        firm.last_sales = 20.0
        firm.last_expected_sales = 200.0
        firm.price = 10.0
        firm.last_revenue = 200.0
        firm.last_wage_bill = 180.0
        firm.last_input_cost = 120.0
        firm.last_transport_cost = 20.0
        firm.last_fixed_overhead = 30.0
        firm.cash = 500.0
        firm.loss_streak = 3
        firm.loan_balance = 100.0

        self.assertFalse(sim._firm_creditworthy(firm, 250.0, bank))

    def test_firm_default_is_resolved_before_owner_extracts_residual_cash(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=160, firms_per_sector=2, seed=90, replacement_enabled=False)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "manufactured")
        owner = sim.entrepreneurs[firm.owner_id]
        owner_wealth_before = owner.wealth
        firm.cash = 50.0
        firm.loan_balance = 200.0
        firm.loan_default_flag = True
        firm.age = sim.config.bankruptcy_grace_period

        sim._resolve_bankruptcy_and_entry()

        self.assertFalse(firm.active)
        self.assertEqual(firm.loan_balance, 0.0)
        self.assertEqual(sim._period_firm_loan_defaults, 1)
        self.assertGreaterEqual(sim._period_bank_writeoffs, 150.0)
        self.assertAlmostEqual(owner.wealth, owner_wealth_before, places=6)

    def test_history_frame_derives_augmented_class_shares_and_fiscal_burdens(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=3, households=320, firms_per_sector=4, seed=73)
        )

        for _ in range(3):
            sim.step()

        frame = history_frame(sim.history)

        self.assertIn("worker_augmented_asset_share", frame.columns)
        self.assertIn("capitalist_augmented_asset_share", frame.columns)
        self.assertIn("government_tax_burden_gdp", frame.columns)
        self.assertIn("worker_consumption_share_gdp", frame.columns)
        self.assertIn("commercial_bank_credit_creation_share_money", frame.columns)
        shares = (
            frame[["worker_augmented_asset_share", "capitalist_augmented_asset_share"]]
            .dropna()
            .sum(axis=1)
        )
        self.assertTrue(((shares - 1.0).abs() <= 1e-6).all())
        self.assertTrue(frame["worker_consumption_share_gdp"].dropna().ge(0.0).all())
        creation_matches = (
            frame["commercial_bank_credit_creation"]
            - frame["household_credit_creation"]
            - frame["firm_credit_creation"]
        ).abs()
        self.assertTrue((creation_matches <= 1e-9).all())

    def test_government_is_inert_when_disabled(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=2,
                households=300,
                firms_per_sector=4,
                seed=67,
                government_enabled=False,
            )
        )

        for _ in range(2):
            snapshot = sim.step()

        self.assertEqual(snapshot.government_tax_revenue, 0.0)
        self.assertEqual(snapshot.government_transfers, 0.0)
        self.assertEqual(snapshot.government_procurement_spending, 0.0)
        self.assertEqual(snapshot.government_bond_issuance, 0.0)
        self.assertEqual(snapshot.government_debt_outstanding, 0.0)

    def test_government_collects_or_spends_when_enabled(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=3,
                households=300,
                firms_per_sector=4,
                seed=67,
                government_enabled=True,
            )
        )

        for _ in range(3):
            snapshot = sim.step()

        self.assertGreaterEqual(snapshot.government_debt_outstanding, 0.0)
        self.assertTrue(
            snapshot.government_tax_revenue > 0.0
            or snapshot.government_transfers > 0.0
            or snapshot.government_procurement_spending > 0.0
        )

    def test_closed_economy_conserves_liquid_money_exactly(self) -> None:
        cfg = SimulationConfig(
            periods=6,
            households=600,
            firms_per_sector=6,
            seed=7,
            annual_birth_rate=0.0,
            annual_base_death_rate=0.0,
            annual_senior_death_rate=0.0,
            max_age_years=200.0,
            replacement_enabled=False,
            central_bank_enabled=False,
        )
        sim = EconomySimulation(cfg)
        initial_liquid_money = _initial_liquid_money(sim)

        for _ in range(cfg.periods):
            sim.step()

        final_liquid_money = sim.history[-1].total_liquid_money
        cumulative_bond_issuance = sum(snapshot.government_bond_issuance for snapshot in sim.history)
        cumulative_credit_creation = sum(snapshot.commercial_bank_credit_creation for snapshot in sim.history)
        self.assertAlmostEqual(
            final_liquid_money,
            initial_liquid_money + cumulative_bond_issuance + cumulative_credit_creation,
            delta=1.0,
        )

    def test_demography_and_reentry_still_preserve_liquid_money(self) -> None:
        cfg = SimulationConfig(
            periods=12,
            households=400,
            firms_per_sector=4,
            seed=7,
            annual_birth_rate=0.35,
            annual_base_death_rate=0.03,
            annual_senior_death_rate=0.15,
            replacement_enabled=True,
            central_bank_enabled=False,
        )
        sim = EconomySimulation(cfg)
        initial_liquid_money = _initial_liquid_money(sim)

        for _ in range(cfg.periods):
            sim.step()

        final_liquid_money = sim.history[-1].total_liquid_money
        cumulative_bond_issuance = sum(snapshot.government_bond_issuance for snapshot in sim.history)
        cumulative_credit_creation = sum(snapshot.commercial_bank_credit_creation for snapshot in sim.history)
        self.assertAlmostEqual(
            final_liquid_money,
            initial_liquid_money + cumulative_bond_issuance + cumulative_credit_creation,
            delta=1.0,
        )

    def test_central_bank_issues_money_when_fisher_target_exceeds_supply(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=300,
                firms_per_sector=4,
                seed=59,
                central_bank_enabled=True,
                government_enabled=False,
                central_bank_rule="fisher",
                central_bank_target_velocity=0.12,
                central_bank_max_issue_share=0.05,
            )
        )
        initial_liquid_money = _initial_liquid_money(sim)

        snapshot = sim.step()

        self.assertGreater(snapshot.central_bank_issuance, 0.0)
        self.assertLessEqual(
            snapshot.total_liquid_money,
            initial_liquid_money + snapshot.commercial_bank_credit_creation + 1e-6,
        )
        self.assertAlmostEqual(snapshot.central_bank_money_supply, snapshot.total_liquid_money, places=6)

    def test_fisher_policy_uses_bank_channel_before_broad_money_changes(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=300,
                firms_per_sector=4,
                seed=59,
                central_bank_enabled=True,
                government_enabled=False,
                central_bank_rule="fisher",
                central_bank_target_velocity=0.12,
                central_bank_max_issue_share=0.05,
            )
        )
        initial_liquid_money = sim._current_total_liquid_money()
        initial_average_reserve_ratio = sum(bank.reserve_ratio for bank in sim.banks) / max(1, len(sim.banks))

        sim._apply_central_bank_policy()

        self.assertNotEqual(sim.central_bank.issuance_this_period, 0.0)
        self.assertAlmostEqual(sim.central_bank.money_supply, initial_liquid_money, places=6)
        average_reserve_ratio = sum(bank.reserve_ratio for bank in sim.banks) / max(1, len(sim.banks))
        self.assertNotEqual(average_reserve_ratio, initial_average_reserve_ratio)

    def test_productivity_dividend_issues_to_workers_of_more_productive_firm(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=2,
                households=300,
                firms_per_sector=4,
                seed=61,
                central_bank_enabled=True,
                central_bank_rule="productivity_dividend",
            )
        )
        sim.step()
        sim._reset_period_counters()

        target_firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food" and firm.workers)
        for firm in sim.firms:
            if not firm.active or not firm.workers:
                continue
            current_output_per_worker = sim._current_output_per_worker_estimate(firm)
            firm.last_worker_count = len(firm.workers)
            firm.last_production = current_output_per_worker * len(firm.workers)
        target_firm.technology *= 1.20

        worker_savings_before = {
            worker_id: sim.households[worker_id].savings
            for worker_id in target_firm.workers
        }
        outsider = next(
            household
            for household in sim.households
            if household.alive and household.id not in target_firm.workers
        )
        outsider_savings_before = outsider.savings
        initial_liquid_money = sim._current_total_liquid_money()

        sim._apply_central_bank_policy()

        self.assertGreater(sim.central_bank.issuance_this_period, 0.0)
        self.assertAlmostEqual(
            sim.central_bank.money_supply,
            initial_liquid_money + sim.central_bank.issuance_this_period,
            places=6,
        )
        worker_deltas = [
            sim.households[worker_id].savings - worker_savings_before[worker_id]
            for worker_id in target_firm.workers
        ]
        self.assertTrue(all(delta > 0.0 for delta in worker_deltas))
        self.assertAlmostEqual(max(worker_deltas), min(worker_deltas), places=6)
        self.assertAlmostEqual(outsider.savings, outsider_savings_before, places=6)

    def test_risky_price_hikes_are_discounted_under_forecast_uncertainty(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=7)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        sector_firms = sim._sector_firms(firm.sector)

        firm.forecast_caution = 1.6
        firm.forecast_error_belief = 0.45
        firm.market_fragility_belief = 0.70
        firm.last_expected_sales = 120.0
        firm.last_sales = 80.0
        firm.last_production = 110.0
        firm.target_inventory = 100.0
        firm.inventory = 135.0
        firm.sales_history = [130.0, 126.0, 120.0, 96.0, 88.0, 80.0]

        market_share = 1.0 / len(sector_firms)
        sector_total_demand = sim._baseline_demand(firm.sector)
        reference_price = max(0.1, firm.price)
        candidate_price = reference_price * 1.08

        expected_sales = sim._expected_demand_for_price(
            firm,
            sector_total_demand,
            market_share,
            candidate_price,
            reference_price,
        )
        _, market_hazard = sim._candidate_market_retention(
            firm,
            candidate_price,
            reference_price,
        )
        prudent_sales = sim._conservative_expected_sales(
            firm,
            expected_sales,
            candidate_price,
            reference_price,
            market_hazard,
        )

        self.assertLess(prudent_sales, expected_sales)
        self.assertLessEqual(sim._firm_max_price_hike_ratio(firm), 1.02)

    def test_baseline_demand_stays_closer_to_structure_during_learning_warmup(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=17)
        )
        sector_key = "manufactured"
        structural_demand = sim._structural_sector_demand(sector_key)
        observed_demand = structural_demand * 0.20
        sim._last_sector_budget_demand_units[sector_key] = observed_demand
        sim._last_sector_potential_demand_units[sector_key] = observed_demand

        sim.period = 2
        early_baseline = sim._baseline_demand(sector_key)

        sim.period = sim.config.firm_learning_warmup_periods + 24
        late_baseline = sim._baseline_demand(sector_key)

        self.assertLess(abs(early_baseline - structural_demand), abs(late_baseline - structural_demand))
        self.assertGreater(early_baseline, late_baseline)

    def test_firm_demand_learning_moves_slower_during_learning_warmup(self) -> None:
        early_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=23)
        )
        late_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=23)
        )
        early_firm = next(firm for firm in early_sim.firms if firm.active and firm.sector == "food")
        late_firm = next(firm for firm in late_sim.firms if firm.active and firm.sector == "food")

        for sim, firm, period in (
            (early_sim, early_firm, 2),
            (late_sim, late_firm, late_sim.config.firm_learning_warmup_periods + 24),
        ):
            sim.period = period
            firm.demand_elasticity = 0.82
            firm.last_expected_sales = 120.0
            firm.last_sales = 72.0
            firm.last_production = 118.0
            firm.target_inventory = 100.0
            firm.inventory = 138.0
            firm.price = 12.0
            firm.last_unit_cost = 9.0
            firm.sales_history = [132.0, 126.0, 119.0, 100.0, 84.0, 72.0]
            firm.market_fragility_belief = 0.60
            firm.forecast_error_belief = 0.40

        early_sim._update_firm_demand_learning(early_firm)
        late_sim._update_firm_demand_learning(late_firm)

        early_elasticity_move = abs(early_firm.demand_elasticity - 0.82)
        late_elasticity_move = abs(late_firm.demand_elasticity - 0.82)
        early_error_move = abs(early_firm.forecast_error_belief - 0.40)
        late_error_move = abs(late_firm.forecast_error_belief - 0.40)

        self.assertGreater(late_elasticity_move, early_elasticity_move)
        self.assertGreater(late_error_move, early_error_move)

    def test_essential_target_price_falls_when_costs_drop_and_affordability_is_stressed(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=2, households=300, firms_per_sector=4, seed=41)
        )
        sim.step()

        sim.history[-1].family_income_to_basket_ratio = 0.62
        sim.history[-1].essential_fulfillment_rate = 0.74

        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        spec = SECTOR_BY_KEY[firm.sector]
        firm.last_unit_cost = 10.0
        firm.price = 12.0
        firm.markup_tolerance = 1.0
        firm.volume_preference = 1.25

        average_unit_cost = 7.0
        variable_unit_cost = 6.4
        baseline_target_price = average_unit_cost * (1.0 + spec.markup)

        adjusted_target_price = sim._target_price_for_firm(
            firm,
            spec,
            average_unit_cost,
            variable_unit_cost,
        )

        self.assertLess(adjusted_target_price, baseline_target_price)
        self.assertLess(adjusted_target_price, firm.price)
        self.assertGreaterEqual(adjusted_target_price, variable_unit_cost * 1.01)

    def test_startup_essential_capacity_covers_structural_need(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=43)
        )

        for sector_key in ("food", "housing", "clothing"):
            structural_need = sim._structural_sector_demand(sector_key)
            survival_floor = len([household for household in sim.households if household.alive]) * sim._essential_basket_share(
                sector_key
            )
            self.assertAlmostEqual(
                sim._startup_essential_target_units(sector_key),
                max(structural_need, survival_floor) * sim._startup_essential_supply_multiplier(sector_key),
                places=6,
            )
            startup_capacity = sum(
                sim._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                for firm in sim.firms
                if firm.active and firm.sector == sector_key
            )
            self.assertGreaterEqual(startup_capacity, structural_need)

    def test_clothing_startup_target_gets_extra_supply_buffer(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=44)
        )

        structural_need = sim._structural_sector_demand("clothing")
        survival_floor = len([household for household in sim.households if household.alive]) * sim._essential_basket_share(
            "clothing"
        )
        base_target = max(structural_need, survival_floor) * max(1.10, sim.config.startup_essential_supply_buffer)

        self.assertGreater(sim._startup_essential_target_units("clothing"), base_target)

    def test_startup_labor_relief_keeps_sector_expected_sales_scaled_to_sector_need(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=43)
        )

        for sector_key in ESSENTIAL_SECTOR_KEYS:
            sector_expected_sales = sum(
                firm.last_expected_sales
                for firm in sim.firms
                if firm.active and firm.sector == sector_key
            )
            startup_target = sim._startup_essential_target_units(sector_key)
            self.assertGreater(sector_expected_sales, 0.0)
            self.assertLessEqual(sector_expected_sales, startup_target)

    def test_startup_essential_prices_refresh_to_current_unit_cost(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=45)
        )

        for sector_key in ESSENTIAL_SECTOR_KEYS:
            firms = [firm for firm in sim.firms if firm.active and firm.sector == sector_key]
            self.assertTrue(firms)
            for firm in firms:
                spec = SECTOR_BY_KEY[sector_key]
                self.assertAlmostEqual(firm.price, firm.last_unit_cost * (1.0 + spec.markup), places=6)

    def test_period_one_produces_enough_essential_baskets_for_population(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=1200, firms_per_sector=8, seed=47)
        )
        initial_population = sum(1 for household in sim.households if household.alive)

        snapshot = sim.step()
        sector_production = {
            sector_key: sum(
                firm.last_production for firm in sim.firms if firm.active and firm.sector == sector_key
            )
            for sector_key in ("food", "housing", "clothing")
        }
        total_essential_need = sum(SECTOR_BY_KEY[sector_key].essential_need for sector_key in sector_production)
        basket_equivalents = min(
            sector_production[sector_key]
            / (SECTOR_BY_KEY[sector_key].essential_need / total_essential_need)
            for sector_key in sector_production
        )

        self.assertGreaterEqual(basket_equivalents, initial_population)

    def test_birth_can_happen_without_partner(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=80,
                firms_per_sector=2,
                seed=11,
                annual_birth_rate=1.0,
                annual_base_death_rate=0.0,
                annual_senior_death_rate=0.0,
                period_base_death_probability=0.0,
                period_senior_death_probability=0.0,
                period_food_subsistence_death_risk=0.0,
                period_severe_hunger_death_risk=0.0,
                period_health_fragility_death_risk=0.0,
            )
        )
        mother = next(
            household
            for household in sim.households
            if household.sex == "F"
            and sim._household_age_years(household) >= sim.config.entry_age_years
            and sim._household_age_years(household) <= sim.config.fertile_age_max_years
        )
        mother.id = 0
        sim.households = [mother]
        sim._next_household_id = 1
        mother.partner_id = None
        mother.children_count = 0
        mother.desired_children = 1
        mother.last_birth_period = -999
        mother.savings = 500.0
        mother.wage_income = 0.0

        with patch.object(sim.rng, "random", return_value=0.0):
            sim._apply_demography(unemployment_rate=0.0)

        self.assertEqual(len(sim.households), 2)
        child = sim.households[-1]
        self.assertEqual(child.mother_id, mother.id)
        self.assertIsNone(child.father_id)

    def test_stable_family_does_not_auto_increase_desired_children_over_time(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=80,
                firms_per_sector=2,
                seed=13,
            )
        )
        mother = next(
            household
            for household in sim.households
            if household.sex == "F"
            and sim._household_age_years(household) >= sim.config.entry_age_years
            and sim._household_age_years(household) <= sim.config.fertile_age_max_years
        )
        mother.id = 0
        sim.households = [mother]
        sim._next_household_id = 1
        mother.partner_id = None
        mother.children_count = 0
        mother.desired_children = 1
        mother.child_desire_pressure = 0.95
        mother.last_birth_period = sim.period - 12

        sim._update_family_child_desires(
            [mother],
            essential_target_units=10.0,
            essential_units_bought=10.0,
            family_remaining_cash=500.0,
        )

        self.assertEqual(mother.desired_children, 1)
        self.assertEqual(mother.child_desire_pressure, 0.95)

    def test_fertile_capable_women_metric_excludes_mothers_inside_birth_interval(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=23,
            )
        )

        mothers = [
            household
            for household in sim.households
            if household.sex == "F"
            and sim._household_age_years(household) >= sim.config.entry_age_years
            and sim._household_age_years(household) <= sim.config.fertile_age_max_years
        ][:2]
        self.assertEqual(len(mothers), 2)
        ready_mother, blocked_mother = mothers

        ready_mother.id = 0
        blocked_mother.id = 1
        sim.households = [ready_mother, blocked_mother]
        sim._next_household_id = 2

        for mother in sim.households:
            mother.partner_id = None
            mother.desired_children = 3
            mother.children_count = 0
            mother.last_available_cash = 500.0
            mother.savings = 500.0
            mother.wage_income = 0.0

        ready_mother.last_birth_period = sim.period - sim.config.birth_interval_periods
        blocked_mother.last_birth_period = sim.period - max(0, sim.config.birth_interval_periods - 1)

        fertile_capable_women, _, _ = sim._fertile_women_reproductive_metrics()
        self.assertEqual(fertile_capable_women, 1)

    def test_history_frame_exposes_reproductive_time_series(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=4,
                households=120,
                firms_per_sector=2,
                seed=17,
            )
        )

        for _ in range(4):
            sim.step()

        frame = history_frame(sim.history, periods_per_year=sim.config.periods_per_year)
        self.assertIn("fertile_families", frame.columns)
        self.assertIn("fertile_families_with_births", frame.columns)
        self.assertIn("fertile_capable_families", frame.columns)
        self.assertIn("fertile_capable_families_low_desire_no_birth", frame.columns)
        self.assertIn("fertile_capable_families_with_births", frame.columns)
        self.assertIn("fertile_capable_family_birth_rate", frame.columns)
        self.assertIn("fertile_capable_family_low_desire_share", frame.columns)
        self.assertIn("fertile_capable_women", frame.columns)
        self.assertIn("fertile_capable_women_low_desire_no_birth", frame.columns)
        self.assertIn("fertile_capable_women_with_births", frame.columns)
        self.assertIn("fertile_capable_women_birth_rate", frame.columns)
        self.assertIn("fertile_capable_women_low_desire_share", frame.columns)

    def test_capable_and_noncapable_birth_rates_are_applied_differently(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=29,
                annual_birth_rate_capable_single=1.0,
                annual_birth_rate_capable_partnered=1.0,
                annual_birth_rate_noncapable=0.0,
                annual_base_death_rate=0.0,
                annual_senior_death_rate=0.0,
                period_base_death_probability=0.0,
                period_senior_death_probability=0.0,
                period_food_subsistence_death_risk=0.0,
                period_severe_hunger_death_risk=0.0,
                period_health_fragility_death_risk=0.0,
            )
        )

        mothers = [
            household
            for household in sim.households
            if household.sex == "F"
            and sim._household_age_years(household) >= sim.config.entry_age_years
            and sim._household_age_years(household) <= sim.config.fertile_age_max_years
        ][:2]
        self.assertEqual(len(mothers), 2)
        capable_mother, noncapable_mother = mothers

        capable_mother.id = 0
        noncapable_mother.id = 1
        sim.households = [capable_mother, noncapable_mother]
        sim._next_household_id = 2

        for mother in sim.households:
            mother.partner_id = None
            mother.children_count = 0
            mother.desired_children = 1
            mother.last_birth_period = sim.period - sim.config.birth_interval_periods
            mother.wage_income = 0.0
            mother.savings = 0.0
            mother.last_available_cash = 0.0

        capable_mother.last_available_cash = 500.0
        capable_mother.savings = 500.0
        noncapable_mother.last_available_cash = 0.0
        noncapable_mother.savings = 0.0

        with patch.object(sim.rng, "random", return_value=0.0):
            sim._apply_demography(unemployment_rate=0.0)

        self.assertEqual(len(sim.households), 3)
        child = sim.households[-1]
        self.assertEqual(child.mother_id, capable_mother.id)

    def test_structural_demand_map_matches_sector_sums(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=17)
        )

        active_households = [household for household in sim.households if household.alive]
        structural_map = sim._compute_structural_demand_map(active_households)

        for spec in SECTOR_SPECS:
            expected = sum(
                sim._household_sector_desired_units(household, spec.key)
                for household in active_households
            )
            self.assertAlmostEqual(structural_map[spec.key], expected, places=9)

    def test_guardian_selection_matches_sorting_rule(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=80, firms_per_sector=2, seed=9)
        )

        child = next(
            household
            for household in sim.households
            if sim._household_age_years(household) < sim.config.entry_age_years
        )
        candidates = [
            household
            for household in sim._active_households()
            if sim._household_age_years(household) >= sim.config.entry_age_years
            and household.id != child.id
        ]
        expected = min(
            candidates,
            key=lambda household: (
                household.dependent_children,
                -sim._household_cash_balance(household),
                sim._household_age_years(household),
                household.id,
            ),
        )

        self.assertEqual(sim._select_guardian_for_child(child).id, expected.id)


if __name__ == "__main__":
    unittest.main()
