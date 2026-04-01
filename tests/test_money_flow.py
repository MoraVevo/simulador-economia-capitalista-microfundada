from __future__ import annotations

import unittest
from unittest.mock import patch

from economy_simulator.domain import ESSENTIAL_SECTOR_KEYS, SECTOR_BY_KEY, SECTOR_SPECS, SimulationConfig
from economy_simulator.engine import EconomySimulation
from economy_simulator.reporting import history_frame


def _initial_liquid_money(sim: EconomySimulation) -> float:
    return (
        sum(household.savings for household in sim.households if household.alive)
        + sum(owner.wealth + owner.vault_cash for owner in sim.entrepreneurs)
        + sum(firm.cash for firm in sim.firms if firm.active)
        + sim.government.treasury_cash
    )


class MoneyFlowTests(unittest.TestCase):
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
            family_cash=120.0,
            family_basic_basket_cost=100.0,
            family_price_sensitivity=1.0,
            family_saving_propensity=0.25,
            family_consumption_multiplier=1.0,
            inflation_pressure=0.0,
        )
        high_buffer_rate = sim._family_savings_rate(
            members,
            family_cash=220.0,
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
            family_cash=220.0,
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
            family_cash=220.0,
            family_basic_basket_cost=100.0,
            family_price_sensitivity=1.0,
            family_saving_propensity=0.25,
            family_consumption_multiplier=1.0,
            inflation_pressure=0.0,
        )

        self.assertGreater(high_buffer_rate, low_buffer_rate)
        self.assertGreater(low_trust_rate, high_buffer_rate)
        self.assertLess(high_impatience_rate, high_buffer_rate)

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
        self.assertGreaterEqual(snapshot.average_food_meals_per_person, 0.0)
        self.assertTrue(0.0 <= snapshot.food_sufficient_share <= 1.0)
        self.assertTrue(0.0 <= snapshot.food_severe_hunger_share <= 1.0)

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
        living_wage_anchor = sim._living_wage_anchor()

        sim.step()

        self.assertGreater(living_wage_anchor, 0.0)
        active_firms = [firm for firm in sim.firms if firm.active]
        self.assertTrue(active_firms)
        wage_floor = living_wage_anchor * sim.config.reservation_wage_floor_share
        self.assertTrue(all(firm.wage_offer >= wage_floor - 1e-9 for firm in active_firms))

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
        self.assertIn("commercial_bank_credit_creation_share_money", frame.columns)
        shares = (
            frame[["worker_augmented_asset_share", "capitalist_augmented_asset_share"]]
            .dropna()
            .sum(axis=1)
        )
        self.assertTrue(((shares - 1.0).abs() <= 1e-6).all())
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
        self.assertAlmostEqual(final_liquid_money, initial_liquid_money, places=6)

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
        self.assertAlmostEqual(final_liquid_money, initial_liquid_money, places=6)

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
        self.assertAlmostEqual(
            snapshot.total_liquid_money,
            initial_liquid_money + snapshot.commercial_bank_credit_creation,
            places=6,
        )
        self.assertAlmostEqual(snapshot.central_bank_money_supply, snapshot.total_liquid_money, places=6)

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
                max(structural_need, survival_floor) * 1.10,
                places=6,
            )
            startup_capacity = sum(
                sim._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                for firm in sim.firms
                if firm.active and firm.sector == sector_key
            )
            self.assertGreaterEqual(startup_capacity, structural_need)

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
            self.assertAlmostEqual(
                sector_expected_sales,
                sim._startup_essential_target_units(sector_key),
                places=6,
            )

    def test_period_one_produces_enough_essential_baskets_for_population(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=1200, firms_per_sector=8, seed=47)
        )

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

        self.assertGreaterEqual(basket_equivalents, snapshot.population)

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

    def test_stable_family_increases_desired_children_over_time(self) -> None:
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

        self.assertEqual(mother.desired_children, 2)
        self.assertLess(mother.child_desire_pressure, 1.0)

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
