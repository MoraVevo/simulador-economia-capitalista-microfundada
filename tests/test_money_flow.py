from __future__ import annotations

from contextlib import ExitStack
import dataclasses
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from economy_simulator.batch_reports import (
    _annualize_core_history,
    _rename_firm_audit_export,
    _rename_family_audit_export,
    _sample_audit_entities,
    _scenario_sample_seed,
    _summary_tables,
)
from economy_simulator.domain import ESSENTIAL_SECTOR_KEYS, Entrepreneur, SECTOR_BY_KEY, SECTOR_SPECS, SimulationConfig
from economy_simulator.engine import EconomySimulation, PUBLIC_ADMINISTRATION_EMPLOYER_ID
from economy_simulator.policies import scenario_policy_presets, social_state_intensive_profile
from economy_simulator.reporting import annual_frame, core_history_frame, family_audit_frame, firm_audit_frame, firm_history_frame, history_frame


def _initial_liquid_money(sim: EconomySimulation) -> float:
    return (
        sum(household.savings for household in sim.households if household.alive)
        + sum(owner.wealth + owner.vault_cash for owner in sim.entrepreneurs)
        + sum(firm.cash for firm in sim.firms if firm.active)
        + sim.government.treasury_cash
    )


class MoneyFlowTests(unittest.TestCase):
    def test_break_even_price_for_sales_does_not_add_markup(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=40, firms_per_sector=1, seed=117)
        )

        self.assertAlmostEqual(sim._break_even_price_for_sales(4.0, 20.0, 10.0), 6.0)

    def test_marginal_price_cut_rejects_cut_when_marginal_revenue_is_below_cost(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=40, firms_per_sector=1, seed=118)
        )

        self.assertFalse(
            sim._marginal_price_cut_is_viable(
                effective_price=8.0,
                prudent_sales=120.0,
                reference_price=10.0,
                reference_prudent_sales=100.0,
                marginal_unit_cost=1.0,
            )
        )
        self.assertTrue(
            sim._marginal_price_cut_is_viable(
                effective_price=9.9,
                prudent_sales=110.0,
                reference_price=10.0,
                reference_prudent_sales=100.0,
                marginal_unit_cost=1.0,
            )
        )

    def test_liquidation_sale_requires_avoided_inventory_loss_to_cover_discount_loss(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=40, firms_per_sector=1, seed=119)
        )
        firm = next(firm for firm in sim.firms if not sim._is_education_sector(firm.sector))
        firm.target_inventory = 1.0
        firm.last_unit_cost = 10.0
        firm.inventory_batches = [10.0] * (sim._firm_inventory_shelf_life_periods() + 2)
        firm.inventory = sum(firm.inventory_batches)

        self.assertTrue(
            sim._liquidation_sale_is_prudent(
                firm,
                effective_price=9.9,
                break_even_price=10.0,
                prudent_sales=5.0,
            )
        )
        self.assertFalse(
            sim._liquidation_sale_is_prudent(
                firm,
                effective_price=1.0,
                break_even_price=100.0,
                prudent_sales=5.0,
            )
        )

    def test_loss_response_thresholds_are_heterogeneous_by_firm_behavior(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=40, firms_per_sector=1, seed=120)
        )
        patient_firm = sim.firms[0]
        reactive_firm = sim.firms[1]
        for firm in (patient_firm, reactive_firm):
            firm.cash = 100.0
            firm.capital = 100.0
            firm.inventory = 10.0

        patient_firm.volume_preference = 1.6
        patient_firm.market_share_ambition = 1.6
        patient_firm.price_aggressiveness = 1.4
        patient_firm.cash_conservatism = 0.7
        patient_firm.forecast_caution = 0.7
        patient_firm.inventory_aversion = 0.8

        reactive_firm.volume_preference = 0.7
        reactive_firm.market_share_ambition = 0.7
        reactive_firm.price_aggressiveness = 0.8
        reactive_firm.cash_conservatism = 1.6
        reactive_firm.forecast_caution = 1.5
        reactive_firm.inventory_aversion = 1.4

        self.assertGreater(
            sim._firm_unit_margin_loss_response_threshold(patient_firm),
            sim._firm_unit_margin_loss_response_threshold(reactive_firm),
        )
        self.assertGreater(
            sim._firm_adaptation_threshold(patient_firm),
            sim._firm_adaptation_threshold(reactive_firm),
        )

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

    def test_default_central_bank_uses_inflation_targeting_not_goods_growth(self) -> None:
        config = SimulationConfig()

        self.assertEqual(config.central_bank_rule, "inflation_targeting")

    def test_inflation_targeting_liquidity_target_expands_when_below_target(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=80,
                firms_per_sector=1,
                seed=131,
                central_bank_rule="inflation_targeting",
                central_bank_target_annual_inflation=0.06,
            )
        )
        current_money = sim._current_total_liquid_money()

        self.assertGreater(sim._target_money_supply_inflation_targeting(), current_money)

    def test_batch_report_summary_uses_last_three_year_averages(self) -> None:
        annual = pd.DataFrame(
            {
                "year": [1, 2, 3, 4],
                "gdp_nominal": [100.0, 200.0, 300.0, 600.0],
                "real_gdp_nominal": [80.0, 160.0, 240.0, 480.0],
                "potential_gdp_nominal": [120.0, 220.0, 320.0, 620.0],
                "inflation_yoy": [0.01, 0.02, 0.03, 0.06],
                "avg_unemployment_rate": [0.10, 0.09, 0.08, 0.05],
                "cpi": [1.0, 1.1, 1.2, 1.3],
                "average_wage": [10.0, 20.0, 30.0, 60.0],
                "real_average_wage": [9.0, 18.0, 27.0, 54.0],
                "full_essential_coverage_share": [0.7, 0.8, 0.9, 1.0],
                "bank_equity": [50.0, 60.0, 70.0, 100.0],
                "government_deficit": [5.0, 6.0, 7.0, 10.0],
                "government_deficit_share_gdp": [0.05, 0.04, 0.03, 0.02],
                "recession_intensity": [0.2, 0.15, 0.10, 0.05],
                "government_countercyclical_spending": [20.0, 18.0, 16.0, 14.0],
                "school_enrollment_share": [0.6, 0.6, 0.6, 0.6],
                "university_enrollment_share": [0.2, 0.2, 0.2, 0.2],
                "low_resource_school_enrollment_share": [0.5, 0.5, 0.5, 0.5],
                "low_resource_university_enrollment_share": [0.1, 0.1, 0.1, 0.1],
                "low_resource_origin_upward_mobility_share": [0.3, 0.3, 0.3, 0.3],
                "university_income_premium": [1.4, 1.4, 1.4, 1.4],
                "poverty_rate_with_university": [0.1, 0.1, 0.1, 0.1],
                "poverty_rate_without_university": [0.4, 0.4, 0.4, 0.4],
                "skilled_job_fill_rate": [0.7, 0.7, 0.7, 0.7],
            }
        )

        summary_rows, _ = _summary_tables(annual)
        summary_map = {label: value for label, value in summary_rows}

        self.assertEqual(summary_map["Ventana del resumen"], "2-4")
        self.assertEqual(summary_map["PIB nominal promedio ultimos 3 anos"], "366.67")
        self.assertEqual(summary_map["PIB real promedio ultimos 3 anos"], "293.33")
        self.assertEqual(summary_map["Desempleo promedio ultimos 3 anos"], "7.3%")
        self.assertEqual(summary_map["Deficit fiscal promedio ultimos 3 anos"], "7.67")

    def test_batch_report_annualization_derives_labor_and_payroll_tax_burdens(self) -> None:
        monthly = pd.DataFrame(
            {
                "year": [1, 1, 2, 2],
                "gdp_nominal": [100.0, 150.0, 200.0, 250.0],
                "real_gdp_nominal": [100.0, 150.0, 200.0, 250.0],
                "population": [100, 100, 100, 100],
                "fertile_women": [20, 20, 20, 20],
                "births": [1, 1, 1, 1],
                "deaths": [0, 0, 0, 0],
                "labor_force": [50, 50, 50, 50],
                "employment_count": [45, 45, 46, 46],
                "unemployment_rate": [0.10, 0.10, 0.08, 0.08],
                "family_income_to_basket_ratio": [1.0, 1.0, 1.1, 1.1],
                "output_gap_share": [0.0, 0.0, 0.0, 0.0],
                "cpi": [1.0, 1.0, 1.0, 1.0],
                "gdp_deflator": [1.0, 1.0, 1.0, 1.0],
                "inflation_rate": [0.0, 0.0, 0.0, 0.0],
                "gdp_growth": [0.0, 0.0, 0.0, 0.0],
                "population_growth": [0.0, 0.0, 0.0, 0.0],
                "average_wage": [10.0, 10.0, 11.0, 11.0],
                "real_average_wage": [10.0, 10.0, 11.0, 11.0],
                "essential_demand_units": [10.0, 10.0, 10.0, 10.0],
                "essential_production_units": [10.0, 10.0, 10.0, 10.0],
                "essential_sales_units": [10.0, 10.0, 10.0, 10.0],
                "people_full_essential_coverage": [90.0, 90.0, 92.0, 92.0],
                "full_essential_coverage_share": [0.9, 0.9, 0.92, 0.92],
                "average_food_meals_per_person": [3.0, 3.0, 3.0, 3.0],
                "bank_equity": [100.0, 100.0, 100.0, 100.0],
                "bank_capital_ratio": [0.1, 0.1, 0.1, 0.1],
                "bank_insolvent_share": [0.0, 0.0, 0.0, 0.0],
                "bank_undercapitalized_share": [0.0, 0.0, 0.0, 0.0],
                "central_bank_money_supply": [1000.0, 1000.0, 1000.0, 1000.0],
                "central_bank_target_money_supply": [1000.0, 1000.0, 1000.0, 1000.0],
                "central_bank_policy_rate": [0.03, 0.03, 0.03, 0.03],
                "central_bank_issuance": [0.0, 0.0, 0.0, 0.0],
                "central_bank_monetary_gap_share": [0.0, 0.0, 0.0, 0.0],
                "average_bank_reserve_ratio": [0.1, 0.1, 0.1, 0.1],
                "government_tax_revenue": [20.0, 30.0, 40.0, 50.0],
                "government_labor_tax_revenue": [8.0, 12.0, 16.0, 20.0],
                "government_payroll_tax_revenue": [5.0, 7.0, 10.0, 12.0],
                "government_total_spending": [25.0, 35.0, 45.0, 55.0],
                "government_deficit": [5.0, 5.0, 5.0, 5.0],
                "government_debt_outstanding": [10.0, 10.0, 12.0, 12.0],
                "government_school_spending": [10.0, 15.0, 18.0, 22.0],
                "government_university_spending": [4.0, 6.0, 8.0, 10.0],
                "government_school_units": [2.0, 3.0, 4.0, 5.0],
                "government_university_units": [1.0, 1.5, 2.0, 2.5],
                "school_average_price": [5.0, 5.0, 5.0, 5.0],
                "university_average_price": [4.0, 4.0, 4.0, 4.0],
                "recession_flag": [0.0, 0.0, 0.0, 0.0],
                "recession_intensity": [0.0, 0.0, 0.0, 0.0],
                "government_countercyclical_spending": [0.0, 0.0, 0.0, 0.0],
                "government_countercyclical_support_multiplier": [1.0, 1.0, 1.0, 1.0],
                "government_countercyclical_procurement_multiplier": [1.0, 1.0, 1.0, 1.0],
                "household_final_consumption_share_gdp": [0.5, 0.5, 0.5, 0.5],
                "government_final_consumption_share_gdp": [0.1, 0.1, 0.1, 0.1],
                "government_infrastructure_spending_share_gdp": [0.02, 0.02, 0.02, 0.02],
                "government_spending_share_gdp": [0.25, 0.25, 0.25, 0.25],
                "government_tax_burden_gdp": [0.2, 0.2, 0.2, 0.2],
                "government_labor_tax_burden_gdp": [0.08, 0.08, 0.08, 0.08],
                "government_payroll_tax_burden_gdp": [0.05, 0.05, 0.05, 0.05],
                "gross_capital_formation_share_gdp": [0.2, 0.2, 0.2, 0.2],
                "investment_knowledge_multiplier": [1.0, 1.0, 1.0, 1.0],
                "public_capital_stock": [10.0, 11.0, 12.0, 13.0],
                "net_exports_share_gdp": [0.0, 0.0, 0.0, 0.0],
                "gdp_expenditure_gap_share_gdp": [0.0, 0.0, 0.0, 0.0],
                "government_deficit_share_gdp": [0.05, 0.05, 0.05, 0.05],
                "school_enrollment_share": [0.8, 0.8, 0.8, 0.8],
                "university_enrollment_share": [0.2, 0.2, 0.2, 0.2],
                "school_completion_share": [0.7, 0.7, 0.7, 0.7],
                "university_completion_share": [0.15, 0.15, 0.15, 0.15],
                "low_resource_school_enrollment_share": [0.6, 0.6, 0.6, 0.6],
                "low_resource_university_enrollment_share": [0.1, 0.1, 0.1, 0.1],
                "low_resource_university_student_share": [0.1, 0.1, 0.1, 0.1],
                "low_resource_origin_upward_mobility_share": [0.2, 0.2, 0.2, 0.2],
                "low_resource_origin_university_completion_share": [0.1, 0.1, 0.1, 0.1],
                "poor_origin_university_mobility_lift": [0.05, 0.05, 0.05, 0.05],
                "school_income_premium": [1.1, 1.1, 1.1, 1.1],
                "university_income_premium": [1.3, 1.3, 1.3, 1.3],
                "poverty_rate_without_university": [0.3, 0.3, 0.3, 0.3],
                "poverty_rate_with_university": [0.1, 0.1, 0.1, 0.1],
                "skilled_job_fill_rate": [0.7, 0.7, 0.7, 0.7],
                "firm_expansion_credit_creation": [1.0, 1.0, 1.0, 1.0],
            }
        )

        annual = _annualize_core_history(monthly)

        self.assertIn("government_labor_tax_burden_gdp", annual.columns)
        self.assertIn("government_payroll_tax_burden_gdp", annual.columns)
        self.assertIn("government_school_unit_cost_ratio_private_price", annual.columns)
        self.assertIn("government_university_unit_cost_ratio_private_price", annual.columns)
        self.assertIn("children_studying_ratio", annual.columns)
        self.assertIn("adults_with_school_credential_ratio", annual.columns)
        self.assertIn("adults_with_university_credential_ratio", annual.columns)
        self.assertAlmostEqual(annual.iloc[0]["government_labor_tax_burden_gdp"], 20.0 / 250.0)
        self.assertAlmostEqual(annual.iloc[1]["government_payroll_tax_burden_gdp"], 22.0 / 450.0)
        self.assertAlmostEqual(annual.iloc[0]["government_school_unit_cost_ratio_private_price"], 1.0, places=6)
        self.assertAlmostEqual(annual.iloc[1]["government_university_unit_cost_ratio_private_price"], 1.0, places=6)
        self.assertAlmostEqual(annual.iloc[0]["children_studying_ratio"], annual.iloc[0]["school_enrollment_share"])
        self.assertAlmostEqual(
            annual.iloc[0]["adults_with_school_credential_ratio"],
            annual.iloc[0]["school_completion_share"],
        )
        self.assertAlmostEqual(
            annual.iloc[0]["adults_with_university_credential_ratio"],
            annual.iloc[0]["university_completion_share"],
        )

    def test_social_state_intensive_profile_uses_broad_labor_tax_base(self) -> None:
        values = social_state_intensive_profile().values

        self.assertGreaterEqual(float(values["government_labor_tax_rate_high"]), 0.45)
        self.assertGreaterEqual(float(values["government_payroll_tax_rate"]), 0.25)
        self.assertGreaterEqual(float(values["public_administration_payroll_share"]), 0.70)
        self.assertGreaterEqual(float(values["public_administration_employment_floor_share"]), 0.04)

    def test_history_frame_uses_gdp_deflator_for_real_gdp_and_inflation(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=4, households=120, firms_per_sector=2, seed=91)
        )
        result = sim.run()

        monthly = history_frame(
            result.history,
            periods_per_year=result.config.periods_per_year,
            target_unemployment=result.config.target_unemployment,
        )

        self.assertIn("gdp_deflator", monthly.columns)
        self.assertAlmostEqual(
            monthly.iloc[-1]["real_gdp_nominal"],
            monthly.iloc[-1]["gdp_nominal"] / monthly.iloc[-1]["gdp_deflator"],
            places=6,
        )
        self.assertAlmostEqual(
            monthly.iloc[1]["inflation_rate"],
            monthly["gdp_deflator"].pct_change().iloc[1],
            places=6,
        )

    def test_annual_frame_sums_monthly_real_gdp_and_uses_gdp_deflator(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=24, households=120, firms_per_sector=2, seed=93)
        )
        result = sim.run()

        monthly = history_frame(
            result.history,
            periods_per_year=result.config.periods_per_year,
            target_unemployment=result.config.target_unemployment,
        )
        annual = annual_frame(monthly, target_unemployment=result.config.target_unemployment)

        first_year = annual.iloc[0]["year"]
        expected_real_gdp = monthly.loc[monthly["year"] == first_year, "real_gdp_nominal"].sum()

        self.assertAlmostEqual(annual.iloc[0]["real_gdp_nominal"], expected_real_gdp, places=6)
        self.assertAlmostEqual(
            annual.iloc[0]["gdp_deflator"],
            annual.iloc[0]["gdp_nominal"] / annual.iloc[0]["real_gdp_nominal"],
            places=6,
        )
        self.assertAlmostEqual(
            annual.iloc[1]["inflation_yoy"],
            annual["gdp_deflator"].pct_change().iloc[1],
            places=6,
        )

    def test_run_stops_early_once_population_is_extinct(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=5, households=60, firms_per_sector=1, seed=17)
        )

        with patch.object(sim, "step", side_effect=[SimpleNamespace(population=0)]) as step_mock:
            result = sim.run()

        self.assertEqual(step_mock.call_count, 1)
        self.assertIs(result.history, sim.history)

    def test_target_inventory_for_goods_depends_on_expected_sales_and_inventory_aversion(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=2, seed=23)
        )
        firm = next(
            firm
            for firm in sim.firms
            if firm.active and firm.sector in ESSENTIAL_SECTOR_KEYS
        )
        spec = SECTOR_BY_KEY[firm.sector]
        expected_sales = 120.0

        firm.inventory_aversion = 0.6
        low_aversion_target = sim._firm_target_inventory_units(firm, expected_sales)

        firm.inventory_aversion = 1.6
        high_aversion_target = sim._firm_target_inventory_units(firm, expected_sales)

        low_expected_multiplier = min(1.45, max(0.70, 1.10 - 0.18 * (0.6 - 1.0)))
        high_expected_multiplier = min(1.45, max(0.70, 1.10 - 0.18 * (1.6 - 1.0)))

        self.assertAlmostEqual(
            low_aversion_target,
            expected_sales * spec.target_inventory_ratio * low_expected_multiplier,
            places=6,
        )
        self.assertAlmostEqual(
            high_aversion_target,
            expected_sales * spec.target_inventory_ratio * high_expected_multiplier,
            places=6,
        )
        self.assertGreater(low_aversion_target, high_aversion_target)

    def test_desired_workers_for_goods_follow_projected_output(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=2, seed=29)
        )
        firm = next(
            firm
            for firm in sim.firms
            if firm.active and firm.sector in ESSENTIAL_SECTOR_KEYS
        )
        expected_sales = 120.0
        firm.inventory = 15.0
        effective_productivity = 3.25

        desired_output = sim._firm_desired_output_from_expected_sales(firm, expected_sales)
        desired_workers = sim._workers_needed_for_units(desired_output, effective_productivity)

        self.assertAlmostEqual(
            desired_output,
            expected_sales + sim._firm_target_inventory_units(firm, expected_sales) - firm.inventory,
            places=6,
        )
        self.assertEqual(
            desired_workers,
            math.ceil(desired_output / effective_productivity),
        )

    def test_target_headcount_waits_when_capacity_signal_is_not_persistent(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=2, seed=30)
        )
        firm = next(
            firm
            for firm in sim.firms
            if firm.active and firm.sector in ESSENTIAL_SECTOR_KEYS
        )
        firm.workers = [0, 1, 2, 3, 4]
        firm.age = 6
        firm.inventory = 0.0
        firm.last_sales = 150.0
        firm.sales_history = [118.0, 121.0, 150.0]
        firm.last_expected_sales = 120.0
        firm.expected_sales_history = [120.0, 120.0, 120.0]
        firm.last_production = 80.0
        firm.employment_inertia = 0.90
        effective_productivity = 25.0
        target_inventory = sim._firm_target_inventory_units(firm, 150.0)

        raw_target = sim._workers_needed_for_units(
            sim._firm_desired_output_from_expected_sales(firm, 150.0),
            effective_productivity,
        )
        target_headcount = sim._target_headcount_for_expected_sales(
            firm,
            150.0,
            effective_productivity,
            target_inventory=target_inventory,
        )

        self.assertGreater(raw_target, len(firm.workers))
        self.assertEqual(target_headcount, len(firm.workers))

    def test_target_headcount_expands_gradually_under_persistent_shortage(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=2, seed=31)
        )
        firm = next(
            firm
            for firm in sim.firms
            if firm.active and firm.sector in ESSENTIAL_SECTOR_KEYS
        )
        firm.workers = [0, 1, 2, 3, 4]
        firm.age = 8
        firm.inventory = 0.0
        firm.last_sales = 170.0
        firm.sales_history = [150.0, 162.0, 170.0]
        firm.last_expected_sales = 120.0
        firm.expected_sales_history = [118.0, 120.0, 122.0]
        firm.last_production = 124.0
        firm.employment_inertia = 0.88
        effective_productivity = 25.0
        target_inventory = sim._firm_target_inventory_units(firm, 170.0)

        raw_target = sim._workers_needed_for_units(
            sim._firm_desired_output_from_expected_sales(firm, 170.0),
            effective_productivity,
        )
        target_headcount = sim._target_headcount_for_expected_sales(
            firm,
            170.0,
            effective_productivity,
            target_inventory=target_inventory,
        )

        self.assertGreater(raw_target, len(firm.workers))
        self.assertGreater(target_headcount, len(firm.workers))
        self.assertLess(target_headcount, raw_target)

    def test_target_headcount_does_not_mass_layoff_after_one_price_bad_month(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=3, seed=32)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        successful_peer = next(
            peer for peer in sim.firms if peer.active and peer.sector == firm.sector and peer.id != firm.id
        )
        firm.workers = list(range(91))
        firm.age = 2
        firm.price = 11.31
        firm.last_unit_cost = 4.0
        firm.inventory = 180.0
        firm.target_inventory = 70.0
        firm.last_sales = 0.0
        firm.last_expected_sales = 120.0
        firm.sales_history = [122.0, 118.0, 0.0]
        firm.expected_sales_history = [120.0, 120.0, 120.0]
        firm.last_profit = -400.0
        firm.cash = 9000.0
        firm.last_wage_bill = 910.0
        firm.loss_streak = 1
        successful_peer.price = 4.54
        successful_peer.last_sales = 130.0
        successful_peer.last_expected_sales = 120.0
        successful_peer.sales_history = [125.0, 128.0, 130.0]
        successful_peer.expected_sales_history = [120.0, 120.0, 120.0]

        target_headcount = sim._target_headcount_for_expected_sales(
            firm,
            expected_sales=7.82,
            effective_productivity=3.0,
            target_inventory=12.0,
        )

        self.assertGreaterEqual(target_headcount, 86)
        self.assertLess(target_headcount, len(firm.workers))

    def test_startup_essential_price_candidates_include_visible_success_cut(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=3, seed=33)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        successful_peer = next(
            peer for peer in sim.firms if peer.active and peer.sector == firm.sector and peer.id != firm.id
        )
        firm.price = 11.31
        firm.last_unit_cost = 4.0
        firm.inventory = 160.0
        firm.target_inventory = 70.0
        firm.last_sales = 0.0
        firm.last_expected_sales = 120.0
        firm.sales_history = [120.0, 0.0]
        firm.expected_sales_history = [120.0, 120.0]
        successful_peer.price = 4.54
        successful_peer.last_sales = 130.0
        successful_peer.last_expected_sales = 120.0
        successful_peer.sales_history = [126.0, 130.0]
        successful_peer.expected_sales_history = [120.0, 120.0]

        candidates = sim._startup_essential_candidate_prices(firm, variable_unit_cost=3.20)

        self.assertLess(min(candidates), firm.price)
        self.assertLessEqual(min(candidates), successful_peer.price * 1.05)

    def test_smoothed_sales_reference_uses_six_month_moving_average(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=2, seed=31, periods_per_year=12)
        )
        firm = next(firm for firm in sim.firms if firm.active)
        firm.sales_history = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        firm.last_sales = 80.0
        firm.last_expected_sales = 75.0

        self.assertAlmostEqual(
            sim._smoothed_sales_reference(firm),
            sum([30.0, 40.0, 50.0, 60.0, 70.0, 80.0]) / 6.0,
            places=6,
        )

    def test_inventory_batches_expire_after_six_months_fifo(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=2, seed=37, periods_per_year=12)
        )
        firm = next(
            firm
            for firm in sim.firms
            if firm.active and firm.sector in ESSENTIAL_SECTOR_KEYS
        )
        firm.inventory_batches = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
        firm.inventory = sum(firm.inventory_batches)
        firm.last_unit_cost = 2.0

        carry_cost, waste_cost = sim._apply_inventory_carry_and_waste(firm)

        self.assertAlmostEqual(carry_cost, 140.0 * 2.0 * sim.config.inventory_carry_cost_share, places=6)
        self.assertAlmostEqual(waste_cost, 5.0 * 2.0, places=6)
        self.assertEqual(firm.inventory_batches, [10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
        self.assertAlmostEqual(firm.inventory, 135.0, places=6)

    def test_candidate_profit_penalizes_old_inventory_through_waste_and_carry(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=60, firms_per_sector=2, seed=41, periods_per_year=12)
        )
        firm = next(
            firm
            for firm in sim.firms
            if firm.active and firm.sector in ESSENTIAL_SECTOR_KEYS
        )
        firm.wage_offer = 8.0
        firm.input_cost_per_unit = 1.5
        firm.transport_cost_per_unit = 0.5
        firm.fixed_overhead = 6.0
        firm.capital = 20.0
        firm.productivity = 5.0
        firm.technology = 1.0
        firm.inventory = 80.0
        firm.last_unit_cost = 4.0

        fresh_batches = [12.0, 12.0, 14.0, 14.0, 14.0, 14.0]
        stale_batches = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

        firm.inventory_batches = fresh_batches.copy()
        fresh_profit = sim._candidate_total_profit(
            firm,
            prudent_sales=0.0,
            effective_price=7.0,
            variable_unit_cost=3.6,
            fixed_cost=firm.fixed_overhead + firm.capital * sim.config.depreciation_rate,
        )

        firm.inventory_batches = stale_batches.copy()
        stale_profit = sim._candidate_total_profit(
            firm,
            prudent_sales=0.0,
            effective_price=7.0,
            variable_unit_cost=3.6,
            fixed_cost=firm.fixed_overhead + firm.capital * sim.config.depreciation_rate,
        )

        self.assertLess(stale_profit, fresh_profit)

    def test_government_education_budget_base_uses_tax_anchor_and_deficit_tolerance(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=19,
                government_structural_deficit_tolerance=0.50,
            )
        )
        sim.history = [
            SimpleNamespace(
                government_tax_revenue=80.0,
                government_procurement_spending=20.0,
                government_education_spending=10.0,
            ),
            SimpleNamespace(
                government_tax_revenue=120.0,
                government_procurement_spending=30.0,
                government_education_spending=10.0,
            ),
        ]
        sim.government.treasury_cash = 25.0
        sim.government.debt_outstanding = 0.0
        sim._period_government_tax_revenue = 0.0

        self.assertAlmostEqual(sim._government_education_budget_base(), 150.0)

    def test_government_structural_budget_anchor_softens_when_debt_burden_is_high(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=20,
                government_structural_deficit_tolerance=0.50,
            )
        )
        sim.history = [
            SimpleNamespace(
                government_tax_revenue=100.0,
                government_procurement_spending=30.0,
                government_education_spending=20.0,
            )
        ]
        sim.government.treasury_cash = 10.0
        sim.government.debt_outstanding = 3600.0

        self.assertAlmostEqual(sim._government_structural_budget_anchor(), 132.0)

    def test_public_administration_target_workers_scale_with_state_size(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=220,
                firms_per_sector=2,
                seed=21,
                public_administration_employment_floor_share=0.02,
                public_administration_employment_state_size_sensitivity=0.20,
                public_administration_employment_cap_share=0.12,
            )
        )
        sim.history = [SimpleNamespace(gdp_nominal=1_000.0)]

        with patch.object(sim, "_government_structural_budget_anchor", return_value=100.0):
            lower_state = sim._public_administration_target_workers()
        with patch.object(sim, "_government_structural_budget_anchor", return_value=300.0):
            higher_state = sim._public_administration_target_workers()

        self.assertGreater(higher_state, lower_state)

    def test_public_administration_budget_splits_between_payroll_and_operations(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=180,
                firms_per_sector=2,
                seed=22,
                public_administration_payroll_share=0.60,
            )
        )
        public_workers = [
            household
            for household in sim.households
            if household.alive and sim._household_labor_capacity(household) > 0.0
        ][:2]
        for household in public_workers:
            household.employed_by = PUBLIC_ADMINISTRATION_EMPLOYER_ID
        sim.government.treasury_cash = 100.0

        with patch.object(sim, "_public_administration_budget", return_value=100.0), patch.object(
            sim,
            "_public_administration_target_workers",
            return_value=4,
        ):
            wage_offer = sim._public_administration_wage_offer()
            total_spent = sim._pay_public_administration_wages()

        self.assertAlmostEqual(total_spent, 2 * wage_offer + 40.0, places=6)
        self.assertAlmostEqual(sim.government.public_administration_spending_this_period, total_spent, places=6)
        self.assertGreater(sum(sim._pending_sector_payments.values()), 0.0)

    def test_government_structural_budget_anchor_respects_final_consumption_floor_share(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=20,
                government_structural_procurement_budget_share=0.10,
                public_school_budget_share=0.10,
                public_university_budget_share=0.05,
                public_administration_budget_share=0.15,
                government_final_consumption_floor_share_gdp=0.20,
                government_structural_deficit_tolerance=0.0,
            )
        )
        sim.history = [
            SimpleNamespace(
                gdp_nominal=1_000.0,
                government_tax_revenue=100.0,
                government_procurement_spending=20.0,
                government_education_spending=10.0,
                government_public_administration_spending=10.0,
            )
        ]
        sim.government.treasury_cash = 25.0
        sim.government.debt_outstanding = 0.0

        self.assertAlmostEqual(sim._government_structural_budget_anchor(), 500.0)

    def test_structural_procurement_uses_endogenous_budget_anchor(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=21,
                government_procurement_gap_share=0.0,
                government_structural_procurement_budget_share=0.50,
                government_structural_deficit_tolerance=0.0,
            )
        )
        sim.history = []
        sim.government.treasury_cash = 200.0
        sim._period_recession_flag = 1.0
        sim._period_recession_intensity = 0.0
        population = len(sim._active_households())
        for sector_key in ESSENTIAL_SECTOR_KEYS:
            sim._period_sector_sales_units[sector_key] = population * sim._essential_basket_share(sector_key)

        with patch.object(sim, "_average_sector_price", return_value=1.0), patch.object(
            sim,
            "_purchase_from_sector",
            side_effect=lambda price_sensitivity, sector_key, desired_units, cash, spending_log: (0.0, desired_units),
        ):
            spent = sim._apply_government_essential_procurement()

        self.assertAlmostEqual(spent, 100.0, places=6)
        self.assertAlmostEqual(sim._period_government_procurement_spending, 100.0, places=6)

    def test_public_education_targets_cover_children_and_adult_catchup(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=23,
                public_school_support_package_share=0.80,
                public_education_low_resource_priority_bonus=0.0,
            )
        )
        child = sim.households[0]
        child.age_periods = int(10 * sim.config.periods_per_year)
        child.school_years_completed = 0.0
        adult = sim.households[1]
        adult.age_periods = int(25 * sim.config.periods_per_year)
        adult.school_years_completed = 0.0

        self.assertAlmostEqual(
            sim._public_education_target_units(child, "school", 1.0),
            0.80,
        )
        self.assertAlmostEqual(
            sim._public_education_target_units(adult, "school", 1.0),
            0.48,
        )

    def test_family_public_education_priority_rewards_need_and_continuity(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=24)
        )
        child = sim.households[0]
        child.age_periods = int(10 * sim.config.periods_per_year)
        child.school_years_completed = 0.0
        child.public_school_support_persistence = 1.0
        comparison_adult = sim.households[1]
        comparison_adult.age_periods = int(35 * sim.config.periods_per_year)
        comparison_adult.school_years_completed = sim.config.school_years_required
        comparison_adult.university_years_completed = sim.config.university_years_required

        with patch.object(sim, "_household_family_resource_coverage", side_effect=lambda _: 0.7):
            priority_score = sim._family_public_education_priority([child])
            neutral_score = sim._family_public_education_priority([comparison_adult])

        self.assertGreater(priority_score, neutral_score)

    def test_households_buy_private_school_when_public_slots_are_unavailable(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=80,
                firms_per_sector=1,
                seed=125,
                government_enabled=True,
                public_school_budget_share=0.0,
                public_school_min_target_units=1.0,
                public_school_support_package_share=1.0,
            )
        )
        student = sim.households[0]
        for household in sim.households:
            household.alive = False
            household.savings = 0.0
            household.wage_income = 0.0
            household.last_income = 0.0
        student.alive = True
        student.age_periods = int(10 * sim.config.periods_per_year)
        student.school_years_completed = 0.0
        student.savings = 50.0
        student.guardian_id = None
        student.mother_id = None
        student.father_id = None

        for firm in sim.firms:
            firm.active = False
            firm.workers.clear()
            firm.inventory = 0.0
            firm.inventory_batches.clear()
        for sector_key in (*ESSENTIAL_SECTOR_KEYS, "school"):
            firm = next(firm for firm in sim.firms if firm.sector == sector_key)
            firm.active = True
            firm.price = 1.0
            firm.inventory = 25.0
            firm.inventory_batches = [] if sector_key == "school" else [25.0]

        sim.period = 1
        sim._reset_period_counters()
        sim._refresh_period_household_caches()
        sim._refresh_period_sector_caches()
        sim._consume_households()

        self.assertGreater(student.last_consumption["school"], 0.0)
        self.assertEqual(sim._period_family_public_education_units.get((student.id, "school"), 0.0), 0.0)

    def test_public_administration_hires_unemployed_households(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=24,
                government_structural_deficit_tolerance=0.0,
                public_administration_budget_share=0.50,
                public_administration_wage_premium=0.0,
            )
        )
        for household in sim.households:
            if household.employed_by is not None:
                sim._release_household_from_employment(household)
        sim.government.treasury_cash = 200.0

        sim._manage_public_administration_workforce()

        public_workers = [
            household for household in sim.households if household.employed_by == PUBLIC_ADMINISTRATION_EMPLOYER_ID
        ]
        self.assertGreater(len(public_workers), 0)
        self.assertEqual(len(public_workers), sim._public_administration_target_workers())

    def test_public_administration_payroll_counts_as_government_spending(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=25,
                public_administration_wage_premium=0.0,
            )
        )
        worker = next(
            household
            for household in sim.households
            if sim._household_labor_capacity(household) > 0.0
        )
        sim._release_household_from_employment(worker)
        worker.employed_by = PUBLIC_ADMINISTRATION_EMPLOYER_ID
        sim.government.treasury_cash = 100.0

        paid = sim._pay_public_administration_wages()

        self.assertGreater(paid, 0.0)
        self.assertAlmostEqual(sim._period_government_public_administration_spending, paid)
        self.assertAlmostEqual(sim.government.public_administration_spending_this_period, paid)
        self.assertAlmostEqual(worker.last_income, paid)

    def test_firm_history_is_disabled_by_default(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=2, households=80, firms_per_sector=2, seed=41)
        )

        result = sim.run()

        self.assertEqual(result.firm_history, [])

    def test_firm_history_tracks_worker_exit_breakdown_and_reemployment_for_audit(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=58)
        )
        source_firm = next(firm for firm in sim.firms if firm.active and len(firm.workers) >= 1)
        target_firm = next(
            firm for firm in sim.firms
            if firm.active and firm.id != source_firm.id
        )
        starting_workers = len(source_firm.workers)
        moved_worker_id = source_firm.workers[0]

        sim._reset_period_counters()
        sim._release_household_from_employment(sim.households[moved_worker_id], exit_reason="quit")
        sim._assign_household_to_employer(sim.households[moved_worker_id], target_firm.id, target_firm.wage_offer)
        sim.period = 1

        snapshots = sim._build_firm_period_snapshots()
        source_snapshot = next(snapshot for snapshot in snapshots if snapshot.firm_id == source_firm.id)

        self.assertEqual(source_snapshot.starting_workers, starting_workers)
        self.assertEqual(source_snapshot.worker_exits, 1)
        self.assertEqual(source_snapshot.worker_quits, 1)
        self.assertEqual(source_snapshot.worker_dismissals, 0)
        self.assertEqual(source_snapshot.exited_workers_reemployed, 1)

    def test_private_dismissal_pays_tenure_based_severance(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=158)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.workers)
        worker = sim.households[firm.workers[0]]
        worker.contract_wage = 10.0
        worker.employment_tenure = 12
        firm.cash = 100.0
        firm.age = 1

        sim._reset_period_counters()
        sim._release_household_from_employment(worker)
        snapshot = next(snapshot for snapshot in sim._build_firm_period_snapshots() if snapshot.firm_id == firm.id)

        self.assertAlmostEqual(firm.cash, 90.0)
        self.assertAlmostEqual(worker.wage_income, 10.0)
        self.assertAlmostEqual(worker.last_income, 10.0)
        self.assertAlmostEqual(firm.last_severance_cost, 10.0)
        self.assertAlmostEqual(snapshot.severance_total, 10.0)
        self.assertEqual(snapshot.worker_dismissals, 1)

    def test_private_quit_does_not_pay_severance(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=159)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.workers)
        worker = sim.households[firm.workers[0]]
        worker.contract_wage = 10.0
        worker.employment_tenure = 12
        firm.cash = 100.0

        sim._reset_period_counters()
        sim._release_household_from_employment(worker, exit_reason="quit")

        self.assertAlmostEqual(firm.cash, 100.0)
        self.assertAlmostEqual(worker.wage_income, 0.0)
        self.assertAlmostEqual(firm.last_severance_cost, 0.0)

    def test_workforce_alignment_releases_low_severance_worker_first(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=140,
                firms_per_sector=2,
                seed=160,
                government_payroll_tax_rate=0.0,
                severance_layoff_payback_periods=1,
            )
        )
        firm = next(firm for firm in sim.firms if firm.active)
        for worker_id in list(firm.workers):
            sim._release_household_from_employment(sim.households[worker_id], exit_reason="quit")
        candidates = [
            household
            for household in sim.households
            if household.alive and sim._household_labor_capacity(household) > 0.0
        ][:2]
        for household in candidates:
            sim._release_household_from_employment(household, exit_reason="quit")
            sim._assign_household_to_employer(household, firm.id, 10.0)
        low_tenure, high_tenure = candidates
        low_tenure.employment_tenure = 1
        high_tenure.employment_tenure = 120
        firm.cash = 1000.0
        firm.last_wage_bill = 20.0
        firm.loss_streak = 0
        firm.desired_workers = 1

        sim._reset_period_counters()
        sim._align_existing_workforce()

        self.assertIsNone(low_tenure.employed_by)
        self.assertEqual(high_tenure.employed_by, firm.id)
        self.assertAlmostEqual(firm.last_severance_cost, 10.0 / 12.0)

    def test_firm_audit_frame_merges_macro_labor_context(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=2,
                households=120,
                firms_per_sector=2,
                seed=59,
                track_firm_history=True,
            )
        )
        result = sim.run()

        firms = firm_history_frame(result)
        macro = history_frame(
            result.history,
            periods_per_year=result.config.periods_per_year,
            target_unemployment=result.config.target_unemployment,
        )
        audit = firm_audit_frame(firms, macro)

        self.assertFalse(audit.empty)
        for column in (
            "starting_workers",
            "worker_exits",
            "worker_quits",
            "worker_dismissals",
            "exited_workers_reemployed",
            "firm_income_total",
            "firm_profit_total",
            "payroll_total",
            "severance_total",
            "desired_workers_next_period",
            "average_wage",
            "unemployment_rate",
            "effective_marginal_unit_cost",
            "price_minus_marginal_unit_cost",
            "sales_realization_ratio",
            "probable_loss_cause",
            "recommended_loss_response",
            "capital",
            "capital_efficiency_percent",
            "technology",
            "technology_level_percent",
            "productivity",
            "effective_worker_productivity_capacity",
        ):
            self.assertIn(column, audit.columns)
        merged = audit.merge(
            firms[["period", "firm_id", "revenue", "desired_workers"]],
            on=["period", "firm_id"],
            how="left",
        )
        self.assertTrue((merged["firm_income_total"] == merged["revenue"]).all())
        self.assertTrue((merged["desired_workers_next_period"] == merged["desired_workers"]).all())

    def test_firm_audit_frame_classifies_probable_loss_cause_for_xlsx(self) -> None:
        firms = pd.DataFrame(
            [
                {
                    "period": 1,
                    "year": 1,
                    "period_in_year": 1,
                    "firm_id": 1,
                    "sector": "food",
                    "workers": 2,
                    "desired_workers": 2,
                    "expected_sales": 100.0,
                    "sales": 100.0,
                    "production": 120.0,
                    "inventory": 10.0,
                    "price": 5.0,
                    "capital": 100.0,
                    "capital_efficiency_percent": 80.0,
                    "technology": 1.2,
                    "technology_level_percent": 12.0,
                    "investment_animal_spirits": 1.15,
                    "technology_investment": 20.0,
                    "industrial_investment_spending": 80.0,
                    "productivity": 3.0,
                    "effective_worker_productivity_capacity": 2.9,
                    "effective_marginal_unit_cost": 6.0,
                    "revenue": 500.0,
                    "profit": -50.0,
                    "total_cost": 550.0,
                },
                {
                    "period": 1,
                    "year": 1,
                    "period_in_year": 1,
                    "firm_id": 2,
                    "sector": "food",
                    "workers": 2,
                    "desired_workers": 2,
                    "expected_sales": 100.0,
                    "sales": 20.0,
                    "production": 80.0,
                    "inventory": 10.0,
                    "price": 9.0,
                    "capital": 100.0,
                    "capital_efficiency_percent": 80.0,
                    "technology": 1.2,
                    "technology_level_percent": 12.0,
                    "investment_animal_spirits": 0.95,
                    "technology_investment": 0.0,
                    "industrial_investment_spending": 0.0,
                    "productivity": 3.0,
                    "effective_worker_productivity_capacity": 2.9,
                    "effective_marginal_unit_cost": 6.0,
                    "revenue": 180.0,
                    "profit": -40.0,
                    "total_cost": 220.0,
                },
                {
                    "period": 1,
                    "year": 1,
                    "period_in_year": 1,
                    "firm_id": 3,
                    "sector": "food",
                    "workers": 2,
                    "desired_workers": 2,
                    "expected_sales": 100.0,
                    "sales": 100.0,
                    "production": 110.0,
                    "inventory": 10.0,
                    "price": 9.0,
                    "capital": 100.0,
                    "capital_efficiency_percent": 80.0,
                    "technology": 1.2,
                    "technology_level_percent": 12.0,
                    "investment_animal_spirits": 1.05,
                    "technology_investment": 40.0,
                    "industrial_investment_spending": 120.0,
                    "productivity": 3.0,
                    "effective_worker_productivity_capacity": 2.9,
                    "effective_marginal_unit_cost": 6.0,
                    "revenue": 900.0,
                    "profit": 100.0,
                    "total_cost": 800.0,
                },
            ]
        )
        macro = pd.DataFrame({"period": [1], "average_wage": [10.0], "unemployment_rate": [0.05]})

        audit = firm_audit_frame(firms, macro)
        causes = dict(zip(audit["firm_id"], audit["probable_loss_cause"]))
        exported = _rename_firm_audit_export(audit)

        self.assertEqual(causes[1], "perdida_probable_por_precio_bajo_vs_costo_marginal")
        self.assertEqual(causes[2], "perdida_probable_por_precio_alto_o_demanda_debil")
        self.assertEqual(causes[3], "sin_perdida")
        self.assertIn("Causa probable de perdida", exported.columns)
        self.assertIn("Respuesta sugerida ante perdida", exported.columns)
        self.assertIn("Produccion realizada", exported.columns)
        self.assertIn("Propension interna a invertir", exported.columns)
        self.assertIn("Propension invertir productividad / ingresos", exported.columns)
        self.assertIn("Propension inversion total / ingresos", exported.columns)
        self.assertIn("Reinversion sobre liquidez disponible", exported.columns)
        self.assertIn("Razon decision inversion maquinaria", exported.columns)
        self.assertIn("Productividad efectiva por trabajador", exported.columns)
        self.assertIn("Capacidad instalada de produccion", exported.columns)
        self.assertIn("Utilizacion de capacidad %", exported.columns)
        self.assertIn("Tecnologia sobre maximo %", exported.columns)

    def test_family_audit_frame_reports_worker_side_metrics(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=60)
        )

        sim.step()
        audit = family_audit_frame(sim)
        exported = _rename_family_audit_export(audit)

        self.assertFalse(audit.empty)
        for column in (
            "total_basic_basket_cost_including_school",
            "private_school_basket_cost",
            "total_family_income",
            "family_employment_rate",
            "family_cash_available",
            "family_cash_spent",
            "family_voluntary_saved_cash",
            "family_involuntary_retained_cash",
            "family_expected_salary",
            "family_accepted_salary",
            "basic_goods_coverage_percent",
            "basic_goods_shortfall_reason",
            "marginal_propensity_to_spend",
            "marginal_propensity_to_save",
            "necessary_essential_demand_units",
            "essential_offer_units",
            "necessary_demand_to_offer_ratio",
        ):
            self.assertIn(column, audit.columns)
        self.assertTrue((audit["total_basic_basket_cost_including_school"] >= audit["private_school_basket_cost"]).all())
        self.assertTrue(((audit["family_employment_rate"] >= 0.0) & (audit["family_employment_rate"] <= 1.0)).all())
        self.assertTrue(
            ((audit["marginal_propensity_to_spend"] + audit["marginal_propensity_to_save"]) - 1.0).abs().max() < 1e-9
        )
        self.assertTrue((audit["family_cash_available"] >= audit["family_cash_spent"]).all())
        self.assertTrue(
            (
                audit["family_cash_available"]
                >= audit["family_voluntary_saved_cash"] + audit["family_involuntary_retained_cash"]
            ).all()
        )
        self.assertTrue((audit["family_expected_salary"] >= 0.0).all())
        self.assertTrue((audit["family_accepted_salary"] >= 0.0).all())
        self.assertTrue((audit["necessary_essential_demand_units"] >= 0.0).all())
        self.assertTrue((audit["essential_offer_units"] >= 0.0).all())
        self.assertTrue((audit["necessary_demand_to_offer_ratio"] >= 0.0).all())
        self.assertTrue((audit["basic_goods_coverage_percent"] >= 0.0).all())
        self.assertTrue((audit["basic_goods_coverage_percent"] <= 100.0).all())
        self.assertTrue(audit["basic_goods_shortfall_reason"].notna().all())
        self.assertIn("Expectativa salarial familia", exported.columns)
        self.assertIn("Salario aceptado familia", exported.columns)

    def test_family_audit_frame_keeps_period_history_when_tracking_enabled(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=2,
                households=120,
                firms_per_sector=2,
                seed=62,
                track_family_history=True,
            )
        )

        sim.run()
        audit = family_audit_frame(sim)

        self.assertFalse(audit.empty)
        self.assertEqual(sorted(audit["period"].unique().tolist()), [1, 2])

    def test_family_audit_propensities_use_period_flows_not_savings_stock(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=63)
        )

        sim.step()
        audit_before = family_audit_frame(sim)
        target_family_id = int(audit_before.iloc[0]["family_id"])
        before_row = audit_before[audit_before["family_id"] == target_family_id].iloc[0]
        for member in sim._family_groups()[target_family_id]:
            member.savings += 1_000_000.0

        audit_after = family_audit_frame(sim)
        after_row = audit_after[audit_after["family_id"] == target_family_id].iloc[0]

        self.assertAlmostEqual(
            before_row["marginal_propensity_to_spend"],
            after_row["marginal_propensity_to_spend"],
        )
        self.assertAlmostEqual(
            before_row["marginal_propensity_to_save"],
            after_row["marginal_propensity_to_save"],
        )

    def test_simulation_config_defaults_to_5000_households(self) -> None:
        config = SimulationConfig()

        self.assertEqual(config.households, 5000)

    def test_family_audit_frame_zero_school_cost_when_government_covers_school(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=61,
                public_school_min_target_units=1.0,
                public_school_support_package_share=1.0,
            )
        )

        sim.step()
        audit = family_audit_frame(sim)

        self.assertFalse(audit.empty)
        self.assertTrue((audit["private_school_basket_cost"] >= 0.0).all())
        self.assertGreaterEqual((audit["private_school_basket_cost"] == 0.0).sum(), 1)

    def test_sample_audit_entities_limits_random_family_sample(self) -> None:
        frame = pd.DataFrame(
            {
                "family_id": list(range(1, 81)),
                "period": [1] * 80,
                "total_family_income": [float(index) for index in range(1, 81)],
            }
        )

        sampled = _sample_audit_entities(
            frame,
            id_column="family_id",
            sample_size=45,
            random_state=_scenario_sample_seed(7, "Guatemala (mas liberal)", "family_audit"),
        )

        self.assertEqual(sampled["family_id"].nunique(), 45)
        self.assertEqual(len(sampled), 45)

    def test_sample_audit_entities_keeps_full_history_for_selected_firms(self) -> None:
        frame = pd.DataFrame(
            {
                "firm_id": [firm_id for firm_id in range(1, 31) for _ in range(3)],
                "period": [1, 2, 3] * 30,
                "sales": [10.0] * 90,
            }
        )

        sampled = _sample_audit_entities(
            frame,
            id_column="firm_id",
            sample_size=20,
            random_state=_scenario_sample_seed(7, "Noruega (economia del bienestar)", "firm_audit"),
        )

        self.assertEqual(sampled["firm_id"].nunique(), 20)
        counts = sampled.groupby("firm_id").size()
        self.assertTrue((counts == 3).all())

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
        self.assertIn("government_spending_share_gdp", frame.columns)
        self.assertIn("government_tax_burden_gdp", frame.columns)
        self.assertIn("firm_expansion_credit_creation", frame.columns)
        self.assertIn("investment_knowledge_multiplier", frame.columns)
        self.assertIn("public_capital_stock", frame.columns)
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
        self.assertIn("children_studying_ratio", frame.columns)
        self.assertIn("university_enrollment_share", frame.columns)
        self.assertIn("school_completion_share", frame.columns)
        self.assertIn("adults_with_school_credential_ratio", frame.columns)
        self.assertIn("university_completion_share", frame.columns)
        self.assertIn("adults_with_university_credential_ratio", frame.columns)
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
        self.assertTrue((frame["children_studying_ratio"] == frame["school_enrollment_share"]).all())
        self.assertTrue((frame["adults_with_school_credential_ratio"] == frame["school_completion_share"]).all())
        self.assertTrue(
            (frame["adults_with_university_credential_ratio"] == frame["university_completion_share"]).all()
        )

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

    def test_basic_basket_shortfall_mortality_prioritizes_vulnerable_members(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=140,
                firms_per_sector=2,
                seed=92,
                period_base_death_probability=0.0,
                period_senior_death_probability=0.0,
            )
        )
        members = sim.households[:4]
        self.assertEqual(len(members), 4)
        baby, child, adult, senior = members
        baby.age_periods = int(1.0 * sim.config.periods_per_year)
        child.age_periods = int(8.0 * sim.config.periods_per_year)
        adult.age_periods = int(35.0 * sim.config.periods_per_year)
        senior.age_periods = int(76.0 * sim.config.periods_per_year)

        for member in members:
            member.health_fragility = 0.0
            member.deprivation_streak = sim.config.starvation_death_periods
            member.severe_hunger_streak = sim.config.starvation_death_periods

        probabilities = {
            "baby": sim._household_death_probability(
                baby,
                unemployment_rate=0.0,
                average_savings=0.0,
                family_resources_ratio=0.35,
            ),
            "child": sim._household_death_probability(
                child,
                unemployment_rate=0.0,
                average_savings=0.0,
                family_resources_ratio=0.35,
            ),
            "adult": sim._household_death_probability(
                adult,
                unemployment_rate=0.0,
                average_savings=0.0,
                family_resources_ratio=0.35,
            ),
            "senior": sim._household_death_probability(
                senior,
                unemployment_rate=0.0,
                average_savings=0.0,
                family_resources_ratio=0.35,
            ),
        }

        self.assertGreater(probabilities["baby"], probabilities["child"])
        self.assertGreater(probabilities["child"], probabilities["adult"])
        self.assertGreater(probabilities["senior"], probabilities["adult"])

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

    def test_university_consumption_happens_before_savings(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=80,
                firms_per_sector=2,
                seed=73,
                government_enabled=False,
            )
        )
        student = next(
            household
            for household in sim.households
            if household.alive
            and sim._is_university_age(household)
            and not sim._household_has_university_credential(household)
        )
        student.school_years_completed = sim.config.school_years_required
        student.university_years_completed = 0.0
        student.higher_education_affinity = 0.99
        student.partner_id = None
        student.guardian_id = None
        student.saving_propensity = 1.0
        student.money_trust = 0.0
        student.consumption_impatience = 0.0
        student.price_sensitivity = 0.0
        student.need_scale = 1.0
        student.savings = 500.0
        sim.households = [student]
        sim._next_household_id = 1

        for firm in sim.firms:
            if not firm.active:
                continue
            firm.inventory = 1000.0
            if firm.sector in ESSENTIAL_SECTOR_KEYS or firm.sector == "university":
                firm.price = 1.0

        original_desired_units = sim._household_sector_desired_units

        def fixed_desired_units(household, sector_key):
            if household.id == student.id and sector_key == "university":
                return 1.0
            return original_desired_units(household, sector_key)

        sim._reset_period_counters()
        sim._refresh_period_household_caches()
        sim._refresh_period_family_cache()
        sim._refresh_period_sector_caches()
        with patch.object(sim, "_household_sector_desired_units", side_effect=fixed_desired_units):
            university_target = sim._household_sector_desired_units(student, "university")
            self.assertGreater(university_target, 0.0)
            student.savings = sim._essential_budget(student) + university_target

            sim._reset_period_counters()
            sim._refresh_period_household_caches()
            sim._refresh_period_family_cache()
            sim._refresh_period_sector_caches()
            with patch.object(EconomySimulation, "_family_savings_rate", return_value=0.45):
                with patch.object(EconomySimulation, "_essential_extra_budget_share", return_value=0.0):
                    sim._consume_households()

        self.assertAlmostEqual(student.last_consumption.get("university", 0.0), university_target, places=6)
        self.assertAlmostEqual(student.savings, 0.0, places=6)

    def test_period_one_wages_do_not_depend_on_living_wage_anchor(self) -> None:
        with patch.object(EconomySimulation, "_living_wage_anchor", return_value=1.0):
            base_sim = EconomySimulation(
                SimulationConfig(periods=1, households=400, firms_per_sector=4, seed=7)
            )
        with patch.object(EconomySimulation, "_living_wage_anchor", return_value=1_000.0):
            anchored_sim = EconomySimulation(
                SimulationConfig(periods=1, households=400, firms_per_sector=4, seed=7)
            )

        base_wages = [firm.wage_offer for firm in base_sim.firms if firm.active and firm.age >= 0]
        anchored_wages = [
            firm.wage_offer for firm in anchored_sim.firms if firm.active and firm.age >= 0
        ]
        self.assertEqual(len(base_wages), len(anchored_wages))
        for base_wage, anchored_wage in zip(base_wages, anchored_wages):
            self.assertAlmostEqual(base_wage, anchored_wage, places=6)

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

        with patch.object(sim, "_baseline_demand", return_value=5.0), patch.object(
            sim, "_firm_effective_productivity", return_value=10.0
        ), patch.object(
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

        self.assertLess(firm.wage_offer, original_wage)
        self.assertGreaterEqual(firm.wage_offer, 0.01)

    def test_update_firm_policies_raises_wage_for_profitable_rejection_signal(self) -> None:
        base_sim = EconomySimulation(
            SimulationConfig(periods=1, households=220, firms_per_sector=2, seed=154)
        )
        rejection_sim = EconomySimulation(
            SimulationConfig(periods=1, households=220, firms_per_sector=2, seed=154)
        )
        base_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == "food")
        rejection_firm = next(firm for firm in rejection_sim.firms if firm.active and firm.sector == "food")

        for sim, firm in ((base_sim, base_firm), (rejection_sim, rejection_firm)):
            firm.wage_offer = 10.0
            firm.price = 14.0
            firm.cash = 240.0
            firm.inventory = 0.0
            firm.input_cost_per_unit = 1.0
            firm.transport_cost_per_unit = 0.5
            firm.last_profit = 40.0
            firm.last_revenue = 160.0
            firm.last_worker_count = 4
            firm.last_production = 20.0
            firm.last_wage_bill = 40.0
            firm.last_input_cost = 15.0
            firm.last_transport_cost = 5.0
            firm.fixed_overhead = 8.0
            firm.last_expected_sales = 30.0
            firm.last_sales = 28.0
            firm.employment_inertia = 0.0
            firm.last_labor_offer_rejections = 3
            firm.last_labor_offer_rejection_wage_floor = 11.5

        base_firm.last_labor_offer_rejections = 0
        base_firm.last_labor_offer_rejection_wage_floor = 0.0

        patchers = (
            patch.object(base_sim, "_baseline_demand", return_value=30.0),
            patch.object(rejection_sim, "_baseline_demand", return_value=30.0),
            patch.object(base_sim, "_firm_effective_productivity", return_value=5.0),
            patch.object(rejection_sim, "_firm_effective_productivity", return_value=5.0),
            patch.object(base_sim, "_firm_learning_maturity", return_value=1.0),
            patch.object(rejection_sim, "_firm_learning_maturity", return_value=1.0),
            patch.object(base_sim, "_sales_anchor", return_value=30.0),
            patch.object(rejection_sim, "_sales_anchor", return_value=30.0),
            patch.object(base_sim, "_sector_wage_pressure_bonus", return_value=0.0),
            patch.object(rejection_sim, "_sector_wage_pressure_bonus", return_value=0.0),
            patch.object(base_sim, "_target_price_for_firm", return_value=14.0),
            patch.object(rejection_sim, "_target_price_for_firm", return_value=14.0),
            patch.object(base_sim, "_price_search_candidates", return_value=[14.0]),
            patch.object(rejection_sim, "_price_search_candidates", return_value=[14.0]),
            patch.object(base_sim, "_inventory_clearance_discount", return_value=0.0),
            patch.object(rejection_sim, "_inventory_clearance_discount", return_value=0.0),
            patch.object(base_sim, "_expected_demand_for_price", return_value=30.0),
            patch.object(rejection_sim, "_expected_demand_for_price", return_value=30.0),
            patch.object(base_sim, "_candidate_market_retention", return_value=(1.0, 0.0)),
            patch.object(rejection_sim, "_candidate_market_retention", return_value=(1.0, 0.0)),
            patch.object(base_sim, "_conservative_expected_sales", return_value=30.0),
            patch.object(rejection_sim, "_conservative_expected_sales", return_value=30.0),
            patch.object(base_sim, "_candidate_price_objective", return_value=0.0),
            patch.object(rejection_sim, "_candidate_price_objective", return_value=0.0),
        )

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            base_sim._update_firm_policies(last_unemployment=0.08)
            rejection_sim._update_firm_policies(last_unemployment=0.08)

        self.assertGreater(rejection_firm.wage_offer, base_firm.wage_offer)

    def test_update_firm_policies_ignore_unprofitable_rejection_signal(self) -> None:
        base_sim = EconomySimulation(
            SimulationConfig(periods=1, households=220, firms_per_sector=2, seed=155)
        )
        rejection_sim = EconomySimulation(
            SimulationConfig(periods=1, households=220, firms_per_sector=2, seed=155)
        )
        base_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == "food")
        rejection_firm = next(firm for firm in rejection_sim.firms if firm.active and firm.sector == "food")

        for sim, firm in ((base_sim, base_firm), (rejection_sim, rejection_firm)):
            firm.wage_offer = 10.0
            firm.price = 11.0
            firm.cash = 240.0
            firm.inventory = 0.0
            firm.input_cost_per_unit = 7.0
            firm.transport_cost_per_unit = 2.0
            firm.last_profit = 20.0
            firm.last_revenue = 120.0
            firm.last_worker_count = 4
            firm.last_production = 20.0
            firm.last_wage_bill = 40.0
            firm.last_input_cost = 15.0
            firm.last_transport_cost = 5.0
            firm.fixed_overhead = 8.0
            firm.last_expected_sales = 30.0
            firm.last_sales = 28.0
            firm.employment_inertia = 0.0
            firm.last_labor_offer_rejections = 3
            firm.last_labor_offer_rejection_wage_floor = 12.0

        base_firm.last_labor_offer_rejections = 0
        base_firm.last_labor_offer_rejection_wage_floor = 0.0

        patchers = (
            patch.object(base_sim, "_baseline_demand", return_value=30.0),
            patch.object(rejection_sim, "_baseline_demand", return_value=30.0),
            patch.object(base_sim, "_firm_effective_productivity", return_value=5.0),
            patch.object(rejection_sim, "_firm_effective_productivity", return_value=5.0),
            patch.object(base_sim, "_firm_learning_maturity", return_value=1.0),
            patch.object(rejection_sim, "_firm_learning_maturity", return_value=1.0),
            patch.object(base_sim, "_sales_anchor", return_value=30.0),
            patch.object(rejection_sim, "_sales_anchor", return_value=30.0),
            patch.object(base_sim, "_sector_wage_pressure_bonus", return_value=0.0),
            patch.object(rejection_sim, "_sector_wage_pressure_bonus", return_value=0.0),
            patch.object(base_sim, "_target_price_for_firm", return_value=11.0),
            patch.object(rejection_sim, "_target_price_for_firm", return_value=11.0),
            patch.object(base_sim, "_price_search_candidates", return_value=[11.0]),
            patch.object(rejection_sim, "_price_search_candidates", return_value=[11.0]),
            patch.object(base_sim, "_inventory_clearance_discount", return_value=0.0),
            patch.object(rejection_sim, "_inventory_clearance_discount", return_value=0.0),
            patch.object(base_sim, "_expected_demand_for_price", return_value=30.0),
            patch.object(rejection_sim, "_expected_demand_for_price", return_value=30.0),
            patch.object(base_sim, "_candidate_market_retention", return_value=(1.0, 0.0)),
            patch.object(rejection_sim, "_candidate_market_retention", return_value=(1.0, 0.0)),
            patch.object(base_sim, "_conservative_expected_sales", return_value=30.0),
            patch.object(rejection_sim, "_conservative_expected_sales", return_value=30.0),
            patch.object(base_sim, "_candidate_price_objective", return_value=0.0),
            patch.object(rejection_sim, "_candidate_price_objective", return_value=0.0),
        )

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            base_sim._update_firm_policies(last_unemployment=0.08)
            rejection_sim._update_firm_policies(last_unemployment=0.08)

        self.assertAlmostEqual(rejection_firm.wage_offer, base_firm.wage_offer, places=6)

    def test_low_liquidity_reduces_reservation_wage_holdout(self) -> None:
        config = SimulationConfig(periods=1, households=80, firms_per_sector=2, seed=153)
        low_liquidity_sim = EconomySimulation(config)
        high_liquidity_sim = EconomySimulation(config)

        def prepare_member(sim: EconomySimulation):
            member = sim.households[0]
            member.age_periods = int(30 * sim.config.periods_per_year)
            member.reservation_wage = 100.0
            member.money_trust = 0.5
            member.employed_by = None
            return member

        low_member = prepare_member(low_liquidity_sim)
        high_member = prepare_member(high_liquidity_sim)

        with patch.object(low_liquidity_sim, "_family_groups", return_value={0: [low_member]}), patch.object(
            low_liquidity_sim,
            "_household_labor_capacity",
            return_value=1.0,
        ), patch.object(low_liquidity_sim, "_essential_budget", return_value=100.0), patch.object(
            low_liquidity_sim,
            "_household_observed_income",
            return_value=60.0,
        ), patch.object(low_liquidity_sim, "_household_cash_balance", return_value=20.0):
            low_liquidity_sim._update_household_reservation_wages()

        with patch.object(high_liquidity_sim, "_family_groups", return_value={0: [high_member]}), patch.object(
            high_liquidity_sim,
            "_household_labor_capacity",
            return_value=1.0,
        ), patch.object(high_liquidity_sim, "_essential_budget", return_value=100.0), patch.object(
            high_liquidity_sim,
            "_household_observed_income",
            return_value=60.0,
        ), patch.object(high_liquidity_sim, "_household_cash_balance", return_value=450.0):
            high_liquidity_sim._update_household_reservation_wages()

        self.assertLess(low_member.reservation_wage, high_member.reservation_wage)

    def test_long_unemployment_and_basic_basket_shortfall_reduce_reservation_wage(self) -> None:
        config = SimulationConfig(periods=1, households=80, firms_per_sector=2, seed=154)
        recent_sim = EconomySimulation(config)
        long_sim = EconomySimulation(config)

        def prepare_member(sim: EconomySimulation, unemployment_duration: int):
            member = sim.households[0]
            member.age_periods = int(30 * sim.config.periods_per_year)
            member.reservation_wage = 100.0
            member.money_trust = 0.5
            member.employed_by = None
            member.unemployment_duration = unemployment_duration
            return member

        recent_member = prepare_member(recent_sim, 0)
        long_member = prepare_member(long_sim, 9)

        for sim, member in ((recent_sim, recent_member), (long_sim, long_member)):
            with patch.object(sim, "_family_groups", return_value={0: [member]}), patch.object(
                sim,
                "_household_labor_capacity",
                return_value=1.0,
            ), patch.object(sim, "_essential_budget", return_value=100.0), patch.object(
                sim,
                "_household_observed_income",
                return_value=20.0,
            ), patch.object(sim, "_household_cash_balance", return_value=40.0):
                sim._update_household_reservation_wages()

        self.assertLess(long_member.reservation_wage, recent_member.reservation_wage)

    def test_reservation_wage_distress_response_is_heterogeneous(self) -> None:
        config = SimulationConfig(periods=1, households=80, firms_per_sector=2, seed=155)
        patient_sim = EconomySimulation(config)
        desperate_sim = EconomySimulation(config)

        def prepare_member(sim: EconomySimulation, sensitivity: float):
            member = sim.households[0]
            member.age_periods = int(30 * sim.config.periods_per_year)
            member.reservation_wage = 100.0
            member.money_trust = 0.5
            member.consumption_impatience = 0.5
            member.price_sensitivity = 1.0
            member.saving_propensity = 0.1
            member.job_change_aversion = 0.5
            member.employed_by = None
            member.unemployment_duration = 9
            member.reservation_wage_distress_sensitivity = sensitivity
            return member

        patient_member = prepare_member(patient_sim, 0.45)
        desperate_member = prepare_member(desperate_sim, 1.75)

        for sim, member in ((patient_sim, patient_member), (desperate_sim, desperate_member)):
            with patch.object(sim, "_family_groups", return_value={0: [member]}), patch.object(
                sim,
                "_household_labor_capacity",
                return_value=1.0,
            ), patch.object(sim, "_essential_budget", return_value=100.0), patch.object(
                sim,
                "_household_observed_income",
                return_value=20.0,
            ), patch.object(sim, "_household_cash_balance", return_value=40.0):
                sim._update_household_reservation_wages()

        self.assertLess(desperate_member.reservation_wage, patient_member.reservation_wage)

    def test_minimum_acceptable_wage_is_lower_for_desperate_unemployed_workers(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=80, firms_per_sector=2, seed=156)
        )
        member = sim.households[0]
        member.age_periods = int(30 * sim.config.periods_per_year)
        member.reservation_wage = 100.0
        member.money_trust = 0.5
        member.consumption_impatience = 0.6
        member.price_sensitivity = 1.0
        member.saving_propensity = 0.1
        member.job_change_aversion = 0.5
        member.unemployment_duration = 8
        member.employed_by = None

        with patch.object(sim, "_family_groups", return_value={0: [member]}), patch.object(
            sim,
            "_household_labor_capacity",
            return_value=1.0,
        ), patch.object(sim, "_essential_budget", return_value=100.0), patch.object(
            sim,
            "_household_observed_income",
            return_value=10.0,
        ), patch.object(sim, "_household_cash_balance", return_value=5.0):
            desperate_floor = sim._household_minimum_acceptable_wage(
                member,
                current_wage=0.0,
                unemployment_rate=0.04,
            )

        with patch.object(sim, "_family_groups", return_value={0: [member]}), patch.object(
            sim,
            "_household_labor_capacity",
            return_value=1.0,
        ), patch.object(sim, "_essential_budget", return_value=100.0), patch.object(
            sim,
            "_household_observed_income",
            return_value=10.0,
        ), patch.object(sim, "_household_cash_balance", return_value=300.0):
            cushioned_floor = sim._household_minimum_acceptable_wage(
                member,
                current_wage=0.0,
                unemployment_rate=0.04,
            )

        self.assertLess(desperate_floor, cushioned_floor)

        member.employed_by = 1
        member.contract_wage = 12.0
        with patch.object(sim, "_family_groups", return_value={0: [member]}), patch.object(
            sim,
            "_household_labor_capacity",
            return_value=1.0,
        ), patch.object(sim, "_essential_budget", return_value=100.0), patch.object(
            sim,
            "_household_observed_income",
            return_value=10.0,
        ), patch.object(sim, "_household_cash_balance", return_value=5.0):
            employed_floor = sim._household_minimum_acceptable_wage(
                member,
                current_wage=12.0,
                unemployment_rate=0.04,
            )

        self.assertGreater(employed_floor, desperate_floor)

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

    def test_startups_begin_with_liquid_buffer_against_assets(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=43)
        )

        for firm in sim.firms:
            if not firm.active or firm.sector == "public_administration":
                continue
            spec = SECTOR_BY_KEY[firm.sector]
            productive_assets = firm.capital + firm.inventory * max(0.1, spec.base_price)
            if productive_assets <= 0.0:
                continue
            self.assertGreaterEqual(
                firm.cash,
                productive_assets * sim.config.startup_liquid_asset_buffer_share,
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

    def test_prior_income_signal_survives_period_reset_for_credit_and_wage_updates(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=177)
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
        covered_spouse.last_income = 0.0
        employed_adult.savings = 0.0
        covered_spouse.savings = 0.0

        for household in sim.households:
            if household.id not in {employed_adult.id, covered_spouse.id}:
                household.alive = False

        sim._refresh_period_household_caches()
        sim._refresh_family_links()
        sim._refresh_period_family_cache()
        sim._reset_household_labor_state()
        sim._refresh_period_household_caches()
        sim._refresh_family_links()
        sim._refresh_period_family_cache()

        projected_income = sim._projected_family_labor_income([employed_adult, covered_spouse])
        participant_ids = sim._labor_force_participant_ids()

        self.assertEqual(employed_adult.previous_income, 5000.0)
        self.assertEqual(sim._household_observed_income(employed_adult), 5000.0)
        self.assertGreater(projected_income, 0.0)
        self.assertIn(employed_adult.id, participant_ids)

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

    def test_match_labor_records_best_underpaid_offer_rejection(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=82)
        )
        firms = [firm for firm in sim.firms if firm.active and firm.sector == "food"][:2]
        high_wage_firm, low_wage_firm = sorted(firms, key=lambda firm: firm.wage_offer, reverse=True)

        for firm in sim.firms:
            if firm not in (high_wage_firm, low_wage_firm):
                firm.active = False
                firm.workers = []

        worker = next(
            household
            for household in sim.households
            if sim.config.entry_age_years <= sim._household_age_years(household) < sim.config.senior_age_years
        )
        worker.employed_by = None

        high_wage_firm.desired_workers = 1
        high_wage_firm.workers = []
        high_wage_firm.wage_offer = 30.0
        high_wage_firm.last_profit = 12.0
        high_wage_firm.last_revenue = 50.0
        high_wage_firm.cash = 200.0

        low_wage_firm.desired_workers = 1
        low_wage_firm.workers = []
        low_wage_firm.wage_offer = 20.0
        low_wage_firm.last_profit = 12.0
        low_wage_firm.last_revenue = 50.0
        low_wage_firm.cash = 200.0
        worker.reservation_wage = max(high_wage_firm.wage_offer, low_wage_firm.wage_offer) + 5.0

        sim._eligible_households = lambda: [worker]  # type: ignore[method-assign]
        with patch.object(sim, "_worker_effective_labor_for_sector", return_value=1.0):
            sim._match_labor()

        self.assertEqual(high_wage_firm.labor_offer_rejections, 1)
        self.assertEqual(low_wage_firm.labor_offer_rejections, 0)
        self.assertAlmostEqual(high_wage_firm.labor_offer_rejection_wage_floor, worker.reservation_wage)

    def test_match_labor_negotiates_between_worker_expectation_and_firm_offer(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=1, seed=83)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        for other_firm in sim.firms:
            if other_firm.id != firm.id:
                other_firm.active = False
                other_firm.workers = []

        worker = next(
            household
            for household in sim.households
            if sim.config.entry_age_years <= sim._household_age_years(household) < sim.config.senior_age_years
        )
        worker.employed_by = None
        worker.reservation_wage = 8.0
        worker.savings = 10.0
        worker.unemployment_duration = 0
        firm.desired_workers = 1
        firm.workers = []
        firm.wage_offer = 12.0
        firm.price = 20.0
        firm.input_cost_per_unit = 1.0
        firm.transport_cost_per_unit = 1.0
        firm.productivity = 1.0
        firm.technology = 1.0
        firm.cash = 500.0
        firm.fixed_overhead = 5.0
        firm.last_profit = 50.0
        firm.last_revenue = 100.0

        sim._eligible_households = lambda: [worker]  # type: ignore[method-assign]
        with patch.object(sim, "_worker_effective_labor_for_sector", return_value=1.0), patch.object(
            sim,
            "_household_minimum_acceptable_wage",
            return_value=8.0,
        ):
            sim._match_labor()

        self.assertEqual(worker.employed_by, firm.id)
        self.assertGreaterEqual(worker.contract_wage, 8.0)
        self.assertLess(worker.contract_wage, firm.wage_offer)
        self.assertAlmostEqual(worker.contract_wage, 10.0, delta=0.75)

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

    def test_wage_taxes_generate_labor_and_payroll_revenue(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=3,
                households=280,
                firms_per_sector=4,
                seed=71,
                government_labor_tax_rate_low=0.18,
                government_labor_tax_rate_mid=0.22,
                government_labor_tax_rate_high=0.26,
                government_labor_tax_bracket_low=0.8,
                government_labor_tax_bracket_high=1.2,
                government_payroll_tax_rate=0.12,
            )
        )

        for _ in range(3):
            snapshot = sim.step()

        self.assertGreater(snapshot.government_labor_tax_revenue, 0.0)
        self.assertGreater(snapshot.government_payroll_tax_revenue, 0.0)
        self.assertGreaterEqual(
            snapshot.government_tax_revenue,
            snapshot.government_labor_tax_revenue + snapshot.government_payroll_tax_revenue,
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

    def test_expansion_credit_need_falls_when_rates_are_high_and_macro_is_unstable(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=65)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        firm.last_revenue = 220.0
        firm.last_sales = 210.0
        firm.last_expected_sales = 190.0
        firm.last_production = 185.0
        firm.last_wage_bill = 70.0
        firm.last_input_cost = 25.0
        firm.last_transport_cost = 10.0
        firm.last_fixed_overhead = 12.0
        firm.capital = 140.0
        firm.cash = 20.0
        firm.loan_balance = 0.0
        firm.loss_streak = 0
        firm.expansion_credit_appetite = 1.35
        firm.stability_sensitivity = 1.25
        firm.investment_animal_spirits = 1.20

        stable_history = [
            SimpleNamespace(
                price_index=1.00,
                unemployment_rate=0.06,
                essential_fulfillment_rate=0.96,
                bank_undercapitalized_share=0.0,
                recession_intensity=0.0,
            ),
            SimpleNamespace(
                price_index=1.01,
                unemployment_rate=0.05,
                essential_fulfillment_rate=0.98,
                bank_undercapitalized_share=0.0,
                recession_intensity=0.0,
            ),
        ]
        unstable_history = [
            SimpleNamespace(
                price_index=1.00,
                unemployment_rate=0.10,
                essential_fulfillment_rate=0.90,
                bank_undercapitalized_share=0.08,
                recession_intensity=0.08,
            ),
            SimpleNamespace(
                price_index=1.08,
                unemployment_rate=0.16,
                essential_fulfillment_rate=0.78,
                bank_undercapitalized_share=0.30,
                recession_intensity=0.30,
            ),
        ]

        sim.history = stable_history
        with patch.object(sim, "_firm_revealed_growth_pressure", return_value=1.0), patch.object(
            sim,
            "_firm_recent_sell_through",
            return_value=1.05,
        ), patch.object(sim, "_firm_forecast_uncertainty", return_value=0.10), patch.object(
            sim,
            "_loan_rate_for_firm",
            return_value=0.03,
        ):
            stable_need = sim._estimate_firm_expansion_credit_need(firm)

        sim.history = unstable_history
        with patch.object(sim, "_firm_revealed_growth_pressure", return_value=1.0), patch.object(
            sim,
            "_firm_recent_sell_through",
            return_value=1.05,
        ), patch.object(sim, "_firm_forecast_uncertainty", return_value=0.10), patch.object(
            sim,
            "_loan_rate_for_firm",
            return_value=0.09,
        ):
            unstable_need = sim._estimate_firm_expansion_credit_need(firm)

        self.assertGreater(stable_need, 0.0)
        self.assertLess(unstable_need, stable_need)

    def test_investment_confidence_is_heterogeneous_under_same_macro_shock(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=66)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        firm.cash = 80.0
        firm.last_wage_bill = 40.0
        firm.last_input_cost = 15.0
        firm.last_transport_cost = 6.0
        firm.last_fixed_overhead = 8.0
        firm.last_revenue = 140.0
        firm.last_profit = 18.0
        firm.investment_animal_spirits = 1.10

        with patch.object(sim, "_firm_forecast_uncertainty", return_value=0.15):
            firm.stability_sensitivity = 0.70
            lower_sensitivity = sim._firm_investment_confidence(firm, macro_stability=0.55)
            firm.stability_sensitivity = 1.60
            higher_sensitivity = sim._firm_investment_confidence(firm, macro_stability=0.55)

        self.assertGreater(lower_sensitivity, higher_sensitivity)

    def test_investment_knowledge_multiplier_rises_with_university_stock_but_stays_bounded(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=120, firms_per_sector=2, seed=68)
        )
        for household in sim.households:
            household.school_years_completed = 0.0
            household.university_years_completed = 0.0

        low_knowledge = sim._investment_knowledge_multiplier()

        adults = [
            household
            for household in sim.households
            if sim._household_age_years(household) >= sim.config.entry_age_years
        ]
        for household in adults[: max(1, len(adults) // 2)]:
            household.school_years_completed = sim.config.school_years_required
            household.university_years_completed = sim.config.university_years_required

        high_knowledge = sim._investment_knowledge_multiplier()

        self.assertGreater(high_knowledge, low_knowledge)
        self.assertGreaterEqual(low_knowledge, sim.config.firm_investment_knowledge_floor)
        self.assertLessEqual(high_knowledge, sim.config.firm_investment_knowledge_ceiling)

    def test_government_infrastructure_investment_builds_public_capital(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=120,
                firms_per_sector=2,
                seed=67,
                government_infrastructure_budget_share=0.20,
                government_spending_efficiency=1.0,
            )
        )
        sim.government.treasury_cash = 100.0

        spent = sim._apply_government_infrastructure_investment()

        self.assertAlmostEqual(spent, 20.0, places=6)
        self.assertAlmostEqual(sim._period_government_infrastructure_spending, 20.0, places=6)
        self.assertAlmostEqual(sim.government.public_capital_stock, 20.0, places=6)
        self.assertAlmostEqual(sim._period_government_public_capital_formation, 20.0, places=6)
        self.assertGreater(sim._public_infrastructure_productivity_multiplier(), 1.0)
        self.assertLess(sim._public_infrastructure_transport_cost_multiplier(), 1.0)

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

    def test_sales_anchor_caps_single_firm_capture_to_operational_scale(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=207)
        )
        sector_key = "leisure"
        target_firm = next(firm for firm in sim.firms if firm.active and firm.sector == sector_key)
        for firm in sim.firms:
            if firm.sector == sector_key and firm.id != target_firm.id:
                firm.active = False

        target_firm.last_worker_count = 12
        target_firm.last_sales = 40.0
        target_firm.last_expected_sales = 5000.0
        target_firm.last_production = 48.0
        target_firm.sales_history = [42.0, 41.0, 39.0, 40.0, 38.0, 40.0]
        target_firm.cash = 220.0
        target_firm.wage_offer = 10.0
        target_firm.fixed_overhead = 35.0
        target_firm.last_wage_bill = 120.0
        target_firm.last_input_cost = 25.0
        target_firm.last_transport_cost = 10.0
        target_firm.last_interest_cost = 0.0
        target_firm.forecast_error_belief = 0.9
        target_firm.market_share_ambition = 1.1

        sector_total_demand = 10000.0
        sales_anchor = sim._sales_anchor(target_firm, sector_total_demand, market_share=1.0)
        effective_supply = sim._firm_effective_supply_signal(target_firm, sector_total_demand)

        self.assertGreater(sales_anchor, target_firm.last_sales)
        self.assertLess(sales_anchor, 150.0)
        self.assertLess(effective_supply, 150.0)
        self.assertLess(effective_supply, 0.10 * target_firm.last_expected_sales)

    def test_sales_anchor_responds_to_observed_demand_without_stockouts(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=208)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        firm.last_sales = 40.0
        firm.last_expected_sales = 40.0
        firm.last_observed_demand = 40.0
        firm.last_stockout_rejections = 0.0
        firm.last_competitive_demand_rejections = 0.0
        firm.last_capacity_shortage_rejections = 0.0
        firm.sales_history = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
        firm.expected_sales_history = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
        firm.observed_demand_history = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]

        base_anchor = sim._sales_anchor(firm, 800.0, market_share=0.25)

        firm.last_observed_demand = 80.0
        firm.observed_demand_history = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
        observed_anchor = sim._sales_anchor(firm, 800.0, market_share=0.25)

        self.assertGreater(observed_anchor, base_anchor)
        self.assertGreater(observed_anchor, firm.last_sales)

    def test_sold_out_firm_reclassifies_missed_preferred_sales_as_stockout(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=218)
        )
        target_firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        for firm in sim.firms:
            if firm.id != target_firm.id:
                firm.active = False
                firm.workers.clear()

        target_firm.sales_this_period = 100.0
        target_firm.stockout_rejections_this_period = 0.0
        target_firm.competitive_demand_rejections_this_period = 80.0
        target_firm.capacity_shortage_rejections_this_period = 0.0
        target_firm.inventory = 0.0
        target_firm.target_inventory = 20.0
        target_firm.last_production = 100.0
        target_firm.last_expected_sales = 90.0
        target_firm.production_history = [100.0, 100.0, 100.0, 100.0]
        target_firm.observed_demand_history = [100.0, 100.0, 100.0, 100.0]
        target_firm.sales_history = [100.0, 100.0, 100.0, 100.0]
        target_firm.expected_sales_history = [90.0, 90.0, 90.0, 90.0]
        target_firm.cash = 800.0
        target_firm.last_wage_bill = 80.0
        target_firm.last_input_cost = 20.0
        target_firm.last_transport_cost = 5.0
        target_firm.last_fixed_overhead = 15.0
        target_firm.last_capital_charge = 2.0
        target_firm.last_severance_cost = 0.0
        sim._cash_before_sales[target_firm.id] = 300.0

        sim._settle_firms()

        self.assertGreater(target_firm.last_stockout_rejections, 0.0)
        self.assertGreater(target_firm.last_stockout_pressure, 0.0)
        self.assertGreaterEqual(target_firm.last_observed_demand, target_firm.last_sales + target_firm.last_stockout_rejections)

    def test_sold_out_pressure_raises_sales_anchor_and_capital_budget(self) -> None:
        base_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=219)
        )
        shortage_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=219)
        )
        base_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == "food")
        shortage_firm = next(firm for firm in shortage_sim.firms if firm.active and firm.sector == "food")

        for firm in (base_firm, shortage_firm):
            firm.inventory = 0.0
            firm.target_inventory = 16.0
            firm.last_sales = 80.0
            firm.last_expected_sales = 80.0
            firm.last_production = 80.0
            firm.sales_history = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
            firm.expected_sales_history = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
            firm.production_history = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
            firm.cash = 900.0
            firm.capital = 75.0
            firm.price = 11.0
            firm.last_revenue = 880.0
            firm.last_profit = 90.0
            firm.last_wage_bill = 120.0
            firm.last_input_cost = 30.0
            firm.last_transport_cost = 8.0
            firm.last_fixed_overhead = 25.0
            firm.last_interest_cost = 0.0
            firm.market_share_ambition = 1.10

        base_firm.last_observed_demand = 80.0
        base_firm.observed_demand_history = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
        shortage_firm.last_observed_demand = 150.0
        shortage_firm.last_competitive_demand_rejections = 70.0
        shortage_firm.observed_demand_history = [150.0, 150.0, 150.0, 150.0, 150.0, 150.0]

        base_anchor = base_sim._sales_anchor(base_firm, 900.0, market_share=0.25)
        shortage_anchor = shortage_sim._sales_anchor(shortage_firm, 900.0, market_share=0.25)
        base_budget = base_sim._firm_desired_capital_goods_budget(base_firm)
        shortage_budget = shortage_sim._firm_desired_capital_goods_budget(shortage_firm)

        self.assertGreater(shortage_anchor, base_anchor)
        self.assertGreater(shortage_budget, base_budget)

    def test_sold_out_firm_expected_sales_cannot_collapse_below_revealed_sales(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=221)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        firm.inventory = 0.0
        firm.target_inventory = 20.0
        firm.last_sales = 80.0
        firm.last_expected_sales = 0.0
        firm.last_production = 80.0
        firm.sales_history = [80.0, 80.0, 80.0, 80.0]
        firm.expected_sales_history = [0.0, 0.0, 0.0, 0.0]
        firm.production_history = [80.0, 80.0, 80.0, 80.0]
        firm.observed_demand_history = [80.0, 80.0, 80.0, 80.0]

        adjusted_expected_sales = sim._firm_revealed_expected_sales_floor(firm, 0.0)

        self.assertGreater(adjusted_expected_sales, firm.last_sales)

    def test_manufacturing_sellout_gets_stronger_expected_sales_floor(self) -> None:
        base_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=222)
        )
        food_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == "food")
        manufactured_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == "manufactured")

        for firm in (food_firm, manufactured_firm):
            firm.inventory = 0.0
            firm.target_inventory = 20.0
            firm.last_sales = 1.0
            firm.last_expected_sales = 0.0
            firm.last_production = 1.0
            firm.sales_history = [1.0, 1.0, 1.0, 1.0]
            firm.expected_sales_history = [0.0, 0.0, 0.0, 0.0]
            firm.production_history = [1.0, 1.0, 1.0, 1.0]
            firm.observed_demand_history = [1.0, 1.0, 1.0, 1.0]

        food_floor = base_sim._firm_revealed_expected_sales_floor(food_firm, 0.0)
        manufactured_floor = base_sim._firm_revealed_expected_sales_floor(manufactured_firm, 0.0)

        self.assertGreater(manufactured_floor, food_floor)

    def test_firm_audit_records_no_machinery_when_investment_market_is_empty(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=3, seed=220)
        )
        target_firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        for firm in sim.firms:
            if firm.sector == "manufactured":
                firm.active = False
                firm.workers.clear()

        target_firm.cash = 1_200.0
        target_firm.sales_this_period = 100.0
        target_firm.last_expected_sales = 95.0
        target_firm.last_production = 100.0
        target_firm.inventory = 0.0
        target_firm.target_inventory = 20.0
        target_firm.last_wage_bill = 80.0
        target_firm.last_input_cost = 10.0
        target_firm.last_transport_cost = 5.0
        target_firm.last_fixed_overhead = 10.0
        target_firm.last_capital_charge = 2.0
        target_firm.last_interest_cost = 0.0
        target_firm.sales_history = [100.0, 100.0, 100.0, 100.0]
        target_firm.expected_sales_history = [95.0, 95.0, 95.0, 95.0]
        target_firm.observed_demand_history = [130.0, 130.0, 130.0, 130.0]
        target_firm.production_history = [100.0, 100.0, 100.0, 100.0]
        sim._cash_before_sales[target_firm.id] = 200.0

        sim._settle_firms()

        self.assertEqual(target_firm.last_industrial_investment_spending, 0.0)
        self.assertGreater(target_firm.last_unfilled_investment_budget, 0.0)
        self.assertEqual(target_firm.last_investment_decision_reason, "no_habia_maquinaria_disponible")

    def test_contraction_is_frozen_when_observed_demand_is_still_strong(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=209)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        firm.workers = [
            household.id
            for household in sim.households
            if household.alive and sim._household_labor_capacity(household) > 0.0
        ][:12]
        firm.age = 4
        firm.cash = 800.0
        firm.capital = 1.0
        firm.productivity = 2.0
        firm.technology = 1.0
        firm.price = 11.0
        firm.last_profit = 18.0
        firm.last_revenue = 220.0
        firm.last_sales = 40.0
        firm.last_expected_sales = 40.0
        firm.last_observed_demand = 78.0
        firm.last_production = 20.0
        firm.last_wage_bill = 120.0
        firm.last_input_cost = 15.0
        firm.last_transport_cost = 10.0
        firm.last_fixed_overhead = 20.0
        firm.last_interest_cost = 0.0
        firm.loss_streak = 0
        firm.sales_history = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
        firm.expected_sales_history = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
        firm.observed_demand_history = [78.0, 78.0, 78.0, 78.0, 78.0, 78.0]

        retained_workers = sim._constrained_headcount_after_contraction(firm, 7, len(firm.workers))

        self.assertEqual(retained_workers, len(firm.workers))

    def test_baseline_demand_stays_closer_to_structure_during_learning_warmup(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=17)
        )
        sector_key = "manufactured"
        structural_demand = sim._structural_sector_demand(sector_key)
        observed_demand = structural_demand * 0.20
        sim._last_sector_sales_units[sector_key] = observed_demand
        sim._last_sector_revealed_unmet_units[sector_key] = 0.0

        sim.period = 2
        early_baseline = sim._baseline_demand(sector_key)

        sim.period = sim.config.firm_learning_warmup_periods + 24
        late_baseline = sim._baseline_demand(sector_key)

        self.assertLess(abs(early_baseline - structural_demand), abs(late_baseline - structural_demand))
        self.assertGreater(early_baseline, late_baseline)

    def test_observed_sector_demand_signal_uses_revealed_unmet_demand(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=19)
        )
        sim._last_sector_sales_units["food"] = 40.0
        sim._last_sector_revealed_unmet_units["food"] = 20.0
        sim._last_sector_sales_units["manufactured"] = 40.0
        sim._last_sector_revealed_unmet_units["manufactured"] = 20.0

        food_signal = sim._observed_sector_demand_signal("food")
        manufactured_signal = sim._observed_sector_demand_signal("manufactured")

        self.assertAlmostEqual(food_signal, 58.0, places=6)
        self.assertAlmostEqual(manufactured_signal, 51.0, places=6)
        self.assertGreater(food_signal, manufactured_signal)

    def test_firm_market_memory_uses_three_year_moving_window(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=21)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        old_regime = [500.0] * 12
        recent_regime = [100.0] * 36
        firm.sales_history = old_regime + recent_regime

        smoothed_reference = sim._smoothed_sales_reference(firm)
        retained_history = sim._firm_memory_slice(firm)

        self.assertEqual(len(retained_history), 36)
        self.assertEqual(retained_history, recent_regime)
        self.assertAlmostEqual(smoothed_reference, 100.0, places=6)

    def test_revealed_shortage_pushes_firm_hiring_and_capacity(self) -> None:
        base_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=29)
        )
        shortage_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=29)
        )
        sector_key = "food"
        target_base_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == sector_key)
        target_shortage_firm = next(firm for firm in shortage_sim.firms if firm.active and firm.sector == sector_key)

        for sim, target_firm in ((base_sim, target_base_firm), (shortage_sim, target_shortage_firm)):
            sim.period = 30
            for firm in sim.firms:
                if firm.sector == sector_key and firm.id != target_firm.id:
                    firm.active = False
            target_firm.last_worker_count = 10
            target_firm.sales_this_period = 0.0
            target_firm.last_sales = 110.0
            target_firm.last_expected_sales = 110.0
            target_firm.last_production = 118.0
            target_firm.sales_history = [112.0, 108.0, 111.0, 109.0, 113.0, 110.0]
            target_firm.cash = 420.0
            target_firm.wage_offer = 10.0
            target_firm.price = 12.0
            target_firm.fixed_overhead = 28.0
            target_firm.inventory = 3.0
            target_firm.target_inventory = 10.0
            target_firm.last_wage_bill = 100.0
            target_firm.last_input_cost = 16.0
            target_firm.last_transport_cost = 8.0
            target_firm.last_interest_cost = 0.0
            target_firm.last_profit = 32.0
            target_firm.last_revenue = 180.0
            target_firm.employment_inertia = 0.0
            target_firm.market_share_ambition = 1.1

        base_sim._last_sector_sales_units[sector_key] = 110.0
        shortage_sim._last_sector_sales_units[sector_key] = 110.0
        base_sim._last_sector_revealed_unmet_units[sector_key] = 0.0
        shortage_sim._last_sector_revealed_unmet_units[sector_key] = 90.0

        base_sim._update_firm_policies(last_unemployment=0.08)
        shortage_sim._update_firm_policies(last_unemployment=0.08)

        self.assertGreater(target_shortage_firm.desired_workers, target_base_firm.desired_workers)
        self.assertGreater(target_shortage_firm.target_inventory, target_base_firm.target_inventory)

    def test_revealed_shortage_raises_reinvestment_after_sales(self) -> None:
        base_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=31)
        )
        shortage_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=31)
        )
        sector_key = "food"
        target_base_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == sector_key)
        target_shortage_firm = next(firm for firm in shortage_sim.firms if firm.active and firm.sector == sector_key)
        base_supplier_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == "manufactured")
        shortage_supplier_firm = next(firm for firm in shortage_sim.firms if firm.active and firm.sector == "manufactured")

        for sim, target_firm, supplier_firm in (
            (base_sim, target_base_firm, base_supplier_firm),
            (shortage_sim, target_shortage_firm, shortage_supplier_firm),
        ):
            for firm in sim.firms:
                if firm.id not in (target_firm.id, supplier_firm.id):
                    firm.active = False
                    firm.workers.clear()
            target_firm.sales_this_period = 150.0
            target_firm.last_worker_count = 12
            target_firm.last_sales = 130.0
            target_firm.last_expected_sales = 125.0
            target_firm.last_production = 136.0
            target_firm.sales_history = [118.0, 122.0, 126.0, 129.0, 131.0, 130.0]
            target_firm.cash = 520.0
            target_firm.capital = 90.0
            target_firm.price = 12.0
            target_firm.wage_offer = 10.0
            target_firm.last_wage_bill = 120.0
            target_firm.last_input_cost = 22.0
            target_firm.last_transport_cost = 9.0
            target_firm.last_fixed_overhead = 28.0
            target_firm.last_capital_charge = 6.0
            target_firm.last_interest_cost = 0.0
            target_firm.last_profit = 0.0
            target_firm.market_share_ambition = 1.15
            sim._cash_before_sales[target_firm.id] = 210.0

            supplier_firm.cash = 600.0
            supplier_firm.inventory = max(120.0, supplier_firm.inventory)
            supplier_firm.price = 18.0
            supplier_firm.last_sales = max(60.0, supplier_firm.last_sales)
            supplier_firm.last_expected_sales = max(60.0, supplier_firm.last_expected_sales)
            supplier_firm.sales_history = [60.0, 62.0, 64.0, 66.0, 68.0, 70.0]
            supplier_firm.expected_sales_history = [60.0, 62.0, 64.0, 66.0, 68.0, 70.0]
            supplier_firm.observed_demand_history = [60.0, 62.0, 64.0, 66.0, 68.0, 70.0]

        base_sim._period_sector_sales_units[sector_key] = 150.0
        shortage_sim._period_sector_sales_units[sector_key] = 150.0
        base_sim._period_sector_revealed_unmet_units[sector_key] = 0.0
        shortage_sim._period_sector_revealed_unmet_units[sector_key] = 120.0

        base_sim._settle_firms()
        shortage_sim._settle_firms()

        self.assertGreater(shortage_sim._period_investment_spending, base_sim._period_investment_spending)
        self.assertGreater(target_shortage_firm.last_technology_investment, target_base_firm.last_technology_investment)

    def test_observed_demand_raises_firm_expansion_credit_need(self) -> None:
        base_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=33)
        )
        observed_sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=33)
        )
        sector_key = "food"
        target_base_firm = next(firm for firm in base_sim.firms if firm.active and firm.sector == sector_key)
        target_observed_firm = next(firm for firm in observed_sim.firms if firm.active and firm.sector == sector_key)

        for sim, target_firm in ((base_sim, target_base_firm), (observed_sim, target_observed_firm)):
            for firm in sim.firms:
                if firm.id != target_firm.id:
                    firm.active = False
                    firm.workers.clear()
            target_firm.sales_this_period = 90.0
            target_firm.last_worker_count = 12
            target_firm.last_sales = 90.0
            target_firm.last_expected_sales = 90.0
            target_firm.last_observed_demand = 90.0
            target_firm.last_production = 90.0
            target_firm.sales_history = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
            target_firm.expected_sales_history = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
            target_firm.observed_demand_history = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
            target_firm.cash = 520.0
            target_firm.capital = 80.0
            target_firm.price = 12.0
            target_firm.wage_offer = 10.0
            target_firm.last_wage_bill = 120.0
            target_firm.last_input_cost = 22.0
            target_firm.last_transport_cost = 9.0
            target_firm.last_fixed_overhead = 28.0
            target_firm.last_capital_charge = 6.0
            target_firm.last_interest_cost = 0.0
            target_firm.last_profit = 24.0
            target_firm.last_revenue = 180.0
            target_firm.market_share_ambition = 1.15

        observed_target = target_observed_firm
        observed_target.last_observed_demand = 150.0
        observed_target.observed_demand_history = [150.0, 150.0, 150.0, 150.0, 150.0, 150.0]
        observed_target.last_competitive_demand_rejections = 20.0

        base_need = base_sim._estimate_firm_expansion_credit_need(target_base_firm)
        observed_need = observed_sim._estimate_firm_expansion_credit_need(observed_target)

        self.assertGreater(observed_need, base_need)

    def test_revealed_shortage_can_trigger_extra_entry(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=2, seed=37)
        )
        sector_key = "food"
        active_food_firms = [firm for firm in sim.firms if firm.active and firm.sector == sector_key]
        incumbent = active_food_firms[0]
        for firm in active_food_firms[1:]:
            firm.active = False
            firm.workers.clear()
        incumbent.last_market_share = 1.0
        incumbent.last_sales = 138.0
        incumbent.last_expected_sales = 138.0
        incumbent.last_production = 138.0
        incumbent.sales_history = [138.0, 138.0, 137.0, 139.0, 138.0, 138.0]
        incumbent.last_worker_count = 12
        incumbent.wage_offer = 10.0
        incumbent.price = 12.0
        incumbent.cash = 420.0
        incumbent.last_wage_bill = 120.0
        incumbent.last_input_cost = 16.0
        incumbent.last_transport_cost = 8.0
        incumbent.last_fixed_overhead = 20.0
        incumbent.last_interest_cost = 0.0
        incumbent.last_profit = 24.0

        for owner in sim.entrepreneurs:
            owner.wealth = max(owner.wealth, 600.0)
            owner.vault_cash = max(owner.vault_cash, 60.0)

        sim._period_sector_sales_units[sector_key] = 100.0
        sim._period_sector_revealed_unmet_units[sector_key] = 20.0

        initial_active_count = len([firm for firm in sim.firms if firm.active and firm.sector == sector_key])
        sim._attempt_endogenous_sector_entry()
        final_active_count = len([firm for firm in sim.firms if firm.active and firm.sector == sector_key])

        self.assertGreater(final_active_count, initial_active_count)

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

    def test_nonessential_target_margin_falls_when_sales_are_lost(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=3, startup_grace_periods=0, seed=44)
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "clothing")
        spec = SECTOR_BY_KEY[firm.sector]

        firm.price = 15.0
        firm.last_unit_cost = 10.0
        firm.last_sales = 12.0
        firm.last_expected_sales = 60.0
        firm.last_competitive_demand_rejections = 30.0
        firm.last_observed_demand = 45.0
        firm.sales_history = [58.0, 46.0, 28.0, 12.0]
        firm.expected_sales_history = [60.0, 58.0, 55.0, 60.0]
        firm.volume_preference = 1.35
        firm.market_share_ambition = 1.20
        firm.price_aggressiveness = 0.85

        baseline_margin = spec.markup * max(0.70, min(1.30, 1.0 - 0.18 * (firm.markup_tolerance - 1.0)))
        adjusted_margin = sim._target_margin_for_firm(firm, spec, average_unit_cost=10.0)

        self.assertLess(adjusted_margin, baseline_margin)
        self.assertLess(adjusted_margin, spec.markup)

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
            real_startup_capacity = sum(
                sim._firm_effective_productivity(firm) * sim._firm_effective_labor_units_for_production(firm)
                for firm in sim.firms
                if firm.active and firm.sector == sector_key
            )
            self.assertGreaterEqual(startup_capacity, structural_need)
            self.assertGreaterEqual(real_startup_capacity, structural_need)

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

    def test_startup_essential_supply_buffer_stays_close_to_base_need(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=300, firms_per_sector=4, seed=46)
        )

        self.assertAlmostEqual(
            sim._startup_essential_supply_multiplier("food"),
            sim.config.startup_essential_supply_buffer,
            places=6,
        )
        self.assertLess(sim._startup_essential_supply_multiplier("food"), 2.0)

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
            self.assertLessEqual(sector_expected_sales, startup_target + 1e-6)

    def test_manufactured_goods_are_heterogeneous_capital_tools(self) -> None:
        spec = SECTOR_BY_KEY["manufactured"]
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=180, firms_per_sector=2, seed=47)
        )
        buyer = next(firm for firm in sim.firms if firm.active and firm.sector == "food")
        suppliers = [firm for firm in sim.firms if firm.active and firm.sector == "manufactured"]
        self.assertGreaterEqual(len(suppliers), 2)
        weaker_supplier, stronger_supplier = suppliers[:2]

        for firm in suppliers:
            firm.inventory = 4.0
            firm.capital_goods_sales_this_period = 0.0
            firm.workers.clear()

        weaker_supplier.price = 120.0
        weaker_supplier.capital = 10.0
        weaker_supplier.technology = 0.82
        stronger_supplier.price = 180.0
        stronger_supplier.capital = 260.0
        stronger_supplier.technology = 1.8

        educated_workers = [
            household
            for household in sim.households
            if household.alive and sim._household_labor_capacity(household) > 0.0
        ][:8]
        for household in educated_workers:
            household.school_years_completed = sim.config.school_years_required
            household.university_years_completed = sim.config.university_years_required
            household.employed_by = stronger_supplier.id
            stronger_supplier.workers.append(household.id)

        spent, units, quality, capital_service = sim._purchase_industrial_investment_goods(
            buyer,
            spec.base_price * 6.0,
        )

        self.assertAlmostEqual(spec.base_productivity, 0.30, places=6)
        self.assertLess(spec.base_price, 200.0)
        self.assertGreater(
            sim._manufacturing_capital_goods_quality(stronger_supplier),
            sim._manufacturing_capital_goods_quality(weaker_supplier),
        )
        self.assertNotEqual(
            sim._manufacturing_capital_goods_labor_required(stronger_supplier),
            sim._manufacturing_capital_goods_labor_required(weaker_supplier),
        )
        self.assertGreaterEqual(
            sim._manufacturing_capital_goods_supported_workers(stronger_supplier),
            sim.config.firm_capital_goods_supported_workers_min,
        )
        self.assertGreaterEqual(units, 1.0)
        self.assertGreater(spent, 0.0)
        self.assertLessEqual(quality, sim.config.firm_capital_goods_quality_ceiling)
        self.assertGreater(capital_service, 0.0)

    def test_manufacturing_capacity_is_three_units_per_ten_workers_until_twenty_workers(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(
                periods=1,
                households=180,
                firms_per_sector=2,
                seed=48,
                government_public_capital_productivity_gain=0.0,
            )
        )
        firm = next(firm for firm in sim.firms if firm.active and firm.sector == "manufactured")
        workers = [
            household
            for household in sim.households
            if household.alive and sim._household_labor_capacity(household) > 0.0
        ][:30]
        firm.workers.clear()
        firm.productivity = SECTOR_BY_KEY["manufactured"].base_productivity
        firm.technology = 1.0

        for household in workers:
            household.employed_by = firm.id
            household.age_periods = int(30 * sim.config.periods_per_year)
            household.health_fragility = 0.0
            household.housing_deprivation_streak = 0
            household.clothing_deprivation_streak = 0
        sim._period_household_labor_capacity_cache.clear()

        firm.workers = [household.id for household in workers[:10]]
        self.assertAlmostEqual(sim._firm_installed_production_capacity_units(firm), 3.0, places=6)

        firm.workers = [household.id for household in workers[:20]]
        self.assertAlmostEqual(sim._firm_installed_production_capacity_units(firm), 6.0, places=6)

        firm.workers = [household.id for household in workers[:30]]
        self.assertAlmostEqual(sim._firm_installed_production_capacity_units(firm), 6.0, places=6)

    def test_competitive_demand_rejections_register_even_without_stockout(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=1, households=160, firms_per_sector=2, seed=49)
        )
        sector_key = "manufactured"
        cheap_firm, expensive_firm = sorted(
            [firm for firm in sim.firms if firm.active and firm.sector == sector_key],
            key=lambda firm: firm.id,
        )

        cheap_firm.price = 5.0
        expensive_firm.price = 6.0
        cheap_firm.inventory = 4.0
        expensive_firm.inventory = 20.0
        cheap_firm.inventory_batches = [4.0]
        expensive_firm.inventory_batches = [20.0]

        sim._refresh_period_sector_caches()
        sim._reset_period_counters()
        sim._cash_before_sales = {firm.id: firm.cash for firm in sim.firms}
        spending_log = {spec.key: 0.0 for spec in SECTOR_SPECS}

        _, bought_total = sim._purchase_from_sector(1.0, sector_key, 10.0, 1_000.0, spending_log)
        sim._settle_firms()

        self.assertEqual(bought_total, 10.0)
        self.assertGreater(cheap_firm.capacity_shortage_rejections_this_period, 0.0)
        self.assertGreater(cheap_firm.last_capacity_shortage_rejections, 0.0)
        self.assertGreater(cheap_firm.last_observed_demand, cheap_firm.last_sales)
        self.assertGreater(sim._firm_capacity_shortage_signal(cheap_firm), 0.0)

        expensive_base = sim._firm_competitive_demand_loss_signal(expensive_firm)
        expensive_firm.price = 8.0
        expensive_higher = sim._firm_competitive_demand_loss_signal(expensive_firm)
        self.assertGreater(expensive_higher, expensive_base)

    def test_firm_snapshot_exposes_installed_production_capacity(self) -> None:
        sim = EconomySimulation(
            SimulationConfig(periods=2, households=180, firms_per_sector=2, seed=53)
        )
        sim.step()

        snapshots = sim._build_firm_period_snapshots()
        food_snapshot = next(
            snapshot
            for snapshot in snapshots
            if snapshot.active and snapshot.sector == "food"
        )

        self.assertGreaterEqual(food_snapshot.installed_production_capacity_units, 0.0)
        self.assertGreaterEqual(food_snapshot.capacity_utilization_rate, 0.0)
        self.assertLessEqual(food_snapshot.capacity_utilization_rate, 1.5)
        self.assertLessEqual(
            food_snapshot.production,
            food_snapshot.installed_production_capacity_units + 1e-6,
        )

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
