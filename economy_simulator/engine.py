from __future__ import annotations

import bisect
import heapq
import math
import random

import numpy as np

from .accelerators import compute_household_baseline_demand_arrays
from .domain import (
    CentralBank,
    CommercialBank,
    DISCRETIONARY_SECTOR_KEYS,
    ESSENTIAL_SECTOR_KEYS,
    SECTOR_BY_KEY,
    SECTOR_SPECS,
    Entrepreneur,
    Firm,
    FirmPeriodSnapshot,
    Government,
    Household,
    PeriodSnapshot,
    SimulationConfig,
    SimulationResult,
)
from .metrics import clamp, gini

PUBLIC_ADMINISTRATION_EMPLOYER_ID = -1
QUALIFIED_SECTOR_KEYS = ("manufactured", "leisure", "school", "university", "public_administration")
ARRAY_BACKED_SECTOR_KEYS = ("food", "housing", "clothing", "manufactured", "leisure")
ARRAY_BACKED_SECTOR_INDEX = {sector_key: index for index, sector_key in enumerate(ARRAY_BACKED_SECTOR_KEYS)}
MERIT_SECTOR_KEYS = ("school", "university")
PURE_DISCRETIONARY_SECTOR_KEYS = ("manufactured", "leisure")


class EconomySimulation:
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.rng = random.Random(self.config.seed)
        self.period = 0
        self._period_active_households_cache: list[Household] | None = None
        self._period_active_firms_by_sector_cache: dict[str, list[Firm]] | None = None
        self._period_ranked_firms_by_sector_cache: dict[str, list[Firm]] | None = None
        self._period_average_sector_price_cache: dict[str, float] | None = None
        self._period_household_age_years_cache: dict[int, tuple[int, float]] = {}
        self._period_household_desired_units_cache: dict[tuple[int, int, str], float] = {}
        self._period_essential_budget_cache: dict[int, tuple[int, float]] = {}
        self._period_household_row_index_cache: dict[int, int] = {}
        self._period_essential_desired_units_matrix: np.ndarray | None = None
        self._period_essential_budget_vector: np.ndarray | None = None
        self._period_household_summary_cache: dict[str, object] | None = None
        self._period_family_groups_cache: dict[int, list[Household]] | None = None
        self._period_family_summary_cache: dict[str, object] | None = None
        self._period_family_resource_coverage_cache: dict[int, float] = {}
        self._period_baseline_demand_cache: dict[tuple[str, bool], float] = {}
        self._period_household_labor_capacity_cache: dict[
            int, tuple[tuple[int, float, int, int], float]
        ] = {}
        self._period_living_wage_anchor_cache: float | None = None
        self.history: list[PeriodSnapshot] = []
        self.firm_history: list[FirmPeriodSnapshot] = []
        self._sector_sales_history = {spec.key: [] for spec in SECTOR_SPECS}
        self._sector_revealed_unmet_history = {spec.key: [] for spec in SECTOR_SPECS}
        self.households = self._build_households()
        self._initialize_household_education()
        self._startup_structural_demand_cache = self._compute_structural_demand_map(
            [household for household in self.households if household.alive]
        )
        startup_population = sum(1 for household in self.households if household.alive)
        self._startup_essential_target_cache = {
            sector_key: max(
                self._startup_structural_demand_cache.get(sector_key, 0.0),
                startup_population * self._essential_basket_share(sector_key),
            )
            * self._startup_essential_supply_multiplier(sector_key)
            for sector_key in ESSENTIAL_SECTOR_KEYS
        }
        self.entrepreneurs = self._build_entrepreneurs()
        self.firms = self._build_firms()
        self.banks = self._build_banks()
        self.bank_by_id = {bank.id: bank for bank in self.banks}
        self.government = Government(
            treasury_cash=0.0,
            bank_id=self.banks[0].id if self.banks else 0,
        )
        self._assign_financial_institutions()
        self._seed_bank_balance_sheets()
        self.government.debt_outstanding = sum(bank.bond_holdings for bank in self.banks)
        initial_monetary_base = self._current_monetary_base()
        self.central_bank = CentralBank(
            money_supply=initial_monetary_base,
            target_money_supply=initial_monetary_base,
            policy_rate=clamp(
                self.config.central_bank_policy_rate_base,
                self.config.central_bank_policy_rate_floor,
                self.config.central_bank_policy_rate_ceiling,
            ),
        )
        self._update_bank_interest_rates()
        self.firm_by_id = {firm.id: firm for firm in self.firms}
        self.firms_by_sector: dict[str, list[Firm]] = {spec.key: [] for spec in SECTOR_SPECS}
        for firm in self.firms:
            self.firms_by_sector[firm.sector].append(firm)
        self._next_household_id = len(self.households)
        self._next_firm_id = len(self.firms)
        self._assign_initial_guardians()
        self._seed_initial_workforce()
        self._startup_structural_demand_cache = None
        self._startup_essential_target_cache = None
        self._startup_goods_monetary_mass = self._current_goods_monetary_mass()
        self._startup_inventory_book_value = self._current_inventory_book_value()

        self._cash_before_sales: dict[int, float] = {}
        self._period_wages = 0.0
        self._period_sales_units = 0.0
        self._period_potential_demand_units = 0.0
        self._period_essential_demand_units = 0.0
        self._period_essential_production_units = 0.0
        self._period_essential_sales_units = 0.0
        self._period_sales_revenue = 0.0
        self._period_production_units = 0.0
        self._period_investment_spending = 0.0
        self._period_startup_fixed_capital_formation = 0.0
        self._period_startup_inventory_investment = 0.0
        self._period_profit = 0.0
        self._pending_sector_payments = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_business_cost_recycled = 0.0
        self._period_business_cost_to_firms = 0.0
        self._period_business_cost_to_households = 0.0
        self._period_business_cost_to_owners = 0.0
        self._period_inheritance_transfers = 0.0
        self._period_bankruptcy_cash_recoveries = 0.0
        self._period_central_bank_issuance = 0.0
        self._period_central_bank_target_money_supply = initial_monetary_base
        self._period_worker_cash_available = 0.0
        self._period_worker_cash_saved = 0.0
        self._period_worker_voluntary_saved = 0.0
        self._period_worker_involuntary_retained = 0.0
        self._period_household_credit_issued = 0.0
        self._period_firm_credit_issued = 0.0
        self._period_firm_expansion_credit_issued = 0.0
        self._period_entrepreneur_spending = 0.0
        self._period_dividends_paid = 0.0
        self._period_government_tax_revenue = 0.0
        self._period_government_labor_tax_revenue = 0.0
        self._period_government_payroll_tax_revenue = 0.0
        self._period_government_corporate_tax_revenue = 0.0
        self._period_government_dividend_tax_revenue = 0.0
        self._period_government_wealth_tax_revenue = 0.0
        self._period_government_transfers = 0.0
        self._period_government_unemployment_support = 0.0
        self._period_government_child_allowance = 0.0
        self._period_government_basic_support = 0.0
        self._period_government_procurement_spending = 0.0
        self._period_government_education_spending = 0.0
        self._period_government_school_spending = 0.0
        self._period_government_university_spending = 0.0
        self._period_government_school_units = 0.0
        self._period_government_university_units = 0.0
        self._period_government_public_administration_spending = 0.0
        self._period_government_infrastructure_spending = 0.0
        self._period_government_public_capital_formation = 0.0
        self._period_government_bond_issuance = 0.0
        self._period_government_deficit = 0.0
        self._period_government_surplus = 0.0
        self._period_recession_flag = 0.0
        self._period_recession_intensity = 0.0
        self._period_government_countercyclical_support_multiplier = 1.0
        self._period_government_countercyclical_procurement_multiplier = 1.0
        self._period_government_countercyclical_spending = 0.0
        self._period_bank_recapitalization = 0.0
        self._period_bank_writeoffs = 0.0
        self._period_bank_loan_restructures = 0
        self._period_household_loan_defaults = 0
        self._period_firm_loan_defaults = 0
        self._period_household_loan_restructures = 0
        self._period_firm_loan_restructures = 0
        self._period_bank_resolution_events = 0
        self._period_bank_undercapitalized_share_signal = 0.0
        self._period_bank_insolvent_share_signal = 0.0
        self._bankruptcies = 0
        self._period_births = 0
        self._period_deaths = 0
        self._last_sector_potential_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._last_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._prior_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._last_sector_budget_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._last_sector_revealed_unmet_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._sector_sales_history = {spec.key: [] for spec in SECTOR_SPECS}
        self._sector_revealed_unmet_history = {spec.key: [] for spec in SECTOR_SPECS}
        self._period_sector_potential_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_budget_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_revealed_unmet_units = {spec.key: 0.0 for spec in SECTOR_SPECS}

    def run(self) -> SimulationResult:
        while self.period < self.config.periods:
            snapshot = self.step()
            if snapshot.population <= 0:
                break
        return SimulationResult(
            config=self.config,
            history=self.history,
            firm_history=self.firm_history,
            households=self.households,
            entrepreneurs=self.entrepreneurs,
            firms=self.firms,
            central_bank=self.central_bank,
            banks=self.banks,
            government=self.government,
        )

    def step(self) -> PeriodSnapshot:
        self.period += 1
        self._reset_period_counters()
        self._refresh_period_household_caches()
        self._refresh_period_sector_caches()
        self._reset_household_labor_state()
        self._age_employment_contracts()
        self._advance_credit_state_timers()
        self._refresh_family_links()
        self._refresh_period_family_cache()
        self._update_household_reservation_wages()

        last_unemployment = self.history[-1].unemployment_rate if self.history else 0.12
        self._update_firm_policies(last_unemployment)
        self._refresh_period_sector_caches()
        self._align_existing_workforce()
        self._match_labor()
        self._apply_central_bank_policy()
        self._accrue_bank_funding_costs()
        self._stabilize_bank_capital_positions()
        self._apply_bank_credit_policy()
        self._manage_public_administration_workforce()
        self._produce_and_pay_wages()
        self._apply_government_household_support()
        self._service_household_loans()
        self._consume_households()
        self._apply_government_essential_procurement()
        self._apply_government_infrastructure_investment()
        self._settle_firms()
        self._finalize_government_period()
        self._resolve_bankruptcy_and_entry()
        self._refresh_period_sector_caches()
        _, _, _, _, _, _, _, _, _, current_unemployment, _ = self._population_metrics()
        self._apply_demography(current_unemployment)
        self._refresh_period_household_caches()
        self._refresh_period_family_cache()
        self._resolve_bank_insolvency()
        self.central_bank.money_supply = self._current_total_liquid_money()

        snapshot = self._build_snapshot()
        self._roll_forward_sector_demand_signals()
        self.history.append(snapshot)
        if self.config.track_firm_history:
            self.firm_history.extend(self._build_firm_period_snapshots())
        return snapshot

    def _reset_period_counters(self) -> None:
        self._cash_before_sales = {}
        self._period_wages = 0.0
        self._period_sales_units = 0.0
        self._period_potential_demand_units = 0.0
        self._period_essential_demand_units = 0.0
        self._period_essential_production_units = 0.0
        self._period_essential_sales_units = 0.0
        self._period_sales_revenue = 0.0
        self._period_production_units = 0.0
        self._period_investment_spending = 0.0
        self._period_startup_fixed_capital_formation = 0.0
        self._period_startup_inventory_investment = 0.0
        self._period_profit = 0.0
        self._pending_sector_payments = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_business_cost_recycled = 0.0
        self._period_business_cost_to_firms = 0.0
        self._period_business_cost_to_households = 0.0
        self._period_business_cost_to_owners = 0.0
        self._period_inheritance_transfers = 0.0
        self._period_bankruptcy_cash_recoveries = 0.0
        self._period_central_bank_issuance = 0.0
        self._period_central_bank_target_money_supply = self.central_bank.money_supply
        self._period_worker_cash_available = 0.0
        self._period_worker_cash_saved = 0.0
        self._period_worker_voluntary_saved = 0.0
        self._period_worker_involuntary_retained = 0.0
        self._period_household_credit_issued = 0.0
        self._period_firm_credit_issued = 0.0
        self._period_firm_expansion_credit_issued = 0.0
        self._period_entrepreneur_spending = 0.0
        self._period_dividends_paid = 0.0
        self._period_government_tax_revenue = 0.0
        self._period_government_labor_tax_revenue = 0.0
        self._period_government_payroll_tax_revenue = 0.0
        self._period_government_corporate_tax_revenue = 0.0
        self._period_government_dividend_tax_revenue = 0.0
        self._period_government_wealth_tax_revenue = 0.0
        self._period_government_transfers = 0.0
        self._period_government_unemployment_support = 0.0
        self._period_government_child_allowance = 0.0
        self._period_government_basic_support = 0.0
        self._period_government_procurement_spending = 0.0
        self._period_government_education_spending = 0.0
        self._period_government_school_spending = 0.0
        self._period_government_university_spending = 0.0
        self._period_government_school_units = 0.0
        self._period_government_university_units = 0.0
        self._period_government_public_administration_spending = 0.0
        self._period_government_infrastructure_spending = 0.0
        self._period_government_public_capital_formation = 0.0
        self._period_government_bond_issuance = 0.0
        self._period_government_deficit = 0.0
        self._period_government_surplus = 0.0
        self._period_recession_flag = 0.0
        self._period_recession_intensity = 0.0
        self._period_government_countercyclical_support_multiplier = 1.0
        self._period_government_countercyclical_procurement_multiplier = 1.0
        self._period_government_countercyclical_spending = 0.0
        self._period_bank_recapitalization = 0.0
        self._period_bank_writeoffs = 0.0
        self._period_bank_loan_restructures = 0
        self._period_household_loan_defaults = 0
        self._period_firm_loan_defaults = 0
        self._period_household_loan_restructures = 0
        self._period_firm_loan_restructures = 0
        self._period_bank_resolution_events = 0
        self._period_bank_undercapitalized_share_signal = 0.0
        self._period_bank_insolvent_share_signal = 0.0
        self._bankruptcies = 0
        self._period_births = 0
        self._period_deaths = 0
        self._period_sector_potential_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_budget_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_revealed_unmet_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_household_desired_units_cache = {}
        self._period_essential_budget_cache = {}
        self.government.tax_revenue_this_period = 0.0
        self.government.labor_tax_revenue = 0.0
        self.government.payroll_tax_revenue = 0.0
        self.government.corporate_tax_revenue = 0.0
        self.government.dividend_tax_revenue = 0.0
        self.government.wealth_tax_revenue = 0.0
        self.government.transfers_this_period = 0.0
        self.government.unemployment_support_this_period = 0.0
        self.government.child_allowance_this_period = 0.0
        self.government.basic_support_this_period = 0.0
        self.government.procurement_spending_this_period = 0.0
        self.government.education_spending_this_period = 0.0
        self.government.school_public_spending_this_period = 0.0
        self.government.university_public_spending_this_period = 0.0
        self.government.public_administration_spending_this_period = 0.0
        self.government.infrastructure_spending_this_period = 0.0
        self.government.bond_issuance_this_period = 0.0
        self.government.deficit_this_period = 0.0
        self.government.surplus_this_period = 0.0
        for bank in self.banks:
            bank.interest_income = 0.0
            bank.interest_expense = 0.0
            bank.profits = 0.0
        for firm in self.firms:
            firm.last_labor_offer_rejections = firm.labor_offer_rejections
            firm.last_labor_offer_rejection_wage_floor = firm.labor_offer_rejection_wage_floor
            firm.labor_offer_rejections = 0
            firm.labor_offer_rejection_wage_floor = 0.0
            firm.sales_this_period = 0.0
            firm.last_technology_investment = 0.0
            firm.last_technology_gain = 0.0

    def _refresh_period_household_caches(self) -> None:
        self._period_active_households_cache = [household for household in self.households if household.alive]
        self._period_household_age_years_cache = {
            household.id: (
                household.age_periods,
                household.age_periods / max(1, self.config.periods_per_year),
            )
            for household in self.households
            if household.alive
        }
        self._period_family_groups_cache = None
        self._period_household_desired_units_cache = {}
        self._period_essential_budget_cache = {}
        self._period_household_row_index_cache = {}
        self._period_essential_desired_units_matrix = None
        self._period_essential_budget_vector = None
        self._period_household_summary_cache = None
        self._period_household_labor_capacity_cache = {}
        self._period_family_summary_cache = None
        self._period_family_resource_coverage_cache = {}
        self._period_baseline_demand_cache = {}
        self._period_living_wage_anchor_cache = None

    def _refresh_period_sector_caches(self) -> None:
        active_firms_by_sector = {
            spec.key: [firm for firm in self.firms_by_sector.get(spec.key, []) if firm.active]
            for spec in SECTOR_SPECS
        }
        self._period_active_firms_by_sector_cache = active_firms_by_sector
        self._period_ranked_firms_by_sector_cache = {
            sector_key: sorted(firms, key=lambda firm: firm.price)
            for sector_key, firms in active_firms_by_sector.items()
        }
        self._period_average_sector_price_cache = {
            sector_key: (
                sum(firm.price for firm in firms) / len(firms)
                if firms
                else SECTOR_BY_KEY[sector_key].base_price
            )
            for sector_key, firms in active_firms_by_sector.items()
        }
        self._period_essential_budget_cache = {}
        self._period_essential_desired_units_matrix = None
        self._period_essential_budget_vector = None
        self._period_household_summary_cache = None
        self._period_family_summary_cache = None
        self._period_baseline_demand_cache = {}
        self._period_living_wage_anchor_cache = None

    def _refresh_period_family_cache(self) -> None:
        active_households = self._period_active_households_cache
        if active_households is None:
            active_households = self._active_households()

        age_lookup = {
            household_id: cached[1]
            for household_id, cached in self._period_household_age_years_cache.items()
        }
        family_root_cache: dict[int, int] = {}

        def adult_root(household: Household) -> int:
            cached_root = family_root_cache.get(household.id)
            if cached_root is not None:
                return cached_root
            partner = self._household_by_id(household.partner_id) if household.partner_id is not None else None
            if partner is not None and partner.alive and age_lookup.get(partner.id, self._household_age_years(partner)) >= self.config.entry_age_years:
                root = min(household.id, partner.id)
            else:
                root = household.id
            family_root_cache[household.id] = root
            return root

        def child_root(household: Household) -> int:
            cached_root = family_root_cache.get(household.id)
            if cached_root is not None:
                return cached_root
            for relative_id in (household.guardian_id, household.mother_id, household.father_id):
                relative = self._household_by_id(relative_id) if relative_id is not None else None
                if relative is not None and relative.alive and age_lookup.get(relative.id, self._household_age_years(relative)) >= self.config.entry_age_years:
                    root = adult_root(relative)
                    family_root_cache[household.id] = root
                    return root
            root = household.guardian_id if household.guardian_id is not None else household.id
            family_root_cache[household.id] = root
            return root

        groups: dict[int, list[Household]] = {}
        for household in active_households:
            age_years = age_lookup.get(household.id, self._household_age_years(household))
            root_id = child_root(household) if age_years < self.config.entry_age_years else adult_root(household)
            groups.setdefault(root_id, []).append(household)

        self._period_family_groups_cache = groups
        self._period_family_summary_cache = None

    def _reset_household_labor_state(self) -> None:
        for household in self.households:
            household.wage_income = 0.0
            household.previous_income = household.last_income
            household.last_income = 0.0
            household.last_available_cash = 0.0
            household.last_consumption = {spec.key: 0.0 for spec in SECTOR_SPECS}
            household.last_perceived_utility = 0.0

    def _household_observed_income(self, household: Household) -> float:
        return max(0.0, household.previous_income, household.last_income)

    def _public_administration_base_wage_offer(self) -> float:
        living_wage_anchor = self._living_wage_anchor()
        premium = max(0.0, self.config.public_administration_wage_premium)
        return max(
            SECTOR_BY_KEY["public_administration"].base_wage,
            living_wage_anchor * (self.config.reservation_wage_floor_share + premium),
        )

    def _recent_nominal_gdp_anchor(self) -> float:
        lookback = max(1, int(self.config.periods_per_year))
        recent_gdp = [
            max(0.0, getattr(snapshot, "gdp_nominal", 0.0))
            for snapshot in self.history[-lookback:]
            if getattr(snapshot, "gdp_nominal", 0.0) > 0.0
        ]
        if recent_gdp:
            return sum(recent_gdp) / len(recent_gdp)
        labor_force = sum(
            1
            for household in self._active_households()
            if self._household_labor_capacity(household) > 0.0
        )
        synthetic_gdp = self._living_wage_anchor() * max(1, labor_force) * 1.6
        return max(1.0, synthetic_gdp, self._period_sales_revenue)

    def _public_administration_state_size_share_gdp(self) -> float:
        recent_gdp = self._recent_nominal_gdp_anchor()
        if recent_gdp <= 0.0:
            return 0.0
        return self._government_structural_budget_anchor() / recent_gdp

    def _public_administration_employment_target_share(self) -> float:
        floor_share = clamp(self.config.public_administration_employment_floor_share, 0.0, 0.25)
        sensitivity = max(0.0, self.config.public_administration_employment_state_size_sensitivity)
        cap_share = clamp(self.config.public_administration_employment_cap_share, floor_share, 0.50)
        state_size_share = max(0.0, self._public_administration_state_size_share_gdp())
        return clamp(floor_share + sensitivity * state_size_share, floor_share, cap_share)

    def _public_administration_payroll_budget(self) -> float:
        return max(0.0, self.config.public_administration_payroll_share) * self._public_administration_budget()

    def _public_administration_target_workers(self) -> int:
        labor_force = len(
            [
                household
                for household in self._active_households()
                if self._household_labor_capacity(household) > 0.0
            ]
        )
        if labor_force <= 0:
            return 0
        return max(
            0,
            int(round(labor_force * self._public_administration_employment_target_share())),
        )

    def _public_administration_wage_offer(self) -> float:
        target_workers = self._public_administration_target_workers()
        base_wage = self._public_administration_base_wage_offer()
        if target_workers <= 0:
            return base_wage
        payroll_budget = self._public_administration_payroll_budget()
        affordability_wage = payroll_budget / max(1, target_workers)
        wage_floor = max(SECTOR_BY_KEY["public_administration"].base_wage, 0.72 * base_wage)
        wage_ceiling = max(wage_floor, 1.15 * base_wage)
        return clamp(max(SECTOR_BY_KEY["public_administration"].base_wage, affordability_wage), wage_floor, wage_ceiling)

    def _current_employer_wage_offer(self, employer_id: int | None) -> float:
        if employer_id is None:
            return 0.0
        if employer_id == PUBLIC_ADMINISTRATION_EMPLOYER_ID:
            return self._public_administration_wage_offer()
        firm = self.firm_by_id.get(employer_id)
        if firm is None or not firm.active:
            return 0.0
        return max(0.0, firm.wage_offer)

    def _update_household_reservation_wages(self) -> None:
        living_wage_anchor = self._living_wage_anchor()
        if living_wage_anchor <= 0.0:
            return

        adjustment_speed = clamp(self.config.reservation_wage_adjustment_speed, 0.05, 0.80)
        floor_share = clamp(self.config.reservation_wage_floor_share, 0.50, 1.20)

        for members in self._family_groups().values():
            earning_members = [
                member
                for member in members
                if self._household_labor_capacity(member) > 0.0
            ]
            if not earning_members:
                continue

            family_basket_cost = sum(self._essential_budget(member) for member in members)
            employed_earners = sum(1 for member in earning_members if member.employed_by is not None)
            supporting_earners = employed_earners if employed_earners > 0 else len(earning_members)
            reproduction_wage = family_basket_cost / max(1, supporting_earners)
            family_income = sum(self._household_observed_income(member) for member in members)
            family_cash = sum(self._household_cash_balance(member) for member in members)
            income_cover = family_income / max(1e-9, family_basket_cost)
            family_stress = clamp(1.0 - income_cover, 0.0, 1.0)
            liquidity_holdout_factor = self._family_reservation_liquidity_factor(
                earning_members,
                family_cash=family_cash,
                family_basic_basket_cost=family_basket_cost,
                family_stress=family_stress,
            )
            reservation_target = max(
                living_wage_anchor * floor_share * liquidity_holdout_factor,
                reproduction_wage * (0.92 + 0.18 * family_stress) * liquidity_holdout_factor,
            )

            for member in earning_members:
                current_wage = self._current_employer_wage_offer(member.employed_by)
                blended_target = (1.0 - adjustment_speed) * member.reservation_wage + adjustment_speed * reservation_target
                member.reservation_wage = clamp(
                    max(current_wage * 0.92, blended_target),
                    0.0,
                    living_wage_anchor * 3.0,
                )

    def _family_reservation_liquidity_factor(
        self,
        earning_members: list[Household],
        *,
        family_cash: float,
        family_basic_basket_cost: float,
        family_stress: float,
    ) -> float:
        if family_basic_basket_cost <= 0.0 or not earning_members:
            return 1.0
        employment_stability = (
            sum(1.0 for member in earning_members if member.employed_by is not None)
            / max(1.0, len(earning_members))
        )
        family_money_trust = sum(member.money_trust for member in earning_members) / max(1, len(earning_members))
        reference_cushion_months = max(0.5, self.config.reservation_wage_reference_cushion_months)
        target_cushion_months = clamp(
            reference_cushion_months
            + 0.70 * (1.0 - employment_stability)
            + 0.55 * (1.0 - family_money_trust)
            + 0.45 * family_stress,
            0.5,
            6.0,
        )
        effective_target_cushion = family_basic_basket_cost * target_cushion_months
        cushion_ratio = family_cash / max(1.0, effective_target_cushion)
        shortage = clamp(1.0 - cushion_ratio, 0.0, 1.0)
        surplus = clamp(cushion_ratio - 1.0, 0.0, 1.5)
        discount_max = clamp(self.config.reservation_wage_liquidity_discount_max, 0.0, 0.45)
        premium_max = clamp(self.config.reservation_wage_liquidity_premium_max, 0.0, 0.30)
        factor = 1.0 - discount_max * shortage + premium_max * surplus
        return clamp(factor, 0.60, 1.20)

    def _sector_wage_pressure_bonus(
        self,
        sector_key: str,
        *,
        vacancy_ratio: float,
        labor_tightness: float,
        living_wage_gap: float,
        wage_room: float,
    ) -> float:
        if sector_key == "leisure":
            return (
                0.10 * vacancy_ratio
                + 0.03 * max(0.0, labor_tightness)
                + 0.05 * max(0.0, living_wage_gap) * (0.50 + 0.50 * max(0.0, wage_room))
            )
        if sector_key == "school":
            return (
                0.08 * vacancy_ratio
                + 0.03 * max(0.0, labor_tightness)
                + 0.05 * max(0.0, living_wage_gap)
            )
        if sector_key == "university":
            return (
                0.12 * vacancy_ratio
                + 0.04 * max(0.0, labor_tightness)
                + 0.06 * max(0.0, living_wage_gap) * (0.60 + 0.40 * max(0.0, wage_room))
            )
        if sector_key == "public_administration":
            return (
                0.06 * vacancy_ratio
                + 0.02 * max(0.0, labor_tightness)
                + 0.04 * max(0.0, living_wage_gap)
            )
        return 0.0

    def _sector_wage_floor_premium(self, sector_key: str) -> float:
        if sector_key == "leisure":
            return 0.05
        if sector_key == "school":
            return 0.08
        if sector_key == "university":
            return 0.16
        if sector_key == "public_administration":
            return max(0.0, self.config.public_administration_wage_premium)
        return 0.0

    def _sector_wage_floor(self, sector_key: str, living_wage_anchor: float | None = None) -> float:
        anchor = self._living_wage_anchor() if living_wage_anchor is None else living_wage_anchor
        return anchor * (self.config.reservation_wage_floor_share + self._sector_wage_floor_premium(sector_key))

    def _release_household_from_employment(self, household: Household) -> None:
        firm_id = household.employed_by
        if firm_id is not None:
            firm = self.firm_by_id.get(firm_id)
            if firm is not None:
                firm.workers = [worker_id for worker_id in firm.workers if worker_id != household.id]
        household.employed_by = None
        household.employment_tenure = 0

    def _firm_hiring_capacity(self, firm: Firm) -> int:
        return max(0, firm.desired_workers)

    def _record_labor_offer_rejection(self, firm: Firm, reservation_wage: float) -> None:
        if reservation_wage <= firm.wage_offer:
            return
        firm.labor_offer_rejections += 1
        if firm.labor_offer_rejection_wage_floor <= 0.0:
            firm.labor_offer_rejection_wage_floor = reservation_wage
        else:
            firm.labor_offer_rejection_wage_floor = min(
                firm.labor_offer_rejection_wage_floor,
                reservation_wage,
            )

    def _profitable_labor_offer_rejection_wage_target(
        self,
        firm: Firm,
        *,
        baseline_demand: float,
        effective_productivity: float,
        target_workers: int,
    ) -> float | None:
        if firm.last_labor_offer_rejections <= 0:
            return None
        observed_rejection_wage = max(
            firm.wage_offer,
            firm.last_labor_offer_rejection_wage_floor,
        )
        if observed_rejection_wage <= firm.wage_offer:
            return None
        current_workers = max(0, firm.last_worker_count)
        if target_workers <= current_workers:
            return None
        current_sales_capacity = firm.inventory + effective_productivity * current_workers
        expected_sales_gap = max(0.0, baseline_demand - current_sales_capacity)
        marginal_sales = min(effective_productivity, expected_sales_gap)
        if marginal_sales <= 0.0:
            return None
        gross_margin_per_unit = max(
            0.0,
            firm.price - firm.input_cost_per_unit - firm.transport_cost_per_unit,
        )
        if gross_margin_per_unit <= 0.0:
            return None
        payroll_tax_rate = self._government_payroll_tax_rate()
        max_profitable_wage = (
            marginal_sales * gross_margin_per_unit / max(1e-9, 1.0 + payroll_tax_rate)
        )
        if max_profitable_wage + 1e-9 < observed_rejection_wage:
            return None
        return observed_rejection_wage

    def _best_available_wage_offer(
        self,
        current_firm_id: int,
        *,
        allowed_sectors: set[str] | None = None,
    ) -> float | None:
        best_wage: float | None = None
        for firm in self.firms:
            if not firm.active or firm.id == current_firm_id:
                continue
            if allowed_sectors is not None and firm.sector not in allowed_sectors:
                continue
            if len(firm.workers) >= self._firm_hiring_capacity(firm):
                continue
            if best_wage is None or firm.wage_offer > best_wage:
                best_wage = firm.wage_offer
        return best_wage

    def _age_employment_contracts(self) -> None:
        contract_periods = max(1, self.config.employment_contract_periods)
        active_households = self._active_households()
        for household in active_households:
            if household.employed_by is None:
                continue
            if household.employed_by == PUBLIC_ADMINISTRATION_EMPLOYER_ID:
                household.employment_tenure += 1
                if household.employment_tenure >= contract_periods:
                    best_wage = self._best_available_wage_offer(PUBLIC_ADMINISTRATION_EMPLOYER_ID)
                    public_wage = self._public_administration_wage_offer()
                    if best_wage is None or best_wage <= public_wage:
                        household.employment_tenure = 0
                        continue
                    self._release_household_from_employment(household)
                continue
            firm = self.firm_by_id.get(household.employed_by)
            if firm is None or not firm.active:
                self._release_household_from_employment(household)
                continue
            household.employment_tenure += 1
            if household.employment_tenure >= contract_periods:
                current_capacity = self._firm_hiring_capacity(firm)
                if len(firm.workers) <= current_capacity:
                    allowed_sectors = None
                    if self._in_essential_protection() and firm.sector in ESSENTIAL_SECTOR_KEYS:
                        allowed_sectors = set(ESSENTIAL_SECTOR_KEYS)
                    best_wage = self._best_available_wage_offer(firm.id, allowed_sectors=allowed_sectors)
                    if best_wage is None or best_wage <= firm.wage_offer:
                        household.employment_tenure = 0
                        continue
                self._release_household_from_employment(household)

    def _align_existing_workforce(self) -> None:
        for firm in self.firms:
            if not firm.active or not firm.workers:
                continue
            target_headcount = max(0, firm.desired_workers)
            if len(firm.workers) <= target_headcount:
                continue
            workers_to_release = len(firm.workers) - target_headcount
            release_order = sorted(
                firm.workers,
                key=lambda worker_id: (
                    self.households[worker_id].employment_tenure,
                    self.households[worker_id].reservation_wage,
                    worker_id,
                ),
                reverse=True,
            )
            for worker_id in release_order[:workers_to_release]:
                self._release_household_from_employment(self.households[worker_id])

    def _active_households(self) -> list[Household]:
        cached = self._period_active_households_cache
        if cached is not None:
            return cached
        return [household for household in self.households if household.alive]

    def _sector_firms(self, sector_key: str, active_only: bool = True) -> list[Firm]:
        firms_by_sector = getattr(self, "firms_by_sector", None)
        if firms_by_sector is None:
            return []
        firms = firms_by_sector.get(sector_key, [])
        if active_only:
            cached = self._period_active_firms_by_sector_cache
            if cached is not None:
                return cached.get(sector_key, [])
            return [firm for firm in firms if firm.active]
        return list(firms)

    def _average_sector_price(self, sector_key: str) -> float:
        cached = self._period_average_sector_price_cache
        if cached is not None and sector_key in cached:
            return cached[sector_key]
        firms = self._sector_firms(sector_key)
        if not firms:
            return SECTOR_BY_KEY[sector_key].base_price
        return sum(firm.price for firm in firms) / len(firms)

    def _roll_forward_sector_demand_signals(self) -> None:
        self._prior_sector_sales_units = self._last_sector_sales_units.copy()
        self._last_sector_potential_demand_units = self._period_sector_potential_demand_units.copy()
        self._last_sector_sales_units = self._period_sector_sales_units.copy()
        self._last_sector_budget_demand_units = self._period_sector_budget_demand_units.copy()
        self._last_sector_revealed_unmet_units = self._period_sector_revealed_unmet_units.copy()
        history_window = self._recent_history_window()
        for spec in SECTOR_SPECS:
            sector_key = spec.key
            self._sector_sales_history[sector_key].append(self._last_sector_sales_units.get(sector_key, 0.0))
            self._sector_revealed_unmet_history[sector_key].append(
                self._last_sector_revealed_unmet_units.get(sector_key, 0.0)
            )
            if len(self._sector_sales_history[sector_key]) > history_window:
                del self._sector_sales_history[sector_key][:-history_window]
            if len(self._sector_revealed_unmet_history[sector_key]) > history_window:
                del self._sector_revealed_unmet_history[sector_key][:-history_window]

    def _sector_market_total(self, sector_key: str) -> float:
        firms = self._sector_firms(sector_key)
        if not firms:
            return 0.0
        competitiveness = sum(1.0 / max(0.1, firm.price) for firm in firms)
        return competitiveness

    def _purchase_from_sector(
        self,
        price_sensitivity: float,
        sector_key: str,
        desired_units: float,
        cash: float,
        spending_log: dict[str, float],
    ) -> tuple[float, float]:
        ranked_firms_cache = self._period_ranked_firms_by_sector_cache or {}
        ranked_firms = ranked_firms_cache.get(sector_key)
        if ranked_firms is None:
            ranked_firms = sorted(self._sector_firms(sector_key), key=lambda firm: firm.price)
        if not ranked_firms or desired_units <= 0.0 or cash <= 0.0:
            self._record_revealed_unmet_demand(sector_key, desired_units, 0.0, cash)
            return cash, 0.0

        elasticity = 1.0 + 0.35 * price_sensitivity
        ranked_available_firms: list[tuple[Firm, float]] = []
        for firm in ranked_firms:
            if firm.inventory <= 0.0 or firm.price <= 0.0:
                continue
            competitiveness = 1.0 / max(0.1, firm.price) ** elasticity
            competitiveness *= 1.0 + min(0.25, firm.inventory / max(1.0, desired_units))
            ranked_available_firms.append((firm, competitiveness))

        if not ranked_available_firms:
            self._record_revealed_unmet_demand(sector_key, desired_units, 0.0, cash)
            return cash, 0.0

        bought_total = 0.0
        suffix_weights = [0.0] * len(ranked_available_firms)
        running_weight = 0.0
        for index in range(len(ranked_available_firms) - 1, -1, -1):
            running_weight += ranked_available_firms[index][1]
            suffix_weights[index] = running_weight
        remaining_desired_units = desired_units
        for index, (firm, weight) in enumerate(ranked_available_firms):
            if cash <= 0.0 or remaining_desired_units <= 0.0:
                break
            remaining_weight = suffix_weights[index]
            if remaining_weight <= 0.0:
                break
            target_units = remaining_desired_units * (weight / remaining_weight)
            affordable_units = cash / firm.price
            units_bought = min(target_units, affordable_units, firm.inventory)
            if units_bought <= 0.0:
                continue
            units_bought = self._remove_inventory_units(firm, units_bought)
            if units_bought <= 0.0:
                continue
            spend = units_bought * firm.price
            cash -= spend
            firm.cash += spend
            firm.sales_this_period += units_bought
            self._period_sales_units += units_bought
            self._period_sales_revenue += spend
            self._period_sector_sales_units[sector_key] += units_bought
            spending_log[sector_key] += units_bought
            bought_total += units_bought
            remaining_desired_units -= units_bought

        self._record_revealed_unmet_demand(sector_key, desired_units, bought_total, cash)
        return cash, bought_total

    def _record_revealed_unmet_demand(
        self,
        sector_key: str,
        desired_units: float,
        bought_units: float,
        residual_cash: float,
    ) -> None:
        if desired_units <= bought_units or residual_cash <= 1e-9:
            return
        average_price = self._average_sector_price(sector_key)
        affordable_unmet_units = residual_cash / max(0.1, average_price)
        revealed_unmet_units = min(max(0.0, desired_units - bought_units), affordable_unmet_units)
        if revealed_unmet_units <= 0.0:
            return
        self._period_sector_revealed_unmet_units[sector_key] += revealed_unmet_units

    def _draw_desired_children(self) -> int:
        choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        weights = [12, 16, 18, 16, 12, 10, 7, 4, 2, 2, 1]
        return self.rng.choices(choices, weights=weights, k=1)[0]

    def _draw_partnership_affinity_code(self) -> int:
        return self.rng.randint(1, max(1, self.config.partnership_affinity_buckets))

    def _draw_initial_partnership_attempt_period(self) -> int:
        retry_periods = max(1, self.config.partnership_retry_periods)
        return self.rng.randint(0, retry_periods - 1)

    def _draw_fertility_multiplier(self) -> float:
        spread = max(0.0, self.config.fertility_heterogeneity_max)
        return self.rng.uniform(max(0.1, 1.0 - spread), 1.0 + spread)

    def _inherit_partnership_affinity_code(self, mother: Household, father: Household | None) -> int:
        if father is None:
            base_code = mother.partnership_affinity_code or self._draw_partnership_affinity_code()
        else:
            candidate_codes = [
                mother.partnership_affinity_code or self._draw_partnership_affinity_code(),
                father.partnership_affinity_code or self._draw_partnership_affinity_code(),
            ]
            base_code = self.rng.choice(candidate_codes)
        if self.rng.random() < 0.10:
            return self._draw_partnership_affinity_code()
        return base_code

    def _schedule_next_partnership_attempt(self, household: Household) -> None:
        household.next_partnership_attempt_period = self.period + max(1, self.config.partnership_retry_periods)

    def _female_age_fertility_factor(self, age_years: float) -> float:
        # Based on ACOG guidance that healthy couples in the 20s/early 30s have about a
        # 25% chance of conception each cycle and around 10% by age 40. We interpolate
        # between those anchors and use the ratio relative to the early-adult baseline.
        if age_years <= 32.0:
            return 1.0
        if age_years >= 40.0:
            return 0.40
        progress = (age_years - 32.0) / 8.0
        return 1.0 - 0.60 * progress

    def _household_by_id(self, household_id: int) -> Household | None:
        if household_id is None:
            return None
        if 0 <= household_id < len(self.households):
            return self.households[household_id]
        return None

    def _household_cash_balance(self, household: Household) -> float:
        return household.savings + household.wage_income

    def _withdraw_household_cash(self, household: Household, amount: float) -> float:
        amount = max(0.0, amount)
        available = self._household_cash_balance(household)
        withdrawn = min(amount, available)
        if withdrawn <= 0.0:
            return 0.0

        wage_draw = min(household.wage_income, withdrawn)
        household.wage_income -= wage_draw
        remaining = withdrawn - wage_draw
        if remaining > 0.0:
            household.savings = max(0.0, household.savings - remaining)
        return withdrawn

    def _advance_credit_state_timers(self) -> None:
        for household in self.households:
            household.loan_restructure_grace_periods = max(0, household.loan_restructure_grace_periods - 1)
            household.credit_exclusion_periods = max(0, household.credit_exclusion_periods - 1)
        for firm in self.firms:
            firm.loan_restructure_grace_periods = max(0, firm.loan_restructure_grace_periods - 1)
            firm.credit_exclusion_periods = max(0, firm.credit_exclusion_periods - 1)
            if firm.loan_balance <= 1e-9:
                firm.loan_default_flag = False

    def _loan_rate_for_household(self, household: Household) -> float:
        bank = self._bank_for_id(self._bank_id_for_household(household))
        if bank is None:
            return max(0.0, self.config.bank_loan_rate)
        return max(0.0, bank.loan_rate)

    def _loan_rate_for_firm(self, firm: Firm) -> float:
        bank = self._bank_for_id(self._bank_id_for_firm(firm))
        if bank is None:
            return max(0.0, self.config.bank_loan_rate)
        return max(0.0, bank.loan_rate)

    def _apply_loan_payment(
        self,
        *,
        balance: float,
        interest_due: float,
        principal_due: float,
        payment: float,
        bank: CommercialBank | None,
    ) -> tuple[float, float, float, float, float]:
        balance = max(0.0, balance)
        payment = max(0.0, payment)
        interest_due = max(0.0, interest_due)
        principal_due = clamp(max(0.0, principal_due), 0.0, balance)

        interest_paid = min(payment, interest_due)
        principal_paid = min(balance, max(0.0, payment - interest_paid))
        unpaid_interest = max(0.0, interest_due - interest_paid)
        shortfall = max(0.0, interest_due + principal_due - (interest_paid + principal_paid))
        new_balance = max(0.0, balance - principal_paid + unpaid_interest)
        if bank is not None and interest_paid > 0.0:
            bank.interest_income += interest_paid
            bank.profits += interest_paid
            bank.reserves += interest_paid
        return new_balance, interest_paid, principal_paid, unpaid_interest, shortfall

    def _withdraw_family_cash(
        self,
        members: list[Household],
        amount: float,
        *,
        priority_member: Household | None = None,
    ) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0 or not members:
            return 0.0

        ordered_members = sorted(
            members,
            key=lambda member: (
                0 if priority_member is not None and member.id == priority_member.id else 1,
                -self._household_cash_balance(member),
                member.id,
            ),
        )
        remaining = amount
        withdrawn = 0.0
        for member in ordered_members:
            if remaining <= 1e-9:
                break
            draw = self._withdraw_household_cash(member, remaining)
            withdrawn += draw
            remaining -= draw
        return withdrawn

    def _current_total_liquid_money(self) -> float:
        return (
            sum((household.savings + household.wage_income) for household in self.households if household.alive)
            + sum(owner.wealth + owner.vault_cash for owner in self.entrepreneurs)
            + sum(firm.cash for firm in self.firms if firm.active)
            + self.government.treasury_cash
        )

    def _current_monetary_base(self) -> float:
        return self._current_total_liquid_money()

    def _build_banks(self) -> list[CommercialBank]:
        bank_count = max(1, self.config.commercial_banks)
        reserve_ratio = clamp(self.config.reserve_ratio, 0.05, 0.95)
        banks = [
            CommercialBank(
                id=bank_id,
                name=f"Banco {bank_id + 1}",
                reserve_ratio=reserve_ratio,
            )
            for bank_id in range(bank_count)
        ]
        return banks

    def _bank_for_id(self, bank_id: int | None) -> CommercialBank | None:
        if bank_id is None:
            return None
        return self.bank_by_id.get(bank_id)

    def _bank_id_for_household(self, household: Household) -> int:
        if household.bank_id in self.bank_by_id:
            return household.bank_id
        return household.id % max(1, len(self.banks))

    def _bank_id_for_entrepreneur(self, owner: Entrepreneur) -> int:
        if owner.bank_id in self.bank_by_id:
            return owner.bank_id
        return owner.id % max(1, len(self.banks))

    def _bank_id_for_firm(self, firm: Firm) -> int:
        if firm.bank_id in self.bank_by_id:
            return firm.bank_id
        return firm.id % max(1, len(self.banks))

    def _owner_total_liquid(self, owner: Entrepreneur) -> float:
        return owner.wealth + owner.vault_cash

    def _withdraw_owner_liquid(self, owner: Entrepreneur, amount: float) -> float:
        amount = max(0.0, amount)
        available = self._owner_total_liquid(owner)
        withdrawn = min(amount, available)
        if withdrawn <= 0.0:
            return 0.0

        from_wealth = min(owner.wealth, withdrawn)
        owner.wealth -= from_wealth
        remaining = withdrawn - from_wealth
        if remaining > 0.0:
            owner.vault_cash = max(0.0, owner.vault_cash - remaining)
        return withdrawn

    def _distribute_government_spending_leakage(self, amount: float) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0 or not self.entrepreneurs:
            return 0.0
        active_owners = [owner for owner in self.entrepreneurs if owner.active]
        if not active_owners:
            active_owners = self.entrepreneurs
        equal_share = amount / max(1, len(active_owners))
        for owner in active_owners:
            owner.wealth += equal_share
        return amount

    def _assign_financial_institutions(self) -> None:
        if not self.banks:
            return

        bank_count = len(self.banks)
        active_households = sorted(
            self._active_households(),
            key=lambda household: (
                household.savings,
                self._household_age_years(household),
                household.id,
            ),
        )
        for index, household in enumerate(active_households):
            bank_index = min(bank_count - 1, index * bank_count // max(1, len(active_households)))
            household.bank_id = self.banks[bank_index].id

        entrepreneurs_sorted = sorted(self.entrepreneurs, key=lambda owner: (owner.wealth, owner.id))
        for index, owner in enumerate(entrepreneurs_sorted):
            bank_index = min(bank_count - 1, index * bank_count // max(1, len(entrepreneurs_sorted)))
            owner.bank_id = self.banks[bank_index].id
            owner.vault_cash = max(0.0, owner.vault_cash)
            if owner.consumption_propensity <= 0.0:
                owner.consumption_propensity = self.rng.uniform(0.60, 1.40)

        sector_offsets = {spec.key: index for index, spec in enumerate(SECTOR_SPECS)}
        for firm in self.firms:
            owner = self.entrepreneurs[firm.owner_id] if 0 <= firm.owner_id < len(self.entrepreneurs) else None
            owner_bank_id = owner.bank_id if owner is not None else firm.id % bank_count
            sector_offset = sector_offsets.get(firm.sector, 0)
            firm.bank_id = self.banks[(owner_bank_id + sector_offset) % bank_count].id

    def _refresh_bank_balance_sheets(self) -> None:
        for bank in self.banks:
            bank.deposits = 0.0
            bank.loans_households = 0.0
            bank.loans_firms = 0.0

        for household in self._active_households():
            bank = self._bank_for_id(self._bank_id_for_household(household))
            if bank is not None:
                bank.deposits += max(0.0, household.savings)
                bank.loans_households += max(0.0, household.loan_balance)

        for owner in self.entrepreneurs:
            bank = self._bank_for_id(self._bank_id_for_entrepreneur(owner))
            if bank is not None:
                bank.deposits += max(0.0, owner.wealth)

        for firm in self.firms:
            if not firm.active:
                continue
            bank = self._bank_for_id(self._bank_id_for_firm(firm))
            if bank is not None:
                bank.deposits += max(0.0, firm.cash)
                bank.loans_firms += max(0.0, firm.loan_balance)

        government_bank = self._bank_for_id(self.government.bank_id)
        if government_bank is not None:
            government_bank.deposits += max(0.0, self.government.treasury_cash)

    def _seed_bank_balance_sheets(self) -> None:
        self._refresh_bank_balance_sheets()
        for bank in self.banks:
            bank.central_bank_borrowing = 0.0
            if not self.config.central_bank_enabled:
                bank.reserves = 0.0
                bank.bond_holdings = 0.0
                continue

            required_reserves = bank.reserve_ratio * bank.deposits
            seed_reserves = required_reserves * max(1.0, self.config.bank_initial_reserve_multiplier)
            target_capital_ratio = clamp(max(self.config.bank_min_capital_ratio, 0.10), 0.01, 0.30)
            target_assets = bank.deposits / max(1e-9, 1.0 - target_capital_ratio)
            if not self.config.government_enabled:
                bank.reserves = max(seed_reserves, target_assets)
                bank.bond_holdings = 0.0
                continue
            bank.reserves = seed_reserves
            bank.bond_holdings = max(0.0, target_assets - bank.reserves)

    def _bank_reserve_requirement(self, bank: CommercialBank) -> float:
        return max(0.0, bank.reserve_ratio * bank.deposits)

    def _bank_lending_capacity(self, bank: CommercialBank) -> float:
        reserve_requirement = self._bank_reserve_requirement(bank)
        return max(0.0, (bank.reserves - reserve_requirement) / max(1e-9, bank.reserve_ratio))

    def _bank_total_liquidity(self, bank: CommercialBank) -> float:
        return bank.reserves + bank.bond_holdings

    def _bank_assets(self, bank: CommercialBank) -> float:
        return bank.reserves + bank.loans_households + bank.loans_firms + bank.bond_holdings

    def _bank_liabilities(self, bank: CommercialBank) -> float:
        return bank.deposits + bank.central_bank_borrowing

    def _bank_equity(self, bank: CommercialBank) -> float:
        return self._bank_assets(bank) - self._bank_liabilities(bank)

    def _bank_risk_weighted_assets(self, bank: CommercialBank) -> float:
        bond_risk_weight = clamp(self.config.bank_bond_risk_weight, 0.0, 1.0)
        return bank.loans_households + bank.loans_firms + bank.bond_holdings * bond_risk_weight

    def _bank_capital_ratio(self, bank: CommercialBank) -> float:
        risk_weighted_assets = self._bank_risk_weighted_assets(bank)
        equity = self._bank_equity(bank)
        if risk_weighted_assets <= 1e-9:
            return 1.0 if equity >= 0.0 else -1.0
        return equity / risk_weighted_assets

    def _bank_warning_capital_ratio(self) -> float:
        minimum_ratio = max(0.01, self.config.bank_min_capital_ratio)
        return clamp(minimum_ratio + max(0.04, 1.00 * minimum_ratio), minimum_ratio + 0.02, 0.30)

    def _bank_comfort_capital_ratio(self) -> float:
        warning_ratio = self._bank_warning_capital_ratio()
        minimum_ratio = max(0.01, self.config.bank_min_capital_ratio)
        return clamp(warning_ratio + max(0.02, 0.75 * minimum_ratio), warning_ratio + 0.01, 0.40)

    def _bank_private_backers(self, bank: CommercialBank) -> list[Entrepreneur]:
        backers = [
            owner
            for owner in self.entrepreneurs
            if owner.active and self._bank_id_for_entrepreneur(owner) == bank.id
        ]
        return sorted(backers, key=lambda owner: (owner.wealth + owner.vault_cash, -owner.id), reverse=True)

    def _bank_private_capital_capacity(self, bank: CommercialBank, *, resolution_mode: bool = False) -> float:
        if not self.entrepreneurs:
            return 0.0
        support_share = 1.00 if resolution_mode else 0.70
        capacity = 0.0
        for owner in self._bank_private_backers(bank):
            committed_floor = self.config.firm_restart_wealth_threshold + (0.0 if resolution_mode else 20.0)
            liquid_base = owner.wealth + (owner.vault_cash if resolution_mode else 0.50 * owner.vault_cash)
            liquid_buffer = max(0.0, liquid_base - committed_floor)
            capacity += liquid_buffer * support_share
        return capacity

    def _restore_bank_capital(
        self,
        bank: CommercialBank,
        target_ratio: float,
        *,
        resolution_mode: bool = False,
    ) -> float:
        target_ratio = max(0.0, target_ratio)
        current_equity = self._bank_equity(bank)
        target_equity = target_ratio * self._bank_risk_weighted_assets(bank)
        capital_shortfall = max(0.0, target_equity - current_equity)
        if capital_shortfall <= 1e-9:
            return 0.0
        support_capacity = self._bank_private_capital_capacity(bank, resolution_mode=resolution_mode)
        if support_capacity <= 1e-9:
            return 0.0
        injected = min(capital_shortfall, support_capacity)
        bank.reserves += injected
        self._period_bank_recapitalization += injected
        return injected

    def _bank_prudential_lending_multiplier(self, bank: CommercialBank) -> float:
        minimum_ratio = max(0.01, self.config.bank_min_capital_ratio)
        warning_ratio = self._bank_warning_capital_ratio()
        comfort_ratio = self._bank_comfort_capital_ratio()
        capital_ratio = self._bank_capital_ratio(bank)
        if capital_ratio <= minimum_ratio:
            capital_scale = 0.0
        elif capital_ratio < warning_ratio:
            progress = (capital_ratio - minimum_ratio) / max(1e-9, warning_ratio - minimum_ratio)
            capital_scale = 0.10 + 0.25 * progress
        elif capital_ratio < comfort_ratio:
            progress = (capital_ratio - warning_ratio) / max(1e-9, comfort_ratio - warning_ratio)
            capital_scale = 0.35 + 0.65 * progress
        else:
            capital_scale = 1.0

        reserve_requirement = self._bank_reserve_requirement(bank)
        reserve_scale = (
            clamp(bank.reserves / max(1e-9, reserve_requirement), 0.0, 1.15)
            if reserve_requirement > 0.0
            else 1.0
        )
        discount_stress = clamp(
            bank.central_bank_borrowing / max(1.0, bank.deposits + bank.reserves),
            0.0,
            2.0,
        )
        discount_scale = 1.0 - 0.45 * min(1.0, discount_stress)
        return clamp(capital_scale * reserve_scale * discount_scale, 0.0, 1.0)

    def _bank_capital_lending_capacity(self, bank: CommercialBank) -> float:
        min_capital_ratio = max(1e-9, self.config.bank_min_capital_ratio)
        bank_equity = self._bank_equity(bank)
        if bank_equity <= 0.0:
            return 0.0
        max_risk_weighted_assets = bank_equity / min_capital_ratio
        current_risk_weighted_assets = self._bank_risk_weighted_assets(bank)
        return max(0.0, max_risk_weighted_assets - current_risk_weighted_assets)

    def _bank_effective_lending_capacity(self, bank: CommercialBank) -> float:
        return max(
            0.0,
            min(
                self._bank_lending_capacity(bank),
                self._bank_capital_lending_capacity(bank),
            ),
        )

    def _bank_discount_window_rate(self) -> float:
        spread = max(0.0, self.config.central_bank_discount_window_spread)
        return clamp(
            self.central_bank.policy_rate + spread,
            self.config.central_bank_policy_rate_floor,
            self.config.central_bank_policy_rate_ceiling + spread,
        )

    def _family_members_for_household(self, household: Household) -> list[Household]:
        root_id = (
            self._family_root_for_child(household)
            if self._household_age_years(household) < self.config.entry_age_years
            else self._family_root_for_adult(household)
        )
        members = self._family_groups().get(root_id)
        if members is not None:
            return members
        return [household] if household.alive else []

    def _projected_family_labor_income(self, members: list[Household]) -> float:
        total_income = 0.0
        for member in members:
            projected_income = self._household_observed_income(member)
            if member.employed_by is not None:
                projected_income = max(
                    projected_income,
                    self._current_employer_wage_offer(member.employed_by),
                )
            total_income += projected_income
        return total_income

    def _household_creditworthy(self, borrower: Household, amount: float, bank: CommercialBank) -> bool:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return False
        if borrower.credit_exclusion_periods > 0 or borrower.loan_delinquency_periods > 0:
            return False

        family_members = self._family_members_for_household(borrower)
        family_projected_income = self._projected_family_labor_income(family_members)
        family_basket_cost = sum(self._essential_budget(member) for member in family_members)
        adult_members = [
            member for member in family_members if self._household_age_years(member) >= self.config.entry_age_years
        ]
        debt_holders = adult_members if adult_members else family_members
        projected_debt = sum(max(0.0, member.loan_balance) for member in debt_holders) + amount
        projected_interest_share = (
            projected_debt * max(0.0, bank.loan_rate) / max(1.0, family_projected_income)
            if family_projected_income > 0.0
            else float("inf")
        )
        projected_debt_to_income = (
            projected_debt / max(1.0, family_projected_income)
            if family_projected_income > 0.0
            else float("inf")
        )

        if family_projected_income <= 0.0:
            return projected_debt <= 0.50 * max(1.0, family_basket_cost)

        return (
            projected_debt_to_income <= max(0.5, self.config.bank_household_max_debt_to_income)
            and projected_interest_share <= clamp(self.config.bank_household_max_interest_share, 0.05, 0.95)
        )

    def _firm_creditworthy(self, firm: Firm, amount: float, bank: CommercialBank) -> bool:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return False
        if firm.credit_exclusion_periods > 0 or firm.loan_delinquency_periods > 0 or firm.loan_default_flag:
            return False

        realized_revenue = max(
            0.0,
            firm.last_revenue,
            firm.last_sales * max(0.0, firm.price),
        )
        sales_realization = firm.last_sales / max(1.0, firm.last_expected_sales)
        expectation_credibility = clamp(0.25 + 0.75 * sales_realization, 0.10, 1.0)
        expected_revenue = max(
            0.0,
            realized_revenue,
            expectation_credibility * firm.last_expected_sales * max(0.0, firm.price),
        )
        operating_cost = max(0.0, firm.last_wage_bill + firm.last_input_cost + firm.last_transport_cost + firm.last_fixed_overhead)
        operating_surplus = expected_revenue - operating_cost
        projected_debt = max(0.0, firm.loan_balance) + amount
        projected_interest_cost = projected_debt * max(0.0, bank.loan_rate)
        projected_debt_to_revenue = projected_debt / max(1.0, expected_revenue)
        interest_coverage = (
            operating_surplus / max(1e-9, projected_interest_cost)
            if projected_interest_cost > 0.0
            else float("inf")
        )

        if expected_revenue <= 0.0:
            return firm.age <= max(1, self.config.startup_grace_periods) and amount <= max(1.0, firm.cash + firm.capital * 0.05)

        # Do not keep scaling firms that are repeatedly missing their market and burning cash,
        # unless they are essential suppliers where scarcity makes short-run support more defensible.
        if (
            firm.sector not in ESSENTIAL_SECTOR_KEYS
            and firm.loss_streak >= 2
            and sales_realization < 0.80
        ):
            modest_bridge = max(
                firm.last_wage_bill,
                firm.last_input_cost + firm.last_transport_cost + firm.last_fixed_overhead,
            )
            if amount > 0.35 * max(1.0, modest_bridge):
                return False

        if projected_debt_to_revenue > max(0.5, self.config.bank_firm_max_debt_to_revenue):
            return False
        if projected_interest_cost > 0.0 and interest_coverage < max(0.5, self.config.bank_firm_min_interest_coverage):
            return False
        if operating_surplus <= 0.0 and firm.cash < 0.75 * max(1.0, operating_cost):
            return False
        return True

    def _household_debt_service_due(self, borrower: Household) -> tuple[float, float]:
        if borrower.loan_balance <= 1e-9:
            return 0.0, 0.0
        interest_due = borrower.loan_balance * self._loan_rate_for_household(borrower)
        principal_share = max(0.0, self.config.household_loan_principal_payment_share)
        if borrower.loan_restructure_grace_periods > 0:
            principal_share = 0.0
        principal_due = borrower.loan_balance * principal_share
        return interest_due, min(borrower.loan_balance, principal_due)

    def _restructure_household_loan(self, borrower: Household) -> None:
        if borrower.loan_balance <= 1e-9:
            borrower.loan_delinquency_periods = 0
            return
        if borrower.loan_restructure_count > 0:
            return
        bank = self._bank_for_id(self._bank_id_for_household(borrower))
        haircut = borrower.loan_balance * max(0.0, self.config.household_loan_restructure_haircut_share)
        borrower.loan_balance = max(0.0, borrower.loan_balance - haircut)
        borrower.loan_restructure_count += 1
        borrower.loan_restructure_grace_periods = max(0, self.config.household_loan_restructure_grace_periods)
        borrower.loan_delinquency_periods = 0
        self._period_bank_loan_restructures += 1
        self._period_household_loan_restructures += 1
        if bank is not None and haircut > 0.0:
            bank.profits -= haircut
            self._period_bank_writeoffs += haircut

    def _resolve_household_default(self, borrower: Household, family_members: list[Household]) -> None:
        if borrower.loan_balance <= 1e-9:
            borrower.loan_delinquency_periods = 0
            return
        bank = self._bank_for_id(self._bank_id_for_household(borrower))
        family_basket_cost = sum(self._essential_budget(member) for member in family_members)
        family_cash = sum(self._household_cash_balance(member) for member in family_members)
        protected_cash = family_basket_cost * max(0.0, self.config.household_default_protected_basket_share)
        exposed_cash = max(0.0, family_cash - protected_cash)
        recovery_target = min(borrower.loan_balance, exposed_cash)
        recovered = self._withdraw_family_cash(
            family_members,
            recovery_target,
            priority_member=borrower,
        )
        remaining_balance = max(0.0, borrower.loan_balance - recovered)
        borrower.loan_balance = 0.0
        borrower.loan_delinquency_periods = 0
        borrower.loan_restructure_count = 0
        borrower.loan_restructure_grace_periods = 0
        borrower.credit_exclusion_periods = max(
            borrower.credit_exclusion_periods,
            self.config.household_default_credit_cooldown_periods,
        )
        self._period_household_loan_defaults += 1
        if remaining_balance > 0.0:
            self._period_bank_writeoffs += remaining_balance
            if bank is not None:
                bank.profits -= remaining_balance

    def _service_household_loans(self) -> None:
        family_groups = self._family_groups()
        for family_members in family_groups.values():
            debtors = [
                member
                for member in family_members
                if (
                    member.alive
                    and self._household_age_years(member) >= self.config.entry_age_years
                    and member.loan_balance > 1e-9
                )
            ]
            if not debtors:
                continue

            family_basket_cost = sum(self._essential_budget(member) for member in family_members)
            family_cash = sum(self._household_cash_balance(member) for member in family_members)
            protected_cash = family_basket_cost * max(0.0, self.config.household_debt_service_protected_basket_share)
            remaining_payment_pool = max(0.0, family_cash - protected_cash)

            ordered_debtors = sorted(
                debtors,
                key=lambda member: (
                    -member.loan_delinquency_periods,
                    -member.loan_balance,
                    member.id,
                ),
            )
            for borrower in ordered_debtors:
                interest_due, principal_due = self._household_debt_service_due(borrower)
                total_due = interest_due + principal_due
                if total_due <= 1e-9:
                    borrower.loan_delinquency_periods = 0
                    continue
                bank = self._bank_for_id(self._bank_id_for_household(borrower))
                scheduled_payment = min(total_due, remaining_payment_pool)
                payment = self._withdraw_family_cash(
                    family_members,
                    scheduled_payment,
                    priority_member=borrower,
                )
                remaining_payment_pool = max(0.0, remaining_payment_pool - payment)
                borrower.loan_balance, _, _, _, shortfall = self._apply_loan_payment(
                    balance=borrower.loan_balance,
                    interest_due=interest_due,
                    principal_due=principal_due,
                    payment=payment,
                    bank=bank,
                )
                if borrower.loan_balance <= 1e-9:
                    borrower.loan_balance = 0.0
                    borrower.loan_delinquency_periods = 0
                    borrower.loan_restructure_count = 0
                    borrower.loan_restructure_grace_periods = 0
                    borrower.credit_exclusion_periods = 0
                    continue
                if shortfall > 1e-6:
                    borrower.loan_delinquency_periods += 1
                else:
                    borrower.loan_delinquency_periods = max(0, borrower.loan_delinquency_periods - 1)

                if (
                    borrower.loan_delinquency_periods >= self.config.household_loan_restructure_delinquency
                    and borrower.loan_restructure_count == 0
                ):
                    self._restructure_household_loan(borrower)
                elif borrower.loan_delinquency_periods >= self.config.household_loan_default_delinquency:
                    self._resolve_household_default(borrower, family_members)

    def _estimate_household_credit_requests(self) -> list[tuple[Household, float]]:
        requests: list[tuple[Household, float]] = []
        for members in self._family_groups().values():
            adult_members = [
                member
                for member in members
                if self._household_age_years(member) >= self.config.entry_age_years
            ]
            if adult_members:
                borrower = max(
                    adult_members,
                    key=lambda member: (self._household_cash_balance(member), member.savings, -member.id),
                )
            else:
                borrower = max(
                    members,
                    key=lambda member: (self._household_cash_balance(member), member.savings, -member.id),
                )

            family_cash = sum(self._household_cash_balance(member) for member in members)
            family_basic_basket_cost = sum(self._essential_budget(member) for member in members)
            cushion = family_basic_basket_cost * clamp(0.10 + 0.20 * (1.0 - borrower.money_trust), 0.10, 0.35)
            loan_need = max(0.0, family_basic_basket_cost + cushion - family_cash)
            if loan_need > 0.0:
                requests.append((borrower, loan_need))
        return requests

    def _economy_investment_stability(self) -> float:
        if not self.history:
            return 1.0
        last_snapshot = self.history[-1]
        observed_inflation = 0.0
        if len(self.history) > 1:
            prior_price_index = max(1e-9, self.history[-2].price_index)
            observed_inflation = max(0.0, (last_snapshot.price_index / prior_price_index) - 1.0)
        target_inflation = self._annual_to_period_growth(self.config.central_bank_target_annual_inflation)
        inflation_gap = max(0.0, observed_inflation - target_inflation)
        unemployment_gap = max(0.0, last_snapshot.unemployment_rate - self.config.target_unemployment)
        essential_gap = max(0.0, 1.0 - last_snapshot.essential_fulfillment_rate)
        bank_fragility = max(0.0, getattr(last_snapshot, "bank_undercapitalized_share", 0.0))
        recession_drag = max(0.0, getattr(last_snapshot, "recession_intensity", 0.0))
        return clamp(
            1.02
            - 0.80 * inflation_gap
            - 0.70 * unemployment_gap
            - 0.55 * essential_gap
            - 0.45 * bank_fragility
            - 0.40 * recession_drag,
            0.35,
            1.15,
        )

    def _firm_investment_confidence(self, firm: Firm, *, macro_stability: float | None = None) -> float:
        macro_stability = (
            self._economy_investment_stability() if macro_stability is None else macro_stability
        )
        forecast_uncertainty = self._firm_forecast_uncertainty(firm)
        cash_cover = clamp(self._firm_cash_cover_ratio(firm) / 1.5, 0.0, 1.25)
        profit_margin = clamp(firm.last_profit / max(1.0, firm.last_revenue), -0.20, 0.30)
        stability_component = clamp(
            1.0 - max(0.0, firm.stability_sensitivity) * max(0.0, 1.0 - macro_stability),
            0.20,
            1.20,
        )
        confidence = (
            stability_component
            + 0.16 * max(0.0, firm.investment_animal_spirits - 1.0)
            + 0.10 * cash_cover
            + 0.12 * max(0.0, profit_margin)
            - 0.18 * forecast_uncertainty
        )
        return clamp(confidence, 0.20, 1.45)

    def _estimate_firm_expansion_credit_need(self, firm: Firm) -> float:
        sell_through = self._firm_recent_sell_through(firm)
        growth_pressure = self._firm_revealed_growth_pressure(firm)
        if sell_through < 0.88 and growth_pressure <= 0.08:
            return 0.0
        profit_margin = clamp(firm.last_profit / max(1.0, firm.last_revenue), -0.25, 0.30)
        if profit_margin < -0.05 and firm.loss_streak >= 2 and growth_pressure < 0.75:
            return 0.0

        macro_stability = self._economy_investment_stability()
        confidence = self._firm_investment_confidence(firm, macro_stability=macro_stability)
        loan_rate = max(0.0, self._loan_rate_for_firm(firm))
        neutral_rate = max(1e-6, self.config.firm_interest_rate_neutral)
        interest_drag = clamp((loan_rate - neutral_rate) / neutral_rate, -0.50, 2.0)
        expansion_signal = clamp(
            0.70 * growth_pressure + 0.55 * max(0.0, sell_through - 0.88),
            0.0,
            1.50,
        )
        base_budget = 0.18 * max(
            20.0,
            firm.capital + 0.60 * firm.last_wage_bill + 0.15 * max(1.0, firm.last_revenue),
        )
        appetite = clamp(firm.expansion_credit_appetite, 0.55, 1.75)
        profit_support = clamp(0.70 + 0.80 * max(0.0, profit_margin + 0.05), 0.55, 1.25)
        leverage_drag = clamp(firm.loan_balance / max(1.0, firm.last_revenue), 0.0, 3.0)
        interest_penalty = max(
            0.0,
            1.0
            - self.config.firm_expansion_credit_interest_sensitivity * max(0.0, interest_drag),
        )
        desired_budget = (
            base_budget
            * expansion_signal
            * appetite
            * confidence
            * profit_support
            * interest_penalty
            * clamp(1.0 - 0.08 * leverage_drag, 0.55, 1.0)
        )
        max_credit = self.config.firm_expansion_credit_max_revenue_share * max(
            1.0,
            firm.last_revenue,
            firm.last_expected_sales * max(0.1, firm.price),
        )
        if firm.sector in ESSENTIAL_SECTOR_KEYS:
            max_credit *= 1.20
        return clamp(desired_budget, 0.0, max_credit)

    def _estimate_firm_credit_requests(self) -> list[tuple[Firm, float, str]]:
        requests: list[tuple[Firm, float, str]] = []
        for firm in self.firms:
            if not firm.active:
                continue
            effective_productivity = max(0.1, self._firm_effective_productivity(firm))
            prudent_sales_anchor = max(
                firm.last_sales,
                min(firm.last_expected_sales, firm.last_sales * 1.30 + 2.0),
            )
            prudent_inventory_target = min(
                max(firm.inventory, 0.0) + max(0.0, prudent_sales_anchor * 0.15),
                max(firm.inventory, firm.target_inventory),
            )
            prudent_output_target = max(0.0, prudent_sales_anchor + prudent_inventory_target - firm.inventory)
            prudent_desired_workers = max(
                len(firm.workers),
                math.ceil(prudent_output_target / effective_productivity) if prudent_output_target > 0.0 else len(firm.workers),
            )
            prudent_desired_workers = min(
                max(1, firm.desired_workers),
                max(prudent_desired_workers, firm.last_worker_count + 2),
            )
            working_capital_target = (
                max(firm.last_wage_bill, prudent_desired_workers * firm.wage_offer)
                + firm.last_input_cost
                + firm.last_transport_cost
                + firm.last_fixed_overhead
                + firm.capital * self.config.depreciation_rate
            )
            buffer_target = 0.15 * max(1.0, prudent_sales_anchor * firm.price)
            working_capital_need = max(0.0, working_capital_target + buffer_target - firm.cash)
            expansion_need = self._estimate_firm_expansion_credit_need(firm)
            total_need = working_capital_need + expansion_need
            if total_need > 0.0:
                purpose = "expansion" if expansion_need > 0.0 else "working_capital"
                requests.append((firm, total_need, purpose))
        return requests

    def _estimate_credit_demand_by_bank(self) -> dict[int, float]:
        demand_by_bank = {bank.id: 0.0 for bank in self.banks}
        for borrower, amount in self._estimate_household_credit_requests():
            demand_by_bank[self._bank_id_for_household(borrower)] += amount
        for firm, amount, _purpose in self._estimate_firm_credit_requests():
            demand_by_bank[self._bank_id_for_firm(firm)] += amount
        return demand_by_bank

    def _distribute_reserve_injection_to_banks(self, amount: float, weights: dict[int, float] | None = None) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0 or not self.banks:
            return 0.0

        if weights is None:
            weights = {
                bank.id: max(1.0, bank.deposits + bank.loans_firms + bank.loans_households)
                for bank in self.banks
            }
        total_weight = sum(max(0.0, weight) for weight in weights.values())
        if total_weight <= 0.0:
            total_weight = float(len(self.banks))
            weights = {bank.id: 1.0 for bank in self.banks}

        for bank in self.banks:
            weight = max(0.0, weights.get(bank.id, 0.0))
            share = weight / total_weight
            bank.reserves += amount * share
        return amount

    def _reconcile_bank_reserves(self) -> None:
        for bank in self.banks:
            reserve_requirement = self._bank_reserve_requirement(bank)
            excess_reserves = max(0.0, bank.reserves - reserve_requirement)
            if excess_reserves > 0.0 and bank.central_bank_borrowing > 0.0:
                repayment = min(bank.central_bank_borrowing, excess_reserves)
                bank.central_bank_borrowing -= repayment
                bank.reserves -= repayment
                reserve_requirement = self._bank_reserve_requirement(bank)
            if bank.reserves >= reserve_requirement:
                continue
            reserve_shortfall = reserve_requirement - bank.reserves
            bond_sale = min(bank.bond_holdings, reserve_shortfall)
            if bond_sale > 0.0:
                bank.bond_holdings -= bond_sale
                bank.reserves += bond_sale
                reserve_shortfall = reserve_requirement - bank.reserves
            if reserve_shortfall > 0.0 and self.config.central_bank_enabled:
                bank.central_bank_borrowing += reserve_shortfall
                bank.reserves += reserve_shortfall

    def _accrue_bank_funding_costs(self) -> None:
        if not self.banks or not self.config.central_bank_enabled:
            return
        discount_window_rate = self._bank_discount_window_rate()
        if discount_window_rate <= 0.0:
            return
        for bank in self.banks:
            if bank.central_bank_borrowing <= 0.0:
                continue
            funding_cost = bank.central_bank_borrowing * discount_window_rate
            if funding_cost <= 0.0:
                continue
            bank.central_bank_borrowing += funding_cost
            bank.interest_expense += funding_cost
            bank.profits -= funding_cost

    def _stabilize_bank_capital_positions(self) -> None:
        if not self.banks:
            return

        self._refresh_bank_balance_sheets()
        self._reconcile_bank_reserves()
        warning_ratio = self._bank_warning_capital_ratio()
        undercapitalized_banks = 0
        interventions_made = False
        for bank in self.banks:
            if self._bank_capital_ratio(bank) >= warning_ratio:
                continue
            undercapitalized_banks += 1
            injected = self._restore_bank_capital(bank, warning_ratio)
            interventions_made = interventions_made or injected > 0.0

        if undercapitalized_banks > 0:
            self._period_bank_undercapitalized_share_signal = max(
                self._period_bank_undercapitalized_share_signal,
                undercapitalized_banks / max(1, len(self.banks)),
            )

        if interventions_made:
            self._refresh_bank_balance_sheets()
            self._reconcile_bank_reserves()
            self._update_bank_interest_rates()

    def _create_household_credit(self, borrower: Household, amount: float, bank: CommercialBank) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return 0.0
        borrower.savings += amount
        borrower.loan_balance += amount
        bank.loans_households += amount
        self._period_household_credit_issued += amount
        return amount

    def _create_firm_credit(
        self,
        firm: Firm,
        amount: float,
        bank: CommercialBank,
        *,
        purpose: str = "working_capital",
    ) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return 0.0
        firm.cash += amount
        firm.loan_balance += amount
        bank.loans_firms += amount
        self._period_firm_credit_issued += amount
        if purpose == "expansion":
            self._period_firm_expansion_credit_issued += amount
        return amount

    def _apply_bank_credit_policy(self) -> None:
        if not self.banks or not self.config.central_bank_enabled:
            return

        self._refresh_bank_balance_sheets()
        household_requests = self._estimate_household_credit_requests()
        firm_requests = self._estimate_firm_credit_requests()
        requests_by_bank: dict[int, list[tuple[str, Household | Firm, float, str | None]]] = {
            bank.id: [] for bank in self.banks
        }

        for borrower, amount in household_requests:
            requests_by_bank[self._bank_id_for_household(borrower)].append(
                ("household", borrower, amount, None)
            )
        for firm, amount, purpose in firm_requests:
            requests_by_bank[self._bank_id_for_firm(firm)].append(("firm", firm, amount, purpose))

        for bank in self.banks:
            bank_requests = requests_by_bank.get(bank.id, [])
            if not bank_requests:
                continue

            approved_requests: list[tuple[str, Household | Firm, float, str | None]] = []
            for request_type, borrower, amount, purpose in bank_requests:
                approved = (
                    self._household_creditworthy(borrower, amount, bank)
                    if request_type == "household"
                    else self._firm_creditworthy(borrower, amount, bank)
                )
                if approved:
                    approved_requests.append((request_type, borrower, amount, purpose))

            total_request = sum(amount for _, _, amount, _ in approved_requests)
            if total_request <= 0.0:
                continue

            capacity = self._bank_effective_lending_capacity(bank)
            if capacity <= 0.0 and bank.bond_holdings > 0.0:
                bond_sale = min(bank.bond_holdings, total_request * bank.reserve_ratio)
                bank.bond_holdings -= bond_sale
                bank.reserves += bond_sale
                capacity = self._bank_effective_lending_capacity(bank)

            if capacity <= 0.0:
                continue

            prudential_scale = self._bank_prudential_lending_multiplier(bank)
            loan_pool = min(total_request, capacity * prudential_scale)
            if loan_pool <= 0.0:
                continue

            allocation_scale = loan_pool / total_request
            for request_type, borrower, amount, purpose in approved_requests:
                granted = amount * allocation_scale
                if granted <= 0.0:
                    continue
                if request_type == "household":
                    self._create_household_credit(borrower, granted, bank)
                else:
                    self._create_firm_credit(borrower, granted, bank, purpose=purpose or "working_capital")

            reserve_requirement = self._bank_reserve_requirement(bank)
            excess_reserves = max(0.0, bank.reserves - reserve_requirement)
            if excess_reserves > 0.0 and prudential_scale >= 0.75:
                bond_purchase = excess_reserves * clamp(self.config.bank_bond_allocation_share, 0.0, 1.0)
                bond_purchase = min(bond_purchase, excess_reserves)
                bank.reserves -= bond_purchase
                bank.bond_holdings += bond_purchase

        self._refresh_bank_balance_sheets()
        self._reconcile_bank_reserves()
        self.central_bank.money_supply = self._current_total_liquid_money()
        self._update_bank_interest_rates()

    def _resolve_bank_insolvency(self) -> None:
        if not self.banks:
            return

        self._refresh_bank_balance_sheets()
        self._reconcile_bank_reserves()
        warning_ratio = self._bank_warning_capital_ratio()
        minimum_ratio = max(0.01, self.config.bank_min_capital_ratio)
        insolvent_banks = 0
        undercapitalized_banks = 0
        interventions_made = False
        for bank in self.banks:
            current_equity = self._bank_equity(bank)
            capital_ratio = self._bank_capital_ratio(bank)
            if capital_ratio < warning_ratio:
                undercapitalized_banks += 1
            if current_equity < 0.0:
                insolvent_banks += 1
            restored = 0.0
            if capital_ratio < warning_ratio:
                restored = self._restore_bank_capital(bank, warning_ratio)
                interventions_made = interventions_made or restored > 0.0
            post_equity = self._bank_equity(bank)
            post_ratio = self._bank_capital_ratio(bank)
            if post_equity < 0.0 or post_ratio < minimum_ratio:
                resolution_injection = self._restore_bank_capital(bank, warning_ratio, resolution_mode=True)
                if resolution_injection > 0.0:
                    self._period_bank_resolution_events += 1
                    interventions_made = True

        if undercapitalized_banks > 0:
            self._period_bank_undercapitalized_share_signal = max(
                self._period_bank_undercapitalized_share_signal,
                undercapitalized_banks / max(1, len(self.banks)),
            )
        if insolvent_banks > 0:
            self._period_bank_insolvent_share_signal = max(
                self._period_bank_insolvent_share_signal,
                insolvent_banks / max(1, len(self.banks)),
            )

        if interventions_made:
            self._refresh_bank_balance_sheets()
            self._reconcile_bank_reserves()
            self._update_bank_interest_rates()

    def _current_price_index_estimate(self) -> float:
        return sum(
            (self._average_sector_price(spec.key) / spec.base_price) * spec.household_demand_share
            for spec in SECTOR_SPECS
        )

    def _sector_price_relative(self, sector_key: str) -> float:
        spec = SECTOR_BY_KEY[sector_key]
        return self._average_sector_price(sector_key) / max(1e-9, spec.base_price)

    def _bundle_price_index(self, weights: dict[str, float], fallback: float | None = None) -> float:
        positive_weights = {sector_key: max(0.0, weight) for sector_key, weight in weights.items() if weight > 0.0}
        if not positive_weights:
            return max(1e-9, fallback if fallback is not None else self._current_price_index_estimate())
        total_weight = sum(positive_weights.values())
        if total_weight <= 0.0:
            return max(1e-9, fallback if fallback is not None else self._current_price_index_estimate())
        return sum(
            self._sector_price_relative(sector_key) * weight
            for sector_key, weight in positive_weights.items()
        ) / total_weight

    def _inventory_price_index_estimate(self, fallback: float | None = None) -> float:
        numerator = 0.0
        denominator = 0.0
        for firm in self.firms:
            if not firm.active or firm.inventory <= 0.0:
                continue
            spec = SECTOR_BY_KEY[firm.sector]
            numerator += max(0.0, firm.inventory) * max(0.1, firm.price)
            denominator += max(0.0, firm.inventory) * max(0.1, spec.base_price)
        if denominator <= 0.0:
            return max(1e-9, fallback if fallback is not None else self._current_price_index_estimate())
        return numerator / denominator

    def _government_consumption_price_index_estimate(
        self,
        procurement_spending: float,
        school_spending: float,
        university_spending: float,
        public_administration_spending: float,
        fallback: float | None = None,
    ) -> float:
        weights = {
            sector_key: procurement_spending * self._essential_basket_share(sector_key)
            for sector_key in ESSENTIAL_SECTOR_KEYS
        }
        weights["school"] = school_spending
        weights["university"] = university_spending
        weights["public_administration"] = public_administration_spending
        return self._bundle_price_index(weights, fallback=fallback)

    def _gdp_deflator_estimate(
        self,
        *,
        gdp_nominal: float,
        household_final_consumption: float,
        government_final_consumption: float,
        gross_fixed_capital_formation: float,
        change_in_inventories: float,
        net_exports: float,
        price_index: float,
        government_procurement_spending: float,
        government_school_spending: float,
        government_university_spending: float,
        government_public_administration_spending: float,
    ) -> float:
        cpi = max(1e-9, price_index)
        government_deflator = self._government_consumption_price_index_estimate(
            government_procurement_spending,
            government_school_spending,
            government_university_spending,
            government_public_administration_spending,
            fallback=cpi,
        )
        investment_deflator = max(1e-9, self._sector_price_relative("manufactured"))
        inventory_deflator = self._inventory_price_index_estimate(fallback=investment_deflator)
        trade_deflator = cpi

        real_household_final_consumption = household_final_consumption / cpi
        real_government_final_consumption = government_final_consumption / government_deflator
        real_gross_fixed_capital_formation = gross_fixed_capital_formation / investment_deflator
        real_change_in_inventories = change_in_inventories / inventory_deflator
        real_net_exports = net_exports / trade_deflator
        real_gdp = (
            real_household_final_consumption
            + real_government_final_consumption
            + real_gross_fixed_capital_formation
            + real_change_in_inventories
            + real_net_exports
        )
        if gdp_nominal <= 0.0 or real_gdp <= 1e-9:
            return cpi
        return gdp_nominal / real_gdp

    def _current_inventory_book_value(self) -> float:
        return sum(
            max(0.0, firm.inventory) * SECTOR_BY_KEY[firm.sector].base_price
            for firm in self.firms
            if firm.active
        )

    def _current_goods_monetary_mass(self) -> float:
        return sum(
            max(0.0, firm.last_production) * SECTOR_BY_KEY[firm.sector].base_price
            for firm in self.firms
            if firm.active
        )

    def _annual_to_period_growth(self, annual_rate: float) -> float:
        annual_rate = max(-0.99, annual_rate)
        periods_per_year = max(1, self.config.periods_per_year)
        return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0

    def _update_bank_interest_rates(
        self,
        monetary_gap_share: float | None = None,
        target_money_supply: float | None = None,
        current_money_supply: float | None = None,
    ) -> None:
        target_period_inflation = self._annual_to_period_growth(self.config.central_bank_target_annual_inflation)
        observed_inflation = 0.0
        essential_gap = 0.0
        unemployment_gap = 0.0
        if monetary_gap_share is None:
            monetary_gap_share = self._current_monetary_gap_share(
                current_money_supply=current_money_supply,
                target_money_supply=target_money_supply,
            )
        if self.history:
            last_snapshot = self.history[-1]
            essential_gap = max(0.0, 1.0 - last_snapshot.essential_fulfillment_rate)
            unemployment_gap = max(0.0, last_snapshot.unemployment_rate - self.config.target_unemployment)
            if len(self.history) > 1:
                prior_price_index = max(1e-9, self.history[-2].price_index)
                observed_inflation = max(0.0, (last_snapshot.price_index / prior_price_index) - 1.0)

        policy_rate = clamp(
            self.config.central_bank_policy_rate_base
            + 0.75 * (observed_inflation - target_period_inflation)
            - 0.25 * unemployment_gap
            - 0.15 * essential_gap
            - max(0.0, self.config.central_bank_monetary_gap_rate_weight) * monetary_gap_share,
            self.config.central_bank_policy_rate_floor,
            self.config.central_bank_policy_rate_ceiling,
        )
        self.central_bank.policy_rate = policy_rate

        for bank in self.banks:
            reserve_requirement = self._bank_reserve_requirement(bank)
            reserve_stress = (
                clamp((reserve_requirement - bank.reserves) / max(1e-9, reserve_requirement), 0.0, 1.0)
                if reserve_requirement > 0.0
                else 0.0
            )
            warning_ratio = self._bank_warning_capital_ratio()
            comfort_ratio = self._bank_comfort_capital_ratio()
            capital_ratio = self._bank_capital_ratio(bank)
            capital_stress = clamp(
                (warning_ratio - capital_ratio) / max(1e-9, warning_ratio),
                0.0,
                2.0,
            )
            capital_relief = clamp(
                (capital_ratio - warning_ratio) / max(1e-9, comfort_ratio - warning_ratio),
                0.0,
                1.0,
            )
            discount_stress = clamp(
                bank.central_bank_borrowing / max(1.0, bank.deposits + bank.reserves),
                0.0,
                2.0,
            )
            liquidity_relief = clamp(
                max(0.0, bank.reserves - reserve_requirement) / max(1.0, bank.deposits),
                0.0,
                1.0,
            )
            loan_floor = max(0.005, self.config.bank_loan_rate, policy_rate + 0.005)
            bank.loan_rate = clamp(
                loan_floor
                + 0.025 * reserve_stress
                + 0.040 * discount_stress
                + 0.050 * capital_stress
                - 0.010 * liquidity_relief
                - 0.004 * capital_relief,
                loan_floor,
                self._bank_discount_window_rate() + 0.10,
            )
            max_deposit_rate = max(0.0, bank.loan_rate - 0.002)
            bank.deposit_rate = clamp(
                policy_rate * self.config.bank_deposit_rate_share
                + 0.004 * liquidity_relief
                - 0.003 * reserve_stress,
                0.0,
                max_deposit_rate,
            )
            bank.deposit_rate = clamp(
                bank.deposit_rate
                - 0.004 * capital_stress
                + 0.002 * capital_relief,
                0.0,
                max_deposit_rate,
            )
            if discount_stress > 0.0:
                bank.deposit_rate = clamp(
                    bank.deposit_rate - 0.002 * discount_stress,
                    0.0,
                    max_deposit_rate,
                )
            bank.bond_yield = max(self.config.bank_bond_yield, max(0.0, policy_rate - 0.002))

    def _structural_nominal_transactions(self) -> float:
        total = 0.0
        for spec in SECTOR_SPECS:
            total += self._structural_sector_demand(spec.key) * self._average_sector_price(spec.key)
        return total

    def _target_money_supply_fisher(self) -> float:
        current_money_supply = self._current_total_liquid_money()
        current_price_index = max(1e-9, self._current_price_index_estimate())
        structural_nominal_transactions = self._structural_nominal_transactions()
        real_transactions = structural_nominal_transactions / current_price_index
        if self.history:
            target_price_index = max(
                current_price_index,
                self.history[-1].price_index * (1.0 + self._annual_to_period_growth(self.config.central_bank_target_annual_inflation)),
            )
        else:
            target_price_index = max(1.0, current_price_index)
        target_nominal_transactions = real_transactions * target_price_index
        return target_nominal_transactions / max(1e-9, self.config.central_bank_target_velocity)

    def _target_money_supply_goods_growth(self) -> float:
        current_money_supply = self._current_total_liquid_money()
        current_goods_mass = max(1e-9, self._current_goods_monetary_mass())
        pass_through = max(0.0, self.config.central_bank_goods_growth_pass_through)

        if len(self.history) >= 2:
            previous_goods_mass = max(1e-9, self.history[-1].goods_monetary_mass)
            prior_goods_mass = max(1e-9, self.history[-2].goods_monetary_mass)
        elif self.history:
            previous_goods_mass = max(1e-9, self.history[-1].goods_monetary_mass)
            prior_goods_mass = max(1e-9, getattr(self, "_startup_goods_monetary_mass", current_goods_mass))
        else:
            previous_goods_mass = current_goods_mass
            prior_goods_mass = max(1e-9, getattr(self, "_startup_goods_monetary_mass", current_goods_mass))

        goods_growth = max(0.0, (previous_goods_mass - prior_goods_mass) / prior_goods_mass)
        target_growth = goods_growth * pass_through
        return current_money_supply * (1.0 + target_growth)

    def _current_target_money_supply(self) -> float:
        rule = self.config.central_bank_rule.strip().lower()
        if rule == "goods_growth":
            return self._target_money_supply_goods_growth()
        if rule == "productivity_dividend":
            return self._current_total_liquid_money()
        return self._target_money_supply_fisher()

    def _current_monetary_gap_share(
        self,
        current_money_supply: float | None = None,
        target_money_supply: float | None = None,
    ) -> float:
        if current_money_supply is None:
            current_money_supply = self._current_total_liquid_money()
        if target_money_supply is None:
            target_money_supply = self._current_target_money_supply()
        target = max(1e-9, target_money_supply)
        return (target_money_supply - current_money_supply) / target

    def _apply_central_bank_reserve_ratio_policy(self, monetary_gap_share: float) -> None:
        if not self.banks or not self.config.central_bank_dynamic_reserve_ratio_enabled:
            return
        floor = clamp(
            self.config.central_bank_reserve_ratio_floor,
            0.01,
            max(0.01, self.config.central_bank_reserve_ratio_ceiling),
        )
        ceiling = clamp(
            self.config.central_bank_reserve_ratio_ceiling,
            floor,
            0.95,
        )
        baseline = clamp(self.config.reserve_ratio, floor, ceiling)
        sensitivity = max(0.0, self.config.central_bank_reserve_ratio_gap_sensitivity)
        adjustment_speed = clamp(self.config.central_bank_reserve_ratio_adjustment_speed, 0.0, 1.0)
        target_ratio = clamp(baseline - sensitivity * monetary_gap_share, floor, ceiling)
        for bank in self.banks:
            bank.reserve_ratio = clamp(
                bank.reserve_ratio + adjustment_speed * (target_ratio - bank.reserve_ratio),
                floor,
                ceiling,
            )

    def _absorb_reserves_from_banks(self, amount: float, weights: dict[int, float] | None = None) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0 or not self.banks:
            return 0.0
        if weights is None:
            weights = {
                bank.id: max(0.0, bank.reserves - self._bank_reserve_requirement(bank))
                for bank in self.banks
            }
        total_weight = sum(max(0.0, weight) for weight in weights.values())
        if total_weight <= 0.0:
            return 0.0

        absorbed_total = 0.0
        for bank in self.banks:
            available = max(0.0, bank.reserves - self._bank_reserve_requirement(bank))
            if available <= 0.0:
                continue
            weight = max(0.0, weights.get(bank.id, 0.0))
            if weight <= 0.0:
                continue
            target_share = amount * (weight / total_weight)
            absorbed = min(available, target_share)
            if absorbed <= 0.0:
                continue
            bank.reserves -= absorbed
            bank.bond_holdings += absorbed
            absorbed_total += absorbed
        return absorbed_total

    def _conduct_open_market_operations(
        self,
        current_money_supply: float,
        target_money_supply: float,
    ) -> float:
        if not self.banks or not self.config.central_bank_enabled:
            return 0.0
        gap = target_money_supply - current_money_supply
        if abs(gap) <= 1e-9:
            return 0.0
        omo_share = clamp(self.config.central_bank_omo_response_share, 0.0, 1.0)
        max_operation = max(0.0, self.config.central_bank_max_issue_share) * max(1.0, current_money_supply)
        desired_operation = min(abs(gap) * omo_share, max_operation)
        if desired_operation <= 0.0:
            return 0.0

        if gap > 0.0:
            weights = self._estimate_credit_demand_by_bank()
            injected = self._distribute_reserve_injection_to_banks(desired_operation, weights)
            if injected <= 0.0:
                return 0.0
            total_bond_holdings = sum(max(0.0, bank.bond_holdings) for bank in self.banks)
            if total_bond_holdings > 0.0:
                for bank in self.banks:
                    share = max(0.0, bank.bond_holdings) / total_bond_holdings
                    bank.bond_holdings = max(0.0, bank.bond_holdings - injected * share)
            return injected

        weights = {
            bank.id: max(0.0, bank.reserves - self._bank_reserve_requirement(bank))
            for bank in self.banks
        }
        absorbed = self._absorb_reserves_from_banks(desired_operation, weights)
        return -absorbed if absorbed > 0.0 else 0.0

    def _central_bank_recipients(self) -> list[Household]:
        recipients: list[Household] = []
        for members in self._family_groups().values():
            adult_members = [
                member
                for member in members
                if self._household_age_years(member) >= self.config.entry_age_years
            ]
            if adult_members:
                recipient = min(adult_members, key=lambda member: member.id)
            else:
                recipient = min(members, key=lambda member: member.id)
            recipients.append(recipient)
        return recipients

    def _distribute_central_bank_transfer(self, amount: float) -> None:
        recipients = self._central_bank_recipients()
        self._distribute_central_bank_transfer_to_households(recipients, amount)

    def _distribute_central_bank_transfer_to_households(
        self,
        recipients: list[Household],
        amount: float,
    ) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return 0.0
        if not recipients:
            return 0.0
        bank_weights: dict[int, float] = {}
        for household in recipients:
            bank_weights[self._bank_id_for_household(household)] = bank_weights.get(
                self._bank_id_for_household(household),
                0.0,
            ) + 1.0
        self._distribute_reserve_injection_to_banks(amount * max(0.0, self.config.reserve_ratio), bank_weights)
        transfer_per_recipient = amount / len(recipients)
        for household in recipients:
            household.savings += transfer_per_recipient
        return amount

    def _government_primary_recipient(self, family_members: list[Household]) -> Household | None:
        if not family_members:
            return None
        adult_members = [
            member
            for member in family_members
            if self._household_age_years(member) >= self.config.entry_age_years
        ]
        target_pool = adult_members if adult_members else family_members
        return min(target_pool, key=lambda member: member.id)

    def _public_education_payroll_recipients(self, sector_key: str) -> list[Household]:
        adult_members = [
            household
            for household in self._active_households()
            if self._household_age_years(household) >= self.config.entry_age_years
        ]
        if sector_key == "school":
            skilled_members = [
                household
                for household in adult_members
                if self._household_has_school_credential(household)
                or self._household_has_university_credential(household)
            ]
        elif sector_key == "university":
            skilled_members = [
                household
                for household in adult_members
                if self._household_has_university_credential(household)
            ]
        else:
            skilled_members = []
        if skilled_members:
            return skilled_members
        if adult_members:
            return adult_members
        return self._central_bank_recipients()

    def _distribute_public_education_service_income(self, sector_key: str, amount: float) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return 0.0
        recipients = self._public_education_payroll_recipients(sector_key)
        if not recipients:
            return self._distribute_government_spending_leakage(amount)
        transfer_per_recipient = amount / max(1, len(recipients))
        for household in recipients:
            household.savings += transfer_per_recipient
        return amount

    def _record_government_tax_revenue(self, amount: float, revenue_type: str) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return 0.0
        self.government.treasury_cash += amount
        self.government.tax_revenue_this_period += amount
        self._period_government_tax_revenue += amount
        if revenue_type == "labor":
            self.government.labor_tax_revenue += amount
            self._period_government_labor_tax_revenue += amount
        elif revenue_type == "payroll":
            self.government.payroll_tax_revenue += amount
            self._period_government_payroll_tax_revenue += amount
        elif revenue_type == "corporate":
            self.government.corporate_tax_revenue += amount
            self._period_government_corporate_tax_revenue += amount
        elif revenue_type == "dividend":
            self.government.dividend_tax_revenue += amount
            self._period_government_dividend_tax_revenue += amount
        elif revenue_type == "wealth":
            self.government.wealth_tax_revenue += amount
            self._period_government_wealth_tax_revenue += amount
        return amount

    def _government_corporate_tax_rate(self, profit_margin: float) -> float:
        if not self.config.government_enabled:
            return 0.0
        if profit_margin >= self.config.government_corporate_tax_margin_high:
            return self.config.government_corporate_tax_rate_high
        if profit_margin >= self.config.government_corporate_tax_margin_mid:
            return self.config.government_corporate_tax_rate_mid
        return self.config.government_corporate_tax_rate_low

    def _government_dividend_tax_rate(self, dividend: float) -> float:
        if not self.config.government_enabled or dividend <= 0.0:
            return 0.0
        living_wage_anchor = max(1.0, self._living_wage_anchor())
        dividend_multiple = dividend / living_wage_anchor
        if dividend_multiple >= self.config.government_dividend_bracket_high:
            return self.config.government_dividend_tax_rate_high
        if dividend_multiple >= self.config.government_dividend_bracket_low:
            return self.config.government_dividend_tax_rate_mid
        return self.config.government_dividend_tax_rate_low

    def _government_labor_tax_rate(self, gross_wage: float) -> float:
        if not self.config.government_enabled or gross_wage <= 0.0:
            return 0.0
        living_wage_anchor = max(1.0, self._living_wage_anchor())
        wage_multiple = gross_wage / living_wage_anchor
        if wage_multiple >= self.config.government_labor_tax_bracket_high:
            return self.config.government_labor_tax_rate_high
        if wage_multiple >= self.config.government_labor_tax_bracket_low:
            return self.config.government_labor_tax_rate_mid
        return self.config.government_labor_tax_rate_low

    def _government_payroll_tax_rate(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        return max(0.0, self.config.government_payroll_tax_rate)

    def _issue_government_bonds(self, amount: float) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0 or not self.config.government_enabled or not self.banks:
            return 0.0
        self._refresh_bank_balance_sheets()
        remaining = amount
        banks_by_capacity = sorted(
            self.banks,
            key=lambda bank: (self._bank_effective_lending_capacity(bank), self._bank_total_liquidity(bank), -bank.id),
            reverse=True,
        )
        for bank in banks_by_capacity:
            if remaining <= 0.0:
                break
            issuance_capacity = self._bank_effective_lending_capacity(bank)
            if issuance_capacity <= 0.0:
                continue
            issued = min(remaining, issuance_capacity)
            bank.bond_holdings += issued
            self.government.treasury_cash += issued
            self.government.debt_outstanding += issued
            self.government.bond_issuance_this_period += issued
            self._period_government_bond_issuance += issued
            remaining -= issued
        self._refresh_bank_balance_sheets()
        self._reconcile_bank_reserves()
        self.central_bank.money_supply = self._current_total_liquid_money()
        return amount - remaining

    def _government_recession_signal(self) -> tuple[bool, float]:
        if not self.config.government_enabled or not self.config.government_countercyclical_enabled:
            return False, 0.0
        if not self.history:
            return False, 0.0
        lookback = max(1, int(self.config.government_recession_lookback_periods))
        trailing = self.history[-lookback:]
        avg_unemployment = sum(snapshot.unemployment_rate for snapshot in trailing) / len(trailing)
        target_employment_rate = max(1e-9, 1.0 - max(0.0, self.config.target_unemployment))
        output_gaps: list[float] = []
        for snapshot in trailing:
            price_index = max(1e-9, snapshot.price_index)
            real_gdp = snapshot.gdp_nominal / price_index
            potential_multiplier = max(
                1.0,
                target_employment_rate / max(1e-9, snapshot.employment_rate),
            )
            potential_nominal = real_gdp * potential_multiplier * price_index
            output_gap = (snapshot.gdp_nominal - potential_nominal) / max(1e-9, potential_nominal)
            output_gaps.append(output_gap)
        avg_output_gap = sum(output_gaps) / len(output_gaps)
        unemployment_trigger = max(
            0.0,
            avg_unemployment
            - (
                max(0.0, self.config.target_unemployment)
                + max(0.0, self.config.government_recession_unemployment_buffer)
            ),
        )
        output_gap_trigger = max(
            0.0,
            -avg_output_gap - max(0.0, self.config.government_recession_output_gap_threshold),
        )
        transfer_weight = max(0.0, self.config.government_countercyclical_transfer_weight)
        procurement_weight = max(0.0, self.config.government_countercyclical_procurement_weight)
        total_weight = max(1e-9, transfer_weight + procurement_weight)
        normalized_unemployment = unemployment_trigger / 0.10
        normalized_output_gap = output_gap_trigger / 0.12
        intensity = clamp(
            (
                transfer_weight * normalized_unemployment
                + procurement_weight * normalized_output_gap
            )
            / total_weight,
            0.0,
            1.0,
        )
        recession_flag = (
            avg_unemployment
            >= max(0.0, self.config.target_unemployment)
            + max(0.0, self.config.government_recession_unemployment_buffer)
            and avg_output_gap <= -max(0.0, self.config.government_recession_output_gap_threshold)
        )
        if not recession_flag and intensity < 0.15:
            intensity = 0.0
        return recession_flag or intensity > 0.0, intensity

    def _government_countercyclical_support_multiplier(self, recession_intensity: float) -> float:
        if recession_intensity <= 0.0:
            return 1.0
        max_multiplier = max(1.0, self.config.government_countercyclical_support_multiplier_max)
        return 1.0 + (max_multiplier - 1.0) * clamp(recession_intensity, 0.0, 1.0)

    def _government_countercyclical_procurement_multiplier(self, recession_intensity: float) -> float:
        if recession_intensity <= 0.0:
            return 1.0
        max_multiplier = max(1.0, self.config.government_countercyclical_procurement_multiplier_max)
        return 1.0 + (max_multiplier - 1.0) * clamp(recession_intensity, 0.0, 1.0)

    def _plan_government_household_support(
        self,
    ) -> list[tuple[Household, float, float, float]]:
        if not self.config.government_enabled:
            return []
        recession_flag, recession_intensity = self._government_recession_signal()
        living_wage_anchor = self._living_wage_anchor()
        if living_wage_anchor <= 0.0:
            return []
        spending_scale = max(0.0, self.config.government_spending_scale)
        support_multiplier = self._government_countercyclical_support_multiplier(recession_intensity)
        self._period_recession_flag = 1.0 if recession_flag else 0.0
        self._period_recession_intensity = recession_intensity
        self._period_government_countercyclical_support_multiplier = support_multiplier

        support_plan: list[tuple[Household, float, float, float]] = []
        for members in self._family_groups().values():
            family_members = [member for member in members if member.alive]
            if not family_members:
                continue
            recipient = self._government_primary_recipient(family_members)
            if recipient is None:
                continue
            family_basket_cost = sum(self._essential_budget(member) for member in family_members)
            family_income = sum(member.last_income for member in family_members)
            unemployed_adults = sum(
                1
                for member in family_members
                if self._household_labor_capacity(member) > 0.0 and member.employed_by is None
            )
            dependent_children = sum(
                1 for member in family_members if self._household_age_years(member) < self.config.entry_age_years
            )
            unemployment_support = (
                unemployed_adults
                * living_wage_anchor
                * max(0.0, self.config.government_unemployment_benefit_share)
            )
            child_allowance = (
                dependent_children
                * living_wage_anchor
                * max(0.0, self.config.government_child_allowance_share)
            )
            remaining_gap = max(0.0, family_basket_cost - family_income - unemployment_support - child_allowance)
            basic_support = remaining_gap * max(0.0, self.config.government_basic_support_gap_share)
            unemployment_support *= spending_scale
            child_allowance *= spending_scale
            basic_support *= spending_scale
            unemployment_support *= support_multiplier
            basic_support *= support_multiplier
            total_support = unemployment_support + child_allowance + basic_support
            if total_support > 0.0:
                support_plan.append((recipient, unemployment_support, child_allowance, basic_support))
        return support_plan

    def _apply_government_household_support(self) -> float:
        support_plan = self._plan_government_household_support()
        planned_support = sum(
            unemployment_support + child_allowance + basic_support
            for _, unemployment_support, child_allowance, basic_support in support_plan
        )
        if planned_support <= 0.0:
            return 0.0

        financing_gap = max(0.0, planned_support - self.government.treasury_cash)
        if financing_gap > 0.0:
            self._issue_government_bonds(financing_gap)

        affordable_scale = min(1.0, self.government.treasury_cash / max(1e-9, planned_support))
        if affordable_scale <= 0.0:
            return 0.0

        spending_efficiency = clamp(self.config.government_spending_efficiency, 0.0, 1.0)
        total_paid = 0.0
        support_multiplier = max(1.0, self._period_government_countercyclical_support_multiplier)
        for recipient, unemployment_support, child_allowance, basic_support in support_plan:
            gross_unemployment_support = unemployment_support * affordable_scale
            gross_child_allowance = child_allowance * affordable_scale
            gross_basic_support = basic_support * affordable_scale
            gross_transfer = gross_unemployment_support + gross_child_allowance + gross_basic_support
            if gross_transfer <= 0.0:
                continue
            effective_transfer = gross_transfer * spending_efficiency
            leakage = gross_transfer - effective_transfer
            recipient.savings += effective_transfer
            if leakage > 0.0:
                self._distribute_government_spending_leakage(leakage)
            self.government.treasury_cash = max(0.0, self.government.treasury_cash - gross_transfer)
            self.government.transfers_this_period += gross_transfer
            self.government.unemployment_support_this_period += gross_unemployment_support
            self.government.child_allowance_this_period += gross_child_allowance
            self.government.basic_support_this_period += gross_basic_support
            self._period_government_transfers += gross_transfer
            self._period_government_unemployment_support += gross_unemployment_support
            self._period_government_child_allowance += gross_child_allowance
            self._period_government_basic_support += gross_basic_support
            total_paid += gross_transfer
            if support_multiplier > 1.0:
                countercyclical_extra = (
                    gross_unemployment_support + gross_basic_support
                ) * (1.0 - 1.0 / support_multiplier)
                self._period_government_countercyclical_spending += max(0.0, countercyclical_extra)
        return total_paid

    def _apply_government_essential_procurement(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        recession_flag = bool(self._period_recession_flag)
        recession_intensity = self._period_recession_intensity
        if not recession_flag and recession_intensity <= 0.0:
            recession_flag, recession_intensity = self._government_recession_signal()
            self._period_recession_flag = 1.0 if recession_flag else 0.0
            self._period_recession_intensity = recession_intensity
        population = len(self._active_households())
        if population <= 0:
            return 0.0

        spending_scale = max(0.0, self.config.government_spending_scale)
        procurement_multiplier = self._government_countercyclical_procurement_multiplier(recession_intensity)
        self._period_government_countercyclical_procurement_multiplier = procurement_multiplier
        spending_efficiency = clamp(self.config.government_spending_efficiency, 0.0, 1.0)
        structural_procurement_budget = (
            max(0.0, self.config.government_structural_procurement_budget_share)
            * self._government_structural_budget_anchor()
        )
        procurement_targets: list[tuple[str, float, float]] = []
        total_budget_needed = 0.0
        for sector_key in ESSENTIAL_SECTOR_KEYS:
            minimum_required_units = population * self._essential_basket_share(sector_key)
            private_units_sold = self._period_sector_sales_units.get(sector_key, 0.0)
            unmet_units = max(0.0, minimum_required_units - private_units_sold)
            average_price = self._average_sector_price(sector_key)
            structural_units = (
                structural_procurement_budget
                * self._essential_basket_share(sector_key)
                * procurement_multiplier
                / max(0.1, average_price)
            )
            gap_units = (
                unmet_units
                * max(0.0, self.config.government_procurement_gap_share)
                * spending_scale
                * procurement_multiplier
            )
            target_units = gap_units + structural_units
            if target_units <= 0.0:
                continue
            budget_needed = target_units * average_price
            procurement_targets.append((sector_key, target_units, budget_needed))
            total_budget_needed += budget_needed

        if total_budget_needed <= 0.0:
            return 0.0

        financing_gap = max(0.0, total_budget_needed - self.government.treasury_cash)
        if financing_gap > 0.0:
            self._issue_government_bonds(financing_gap)

        total_spent = 0.0
        spending_log = {spec.key: 0.0 for spec in SECTOR_SPECS}
        for sector_key, target_units, budget_needed in procurement_targets:
            if self.government.treasury_cash <= 0.0:
                break
            gross_budget_cap = min(self.government.treasury_cash, budget_needed)
            effective_budget_cap = gross_budget_cap * spending_efficiency
            if effective_budget_cap <= 0.0:
                continue
            remaining_cash, _ = self._purchase_from_sector(
                max(0.1, self.config.government_procurement_price_sensitivity),
                sector_key,
                target_units,
                effective_budget_cap,
                spending_log,
            )
            effective_spent = effective_budget_cap - remaining_cash
            if effective_spent <= 0.0:
                continue
            gross_spent = effective_spent / max(1e-9, spending_efficiency)
            leakage = gross_spent - effective_spent
            if leakage > 0.0:
                self._distribute_government_spending_leakage(leakage)
            self.government.treasury_cash = max(0.0, self.government.treasury_cash - gross_spent)
            self.government.procurement_spending_this_period += gross_spent
            self._period_government_procurement_spending += gross_spent
            total_spent += gross_spent
            if procurement_multiplier > 1.0:
                countercyclical_extra = gross_spent * (1.0 - 1.0 / procurement_multiplier)
                self._period_government_countercyclical_spending += max(0.0, countercyclical_extra)
        return total_spent

    def _apply_government_infrastructure_investment(self) -> float:
        depreciation_rate = clamp(self.config.government_public_capital_depreciation_rate, 0.0, 0.25)
        self.government.public_capital_stock = max(
            0.0,
            self.government.public_capital_stock * (1.0 - depreciation_rate),
        )
        if not self.config.government_enabled:
            return 0.0

        infrastructure_budget = (
            max(0.0, self.config.government_infrastructure_budget_share)
            * self._government_structural_budget_anchor()
        )
        if infrastructure_budget <= 0.0:
            return 0.0

        financing_gap = max(0.0, infrastructure_budget - self.government.treasury_cash)
        if financing_gap > 0.0:
            self._issue_government_bonds(financing_gap)

        affordable_budget = min(self.government.treasury_cash, infrastructure_budget)
        spending_efficiency = clamp(self.config.government_spending_efficiency, 0.0, 1.0)
        effective_budget = affordable_budget * spending_efficiency
        if effective_budget <= 0.0:
            return 0.0

        manufactured_share = 0.60
        housing_share = 0.40
        manufactured_spend = effective_budget * manufactured_share
        housing_spend = effective_budget * housing_share
        self._distribute_payment_to_sector("manufactured", manufactured_spend, book_profit=True)
        self._distribute_payment_to_sector("housing", housing_spend, book_profit=True)
        realized_investment = manufactured_spend + housing_spend
        gross_spent = realized_investment / max(1e-9, spending_efficiency)
        leakage = gross_spent - realized_investment
        if leakage > 0.0:
            self._distribute_government_spending_leakage(leakage)
        self.government.treasury_cash = max(0.0, self.government.treasury_cash - gross_spent)
        self.government.infrastructure_spending_this_period += gross_spent
        self._period_government_infrastructure_spending += gross_spent
        self._period_government_public_capital_formation += realized_investment
        self.government.public_capital_stock += realized_investment
        return gross_spent

    def _collect_government_wealth_tax(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        living_wage_anchor = self._living_wage_anchor()
        if living_wage_anchor <= 0.0:
            return 0.0
        threshold = living_wage_anchor * max(0.0, self.config.government_wealth_tax_threshold_multiple)
        tax_rate = max(0.0, self.config.government_wealth_tax_rate)
        total_collected = 0.0
        for owner in self.entrepreneurs:
            if not owner.active:
                continue
            taxable_wealth = max(0.0, self._owner_total_liquid(owner) - threshold)
            if taxable_wealth <= 0.0:
                continue
            due = taxable_wealth * tax_rate
            collected = self._withdraw_owner_liquid(owner, due)
            total_collected += self._record_government_tax_revenue(collected, "wealth")
        return total_collected

    def _government_recent_tax_anchor(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        lookback = max(1, int(self.config.periods_per_year))
        tax_values = [max(0.0, snapshot.government_tax_revenue) for snapshot in self.history[-lookback:]]
        current_tax = max(0.0, self._period_government_tax_revenue)
        if current_tax > 0.0:
            tax_values.append(current_tax)
        if not tax_values:
            return 0.0
        return sum(tax_values) / len(tax_values)

    def _government_recent_direct_spending_anchor(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        lookback = max(1, int(self.config.periods_per_year))
        direct_spending = [
            max(
                0.0,
                snapshot.government_procurement_spending
                + snapshot.government_education_spending
                + getattr(snapshot, "government_public_administration_spending", 0.0)
                + getattr(snapshot, "government_infrastructure_spending", 0.0),
            )
            for snapshot in self.history[-lookback:]
        ]
        current_direct_spending = max(
            0.0,
            self._period_government_procurement_spending
            + self._period_government_education_spending
            + self._period_government_public_administration_spending
            + self._period_government_infrastructure_spending,
        )
        if current_direct_spending > 0.0:
            direct_spending.append(current_direct_spending)
        if not direct_spending:
            return 0.0
        return sum(direct_spending) / len(direct_spending)

    def _government_recent_gdp_anchor(self) -> float:
        if not self.history:
            return 0.0
        lookback = max(1, int(self.config.periods_per_year))
        gdp_values = [max(0.0, getattr(snapshot, "gdp_nominal", 0.0)) for snapshot in self.history[-lookback:]]
        if not gdp_values:
            return 0.0
        return sum(gdp_values) / len(gdp_values)

    def _government_direct_budget_share_sum(self) -> float:
        return max(
            0.0,
            max(0.0, self.config.government_structural_procurement_budget_share)
            + max(0.0, self.config.public_school_budget_share)
            + max(0.0, self.config.public_university_budget_share)
            + max(0.0, self.config.public_administration_budget_share),
        )

    def _government_structural_budget_anchor(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        recent_tax = self._government_recent_tax_anchor()
        recent_direct_spending = self._government_recent_direct_spending_anchor()
        recent_gdp = self._government_recent_gdp_anchor()
        annualized_tax = recent_tax * max(1, int(self.config.periods_per_year))
        debt_ratio = self.government.debt_outstanding / max(1e-9, annualized_tax)
        debt_brake = clamp(1.0 - 0.18 * max(0.0, debt_ratio - 1.0), 0.35, 1.0)
        deficit_tolerance = clamp(self.config.government_structural_deficit_tolerance, 0.0, 2.0)
        structural_budget = recent_tax * (1.0 + deficit_tolerance * debt_brake)
        direct_budget_share_sum = self._government_direct_budget_share_sum()
        direct_consumption_floor = max(0.0, self.config.government_final_consumption_floor_share_gdp)
        direct_budget_floor = 0.0
        if recent_gdp > 0.0 and direct_budget_share_sum > 1e-9 and direct_consumption_floor > 0.0:
            direct_budget_floor = recent_gdp * direct_consumption_floor / direct_budget_share_sum
        return max(
            0.0,
            self.government.treasury_cash,
            recent_direct_spending,
            structural_budget,
            direct_budget_floor,
        )

    def _public_administration_budget(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        return (
            max(0.0, self.config.public_administration_budget_share)
            * self._government_structural_budget_anchor()
        )

    def _public_administration_worker_score(self, household: Household) -> tuple[float, float, float]:
        return (
            self._worker_effective_labor_for_sector(household, "public_administration"),
            -household.reservation_wage,
            -self._household_age_years(household),
        )

    def _manage_public_administration_workforce(self) -> None:
        public_workers = [
            household
            for household in self._active_households()
            if household.employed_by == PUBLIC_ADMINISTRATION_EMPLOYER_ID
        ]
        if not self.config.government_enabled:
            for household in public_workers:
                self._release_household_from_employment(household)
            return
        target_workers = self._public_administration_target_workers()
        if len(public_workers) > target_workers:
            release_order = sorted(
                public_workers,
                key=lambda household: (
                    self._worker_effective_labor_for_sector(household, "public_administration"),
                    -household.reservation_wage,
                    household.id,
                ),
            )
            for household in release_order[: len(public_workers) - target_workers]:
                self._release_household_from_employment(household)
            public_workers = [
                household
                for household in self._active_households()
                if household.employed_by == PUBLIC_ADMINISTRATION_EMPLOYER_ID
            ]
        if len(public_workers) >= target_workers:
            return
        wage_offer = self._public_administration_wage_offer()
        candidates = [
            household
            for household in self._active_households()
            if self._household_labor_capacity(household) > 0.0
            and household.employed_by is None
            and household.reservation_wage <= wage_offer
        ]
        candidates.sort(key=self._public_administration_worker_score, reverse=True)
        vacancies = target_workers - len(public_workers)
        for household in candidates[:vacancies]:
            household.employed_by = PUBLIC_ADMINISTRATION_EMPLOYER_ID
            household.employment_tenure = 0

    def _pay_public_administration_wages(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        public_workers = [
            household
            for household in self._active_households()
            if household.employed_by == PUBLIC_ADMINISTRATION_EMPLOYER_ID
        ]
        if not public_workers:
            return 0.0
        wage_offer = self._public_administration_wage_offer()
        planned_payroll = wage_offer * len(public_workers)
        administration_budget = self._public_administration_budget()
        nonpayroll_budget = max(0.0, administration_budget - self._public_administration_payroll_budget())
        planned_total_spending = planned_payroll + nonpayroll_budget
        if planned_total_spending <= 0.0:
            return 0.0
        financing_gap = max(0.0, planned_total_spending - self.government.treasury_cash)
        if financing_gap > 0.0:
            self._issue_government_bonds(financing_gap)
        affordable_scale = min(1.0, self.government.treasury_cash / max(1e-9, planned_total_spending))
        if affordable_scale <= 0.0:
            return 0.0
        effective_wage = wage_offer * affordable_scale
        total_paid = effective_wage * len(public_workers)
        for household in public_workers:
            labor_tax = min(
                effective_wage,
                effective_wage * self._government_labor_tax_rate(effective_wage),
            )
            net_wage = effective_wage - labor_tax
            household.wage_income += net_wage
            household.last_income += net_wage
            if labor_tax > 0.0:
                self._record_government_tax_revenue(labor_tax, "labor")
        self.government.treasury_cash = max(0.0, self.government.treasury_cash - total_paid)
        self.government.public_administration_spending_this_period += total_paid
        self._period_government_public_administration_spending += total_paid
        self._period_wages += total_paid
        self._period_sales_revenue += total_paid
        nonpayroll_spending = nonpayroll_budget * affordable_scale
        if nonpayroll_spending > 0.0:
            self.government.treasury_cash = max(0.0, self.government.treasury_cash - nonpayroll_spending)
            self.government.public_administration_spending_this_period += nonpayroll_spending
            self._period_government_public_administration_spending += nonpayroll_spending
            self._queue_sector_payment("manufactured", nonpayroll_spending * 0.60)
            self._queue_sector_payment("housing", nonpayroll_spending * 0.40)
        return total_paid + nonpayroll_spending

    def _finalize_government_period(self) -> None:
        spending = (
            self._period_government_transfers
            + self._period_government_procurement_spending
            + self._period_government_education_spending
            + self._period_government_public_administration_spending
            + self._period_government_infrastructure_spending
        )
        tax_revenue = self._period_government_tax_revenue
        net_balance = tax_revenue - spending
        self._period_government_deficit = max(0.0, -net_balance)
        self._period_government_surplus = max(0.0, net_balance)
        self.government.deficit_this_period = self._period_government_deficit
        self.government.surplus_this_period = self._period_government_surplus
        self.government.cumulative_deficit += self._period_government_deficit - self._period_government_surplus

    def _current_output_per_worker_estimate(self, firm: Firm) -> float:
        if not firm.workers:
            return 0.0
        effective_labor_units = sum(
            self._worker_effective_labor_for_sector(self.households[worker_id], firm.sector)
            for worker_id in firm.workers
        )
        if effective_labor_units <= 0.0:
            return 0.0
        output_units = self._firm_effective_productivity(firm) * effective_labor_units
        return output_units / len(firm.workers)

    def _apply_productivity_dividend_policy(self) -> float:
        if not self.history:
            return 0.0
        dividend_share = max(0.0, self.config.central_bank_productivity_dividend_share)
        if dividend_share <= 0.0:
            return 0.0

        total_issuance = 0.0
        for firm in self.firms:
            if not firm.active or not firm.workers:
                continue
            if firm.last_worker_count <= 0 or firm.last_production <= 0.0:
                continue

            previous_output_per_worker = firm.last_production / max(1, firm.last_worker_count)
            current_output_per_worker = self._current_output_per_worker_estimate(firm)
            productivity_gain_per_worker = max(0.0, current_output_per_worker - previous_output_per_worker)
            if productivity_gain_per_worker <= 0.0:
                continue

            monetary_gain_per_worker = productivity_gain_per_worker * max(0.1, firm.price) * dividend_share
            firm_issuance = monetary_gain_per_worker * len(firm.workers)
            recipients = [
                self.households[worker_id]
                for worker_id in firm.workers
                if self.households[worker_id].alive
            ]
            issued = self._distribute_central_bank_transfer_to_households(recipients, firm_issuance)
            total_issuance += issued
        return total_issuance

    def _apply_central_bank_policy(self) -> None:
        current_money_supply = self._current_total_liquid_money()
        self.central_bank.money_supply = current_money_supply
        if not self.config.central_bank_enabled:
            self.central_bank.target_money_supply = current_money_supply
            self.central_bank.issuance_this_period = 0.0
            self._period_central_bank_target_money_supply = current_money_supply
            self._update_bank_interest_rates()
            return

        rule = self.config.central_bank_rule.strip().lower()
        if rule == "productivity_dividend":
            issuance = self._apply_productivity_dividend_policy()
            current_money_supply += issuance
            target_money_supply = current_money_supply
        elif rule == "goods_growth":
            target_money_supply = self._target_money_supply_goods_growth()
            monetary_gap_share = self._current_monetary_gap_share(
                current_money_supply=current_money_supply,
                target_money_supply=target_money_supply,
            )
            self._apply_central_bank_reserve_ratio_policy(monetary_gap_share)
            issuance = self._conduct_open_market_operations(current_money_supply, target_money_supply)
        else:
            target_money_supply = self._target_money_supply_fisher()
            monetary_gap_share = self._current_monetary_gap_share(
                current_money_supply=current_money_supply,
                target_money_supply=target_money_supply,
            )
            self._apply_central_bank_reserve_ratio_policy(monetary_gap_share)
            issuance = self._conduct_open_market_operations(current_money_supply, target_money_supply)
        self.central_bank.cumulative_issuance += issuance
        self.central_bank.money_supply = self._current_total_liquid_money()
        self.central_bank.target_money_supply = target_money_supply
        self.central_bank.issuance_this_period = issuance
        self._period_central_bank_issuance = issuance
        self._period_central_bank_target_money_supply = target_money_supply
        self._update_bank_interest_rates(
            monetary_gap_share=self._current_monetary_gap_share(
                current_money_supply=self.central_bank.money_supply,
                target_money_supply=target_money_supply,
            ),
            target_money_supply=target_money_supply,
            current_money_supply=self.central_bank.money_supply,
        )

    def _supplier_firm_weight(self, firm: Firm) -> float:
        expected_sales = max(1.0, firm.last_expected_sales, firm.last_sales)
        return max(1.0, expected_sales + 0.10 * firm.capital + 0.50 * len(firm.workers))

    def _queue_sector_payment(self, sector_key: str, amount: float) -> None:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return
        self._pending_sector_payments[sector_key] = self._pending_sector_payments.get(sector_key, 0.0) + amount

    def _book_post_settlement_revenue(self, firm: Firm, amount: float) -> None:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return
        firm.last_revenue += amount
        firm.last_profit += amount
        self._period_profit += amount

    def _distribute_payment_to_sector(
        self,
        sector_key: str,
        amount: float,
        *,
        book_profit: bool = False,
        track_operating_cost: bool = False,
    ) -> None:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return
        firms = self._sector_firms(sector_key)
        if not firms:
            firms = [firm for firm in self.firms if firm.active]
        if not firms:
            if not self.entrepreneurs:
                return
            equal_share = amount / len(self.entrepreneurs)
            for owner in self.entrepreneurs:
                owner.wealth += equal_share
            if track_operating_cost:
                self._period_business_cost_recycled += amount
                self._period_business_cost_to_owners += amount
            return
        weights = [self._supplier_firm_weight(firm) for firm in firms]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            equal_share = amount / len(firms)
            for firm in firms:
                firm.cash += equal_share
                if book_profit:
                    self._book_post_settlement_revenue(firm, equal_share)
        else:
            for firm, weight in zip(firms, weights):
                payment = amount * (weight / total_weight)
                firm.cash += payment
                if book_profit:
                    self._book_post_settlement_revenue(firm, payment)

        if track_operating_cost:
            self._period_business_cost_recycled += amount
            self._period_business_cost_to_firms += amount

    def _flush_pending_sector_payments(self) -> None:
        pending_payments = self._pending_sector_payments.copy()
        self._pending_sector_payments = {spec.key: 0.0 for spec in SECTOR_SPECS}
        for sector_key, amount in pending_payments.items():
            self._distribute_payment_to_sector(
                sector_key,
                amount,
                track_operating_cost=True,
            )

    def _draw_household_sector_preference_weights(self) -> dict[str, float]:
        return {
            "food": self.rng.uniform(0.90, 1.12),
            "housing": self.rng.uniform(0.88, 1.10),
            "clothing": self.rng.uniform(0.75, 1.25),
            "manufactured": self.rng.uniform(0.45, 1.75),
            "leisure": self.rng.uniform(0.35, 1.85),
            "school": self.rng.uniform(0.80, 1.55),
            "university": self.rng.uniform(0.55, 1.85),
        }

    def _school_years_progress_from_age(self, age_years: float) -> float:
        if age_years <= self.config.school_age_min_years:
            return 0.0
        school_span = max(1.0, self.config.school_age_max_years - self.config.school_age_min_years)
        progress = clamp((age_years - self.config.school_age_min_years) / school_span, 0.0, 1.0)
        return progress * self.config.school_years_required

    def _initialize_household_education(self) -> None:
        school_completion_share = clamp(self.config.initial_school_completion_share, 0.0, 1.0)
        university_threshold = 1.0 - clamp(self.config.initial_university_completion_share, 0.0, 1.0)
        for household in self.households:
            age_years = self._household_age_years(household)
            school_progress = self._school_years_progress_from_age(age_years)
            if age_years < self.config.entry_age_years:
                household.school_years_completed = clamp(
                    school_progress * self.rng.uniform(0.92, 1.04),
                    0.0,
                    self.config.school_years_required,
                )
                household.university_years_completed = 0.0
                continue

            if self.rng.random() < school_completion_share:
                household.school_years_completed = self.config.school_years_required
            else:
                household.school_years_completed = clamp(
                    self.config.school_years_required * self.rng.uniform(0.45, 0.95),
                    0.0,
                    self.config.school_years_required,
                )

            if (
                household.higher_education_affinity >= university_threshold
                and household.school_years_completed >= self.config.school_years_required
                and age_years >= self.config.university_age_min_years
            ):
                if age_years >= self.config.university_age_min_years + self.config.university_years_required:
                    household.university_years_completed = self.config.university_years_required
                else:
                    progress = clamp(
                        (age_years - self.config.university_age_min_years)
                        / max(1.0, self.config.university_years_required),
                        0.0,
                        1.0,
                    )
                    household.university_years_completed = self.config.university_years_required * progress
            else:
                household.university_years_completed = 0.0

    def _household_has_school_credential(self, household: Household) -> bool:
        return household.school_years_completed >= self.config.school_years_required

    def _household_has_university_credential(self, household: Household) -> bool:
        return household.university_years_completed >= self.config.university_years_required

    def _is_school_age(self, household: Household) -> bool:
        age_years = self._household_age_years(household)
        return self.config.school_age_min_years <= age_years < self.config.school_age_max_years

    def _is_university_age(self, household: Household) -> bool:
        age_years = self._household_age_years(household)
        return self.config.university_age_min_years <= age_years < self.config.university_age_max_years

    def _public_education_target_units(
        self,
        household: Household,
        sector_key: str,
        target_units: float,
    ) -> float:
        target_units = max(0.0, target_units)
        if target_units <= 0.0:
            return 0.0
        age_years = self._household_age_years(household)
        family_resource_coverage = self._household_family_resource_coverage(household)
        low_resource_bonus = (
            max(0.0, self.config.public_education_low_resource_priority_bonus)
            if family_resource_coverage < 1.0
            else 0.0
        )
        if sector_key == "school":
            if self._is_school_age(household):
                coverage_share = max(
                    max(0.0, self.config.public_school_min_target_units),
                    max(0.0, self.config.public_school_support_package_share),
                )
                coverage_share += low_resource_bonus
                coverage_share += (
                    max(0.0, self.config.public_school_support_continuity_bonus)
                    * household.public_school_support_persistence
                )
            elif (
                age_years < self.config.adult_school_catchup_age_max_years
                and not self._household_has_school_credential(household)
            ):
                coverage_share = max(
                    max(0.0, self.config.adult_school_catchup_target_units),
                    0.60 * max(0.0, self.config.public_school_support_package_share),
                )
                coverage_share += 0.5 * low_resource_bonus
            else:
                coverage_share = 0.0
        elif sector_key == "university":
            if self._is_university_age(household) and self._household_has_school_credential(household):
                coverage_share = max(
                    max(0.0, self.config.public_university_min_target_units),
                    max(0.0, self.config.public_university_support_package_share),
                )
                coverage_share += low_resource_bonus
                coverage_share += (
                    max(0.0, self.config.public_university_support_continuity_bonus)
                    * household.public_university_support_persistence
                )
            elif (
                age_years < self.config.adult_university_catchup_age_max_years
                and self._household_has_school_credential(household)
                and not self._household_has_university_credential(household)
            ):
                coverage_share = max(
                    max(0.0, self.config.adult_university_catchup_target_units),
                    0.55 * max(0.0, self.config.public_university_support_package_share),
                )
                coverage_share += 0.5 * low_resource_bonus
            else:
                coverage_share = 0.0
        else:
            return 0.0
        return target_units * clamp(coverage_share, 0.0, 1.0)

    def _university_track_threshold(self) -> float:
        return 1.0 - clamp(self.config.initial_university_completion_share, 0.0, 1.0)

    def _household_is_university_track(self, household: Household) -> bool:
        return household.higher_education_affinity >= self._university_track_threshold()

    def _household_university_track_factor(self, household: Household) -> float:
        threshold = self._university_track_threshold()
        distance = household.higher_education_affinity - threshold
        factor = clamp(0.45 + 1.60 * (distance + 0.18), 0.15, 1.20)
        if self._household_family_resource_coverage(household) < 1.0:
            factor += 0.10
        factor += 0.20 * household.public_university_support_persistence
        return clamp(factor, 0.0, 1.25)

    def _household_family_resource_coverage(self, household: Household) -> float:
        cached = self._period_family_resource_coverage_cache.get(household.id)
        if cached is not None:
            return cached
        root_id = (
            self._family_root_for_child(household)
            if self._household_age_years(household) < self.config.entry_age_years
            else self._family_root_for_adult(household)
        )
        family_members = [member for member in self._family_groups().get(root_id, []) if member.alive]
        family_resources = sum(self._household_cash_balance(member) for member in family_members)
        family_basket_cost = sum(self._essential_budget(member) for member in family_members)
        coverage = family_resources / max(1.0, family_basket_cost)
        for member in family_members:
            self._period_family_resource_coverage_cache[member.id] = coverage
        return coverage

    def _household_education_market_factor(self, household: Household, *, advanced: bool) -> float:
        coverage = self._household_family_resource_coverage(household)
        if advanced:
            return clamp((coverage - 0.85) / 0.90, 0.0, 1.10)
        return clamp((coverage - 0.05) / 0.55, 0.0, 1.15)

    def _public_education_service_units(self, sector_key: str, spending: float) -> float:
        if spending <= 0.0 or sector_key not in ("school", "university"):
            return 0.0
        reference_price = max(0.1, self._average_sector_price(sector_key))
        return spending * max(0.1, self.config.government_spending_efficiency) / reference_price

    def _public_education_supply_signal(self, sector_key: str, *, use_current_period: bool = False) -> float:
        if sector_key == "school":
            spending = (
                self._period_government_school_spending
                if use_current_period
                else (self.history[-1].government_school_spending if self.history else 0.0)
            )
        elif sector_key == "university":
            spending = (
                self._period_government_university_spending
                if use_current_period
                else (self.history[-1].government_university_spending if self.history else 0.0)
            )
        else:
            return 0.0
        return self._public_education_service_units(sector_key, spending)

    def _school_service_target_units(self, household: Household) -> float:
        if not self._is_school_age(household):
            return 0.0
        preference = self._household_sector_preference(household, "school")
        continuity_bonus = 0.08 * household.public_school_support_persistence
        return clamp(0.92 + 0.16 * preference + continuity_bonus, 0.70, 1.20)

    def _university_service_target_units(self, household: Household) -> float:
        if not self._is_university_age(household):
            return 0.0
        if self._household_has_university_credential(household):
            return 0.0
        if not self._household_has_school_credential(household):
            return 0.0
        preference = self._household_sector_preference(household, "university")
        market_factor = clamp(
            0.55 + 0.45 * min(1.5, self._household_family_resource_coverage(household)) / 1.5,
            0.45,
            1.05,
        )
        track_factor = self._household_university_track_factor(household)
        continuity_bonus = 0.10 * household.public_university_support_persistence
        return clamp((0.72 + 0.22 * preference + continuity_bonus) * market_factor * track_factor, 0.0, 1.30)

    def _household_skill_multiplier(self, household: Household, sector_key: str) -> float:
        if sector_key in ESSENTIAL_SECTOR_KEYS:
            return 1.0
        has_school = self._household_has_school_credential(household)
        has_university = self._household_has_university_credential(household)
        if sector_key == "manufactured":
            return 1.22 if has_university else 1.05 if has_school else 0.88
        if sector_key == "leisure":
            return 1.32 if has_university else 1.08 if has_school else 0.82
        if sector_key == "school":
            return 1.25 if has_university else 1.12 if has_school else 0.78
        if sector_key == "university":
            return 1.45 if has_university else 1.05 if has_school else 0.72
        if sector_key == "public_administration":
            return 1.38 if has_university else 1.12 if has_school else 0.80
        return 1.0

    def _ensure_household_demand_shares(self, household: Household) -> None:
        if household.essential_shares and household.discretionary_shares:
            return

        essential_weights = {
            key: SECTOR_BY_KEY[key].essential_need * self._household_sector_preference(household, key)
            for key in ESSENTIAL_SECTOR_KEYS
        }
        total_essential_weight = sum(essential_weights.values())
        if total_essential_weight <= 0.0:
            total_essential_weight = sum(SECTOR_BY_KEY[key].essential_need for key in ESSENTIAL_SECTOR_KEYS)
            household.essential_shares = {
                key: SECTOR_BY_KEY[key].essential_need / max(1e-9, total_essential_weight)
                for key in ESSENTIAL_SECTOR_KEYS
            }
        else:
            household.essential_shares = {
                key: weight / total_essential_weight
                for key, weight in essential_weights.items()
            }

        discretionary_weights = {
            key: SECTOR_BY_KEY[key].household_demand_share * self._household_sector_preference(household, key)
            for key in DISCRETIONARY_SECTOR_KEYS
        }
        total_discretionary_weight = sum(discretionary_weights.values())
        if total_discretionary_weight <= 0.0:
            total_discretionary_weight = sum(
                SECTOR_BY_KEY[key].household_demand_share for key in DISCRETIONARY_SECTOR_KEYS
            )
            household.discretionary_shares = {
                key: SECTOR_BY_KEY[key].household_demand_share / max(1e-9, total_discretionary_weight)
                for key in DISCRETIONARY_SECTOR_KEYS
            }
        else:
            household.discretionary_shares = {
                key: weight / total_discretionary_weight
                for key, weight in discretionary_weights.items()
            }

    def _household_sector_preference(self, household: Household, sector_key: str) -> float:
        return max(0.2, household.sector_preference_weights.get(sector_key, 1.0))

    def _household_essential_share(self, household: Household, sector_key: str) -> float:
        self._ensure_household_demand_shares(household)
        return household.essential_shares.get(sector_key, 0.0)

    def _household_discretionary_share(self, household: Household, sector_key: str) -> float:
        self._ensure_household_demand_shares(household)
        return household.discretionary_shares.get(sector_key, 0.0)

    def _ensure_period_essential_household_arrays(self) -> None:
        if (
            self._period_essential_desired_units_matrix is not None
            and self._period_essential_budget_vector is not None
            and self._period_household_row_index_cache
        ):
            return

        active_households = self._period_active_households_cache
        if active_households is None:
            active_households = self._active_households()

        household_count = len(active_households)
        self._period_household_row_index_cache = {
            household.id: index
            for index, household in enumerate(active_households)
        }
        if household_count == 0:
            self._period_essential_desired_units_matrix = np.empty((0, len(ARRAY_BACKED_SECTOR_KEYS)), dtype=np.float64)
            self._period_essential_budget_vector = np.empty(0, dtype=np.float64)
            return

        age_years = np.empty(household_count, dtype=np.float64)
        need_scale = np.empty(household_count, dtype=np.float64)
        food_share = np.empty(household_count, dtype=np.float64)
        housing_share = np.empty(household_count, dtype=np.float64)
        clothing_share = np.empty(household_count, dtype=np.float64)
        manufactured_share = np.empty(household_count, dtype=np.float64)
        leisure_share = np.empty(household_count, dtype=np.float64)

        for index, household in enumerate(active_households):
            self._ensure_household_demand_shares(household)
            age_years[index] = self._household_age_years(household)
            need_scale[index] = household.need_scale
            food_share[index] = household.essential_shares.get("food", 0.0)
            housing_share[index] = household.essential_shares.get("housing", 0.0)
            clothing_share[index] = household.essential_shares.get("clothing", 0.0)
            manufactured_share[index] = household.discretionary_shares.get("manufactured", 0.0)
            leisure_share[index] = household.discretionary_shares.get("leisure", 0.0)

        discretionary_scale = (
            self.config.nonessential_demand_multiplier
            * sum(SECTOR_BY_KEY[key].household_demand_share for key in DISCRETIONARY_SECTOR_KEYS)
        )

        desired_units, essential_budgets = compute_household_baseline_demand_arrays(
            age_years,
            need_scale,
            food_share,
            housing_share,
            clothing_share,
            manufactured_share,
            leisure_share,
            entry_age_years=self.config.entry_age_years,
            senior_age_years=self.config.senior_age_years,
            max_age_years=self.config.max_age_years,
            child_consumption_multiplier=self.config.child_consumption_multiplier,
            senior_consumption_multiplier=self.config.senior_consumption_multiplier,
            discretionary_scale=discretionary_scale,
            food_price=self._average_sector_price("food"),
            housing_price=self._average_sector_price("housing"),
            clothing_price=self._average_sector_price("clothing"),
        )
        self._period_essential_desired_units_matrix = desired_units
        self._period_essential_budget_vector = essential_budgets

    def _household_sector_desired_units(self, household: Household, sector_key: str) -> float:
        if sector_key in ARRAY_BACKED_SECTOR_KEYS:
            self._ensure_period_essential_household_arrays()
            row_index = self._period_household_row_index_cache.get(household.id)
            if (
                row_index is not None
                and self._period_essential_desired_units_matrix is not None
            ):
                sector_index = ARRAY_BACKED_SECTOR_INDEX[sector_key]
                return float(self._period_essential_desired_units_matrix[row_index, sector_index])

        cache_key = (
            household.id,
            household.age_periods,
            household.school_years_completed,
            household.university_years_completed,
            household.higher_education_affinity,
            sector_key,
        )
        cached = self._period_household_desired_units_cache.get(cache_key)
        if cached is not None:
            return cached
        if sector_key == "school":
            value = self._school_service_target_units(household)
            self._period_household_desired_units_cache[cache_key] = value
            return value
        if sector_key == "university":
            value = self._university_service_target_units(household)
            self._period_household_desired_units_cache[cache_key] = value
            return value
        base_units = household.need_scale * self._household_consumption_multiplier(household)
        if sector_key in ESSENTIAL_SECTOR_KEYS:
            value = base_units * self._household_essential_share(household, sector_key)
            self._period_household_desired_units_cache[cache_key] = value
            return value
        discretionary_scale = (
            self.config.nonessential_demand_multiplier
            * sum(SECTOR_BY_KEY[key].household_demand_share for key in DISCRETIONARY_SECTOR_KEYS)
        )
        value = base_units * discretionary_scale * self._household_discretionary_share(household, sector_key)
        self._period_household_desired_units_cache[cache_key] = value
        return value

    def _essential_marginal_utility(self, coverage_ratio: float) -> float:
        coverage_ratio = max(0.0, coverage_ratio)
        # Essential goods start with very high utility, but additional units
        # quickly lose value once baseline need is covered.
        return 4.0 / (1.0 + 12.0 * (coverage_ratio ** 2))

    def _essential_extra_budget_share(self, essential_coverage: float) -> float:
        essential_coverage = max(0.0, essential_coverage)
        return clamp(0.60 * math.exp(-4.0 * max(0.0, essential_coverage - 1.0)), 0.05, 0.60)

    def _coverage_saturation(self, coverage_ratio: float, intensity: float = 2.0) -> float:
        coverage_ratio = max(0.0, coverage_ratio)
        intensity = max(0.1, intensity)
        return 1.0 - math.exp(-intensity * coverage_ratio)

    def _family_essential_coverage_ratio(
        self,
        essential_target_by_sector: dict[str, float],
        purchased_units_by_sector: dict[str, float],
    ) -> float:
        total_target = sum(max(0.0, target) for target in essential_target_by_sector.values())
        if total_target <= 0.0:
            return 1.0

        weighted_coverage = 0.0
        for sector_key, target_units in essential_target_by_sector.items():
            if target_units <= 0.0:
                continue
            sector_coverage = purchased_units_by_sector.get(sector_key, 0.0) / target_units
            weighted_coverage += target_units * sector_coverage
        return clamp(weighted_coverage / total_target, 0.0, 3.0)

    def _extra_essential_gap_units(self, target_units: float, purchased_units: float) -> float:
        if target_units <= 0.0:
            return 0.0
        coverage_cap = max(1.0, self.config.extra_essential_coverage_cap)
        capped_target_units = target_units * coverage_cap
        return max(0.0, capped_target_units - max(0.0, purchased_units))

    def _discretionary_sector_utility_weight(
        self,
        sector_key: str,
        base_preference_units: float,
        essential_coverage: float,
        family_remaining_cash: float,
        family_basic_basket_cost: float,
    ) -> float:
        if base_preference_units <= 0.0:
            return 0.0

        comfort_ratio = clamp((essential_coverage - 0.90) / 0.55, 0.0, 2.0)
        liquidity_ratio = clamp(
            family_remaining_cash / max(1.0, family_basic_basket_cost),
            0.0,
            2.5,
        )
        if sector_key == "leisure":
            utility_multiplier = 0.95 + 1.10 * comfort_ratio + 0.40 * liquidity_ratio
        elif sector_key == "school":
            utility_multiplier = 1.35 + 1.75 * comfort_ratio + 0.70 * liquidity_ratio
        elif sector_key == "university":
            utility_multiplier = 1.45 + 1.95 * comfort_ratio + 0.90 * liquidity_ratio
        else:
            utility_multiplier = 0.82 + 0.45 * comfort_ratio + 0.16 * liquidity_ratio
        return base_preference_units * utility_multiplier

    def _purchase_discretionary_bundle(
        self,
        *,
        sector_keys: list[str],
        sector_preference_units: dict[str, float],
        essential_coverage: float,
        family_remaining_cash: float,
        family_basic_basket_cost: float,
        family_price_sensitivity: float,
        budget_neutral: float,
        budget_effective: float,
        cash: float,
        spending_log: dict[str, float],
        purchased_units_by_sector: dict[str, float],
    ) -> float:
        if budget_neutral <= 0.0 or budget_effective <= 0.0 or cash <= 0.0 or not sector_keys:
            return cash

        bundle_utility_weights = {
            sector_key: self._discretionary_sector_utility_weight(
                sector_key,
                sector_preference_units.get(sector_key, 0.0),
                essential_coverage,
                family_remaining_cash,
                family_basic_basket_cost,
            )
            for sector_key in sector_keys
        }
        total_utility_weight = sum(bundle_utility_weights.values())
        if total_utility_weight <= 0.0:
            bundle_utility_weights = {
                sector_key: max(0.0, sector_preference_units.get(sector_key, 0.0))
                for sector_key in sector_keys
            }
            total_utility_weight = sum(bundle_utility_weights.values())
        if total_utility_weight <= 0.0:
            bundle_utility_weights = {sector_key: 1.0 for sector_key in sector_keys}
            total_utility_weight = float(len(sector_keys))

        for sector_key in sector_keys:
            if cash <= 0.0:
                continue
            spec = SECTOR_BY_KEY[sector_key]
            share = bundle_utility_weights[sector_key] / total_utility_weight
            neutral_spend = budget_neutral * share
            intended_spend = budget_effective * share
            average_price = self._average_sector_price(sector_key)
            desired_units = intended_spend / max(0.1, average_price)
            desired_units_neutral = neutral_spend / max(0.1, spec.base_price)
            self._period_potential_demand_units += desired_units
            self._period_sector_potential_demand_units[sector_key] += desired_units
            self._period_sector_budget_demand_units[sector_key] += desired_units_neutral
            sector_firms = self._sector_firms(sector_key)
            if not sector_firms:
                continue
            cash, units_bought = self._purchase_from_sector(
                family_price_sensitivity,
                sector_key,
                desired_units,
                cash,
                spending_log,
            )
            purchased_units_by_sector[sector_key] += units_bought
        return cash

    def _essential_affordability_pressure(self) -> float:
        if not self.history or len(self.history) < max(1, self.config.startup_grace_periods):
            return 0.0
        last_snapshot = self.history[-1]
        basket_gap = max(0.0, 1.0 - last_snapshot.family_income_to_basket_ratio)
        scarcity_relief = clamp((1.0 - last_snapshot.essential_fulfillment_rate) / 0.80, 0.0, 1.0)
        return clamp(basket_gap * (1.0 - 0.80 * scarcity_relief), 0.0, 1.0)

    def _firm_cost_decline_ratio(self, firm: Firm, average_unit_cost: float) -> float:
        prior_unit_cost = max(0.1, firm.last_unit_cost)
        return clamp((prior_unit_cost - average_unit_cost) / prior_unit_cost, 0.0, 0.80)

    def _target_margin_for_firm(self, firm: Firm, spec, average_unit_cost: float) -> float:
        target_margin = spec.markup * clamp(
            1.0 - 0.18 * (firm.markup_tolerance - 1.0),
            0.70,
            1.30,
        )
        if spec.key not in ESSENTIAL_SECTOR_KEYS:
            return target_margin

        cost_decline_ratio = self._firm_cost_decline_ratio(firm, average_unit_cost)
        affordability_pressure = self._essential_affordability_pressure()
        return clamp(
            target_margin * (1.0 - 0.50 * affordability_pressure - 0.35 * cost_decline_ratio),
            0.02,
            0.18,
        )

    def _target_price_for_firm(self, firm: Firm, spec, average_unit_cost: float, variable_unit_cost: float) -> float:
        target_margin = self._target_margin_for_firm(firm, spec, average_unit_cost)
        target_price = average_unit_cost * (1.0 + target_margin)
        if spec.key not in ESSENTIAL_SECTOR_KEYS:
            return target_price

        cost_decline_ratio = self._firm_cost_decline_ratio(firm, average_unit_cost)
        affordability_pressure = self._essential_affordability_pressure()
        pass_through_strength = clamp(
            0.30
            + 0.45 * cost_decline_ratio
            + 0.35 * affordability_pressure
            + 0.20 * max(0.0, firm.volume_preference - 1.0),
            0.15,
            0.95,
        )
        if cost_decline_ratio > 0.0:
            pass_through_price = firm.price * (1.0 - pass_through_strength * cost_decline_ratio)
            target_price = min(target_price, pass_through_price)
        return max(variable_unit_cost * 1.01, target_price)

    def _candidate_price_objective(
        self,
        firm: Firm,
        spec,
        effective_price: float,
        prudent_sales: float,
        candidate_profit: float,
        future_market_value: float,
        market_hazard: float,
        variable_unit_cost: float,
        fixed_cost: float,
    ) -> float:
        affordability_pressure = self._essential_affordability_pressure() if spec.key in ESSENTIAL_SECTOR_KEYS else 0.0

        # Rational objective: expected monthly profit plus retained market value,
        # minus fragility risk. Inventory costs are charged explicitly elsewhere.
        future_weight = (
            0.65
            + 0.90 * self._firm_future_market_weight(firm)
            + 0.20 * max(0.0, firm.market_share_ambition - 1.0)
            + (0.45 * affordability_pressure if spec.key in ESSENTIAL_SECTOR_KEYS else 0.0)
        )
        hazard_penalty = market_hazard * (
            fixed_cost * (0.80 + 0.30 * firm.cash_conservatism)
            + 0.30 * future_market_value
        )
        return candidate_profit + future_weight * future_market_value - hazard_penalty

    def _candidate_total_profit(
        self,
        firm: Firm,
        prudent_sales: float,
        effective_price: float,
        variable_unit_cost: float,
        fixed_cost: float,
    ) -> float:
        projected_revenue = max(0.0, prudent_sales) * max(0.0, effective_price)
        effective_productivity = self._firm_effective_productivity(firm)
        if self._is_education_sector(firm.sector):
            projected_output = self._education_service_target_units(firm, prudent_sales)
            desired_workers = self._workers_needed_for_units(
                projected_output,
                effective_productivity,
                productivity_floor=0.25,
            )
        else:
            projected_output = self._firm_desired_output_from_expected_sales(firm, prudent_sales)
            desired_workers = self._workers_needed_for_units(projected_output, effective_productivity)

        wage_bill = desired_workers * firm.wage_offer
        payroll_tax = wage_bill * self._government_payroll_tax_rate()
        input_cost = projected_output * firm.input_cost_per_unit
        transport_cost = (
            projected_output
            * firm.transport_cost_per_unit
            * self._public_infrastructure_transport_cost_multiplier()
        )
        projected_inventory, projected_waste_units = self._project_inventory_end_of_period(
            firm,
            projected_output,
            prudent_sales,
        )
        inventory_cost_basis = self._inventory_cost_basis(
            firm,
            variable_unit_cost + fixed_cost / max(1.0, prudent_sales if prudent_sales > 0.0 else projected_output),
        )
        carry_cost = projected_inventory * inventory_cost_basis * max(0.0, self.config.inventory_carry_cost_share)
        waste_cost = projected_waste_units * inventory_cost_basis
        total_cost = (
            wage_bill
            + payroll_tax
            + input_cost
            + transport_cost
            + fixed_cost
            + carry_cost
            + waste_cost
        )
        return projected_revenue - total_cost

    def _compute_structural_demand_map(
        self,
        active_households: list[Household] | None = None,
    ) -> dict[str, float]:
        households = active_households
        if households is None:
            households = [household for household in self.households if household.alive]

        if households is self._period_active_households_cache:
            self._ensure_period_essential_household_arrays()
            if self._period_essential_desired_units_matrix is not None and len(households) == len(self._period_household_row_index_cache):
                totals = {spec.key: 0.0 for spec in SECTOR_SPECS}
                matrix = self._period_essential_desired_units_matrix
                for sector_key in ARRAY_BACKED_SECTOR_KEYS:
                    totals[sector_key] = float(matrix[:, ARRAY_BACKED_SECTOR_INDEX[sector_key]].sum())
                for sector_key in ("school", "university"):
                    totals[sector_key] = sum(
                        self._household_sector_desired_units(household, sector_key)
                        for household in households
                    )
                return totals

        totals = {spec.key: 0.0 for spec in SECTOR_SPECS}
        for household in households:
            for spec in SECTOR_SPECS:
                totals[spec.key] += self._household_sector_desired_units(household, spec.key)
        return totals

    def _structural_sector_demand(self, spec_key: str) -> float:
        if self._startup_structural_demand_cache is not None:
            return self._startup_structural_demand_cache.get(spec_key, 0.0)
        active_households = self._period_active_households_cache
        if active_households is not None:
            cache_key = (spec_key, "__structural__")
            cached = self._period_baseline_demand_cache.get(cache_key)
            if cached is not None:
                return cached
            totals = self._compute_structural_demand_map(active_households)
            for key, value in totals.items():
                self._period_baseline_demand_cache[(key, "__structural__")] = value
            return totals.get(spec_key, 0.0)
        return sum(
            self._household_sector_desired_units(household, spec_key)
            for household in self._active_households()
        )

    def _essential_basket_share(self, sector_key: str) -> float:
        total_essential_need = sum(SECTOR_BY_KEY[key].essential_need for key in ESSENTIAL_SECTOR_KEYS)
        return SECTOR_BY_KEY[sector_key].essential_need / max(1e-9, total_essential_need)

    def _startup_essential_supply_multiplier(self, sector_key: str) -> float:
        multiplier = max(1.10, self.config.startup_essential_supply_buffer)
        if sector_key == "clothing":
            multiplier *= max(1.0, self.config.startup_clothing_supply_multiplier)
        return multiplier

    def _startup_essential_target_units(self, sector_key: str) -> float:
        if self._startup_essential_target_cache is not None:
            return self._startup_essential_target_cache.get(sector_key, 0.0)
        structural_demand = self._structural_sector_demand(sector_key)
        survival_floor = len(self._active_households()) * self._essential_basket_share(sector_key)
        return max(structural_demand, survival_floor) * self._startup_essential_supply_multiplier(sector_key)

    def _essential_budget(self, household: Household) -> float:
        self._ensure_period_essential_household_arrays()
        row_index = self._period_household_row_index_cache.get(household.id)
        if row_index is not None and self._period_essential_budget_vector is not None:
            return float(self._period_essential_budget_vector[row_index])
        cached = self._period_essential_budget_cache.get(household.id)
        if cached is not None and cached[0] == household.age_periods:
            return cached[1]
        budget = 0.0
        for sector_key in ESSENTIAL_SECTOR_KEYS:
            price = self._average_sector_price(sector_key)
            desired_units = self._household_sector_desired_units(household, sector_key)
            budget += desired_units * price
        self._period_essential_budget_cache[household.id] = (household.age_periods, budget)
        return budget

    def _select_guardian_for_child(self, child: Household, exclude_id: int | None = None) -> Household | None:
        return min(
            (
                household
                for household in self._active_households()
                if self._household_age_years(household) >= self.config.entry_age_years
                and household.id != child.id
                and household.id != exclude_id
            ),
            key=lambda household: (
                household.dependent_children,
                -self._household_cash_balance(household),
                self._household_age_years(household),
                household.id,
            ),
            default=None,
        )

    def _assign_guardian(self, child: Household, guardian: Household) -> None:
        child.guardian_id = guardian.id
        guardian.dependent_children += 1

    def _release_guardian_dependency(self, household: Household) -> None:
        guardian_ids = {
            household.guardian_id,
            household.mother_id,
            household.father_id,
        }
        household.guardian_id = None
        for guardian_id in guardian_ids:
            if guardian_id is None:
                continue
            guardian = self._household_by_id(guardian_id)
            if guardian is not None and guardian.dependent_children > 0:
                guardian.dependent_children -= 1

    def _clear_partner_link(self, household: Household) -> None:
        partner_id = household.partner_id
        household.partner_id = None
        household.partnership_start_period = -999
        self._schedule_next_partnership_attempt(household)
        if partner_id is None:
            return
        partner = self._household_by_id(partner_id)
        if partner is not None and partner.partner_id == household.id:
            partner.partner_id = None
            partner.partnership_start_period = -999
            self._schedule_next_partnership_attempt(partner)

    def _adult_partnership_candidates(self, sex: str) -> list[Household]:
        return [
            household
            for household in self._active_households()
            if household.sex == sex
            and household.partner_id is None
            and self.period >= household.next_partnership_attempt_period
            and self.config.entry_age_years <= self._household_age_years(household) <= self.config.fertile_age_max_years
        ]

    def _pair_score(self, first: Household, second: Household) -> float:
        age_gap = abs(self._household_age_years(first) - self._household_age_years(second))
        savings_gap = abs(first.savings - second.savings)
        desire_bonus = (first.desired_children + second.desired_children) / 2.0
        return desire_bonus + 1.5 / (1.0 + age_gap) + 1.0 / (1.0 + savings_gap)

    def _pair_match_probability(self, first: Household, second: Household, age_gap: float) -> float:
        if first.partnership_affinity_code <= 0 or second.partnership_affinity_code <= 0:
            return 0.0
        if first.partnership_affinity_code != second.partnership_affinity_code:
            return 0.0
        probability = self.config.partnership_base_match_probability
        neutral_years = max(0.0, self.config.partnership_age_gap_neutral_years)
        soft_cap = max(0.0, self.config.partnership_age_gap_soft_cap_years)
        if age_gap <= neutral_years:
            return clamp(probability, 0.0, 1.0)
        if age_gap > soft_cap:
            probability *= self.config.partnership_age_gap_hard_penalty
        elif soft_cap > neutral_years:
            scaled_gap = (age_gap - neutral_years) / max(1e-9, soft_cap - neutral_years)
            probability *= 1.0 - self.config.partnership_age_gap_soft_penalty * scaled_gap
        return clamp(probability, 0.0, 1.0)

    def _refresh_family_links(self) -> None:
        active_households = self._active_households()
        if not active_households:
            return

        age_lookup: dict[int, float] = {}
        males: list[Household] = []
        females: list[Household] = []
        entry_age_years = self.config.entry_age_years
        fertile_age_max_years = self.config.fertile_age_max_years
        households_by_id = self.households

        for household in active_households:
            age_years = self._household_age_years(household)
            age_lookup[household.id] = age_years

            partner_id = household.partner_id
            if partner_id is not None:
                partner = (
                    households_by_id[partner_id]
                    if 0 <= partner_id < len(households_by_id)
                    else None
                )
                if partner is None or not partner.alive or partner.partner_id != household.id:
                    household.partner_id = None

            if household.partner_id is not None:
                continue
            if not (entry_age_years <= age_years <= fertile_age_max_years):
                continue
            if self.period < household.next_partnership_attempt_period:
                continue
            if household.sex == "M":
                males.append(household)
            elif household.sex == "F":
                females.append(household)
        if not males or not females:
            return

        males.sort(key=lambda household: (age_lookup[household.id], -household.savings, household.id))
        females.sort(key=lambda household: (age_lookup[household.id], -household.savings, household.id))
        female_ages = [age_lookup[household.id] for household in females]
        female_savings = [household.savings for household in females]
        female_desired_children = [household.desired_children for household in females]
        next_unmatched = list(range(len(females) + 1))

        # Skip women already matched earlier in this greedy pass without rescanning them.
        def next_available_female(index: int) -> int:
            while next_unmatched[index] != index:
                next_unmatched[index] = next_unmatched[next_unmatched[index]]
                index = next_unmatched[index]
            return index

        for male in males:
            if male.partner_id is not None:
                continue

            male_age = age_lookup[male.id]
            lower_index = bisect.bisect_left(female_ages, male_age - 14.0)
            upper_index = bisect.bisect_right(female_ages, male_age + 14.0)
            best_female_index = -1
            best_score = 0.9
            male_desired_children = male.desired_children
            male_savings = male.savings
            candidate_index = next_available_female(lower_index)
            while candidate_index < upper_index:
                female_age = female_ages[candidate_index]
                female_desire = female_desired_children[candidate_index]
                if male.partnership_affinity_code != females[candidate_index].partnership_affinity_code:
                    candidate_index = next_available_female(candidate_index + 1)
                    continue
                if male_desired_children != 0 or female_desire != 0:
                    age_gap = abs(male_age - female_age)
                    savings_gap = abs(male_savings - female_savings[candidate_index])
                    desire_bonus = 0.5 * (male_desired_children + female_desire)
                    if age_gap > self.config.partnership_age_gap_soft_cap_years:
                        desire_bonus *= self.config.partnership_age_gap_hard_penalty
                    score = desire_bonus + 1.5 / (1.0 + age_gap) + 1.0 / (1.0 + savings_gap)
                    if score > best_score:
                        best_score = score
                        best_female_index = candidate_index
                candidate_index = next_available_female(candidate_index + 1)

            if best_female_index < 0:
                self._schedule_next_partnership_attempt(male)
                continue

            best_female = females[best_female_index]
            age_gap = abs(male_age - female_ages[best_female_index])
            if self.rng.random() >= self._pair_match_probability(male, best_female, age_gap):
                self._schedule_next_partnership_attempt(male)
                continue
            next_unmatched[best_female_index] = next_available_female(best_female_index + 1)
            male.partner_id = best_female.id
            best_female.partner_id = male.id
            male.partnership_start_period = self.period
            best_female.partnership_start_period = self.period
            family_desire = round((male_desired_children + female_desired_children[best_female_index]) / 2.0)
            male.desired_children = family_desire
            best_female.desired_children = family_desire
            if male.last_birth_period < 0:
                male.last_birth_period = self.period - self.config.birth_interval_periods
            if best_female.last_birth_period < 0:
                best_female.last_birth_period = self.period - self.config.birth_interval_periods

    def _assign_initial_guardians(self) -> None:
        adult_guardians = [
            household
            for household in self.households
            if household.alive and self._household_age_years(household) >= self.config.entry_age_years
        ]
        children = [household for household in self.households if self._household_age_years(household) < self.config.entry_age_years]
        guardian_heap = [
            (
                household.dependent_children,
                -self._household_cash_balance(household),
                self._household_age_years(household),
                household.id,
            )
            for household in adult_guardians
        ]
        heapq.heapify(guardian_heap)

        def next_guardian() -> Household | None:
            while guardian_heap:
                _, _, _, guardian_id = guardian_heap[0]
                guardian = self._household_by_id(guardian_id)
                if guardian is None or not guardian.alive:
                    heapq.heappop(guardian_heap)
                    continue
                current_key = (
                    guardian.dependent_children,
                    -self._household_cash_balance(guardian),
                    self._household_age_years(guardian),
                    guardian.id,
                )
                if guardian_heap[0] != current_key:
                    heapq.heappop(guardian_heap)
                    continue
                return guardian
            return None

        for child in children:
            guardian = next_guardian()
            if guardian is not None:
                self._assign_guardian(child, guardian)
                heapq.heappush(
                    guardian_heap,
                    (
                        guardian.dependent_children,
                        -self._household_cash_balance(guardian),
                        self._household_age_years(guardian),
                        guardian.id,
                    ),
                )
                partner = self._household_by_id(guardian.partner_id) if guardian.partner_id is not None else None
                if partner is not None and partner.alive:
                    if guardian.sex == "F":
                        child.mother_id = guardian.id
                        child.father_id = partner.id
                    else:
                        child.mother_id = partner.id
                        child.father_id = guardian.id
                    guardian.children_count += 1
                    partner.children_count += 1
                    partner.dependent_children += 1
                    heapq.heappush(
                        guardian_heap,
                        (
                            partner.dependent_children,
                            -self._household_cash_balance(partner),
                            self._household_age_years(partner),
                            partner.id,
                        ),
                    )

    def _reassign_orphans(self) -> None:
        for child in self._active_households():
            if self._household_age_years(child) >= self.config.entry_age_years:
                continue
            guardian = self._household_by_id(child.guardian_id) if child.guardian_id is not None else None
            if guardian is not None and guardian.alive:
                continue

            previous_guardian_id = child.guardian_id
            for parent_id in (child.mother_id, child.father_id):
                parent = self._household_by_id(parent_id) if parent_id is not None else None
                if parent is not None and parent.alive:
                    if previous_guardian_id is not None and previous_guardian_id != parent.id:
                        old_guardian = self._household_by_id(previous_guardian_id)
                        if old_guardian is not None and old_guardian.dependent_children > 0:
                            old_guardian.dependent_children -= 1
                    child.guardian_id = parent.id
                    parent.dependent_children += 1
                    break
            else:
                new_guardian = self._select_guardian_for_child(child)
                if new_guardian is None:
                    continue
                if previous_guardian_id is not None and previous_guardian_id != new_guardian.id:
                    old_guardian = self._household_by_id(previous_guardian_id)
                    if old_guardian is not None and old_guardian.dependent_children > 0:
                        old_guardian.dependent_children -= 1
                child.guardian_id = new_guardian.id
                new_guardian.dependent_children += 1

    def _child_supporters(self, child: Household) -> list[Household]:
        supporters: list[Household] = []
        supporter_ids: set[int] = set()
        for parent_id in (child.mother_id, child.father_id):
            parent = self._household_by_id(parent_id) if parent_id is not None else None
            if parent is None or not parent.alive:
                continue
            if self._household_age_years(parent) < self.config.entry_age_years:
                continue
            if parent.id in supporter_ids:
                continue
            supporters.append(parent)
            supporter_ids.add(parent.id)

        if supporters:
            return supporters

        guardian = self._household_by_id(child.guardian_id) if child.guardian_id is not None else None
        if guardian is not None and guardian.alive and self._household_age_years(guardian) >= self.config.entry_age_years:
            return [guardian]
        return []

    def _apply_parental_support(self) -> None:
        children_by_supporters: dict[tuple[int, ...], list[Household]] = {}
        supporter_cache: dict[tuple[int, ...], list[Household]] = {}
        for child in self._active_households():
            if self._household_age_years(child) >= self.config.entry_age_years:
                continue
            supporters = self._child_supporters(child)
            if not supporters:
                continue
            supporter_key = tuple(sorted(supporter.id for supporter in supporters))
            children_by_supporters.setdefault(supporter_key, []).append(child)
            supporter_cache[supporter_key] = supporters

        for supporter_ids, children in children_by_supporters.items():
            supporters = supporter_cache[supporter_ids]
            supporter_available: dict[int, float] = {}
            total_available = 0.0
            for supporter in supporters:
                available = max(0.0, self._household_cash_balance(supporter) - self._essential_budget(supporter))
                if available <= 0.0:
                    continue
                supporter_available[supporter.id] = available
                total_available += available
            if total_available <= 0.0:
                continue

            child_needs = {
                child.id: self._essential_budget(child)
                for child in children
            }
            total_need = sum(child_needs.values())
            if total_need <= 0.0:
                continue

            support_total = min(total_available, total_need)
            supporter_shares = {
                supporter_id: available / total_available
                for supporter_id, available in supporter_available.items()
            }
            for child in children:
                child_support = support_total * (child_needs[child.id] / total_need)
                if child_support <= 0.0:
                    continue
                for supporter in supporters:
                    share = supporter_shares.get(supporter.id, 0.0)
                    if share <= 0.0:
                        continue
                    transferred = self._withdraw_household_cash(supporter, child_support * share)
                    if transferred > 0.0:
                        child.savings += transferred

    def _eligible_households(self) -> list[Household]:
        labor_force_participants = self._labor_force_participant_ids()
        return [
            household
            for household in self._active_households()
            if household.id in labor_force_participants
            and household.employed_by is None
        ]

    def _household_labor_capacity(self, household: Household) -> float:
        state_key = (
            household.age_periods,
            household.health_fragility,
            household.housing_deprivation_streak,
            household.clothing_deprivation_streak,
        )
        cached = self._period_household_labor_capacity_cache.get(household.id)
        if cached is not None and cached[0] == state_key:
            return cached[1]
        age_years = self._household_age_years(household)
        if age_years < self.config.entry_age_years:
            capacity = 0.0
        elif age_years < self.config.senior_age_years:
            base_capacity = 1.0
        elif age_years >= self.config.max_age_years:
            capacity = 0.0
        else:
            retirement_span = max(1.0, self.config.retirement_age_years - self.config.senior_age_years)
            progress = clamp((age_years - self.config.senior_age_years) / retirement_span, 0.0, 1.0)
            base_capacity = clamp(
                1.0 - progress * (1.0 - self.config.senior_productivity_floor),
                self.config.senior_productivity_floor,
                1.0,
            )
        if age_years >= self.config.entry_age_years and age_years < self.config.max_age_years:
            fragility_penalty = clamp(0.12 * household.health_fragility, 0.0, 0.45)
            housing_penalty = clamp(0.03 * household.housing_deprivation_streak, 0.0, 0.15)
            clothing_penalty = clamp(0.02 * household.clothing_deprivation_streak, 0.0, 0.10)
            capacity = clamp(base_capacity * (1.0 - fragility_penalty - housing_penalty - clothing_penalty), 0.0, 1.0)
        self._period_household_labor_capacity_cache[household.id] = (state_key, capacity)
        return capacity

    def _worker_effective_labor_for_sector(self, household: Household, sector_key: str) -> float:
        return self._household_labor_capacity(household) * self._household_skill_multiplier(household, sector_key)

    def _in_startup_grace(self) -> bool:
        return self.period <= max(0, self.config.startup_grace_periods)

    def _in_essential_protection(self) -> bool:
        protection_periods = max(0, self.config.essential_protection_periods)
        return protection_periods > 0 and 0 < self.period <= protection_periods

    def _market_learning_maturity(self) -> float:
        warmup_periods = max(0, self.config.firm_learning_warmup_periods)
        if warmup_periods <= 0:
            return 1.0
        observed_periods = max(0, self.period - 1)
        return clamp(observed_periods / warmup_periods, 0.0, 1.0)

    def _firm_learning_maturity(self, firm: Firm) -> float:
        market_maturity = self._market_learning_maturity()
        if market_maturity >= 1.0:
            return 1.0
        warmup_periods = max(1, self.config.firm_learning_warmup_periods)
        firm_experience = clamp(max(0, firm.age - 1) / warmup_periods, 0.0, 1.0)
        return clamp(0.85 * market_maturity + 0.15 * firm_experience, 0.0, 1.0)

    def _firm_market_memory_periods(self) -> int:
        return max(
            6,
            int(round(max(0.5, self.config.firm_market_memory_years) * max(1, self.config.periods_per_year))),
        )

    def _firm_sales_forecast_window_periods(self) -> int:
        return max(2, int(round(max(1, self.config.periods_per_year) * 0.5)))

    def _firm_inventory_shelf_life_periods(self) -> int:
        configured_months = max(1, self.config.inventory_shelf_life_months)
        return max(1, int(round(max(1, self.config.periods_per_year) * configured_months / 12.0)))

    def _firm_memory_slice(self, firm: Firm) -> list[float]:
        memory_periods = self._firm_market_memory_periods()
        if not firm.sales_history:
            return []
        return firm.sales_history[-memory_periods:]

    def _firm_recent_average(
        self,
        series: list[float],
        *,
        fallback: float,
        floor: float = 0.0,
    ) -> float:
        if not series:
            return max(floor, fallback)
        window = min(len(series), self._firm_sales_forecast_window_periods())
        if window <= 0:
            return max(floor, fallback)
        return max(floor, sum(series[-window:]) / window)

    def _recent_history_window(self) -> int:
        return self._firm_sales_forecast_window_periods()

    def _recent_snapshots(self) -> list[PeriodSnapshot]:
        window = self._recent_history_window()
        if window <= 0 or not self.history:
            return []
        return self.history[-window:]

    def _recent_sector_signal_average(
        self,
        history_map: dict[str, list[float]],
        sector_key: str,
        *,
        fallback: float,
    ) -> float:
        series = history_map.get(sector_key, [])
        return self._firm_recent_average(series, fallback=fallback, floor=0.0)

    def _firm_recent_prior_sales_windows(self, firm: Firm) -> tuple[list[float], list[float]]:
        history = self._firm_memory_slice(firm)
        if len(history) < 4:
            return history, []
        recent_window_size = max(3, min(len(history) // 3, max(3, self.config.periods_per_year)))
        recent_window = history[-recent_window_size:]
        prior_window = history[:-recent_window_size]
        if prior_window:
            prior_window = prior_window[-recent_window_size:]
        return recent_window, prior_window

    def _sector_demand_elasticity_prior(self, sector_key: str) -> float:
        if sector_key == "food":
            return 0.82
        if sector_key == "housing":
            return 0.76
        if sector_key == "clothing":
            return 0.92
        if sector_key == "manufactured":
            return 1.18
        if sector_key == "school":
            return 0.96
        if sector_key == "university":
            return 1.08
        return 1.32

    def _smoothed_sales_reference(self, firm: Firm) -> float:
        history = self._firm_memory_slice(firm)
        if not history:
            return max(1.0, firm.last_sales, firm.last_expected_sales)
        forecast_window_size = min(len(history), self._firm_sales_forecast_window_periods())
        forecast_window = history[-forecast_window_size:]
        return max(1.0, sum(forecast_window) / len(forecast_window))

    def _smoothed_expected_sales_reference(self, firm: Firm) -> float:
        return self._firm_recent_average(
            firm.expected_sales_history,
            fallback=max(1.0, firm.last_expected_sales, self._smoothed_sales_reference(firm)),
            floor=1.0,
        )

    def _smoothed_production_reference(self, firm: Firm) -> float:
        return self._firm_recent_average(
            firm.production_history,
            fallback=max(0.0, firm.last_production, self._smoothed_sales_reference(firm)),
            floor=0.0,
        )

    def _inventory_cost_basis(self, firm: Firm, unit_cost: float | None = None) -> float:
        if unit_cost is not None:
            return max(0.1, unit_cost)
        return max(0.1, firm.last_unit_cost, firm.input_cost_per_unit + firm.transport_cost_per_unit)

    def _ensure_inventory_batches(self, firm: Firm) -> None:
        if self._is_education_sector(firm.sector):
            firm.inventory_batches = []
            return
        total_batches = sum(max(0.0, batch) for batch in firm.inventory_batches)
        inventory_units = max(0.0, firm.inventory)
        if inventory_units <= 1e-9:
            firm.inventory = 0.0
            firm.inventory_batches = []
            return
        if not firm.inventory_batches or abs(total_batches - inventory_units) > 1e-6:
            firm.inventory_batches = [inventory_units]
            return
        firm.inventory_batches = [max(0.0, batch) for batch in firm.inventory_batches if batch > 1e-9]

    def _sync_inventory_from_batches(self, firm: Firm) -> None:
        if self._is_education_sector(firm.sector):
            return
        firm.inventory = sum(max(0.0, batch) for batch in firm.inventory_batches)

    def _add_inventory_units(self, firm: Firm, units: float) -> None:
        if units <= 0.0:
            return
        if self._is_education_sector(firm.sector):
            firm.inventory = max(0.0, units)
            return
        self._ensure_inventory_batches(firm)
        firm.inventory_batches.append(max(0.0, units))
        self._sync_inventory_from_batches(firm)

    def _remove_inventory_units(self, firm: Firm, units: float) -> float:
        if units <= 0.0:
            return 0.0
        if self._is_education_sector(firm.sector):
            removed = min(max(0.0, units), max(0.0, firm.inventory))
            firm.inventory = max(0.0, firm.inventory - removed)
            return removed
        self._ensure_inventory_batches(firm)
        remaining = max(0.0, units)
        removed = 0.0
        updated_batches: list[float] = []
        for batch in firm.inventory_batches:
            if remaining <= 1e-9:
                updated_batches.append(batch)
                continue
            take = min(batch, remaining)
            batch -= take
            remaining -= take
            removed += take
            if batch > 1e-9:
                updated_batches.append(batch)
        firm.inventory_batches = updated_batches
        self._sync_inventory_from_batches(firm)
        return removed

    def _project_inventory_end_of_period(
        self,
        firm: Firm,
        production_units: float,
        sales_units: float,
    ) -> tuple[float, float]:
        if self._is_education_sector(firm.sector):
            end_inventory = max(0.0, production_units - sales_units)
            return end_inventory, 0.0
        self._ensure_inventory_batches(firm)
        projected_batches = [max(0.0, batch) for batch in firm.inventory_batches if batch > 1e-9]
        if production_units > 0.0:
            projected_batches.append(max(0.0, production_units))
        remaining_sales = max(0.0, sales_units)
        for index in range(len(projected_batches)):
            if remaining_sales <= 1e-9:
                break
            take = min(projected_batches[index], remaining_sales)
            projected_batches[index] -= take
            remaining_sales -= take
        projected_batches = [batch for batch in projected_batches if batch > 1e-9]
        shelf_life = self._firm_inventory_shelf_life_periods()
        expiring_units = sum(projected_batches[:-shelf_life]) if len(projected_batches) > shelf_life else 0.0
        end_inventory = sum(projected_batches[-shelf_life:]) if shelf_life > 0 else 0.0
        return max(0.0, end_inventory), max(0.0, expiring_units)

    def _apply_inventory_carry_and_waste(self, firm: Firm) -> tuple[float, float]:
        if self._is_education_sector(firm.sector):
            firm.last_inventory_carry_cost = 0.0
            firm.last_inventory_waste_cost = 0.0
            return 0.0, 0.0
        self._ensure_inventory_batches(firm)
        carry_cost = max(0.0, firm.inventory) * self._inventory_cost_basis(firm) * max(
            0.0,
            self.config.inventory_carry_cost_share,
        )
        shelf_life = self._firm_inventory_shelf_life_periods()
        waste_units = sum(firm.inventory_batches[:-shelf_life]) if len(firm.inventory_batches) > shelf_life else 0.0
        if waste_units > 0.0:
            firm.inventory_batches = firm.inventory_batches[-shelf_life:]
            self._sync_inventory_from_batches(firm)
        waste_cost = waste_units * self._inventory_cost_basis(firm)
        firm.last_inventory_carry_cost = carry_cost
        firm.last_inventory_waste_cost = waste_cost
        return carry_cost, waste_cost

    def _household_consumption_multiplier(self, household: Household) -> float:
        age_years = self._household_age_years(household)
        if age_years < self.config.entry_age_years:
            progress = clamp(age_years / max(1.0, self.config.entry_age_years), 0.0, 1.0)
            return clamp(
                self.config.child_consumption_multiplier
                + (1.0 - self.config.child_consumption_multiplier) * progress,
                self.config.child_consumption_multiplier,
                1.0,
            )
        if age_years < self.config.senior_age_years:
            return 1.0
        if age_years >= self.config.max_age_years:
            return self.config.senior_consumption_multiplier

        senior_span = max(1.0, self.config.max_age_years - self.config.senior_age_years)
        progress = clamp((age_years - self.config.senior_age_years) / senior_span, 0.0, 1.0)
        return clamp(
            self.config.senior_consumption_multiplier - 0.10 * progress,
            0.65,
            1.0,
        )

    def _household_food_sufficient_meals(self, household: Household) -> float:
        return (
            self.config.period_days
            * self.config.food_meals_per_day_sufficient
            * household.need_scale
            * self._household_consumption_multiplier(household)
        )

    def _household_food_subsistence_meals(self, household: Household) -> float:
        return (
            self.config.period_days
            * self.config.food_meals_per_day_subsistence
            * household.need_scale
            * self._household_consumption_multiplier(household)
        )

    def _household_food_severe_hunger_meals(self, household: Household) -> float:
        return (
            self.config.period_days
            * self.config.food_meals_per_day_severe
            * household.need_scale
            * self._household_consumption_multiplier(household)
        )

    def _coverage_from_units(self, consumed_units: float, target_units: float) -> float:
        if target_units <= 0.0:
            return 1.0
        return clamp(consumed_units / target_units, 0.0, 3.0)

    def _household_sector_coverage(self, household: Household, sector_key: str) -> float:
        target_units = self._household_sector_desired_units(household, sector_key)
        consumed_units = household.last_consumption.get(sector_key, 0.0)
        return self._coverage_from_units(consumed_units, target_units)

    def _household_food_meals_consumed(self, household: Household) -> float:
        return self._household_sector_coverage(household, "food") * self._household_food_sufficient_meals(household)

    def _household_perceived_utility(
        self,
        household: Household,
        *,
        family_remaining_cash: float,
        family_basic_basket_cost: float,
    ) -> float:
        coverages_by_sector = {
            spec.key: self._household_sector_coverage(household, spec.key)
            for spec in SECTOR_SPECS
        }
        target_units_by_sector = {
            spec.key: self._household_sector_desired_units(household, spec.key)
            for spec in SECTOR_SPECS
        }
        return self._household_perceived_utility_from_inputs(
            household,
            coverages_by_sector=coverages_by_sector,
            target_units_by_sector=target_units_by_sector,
            family_remaining_cash=family_remaining_cash,
            family_basic_basket_cost=family_basic_basket_cost,
        )

    def _household_perceived_utility_from_inputs(
        self,
        household: Household,
        *,
        coverages_by_sector: dict[str, float],
        target_units_by_sector: dict[str, float],
        family_remaining_cash: float,
        family_basic_basket_cost: float,
    ) -> float:
        essential_coverages = [
            coverages_by_sector.get(sector_key, 1.0)
            for sector_key in ESSENTIAL_SECTOR_KEYS
        ]
        average_essential_coverage = (
            sum(essential_coverages) / len(essential_coverages)
            if essential_coverages
            else 1.0
        )
        liquidity_security = clamp(
            family_remaining_cash / max(1.0, family_basic_basket_cost),
            0.0,
            2.0,
        )

        utility = 0.0
        for spec in SECTOR_SPECS:
            target_units = target_units_by_sector.get(spec.key, 0.0)
            if spec.key in ("school", "university") and target_units <= 0.0:
                continue
            coverage = coverages_by_sector.get(spec.key, 1.0)
            preference = self._household_sector_preference(household, spec.key)
            preference_scalar = clamp(0.55 + 0.45 * preference, 0.35, 1.60)
            if spec.key in ESSENTIAL_SECTOR_KEYS:
                sector_utility = spec.essential_need * self._coverage_saturation(coverage, intensity=3.0)
            else:
                affluence_gain = max(0.0, average_essential_coverage - 0.90)
                if spec.key == "leisure":
                    utility_multiplier = 0.95 + 0.95 * affluence_gain + 0.25 * liquidity_security
                elif spec.key == "school":
                    utility_multiplier = 1.15 + 1.55 * affluence_gain + 0.55 * liquidity_security
                elif spec.key == "university":
                    utility_multiplier = 1.20 + 1.85 * affluence_gain + 0.75 * liquidity_security
                else:
                    utility_multiplier = 0.88 + 0.45 * affluence_gain + 0.14 * liquidity_security
                sector_utility = (
                    spec.discretionary_weight
                    * self._coverage_saturation(coverage, intensity=1.7)
                    * utility_multiplier
                )
            utility += preference_scalar * sector_utility

        utility += 0.08 * liquidity_security
        return max(0.0, utility)

    def _food_subsistence_coverage_ratio(self) -> float:
        return self.config.food_meals_per_day_subsistence / max(1e-9, self.config.food_meals_per_day_sufficient)

    def _food_severe_hunger_coverage_ratio(self) -> float:
        return self.config.food_meals_per_day_severe / max(1e-9, self.config.food_meals_per_day_sufficient)

    def _allocate_family_consumption_units(
        self,
        family_members: list[Household],
        purchased_units_by_sector: dict[str, float],
        target_units_by_member: dict[int, dict[str, float]] | None = None,
    ) -> dict[int, dict[str, float]]:
        allocation = {
            member.id: {spec.key: 0.0 for spec in SECTOR_SPECS}
            for member in family_members
        }
        for spec in SECTOR_SPECS:
            sector_key = spec.key
            total_target = sum(
                (
                    target_units_by_member.get(member.id, {}).get(sector_key, 0.0)
                    if target_units_by_member is not None
                    else self._household_sector_desired_units(member, sector_key)
                )
                for member in family_members
            )
            units_bought = max(0.0, purchased_units_by_sector.get(sector_key, 0.0))
            if units_bought <= 0.0 or total_target <= 0.0:
                continue
            for member in family_members:
                member_target = (
                    target_units_by_member.get(member.id, {}).get(sector_key, 0.0)
                    if target_units_by_member is not None
                    else self._household_sector_desired_units(member, sector_key)
                )
                if member_target <= 0.0:
                    continue
                allocation[member.id][sector_key] = units_bought * member_target / total_target
        return allocation

    def _family_root_for_adult(self, household: Household) -> int:
        partner = self._household_by_id(household.partner_id) if household.partner_id is not None else None
        if partner is not None and partner.alive and self._household_age_years(partner) >= self.config.entry_age_years:
            return min(household.id, partner.id)
        return household.id

    def _family_root_for_child(self, household: Household) -> int:
        for relative_id in (household.guardian_id, household.mother_id, household.father_id):
            relative = self._household_by_id(relative_id) if relative_id is not None else None
            if relative is not None and relative.alive and self._household_age_years(relative) >= self.config.entry_age_years:
                return self._family_root_for_adult(relative)
        return household.guardian_id if household.guardian_id is not None else household.id

    def _family_groups(self) -> dict[int, list[Household]]:
        cached = self._period_family_groups_cache
        if cached is not None:
            return cached

        groups: dict[int, list[Household]] = {}
        for household in self._active_households():
            root_id = (
                self._family_root_for_child(household)
                if self._household_age_years(household) < self.config.entry_age_years
                else self._family_root_for_adult(household)
            )
            groups.setdefault(root_id, []).append(household)
        self._period_family_groups_cache = groups
        return groups

    def _family_groups_consumption_order(self) -> list[list[Household]]:
        family_groups = list(self._family_groups().values())
        if len(family_groups) <= 1:
            return family_groups
        if not self.config.government_enabled:
            self.rng.shuffle(family_groups)
            return family_groups
        ranked_groups = [
            (
                self._family_public_education_priority(group),
                self.rng.random(),
                group,
            )
            for group in family_groups
        ]
        ranked_groups.sort(key=lambda item: (-item[0], item[1]))
        return [group for _, _, group in ranked_groups]

    def _family_public_education_priority(self, family_members: list[Household]) -> float:
        score = 0.0
        if not family_members:
            return score
        family_resource_coverage = min(
            (self._household_family_resource_coverage(member) for member in family_members),
            default=1.0,
        )
        for member in family_members:
            if self._is_school_age(member) and not self._household_has_school_credential(member):
                score += 4.0
                score += 2.0 * member.public_school_support_persistence
                if family_resource_coverage < 1.0:
                    score += 2.0
            if (
                self._is_university_age(member)
                and self._household_has_school_credential(member)
                and not self._household_has_university_credential(member)
            ):
                score += 3.0 * max(0.25, self._household_university_track_factor(member))
                score += 3.0 * member.public_university_support_persistence
                if family_resource_coverage < 1.0:
                    score += 2.5
        score += clamp(1.0 - family_resource_coverage, 0.0, 1.0)
        return score

    def _period_family_summary(self) -> dict[str, object]:
        cached = self._period_family_summary_cache
        if cached is not None:
            return cached

        groups = self._family_groups()
        family_income_ratio_by_household: dict[int, float] = {}
        family_resources_ratio_by_household: dict[int, float] = {}
        family_resources_below_basket_by_household: dict[int, bool] = {}

        family_units = len(groups)
        total_family_income = 0.0
        total_family_resources = 0.0
        total_family_basket = 0.0
        income_below_count = 0
        resources_below_count = 0
        fertile_families = 0
        fertile_families_with_births = 0
        fertile_capable_families = 0
        fertile_capable_families_low_desire_no_birth = 0
        fertile_capable_families_with_births = 0
        fertile_capable_women = 0
        fertile_capable_women_low_desire_no_birth = 0
        fertile_capable_women_with_births = 0

        for members in groups.values():
            basket_cost = 0.0
            family_income = 0.0
            family_resources = 0.0
            fertile_mothers: list[Household] = []
            fertile_capable_women_candidates: list[Household] = []

            for member in members:
                basket_cost += self._essential_budget(member)
                family_income += member.last_income
                family_resources += member.last_available_cash
                if member.sex != "F":
                    continue
                age_years = self._household_age_years(member)
                if age_years < self.config.entry_age_years or not self._is_fertile(member):
                    continue
                fertile_mothers.append(member)
                birth_happened = member.last_birth_period == self.period
                months_since_last_birth = self.period - member.last_birth_period
                spacing_ready = (
                    months_since_last_birth >= self.config.birth_interval_periods
                    or birth_happened
                )
                if spacing_ready:
                    fertile_capable_women_candidates.append(member)

            income_ratio = family_income / max(1e-9, basket_cost)
            resources_ratio = family_resources / max(1e-9, basket_cost)
            resources_below_basket = family_resources < basket_cost

            total_family_income += family_income
            total_family_resources += family_resources
            total_family_basket += basket_cost
            if family_income < basket_cost:
                income_below_count += 1
            if resources_below_basket:
                resources_below_count += 1

            for member in members:
                family_income_ratio_by_household[member.id] = income_ratio
                family_resources_ratio_by_household[member.id] = resources_ratio
                family_resources_below_basket_by_household[member.id] = resources_below_basket

            if fertile_mothers:
                fertile_families += 1
                birth_happened = any(member.last_birth_period == self.period for member in fertile_mothers)
                if birth_happened:
                    fertile_families_with_births += 1

                if not resources_below_basket:
                    fertile_capable_families += 1
                    reference_mother = fertile_mothers[0]
                    low_desire_no_birth = (
                        reference_mother.children_count >= reference_mother.desired_children
                        and reference_mother.last_birth_period < self.period
                    )
                    if low_desire_no_birth:
                        fertile_capable_families_low_desire_no_birth += 1
                    if birth_happened:
                        fertile_capable_families_with_births += 1

            if not resources_below_basket:
                for member in fertile_capable_women_candidates:
                    birth_happened = member.last_birth_period == self.period
                    fertile_capable_women += 1
                    if birth_happened:
                        fertile_capable_women_with_births += 1
                    months_since_last_birth = self.period - member.last_birth_period
                    low_desire_no_birth = (
                        not birth_happened
                        and months_since_last_birth >= self.config.birth_interval_periods
                        and member.children_count >= max(0, member.desired_children)
                        and member.last_birth_period < self.period
                    )
                    if low_desire_no_birth:
                        fertile_capable_women_low_desire_no_birth += 1

        average_family_income = total_family_income / max(1, family_units)
        average_family_resources = total_family_resources / max(1, family_units)
        average_family_basic_basket_cost = total_family_basket / max(1, family_units)
        family_income_to_basket_ratio = average_family_income / max(1e-9, average_family_basic_basket_cost)
        family_resources_to_basket_ratio = average_family_resources / max(1e-9, average_family_basic_basket_cost)

        summary = {
            "family_income_ratio_by_household": family_income_ratio_by_household,
            "family_resources_ratio_by_household": family_resources_ratio_by_household,
            "family_resources_below_basket_by_household": family_resources_below_basket_by_household,
            "family_economic_metrics": (
                family_units,
                average_family_income,
                average_family_resources,
                average_family_basic_basket_cost,
                family_income_to_basket_ratio,
                family_resources_to_basket_ratio,
                income_below_count / max(1, family_units),
                resources_below_count / max(1, family_units),
            ),
            "family_reproductive_metrics": (
                fertile_families,
                fertile_families_with_births,
                fertile_capable_families,
                fertile_capable_families_low_desire_no_birth,
                fertile_capable_families_with_births,
            ),
            "fertile_women_reproductive_metrics": (
                fertile_capable_women,
                fertile_capable_women_low_desire_no_birth,
                fertile_capable_women_with_births,
            ),
        }
        self._period_family_summary_cache = summary
        return summary

    def _family_status_maps(self) -> tuple[dict[int, float], dict[int, float], dict[int, bool]]:
        summary = self._period_family_summary()
        return (
            summary["family_income_ratio_by_household"],
            summary["family_resources_ratio_by_household"],
            summary["family_resources_below_basket_by_household"],
        )

    def _family_economic_metrics(self) -> tuple[int, float, float, float, float, float, float, float]:
        return self._period_family_summary()["family_economic_metrics"]

    def _family_reproductive_metrics(self) -> tuple[int, int, int, int, int]:
        return self._period_family_summary()["family_reproductive_metrics"]

    def _fertile_women_reproductive_metrics(self) -> tuple[int, int, int]:
        return self._period_family_summary()["fertile_women_reproductive_metrics"]

    def _period_household_summary(self) -> dict[str, object]:
        cached = self._period_household_summary_cache
        if cached is not None:
            return cached

        active_households = self._active_households()
        age_years_by_household: dict[int, float] = {}
        labor_capacity_by_household: dict[int, float] = {}
        essential_budget_by_household: dict[int, float] = {}
        cash_balance_by_household: dict[int, float] = {}
        female_ids: set[int] = set()
        fertile_ids: set[int] = set()
        child_ids: set[int] = set()
        adult_ids: set[int] = set()
        senior_ids: set[int] = set()
        employed_ids: set[int] = set()
        prime_age_ids: set[int] = set()
        total_essential_basket = 0.0
        labor_capacity_count = 0
        age_period_total = 0

        for household in active_households:
            household_id = household.id
            age_years = self._household_age_years(household)
            labor_capacity = self._household_labor_capacity(household)
            essential_budget = self._essential_budget(household)
            cash_balance = self._household_cash_balance(household)

            age_years_by_household[household_id] = age_years
            labor_capacity_by_household[household_id] = labor_capacity
            essential_budget_by_household[household_id] = essential_budget
            cash_balance_by_household[household_id] = cash_balance
            total_essential_basket += essential_budget
            age_period_total += household.age_periods

            if labor_capacity > 0.0:
                labor_capacity_count += 1
            if household.sex == "F":
                female_ids.add(household_id)
                if self.config.fertile_age_min_years <= age_years <= self.config.fertile_age_max_years:
                    fertile_ids.add(household_id)
            if age_years < self.config.entry_age_years:
                child_ids.add(household_id)
            elif age_years < self.config.senior_age_years:
                adult_ids.add(household_id)
                if labor_capacity > 0.0:
                    prime_age_ids.add(household_id)
            else:
                senior_ids.add(household_id)
            if household.employed_by is not None:
                employed_ids.add(household_id)

        summary = {
            "active_households": active_households,
            "age_years_by_household": age_years_by_household,
            "labor_capacity_by_household": labor_capacity_by_household,
            "essential_budget_by_household": essential_budget_by_household,
            "cash_balance_by_household": cash_balance_by_household,
            "female_ids": female_ids,
            "fertile_ids": fertile_ids,
            "child_ids": child_ids,
            "adult_ids": adult_ids,
            "senior_ids": senior_ids,
            "employed_ids": employed_ids,
            "prime_age_ids": prime_age_ids,
            "population": len(active_households),
            "total_essential_basket": total_essential_basket,
            "labor_capacity_count": labor_capacity_count,
            "average_age": (
                age_period_total / max(1, len(active_households)) / max(1, self.config.periods_per_year)
            ),
        }
        self._period_household_summary_cache = summary
        return summary

    def _living_wage_anchor(self) -> float:
        cached = self._period_living_wage_anchor_cache
        if cached is not None:
            return cached

        household_summary = self._period_household_summary()
        labor_force = household_summary["labor_capacity_count"]
        total_essential_basket = household_summary["total_essential_basket"]
        anchor = total_essential_basket / labor_force if labor_force > 0 else 0.0
        self._period_living_wage_anchor_cache = anchor
        return anchor

    def _labor_force_participant_ids(self) -> set[int]:
        participant_ids: set[int] = set()
        family_groups = self._family_groups()
        living_wage_anchor = max(1.0, self._living_wage_anchor())
        household_summary = self._period_household_summary()
        age_years_by_household = household_summary["age_years_by_household"]
        labor_capacity_by_household = household_summary["labor_capacity_by_household"]
        essential_budget_by_household = household_summary["essential_budget_by_household"]
        cash_balance_by_household = household_summary["cash_balance_by_household"]
        prime_age_ids = household_summary["prime_age_ids"]

        for members in family_groups.values():
            family_members = [member for member in members if member.alive]
            if not family_members:
                continue

            prime_age_candidates: list[Household] = []
            employed_participants = 0
            dependent_children = 0
            family_income = 0.0
            family_liquid = 0.0
            family_basket = 0.0

            for member in family_members:
                age_years = age_years_by_household.get(member.id, 0.0)
                family_income += self._household_observed_income(member)
                family_liquid += max(0.0, cash_balance_by_household.get(member.id, 0.0))
                family_basket += max(0.0, essential_budget_by_household.get(member.id, 0.0))
                if age_years < self.config.entry_age_years:
                    dependent_children += 1
                    continue
                if labor_capacity_by_household.get(member.id, 0.0) <= 0.0:
                    continue
                if member.employed_by is not None:
                    participant_ids.add(member.id)
                    employed_participants += 1
                if member.id in prime_age_ids:
                    prime_age_candidates.append(member)

            if not prime_age_candidates:
                continue

            income_cover = family_income / max(1e-9, family_basket)
            liquidity_cover = family_liquid / max(1e-9, family_basket)
            required_earners = max(1, math.ceil(family_basket / living_wage_anchor))
            dependent_pressure = dependent_children / max(1, len(prime_age_candidates))
            low_earner_coverage = (
                employed_participants < min(required_earners, len(prime_age_candidates))
                and income_cover < 1.35
                and liquidity_cover < (2.40 + 0.25 * dependent_pressure)
            )
            family_needs_more_workers = (
                income_cover < 1.12
                or liquidity_cover < (1.60 + 0.25 * dependent_pressure)
                or low_earner_coverage
            )

            if family_needs_more_workers:
                for member in prime_age_candidates:
                    participant_ids.add(member.id)

        return participant_ids

    def _population_metrics(self) -> tuple[int, int, int, int, int, int, int, int, float, float, float]:
        household_summary = self._period_household_summary()
        active_households = household_summary["active_households"]
        population = household_summary["population"]
        women = len(household_summary["female_ids"])
        men = population - women
        fertile_women = len(household_summary["fertile_ids"])
        children = len(household_summary["child_ids"])
        adults = len(household_summary["adult_ids"])
        seniors = len(household_summary["senior_ids"])
        labor_force = 0
        employed = 0
        labor_force_participants = self._labor_force_participant_ids()
        for household in active_households:
            if household.id in labor_force_participants:
                labor_force += 1
                if household.employed_by is not None:
                    employed += 1
        employment_rate = employed / max(1, labor_force)
        unemployment_rate = 1.0 - employment_rate if labor_force > 0 else 0.0
        average_age = household_summary["average_age"]
        return (
            population,
            women,
            men,
            fertile_women,
            children,
            adults,
            seniors,
            labor_force,
            employment_rate,
            unemployment_rate,
            average_age,
        )

    def _annual_to_period_probability(self, annual_probability: float) -> float:
        annual_probability = clamp(annual_probability, 0.0, 0.999999)
        periods_per_year = max(1, self.config.periods_per_year)
        return 1.0 - (1.0 - annual_probability) ** (1.0 / periods_per_year)

    def _household_age_years(self, household: Household) -> float:
        cached = self._period_household_age_years_cache.get(household.id)
        if cached is not None and cached[0] == household.age_periods:
            return cached[1]
        age_years = household.age_periods / max(1, self.config.periods_per_year)
        self._period_household_age_years_cache[household.id] = (household.age_periods, age_years)
        return age_years

    def _is_fertile(self, household: Household) -> bool:
        age_years = self._household_age_years(household)
        return self.config.fertile_age_min_years <= age_years <= self.config.fertile_age_max_years

    def _birth_household(self, mother: Household, father: Household | None) -> None:
        parent_ids = {mother.id}
        if father is not None:
            parent_ids.add(father.id)
        if father is None:
            guardian = mother
        else:
            guardian = mother if mother.savings + mother.wage_income >= father.savings + father.wage_income else father
        inherited_bank_id = guardian.bank_id
        if inherited_bank_id not in self.bank_by_id:
            inherited_bank_id = mother.bank_id if mother.bank_id in self.bank_by_id else 0
        self.households.append(
                Household(
                    id=self._next_household_id,
                    sex=self.rng.choice(("F", "M")),
                    savings=0.0,
                reservation_wage=self.rng.uniform(
                    self.config.newborn_reservation_wage_min,
                    self.config.newborn_reservation_wage_max,
                ),
                saving_propensity=self.rng.uniform(0.02, 0.12),
                higher_education_affinity=self.rng.random(),
                money_trust=self.rng.uniform(0.40, 0.75),
                consumption_impatience=self.rng.uniform(0.25, 0.85),
                price_sensitivity=self.rng.uniform(0.7, 1.3),
                need_scale=self.rng.uniform(0.9, 1.05),
                sector_preference_weights=self._draw_household_sector_preference_weights(),
                age_periods=0,
                partnership_affinity_code=self._inherit_partnership_affinity_code(mother, father),
                next_partnership_attempt_period=int(round(self.config.entry_age_years * self.config.periods_per_year)),
                fertility_multiplier=self._draw_fertility_multiplier(),
                bank_id=inherited_bank_id,
                guardian_id=guardian.id,
                mother_id=mother.id,
                father_id=father.id if father is not None else None,
                desired_children=self._draw_desired_children(),
            )
        )
        for parent_id in parent_ids:
            parent = self._household_by_id(parent_id)
            if parent is not None:
                parent.dependent_children += 1
        mother.children_count += 1
        if father is not None:
            father.children_count += 1
        mother.last_birth_period = self.period
        mother.child_desire_pressure = 0.0
        if father is not None:
            father.last_birth_period = self.period
            father.child_desire_pressure = 0.0
        self._next_household_id += 1
        self._period_births += 1

    def _household_death_probability(self, household: Household, unemployment_rate: float, average_savings: float) -> float:
        age_years = self._household_age_years(household)
        if age_years >= self.config.max_age_years:
            return 1.0

        age_span = max(1.0, self.config.max_age_years - self.config.fertile_age_min_years)
        age_pressure = clamp((age_years - self.config.fertile_age_min_years) / age_span, 0.0, 1.0)
        period_probability = (
            self.config.period_base_death_probability
            + self.config.period_senior_death_probability * (age_pressure ** 3)
        )

        vulnerability_multiplier = 1.0
        if age_years < self.config.entry_age_years or age_years >= self.config.senior_age_years:
            vulnerability_multiplier = 1.25

        if household.severe_hunger_streak >= self.config.starvation_death_periods:
            period_probability += min(
                0.28,
                self.config.period_severe_hunger_death_risk
                * (1.0 + 0.30 * max(0, household.severe_hunger_streak - self.config.starvation_death_periods))
                * vulnerability_multiplier,
            )
        elif household.severe_hunger_streak > 0:
            period_probability += min(
                0.04,
                0.0035
                * household.severe_hunger_streak
                * vulnerability_multiplier,
            )
        elif household.deprivation_streak > 0:
            period_probability += min(
                0.06,
                1.5 * self.config.period_food_subsistence_death_risk
                * household.deprivation_streak
                * vulnerability_multiplier,
            )

        period_probability += (
            self.config.period_health_fragility_death_risk
            * clamp(household.health_fragility / 3.0, 0.0, 1.2)
            * vulnerability_multiplier
        )

        return clamp(period_probability, 0.0, 0.95)

    def _household_estate_heirs(self, household: Household) -> list[Household]:
        heir_ids: set[int] = set()
        for relative_id in (
            household.partner_id,
            household.guardian_id,
            household.mother_id,
            household.father_id,
        ):
            relative = self._household_by_id(relative_id) if relative_id is not None else None
            if relative is None or not relative.alive or relative.id == household.id:
                continue
            heir_ids.add(relative.id)

        for candidate in self._active_households():
            if candidate.id == household.id:
                continue
            if not candidate.alive:
                continue
            if candidate.mother_id == household.id or candidate.father_id == household.id:
                if self._household_age_years(candidate) >= self.config.entry_age_years:
                    heir_ids.add(candidate.id)
                    continue
                guardian = self._household_by_id(candidate.guardian_id) if candidate.guardian_id is not None else None
                if guardian is not None and guardian.alive and guardian.id != household.id:
                    heir_ids.add(guardian.id)

        heirs = [
            self.households[heir_id]
            for heir_id in sorted(heir_ids)
            if heir_id != household.id
        ]
        if heirs:
            return heirs
        return [
            member
            for member in self._active_households()
            if member.alive
            if member.id != household.id
            and self._household_age_years(member) >= self.config.entry_age_years
        ]

    def _transfer_household_estate(self, household: Household) -> None:
        estate = max(0.0, household.savings + household.wage_income)
        if estate <= 0.0:
            return
        heirs = self._household_estate_heirs(household)
        if not heirs:
            return
        share = estate / len(heirs)
        for heir in heirs:
            heir.savings += share
        self._period_inheritance_transfers += estate

    def _apply_demography(self, unemployment_rate: float) -> None:
        active_households = self._active_households()
        if not active_households:
            return

        average_savings = sum(household.savings for household in active_households) / len(active_households)
        prosperity = clamp(
            0.70 + 0.45 * (1.0 - unemployment_rate) + average_savings / 180.0,
            0.50,
            1.50,
        )

        (
            family_income_ratio_by_household,
            family_resources_ratio_by_household,
            family_resources_below_basket_by_household,
        ) = self._family_status_maps()
        economically_capable_by_household = {
            household_id: family_resources_ratio_by_household.get(household_id, 0.0)
            >= self.config.birth_capable_resource_ratio_min
            for household_id in family_resources_ratio_by_household
        }

        for household in active_households:
            previous_age_years = self._household_age_years(household)
            household.age_periods += 1
            current_age_years = self._household_age_years(household)
            if previous_age_years < self.config.entry_age_years <= current_age_years:
                if household.origin_record_period < 0:
                    household.origin_record_period = self.period
                    household.low_resource_origin = family_resources_below_basket_by_household.get(
                        household.id,
                        False,
                    )
                    household.origin_family_income_to_basket_ratio = family_income_ratio_by_household.get(
                        household.id,
                        0.0,
                    )
                    household.origin_family_resources_to_basket_ratio = family_resources_ratio_by_household.get(
                        household.id,
                        0.0,
                    )
                self._release_guardian_dependency(household)
            death_probability = self._household_death_probability(household, unemployment_rate, average_savings)
            if self.rng.random() < death_probability:
                self._transfer_household_estate(household)
                self._clear_partner_link(household)
                if current_age_years < self.config.entry_age_years:
                    self._release_guardian_dependency(household)
                self._release_household_from_employment(household)
                household.alive = False
                household.wage_income = 0.0
                household.savings = 0.0
                household.deprivation_streak = 0
                household.severe_hunger_streak = 0
                household.housing_deprivation_streak = 0
                household.clothing_deprivation_streak = 0
                household.health_fragility = 0.0
                self._period_deaths += 1

        self._period_active_households_cache = [
            household for household in self.households if household.alive
        ]
        self._reassign_orphans()
        fertile_mothers = [
            household
            for household in self._active_households()
            if household.sex == "F"
            and self._household_age_years(household) <= self.config.fertile_age_max_years
            and self._is_fertile(household)
            and household.children_count < max(0, household.desired_children)
            and (self.period - household.last_birth_period) >= self.config.birth_interval_periods
        ]
        for mother in fertile_mothers:
            father = self._household_by_id(mother.partner_id) if mother.partner_id is not None else None
            if father is not None and not father.alive:
                father = None
            economically_capable = economically_capable_by_household.get(mother.id, False)
            if economically_capable and father is not None:
                annual_birth_rate = self.config.annual_birth_rate_capable_partnered
            elif economically_capable:
                annual_birth_rate = self.config.annual_birth_rate_capable_single
            else:
                annual_birth_rate = self.config.annual_birth_rate_noncapable
            if father is not None:
                relationship_start = max(mother.partnership_start_period, father.partnership_start_period)
                months_together = max(0, self.period - relationship_start)
                years_together = months_together / max(1, self.config.periods_per_year)
                ramp_years = max(1e-9, self.config.partnered_birth_ramp_years)
                ramp_floor = clamp(self.config.partnered_birth_ramp_floor, 0.0, 1.0)
                relationship_birth_multiplier = clamp(
                    ramp_floor + (1.0 - ramp_floor) * min(1.0, years_together / ramp_years),
                    ramp_floor,
                    1.0,
                )
                annual_birth_rate *= relationship_birth_multiplier
            birth_probability = self._annual_to_period_probability(annual_birth_rate * prosperity)
            age_fertility_factor = self._female_age_fertility_factor(self._household_age_years(mother))
            fertility_multiplier = max(0.1, mother.fertility_multiplier)
            adjusted_probability = birth_probability * age_fertility_factor * fertility_multiplier
            if self.rng.random() < clamp(adjusted_probability, 0.0, 0.95):
                self._birth_household(mother, father)

    def _update_family_child_desires(
        self,
        family_members: list[Household],
        *,
        essential_target_by_sector: dict[str, float] | None = None,
        purchased_units_by_sector: dict[str, float] | None = None,
        essential_target_units: float | None = None,
        essential_units_bought: float | None = None,
        family_remaining_cash: float,
    ) -> None:
        family_basic_basket_cost = sum(self._essential_budget(member) for member in family_members)
        cash_buffer_ratio = family_remaining_cash / max(1.0, family_basic_basket_cost)
        if essential_target_by_sector is None:
            essential_target_by_sector = {}
        if purchased_units_by_sector is None:
            purchased_units_by_sector = {}
        if essential_target_units is not None and essential_units_bought is not None and not essential_target_by_sector:
            blended_coverage = essential_units_bought / essential_target_units if essential_target_units > 0.0 else 1.0
            for sector_key in ESSENTIAL_SECTOR_KEYS:
                essential_target_by_sector[sector_key] = 1.0
                purchased_units_by_sector[sector_key] = blended_coverage
        food_target = max(1e-9, essential_target_by_sector.get("food", 0.0))
        housing_target = max(1e-9, essential_target_by_sector.get("housing", 0.0))
        clothing_target = max(1e-9, essential_target_by_sector.get("clothing", 0.0))
        food_coverage = purchased_units_by_sector.get("food", 0.0) / food_target if food_target > 0.0 else 1.0
        housing_coverage = purchased_units_by_sector.get("housing", 0.0) / housing_target if housing_target > 0.0 else 1.0
        clothing_coverage = purchased_units_by_sector.get("clothing", 0.0) / clothing_target if clothing_target > 0.0 else 1.0
        stability = clamp(
            0.45 * min(food_coverage, 1.25)
            + 0.12 * min(housing_coverage, 1.25)
            + 0.08 * min(clothing_coverage, 1.25)
            + 0.35 * min(cash_buffer_ratio, 1.5),
            0.0,
            1.5,
        )

        for member in family_members:
            if member.sex != "F":
                continue
            if self._household_age_years(member) < self.config.entry_age_years:
                continue
            if not self._is_fertile(member):
                continue

            partner = self._household_by_id(member.partner_id) if member.partner_id is not None else None
            if partner is not None and partner.alive and self._household_age_years(partner) >= self.config.entry_age_years:
                if partner.desired_children < member.desired_children:
                    partner.desired_children = member.desired_children

    def _capital_efficiency(self, capital: float) -> float:
        return 1.0 + math.log1p(max(0.0, capital) / self.config.capital_scale)

    def _public_infrastructure_log_scale(self) -> float:
        scale = max(1.0, self.config.government_public_capital_scale)
        government = getattr(self, "government", None)
        public_capital_stock = getattr(government, "public_capital_stock", 0.0)
        return math.log1p(max(0.0, public_capital_stock) / scale)

    def _public_infrastructure_productivity_multiplier(self) -> float:
        gain = max(0.0, self.config.government_public_capital_productivity_gain)
        return 1.0 + gain * self._public_infrastructure_log_scale()

    def _public_infrastructure_transport_cost_multiplier(self) -> float:
        gain = max(0.0, self.config.government_public_capital_transport_gain)
        return clamp(1.0 - gain * self._public_infrastructure_log_scale(), 0.70, 1.0)

    def _investment_knowledge_multiplier(self) -> float:
        active_households = self._active_households()
        if not active_households:
            return 1.0
        adult_households = [
            household
            for household in active_households
            if self._household_age_years(household) >= self.config.entry_age_years
        ]
        labor_force_households = [
            household
            for household in active_households
            if household.alive and self._household_labor_capacity(household) > 0.0
        ]
        university_completion_share = (
            sum(1 for household in adult_households if self._household_has_university_credential(household))
            / max(1, len(adult_households))
        )
        skilled_labor_share = (
            sum(1 for household in labor_force_households if self._household_has_university_credential(household))
            / max(1, len(labor_force_households))
        )
        knowledge_share = clamp(
            self.config.firm_investment_knowledge_university_weight * university_completion_share
            + self.config.firm_investment_knowledge_skill_weight * skilled_labor_share,
            0.0,
            1.0,
        )
        return clamp(
            self.config.firm_investment_knowledge_floor
            + (
                self.config.firm_investment_knowledge_ceiling
                - self.config.firm_investment_knowledge_floor
            )
            * math.sqrt(knowledge_share),
            self.config.firm_investment_knowledge_floor,
            self.config.firm_investment_knowledge_ceiling,
        )

    def _is_education_sector(self, sector_key: str) -> bool:
        return sector_key in ("school", "university")

    def _education_total_levels(self, sector_key: str) -> int:
        if sector_key == "school":
            return max(1, round(self.config.school_years_required))
        if sector_key == "university":
            return max(1, round(self.config.university_years_required))
        return 1

    def _education_level_span_floor(self, sector_key: str) -> int:
        total_levels = self._education_total_levels(sector_key)
        if sector_key == "school":
            return min(total_levels, max(3, math.ceil(total_levels * 0.25)))
        if sector_key == "university":
            return min(total_levels, 1)
        return 1

    def _education_students_per_classroom(self, sector_key: str) -> float:
        if sector_key == "school":
            return max(8.0, self.config.school_students_per_classroom)
        if sector_key == "university":
            return max(6.0, self.config.university_students_per_classroom)
        return 1.0

    def _education_classroom_capital_cost(self, sector_key: str) -> float:
        if sector_key == "school":
            return max(0.1, self.config.school_classroom_capital_cost)
        if sector_key == "university":
            return max(0.1, self.config.university_classroom_capital_cost)
        return 1.0

    def _education_support_staff_ratio(self, sector_key: str) -> float:
        if sector_key == "school":
            return max(0.0, self.config.school_support_staff_ratio)
        if sector_key == "university":
            return max(0.0, self.config.university_support_staff_ratio)
        return 0.0

    def _education_operational_complexity(self, sector_key: str, level_span: float) -> float:
        total_levels = self._education_total_levels(sector_key)
        span_share = clamp(level_span / max(1.0, total_levels), 0.05, 1.0)
        if sector_key == "school":
            return 0.82 + 0.42 * span_share
        if sector_key == "university":
            return 0.90 + 0.50 * span_share
        return 1.0

    def _draw_education_level_span(self, sector_key: str, package_scale: float) -> float:
        total_levels = self._education_total_levels(sector_key)
        level_floor = self._education_level_span_floor(sector_key)
        scale_signal = clamp(0.28 + 0.62 * math.sqrt(max(0.01, package_scale)), 0.20, 1.0)
        mean_span = total_levels * scale_signal
        span = round(mean_span * self.rng.uniform(0.88, 1.12))
        return float(clamp(span, float(level_floor), float(total_levels)))

    def _education_facility_capacity(
        self,
        sector_key: str,
        capital: float,
        level_span: float,
    ) -> float:
        classroom_cost = self._education_classroom_capital_cost(sector_key)
        classroom_count = max(1.0, capital / classroom_cost)
        classroom_size = self._education_students_per_classroom(sector_key)
        complexity = self._education_operational_complexity(sector_key, level_span)
        return max(1.0, classroom_count * classroom_size / max(0.25, complexity))

    def _education_firm_capacity(self, firm: Firm) -> float:
        if not self._is_education_sector(firm.sector):
            return float("inf")
        level_span = firm.education_level_span
        if level_span <= 0.0:
            level_span = float(self._education_level_span_floor(firm.sector))
        return self._education_facility_capacity(firm.sector, firm.capital, level_span)

    def _education_service_target_units(self, firm: Firm, expected_sales: float) -> float:
        capacity = self._education_firm_capacity(firm)
        buffer_multiplier = 1.0 + 0.35 * SECTOR_BY_KEY[firm.sector].target_inventory_ratio
        return clamp(
            max(1.0, expected_sales * buffer_multiplier),
            1.0,
            max(1.0, capacity),
        )

    def _firm_inventory_buffer_multiplier(self, firm: Firm) -> float:
        return clamp(
            1.10 - 0.18 * (firm.inventory_aversion - 1.0),
            0.70,
            1.45,
        )

    def _firm_target_inventory_units(self, firm: Firm, expected_sales: float) -> float:
        if self._is_education_sector(firm.sector):
            return self._education_service_target_units(firm, expected_sales)
        spec = SECTOR_BY_KEY[firm.sector]
        if spec.target_inventory_ratio <= 0.0:
            return 0.0
        return max(
            1.0,
            max(0.0, expected_sales) * spec.target_inventory_ratio * self._firm_inventory_buffer_multiplier(firm),
        )

    def _firm_desired_output_from_expected_sales(self, firm: Firm, expected_sales: float) -> float:
        target_inventory = self._firm_target_inventory_units(firm, expected_sales)
        return max(0.0, max(0.0, expected_sales) + target_inventory - max(0.0, firm.inventory))

    def _workers_needed_for_units(
        self,
        units: float,
        effective_productivity: float,
        *,
        productivity_floor: float = 0.1,
    ) -> int:
        return max(1, math.ceil(max(0.0, units) / max(productivity_floor, effective_productivity)))

    def _education_entry_package_estimate(
        self,
        spec,
        demand_units: float,
        package_scale: float,
    ) -> tuple[float, float, float]:
        effective_demand = max(1.0, demand_units)
        level_span = self._education_total_levels(spec.key) * clamp(
            0.28 + 0.57 * math.sqrt(max(0.01, package_scale)),
            0.20,
            1.0,
        )
        complexity = self._education_operational_complexity(spec.key, level_span)
        target_capacity = effective_demand * (1.0 + 0.35 * spec.target_inventory_ratio)
        classroom_size = self._education_students_per_classroom(spec.key)
        classrooms = max(1.0, math.ceil(target_capacity * complexity / max(1.0, classroom_size)))
        capital_budget = classrooms * self._education_classroom_capital_cost(spec.key)
        support_ratio = self._education_support_staff_ratio(spec.key)
        students_per_worker = classroom_size / max(1.1, 1.0 + support_ratio + 0.25 * complexity)
        desired_workers = max(1, math.ceil(target_capacity / max(0.25, students_per_worker)))
        input_cost_per_unit = spec.base_price * (0.075 + 0.035 * complexity)
        transport_cost_per_unit = spec.base_price * (0.018 + 0.010 * complexity)
        fixed_overhead = classrooms * spec.base_wage * (0.42 + 0.18 * complexity)
        inventory_units = min(
            target_capacity,
            effective_demand * clamp(self.config.startup_expected_sales_share, 0.35, 1.0),
        )
        inventory_budget = inventory_units * spec.base_price
        cash_budget = self._entry_cash_budget(
            max(0.10, package_scale),
            desired_workers * spec.base_wage,
            effective_demand * input_cost_per_unit,
            effective_demand * transport_cost_per_unit,
            fixed_overhead,
        )
        return cash_budget, capital_budget, inventory_budget

    def _draw_education_blueprint(
        self,
        spec,
        demand_units: float,
        package_scale: float,
    ) -> dict[str, float]:
        effective_demand = max(1.0, demand_units)
        level_span = self._draw_education_level_span(spec.key, package_scale)
        complexity = self._education_operational_complexity(spec.key, level_span)
        target_capacity = effective_demand * self.rng.uniform(1.01, 1.08 + 0.10 * spec.target_inventory_ratio)
        classroom_size = self._education_students_per_classroom(spec.key)
        classrooms = max(1.0, math.ceil(target_capacity * complexity / max(1.0, classroom_size)))
        capital_budget = classrooms * self._education_classroom_capital_cost(spec.key) * self.rng.uniform(0.95, 1.10)
        support_ratio = self._education_support_staff_ratio(spec.key)
        students_per_worker = classroom_size / max(
            1.1,
            1.0 + support_ratio + 0.22 * complexity + self.rng.uniform(-0.05, 0.05),
        )
        input_cost_per_unit = spec.base_price * (
            0.07 + 0.04 * complexity + self.rng.uniform(-0.01, 0.015)
        )
        transport_cost_per_unit = spec.base_price * (
            0.018 + 0.012 * complexity + self.rng.uniform(-0.004, 0.006)
        )
        fixed_overhead = classrooms * spec.base_wage * (
            0.40 + 0.22 * complexity + self.rng.uniform(-0.05, 0.08)
        )
        return {
            "level_span": max(1.0, level_span),
            "capital_budget": max(1.0, capital_budget),
            "students_per_worker": max(0.5, students_per_worker),
            "input_cost_per_unit": max(0.05, input_cost_per_unit),
            "transport_cost_per_unit": max(0.0, transport_cost_per_unit),
            "fixed_overhead": max(0.1, fixed_overhead),
            "target_capacity": max(1.0, target_capacity),
        }

    def _firm_effective_productivity(self, firm: Firm) -> float:
        infrastructure_multiplier = self._public_infrastructure_productivity_multiplier()
        if self._is_education_sector(firm.sector):
            return max(
                0.5,
                firm.productivity
                * firm.technology
                * (0.92 + 0.08 * self._capital_efficiency(firm.capital))
                * infrastructure_multiplier,
            )
        return max(
            0.1,
            firm.productivity
            * firm.technology
            * self._capital_efficiency(firm.capital)
            * infrastructure_multiplier,
        )

    def _firm_costing_units(self, firm: Firm, realized_output: float) -> float:
        if self._is_education_sector(firm.sector):
            target_service_units = max(
                1.0,
                min(
                    self._education_firm_capacity(firm),
                    max(realized_output, firm.target_inventory, firm.last_expected_sales, firm.last_sales),
                ),
            )
            return target_service_units
        spec = SECTOR_BY_KEY[firm.sector]
        expected_anchor = max(0.0, firm.last_expected_sales, firm.last_sales)
        target_anchor = 0.0
        if spec.target_inventory_ratio > 0.0 and firm.target_inventory > 0.0:
            target_anchor = firm.target_inventory / spec.target_inventory_ratio
        normal_scale = max(expected_anchor, target_anchor)
        if normal_scale <= 0.0:
            return max(1.0, realized_output)
        if realized_output <= 0.0:
            return max(1.0, normal_scale)
        return max(realized_output, 0.35 * normal_scale)

    def _observed_sector_demand_signal(self, spec_key: str, use_current_period: bool = False) -> float:
        if use_current_period:
            sales_units = self._period_sector_sales_units.get(spec_key, 0.0)
            revealed_unmet_units = self._period_sector_revealed_unmet_units.get(spec_key, 0.0)
        else:
            sales_units = self._recent_sector_signal_average(
                self._sector_sales_history,
                spec_key,
                fallback=self._last_sector_sales_units.get(spec_key, 0.0),
            )
            revealed_unmet_units = self._recent_sector_signal_average(
                self._sector_revealed_unmet_history,
                spec_key,
                fallback=self._last_sector_revealed_unmet_units.get(spec_key, 0.0),
            )
        shortage_visibility = 0.55
        if spec_key in ESSENTIAL_SECTOR_KEYS:
            shortage_visibility = 0.90
        elif spec_key in MERIT_SECTOR_KEYS:
            shortage_visibility = 0.75
        # Firms do not observe the whole latent market. They see realized sales
        # plus shoppers with money leaving empty-handed after trying to buy.
        return max(0.0, sales_units + shortage_visibility * revealed_unmet_units)

    def _sector_revealed_shortage_signal(self, sector_key: str, use_current_period: bool = False) -> float:
        if use_current_period:
            sales_units = self._period_sector_sales_units.get(sector_key, 0.0)
            revealed_unmet_units = self._period_sector_revealed_unmet_units.get(sector_key, 0.0)
        else:
            sales_units = self._recent_sector_signal_average(
                self._sector_sales_history,
                sector_key,
                fallback=self._last_sector_sales_units.get(sector_key, 0.0),
            )
            revealed_unmet_units = self._recent_sector_signal_average(
                self._sector_revealed_unmet_history,
                sector_key,
                fallback=self._last_sector_revealed_unmet_units.get(sector_key, 0.0),
            )
        denominator = max(1.0, sales_units + revealed_unmet_units)
        return clamp(revealed_unmet_units / denominator, 0.0, 1.0)

    def _sector_revealed_expansion_pressure(self, sector_key: str, use_current_period: bool = False) -> float:
        if use_current_period:
            sales_units = self._period_sector_sales_units.get(sector_key, 0.0)
            revealed_unmet_units = self._period_sector_revealed_unmet_units.get(sector_key, 0.0)
        else:
            sales_units = self._recent_sector_signal_average(
                self._sector_sales_history,
                sector_key,
                fallback=self._last_sector_sales_units.get(sector_key, 0.0),
            )
            revealed_unmet_units = self._recent_sector_signal_average(
                self._sector_revealed_unmet_history,
                sector_key,
                fallback=self._last_sector_revealed_unmet_units.get(sector_key, 0.0),
            )
        if sales_units <= 0.0 and revealed_unmet_units <= 0.0:
            return 0.0
        shortage_share = revealed_unmet_units / max(1.0, sales_units + revealed_unmet_units)
        walkaway_intensity = revealed_unmet_units / max(1.0, sales_units)
        return clamp(0.65 * shortage_share + 0.35 * walkaway_intensity, 0.0, 1.5)

    def _firm_revealed_growth_pressure(self, firm: Firm, use_current_period: bool = False) -> float:
        sector_pressure = self._sector_revealed_expansion_pressure(
            firm.sector,
            use_current_period=use_current_period,
        )
        if sector_pressure <= 0.0:
            return 0.0
        sell_through = clamp((self._firm_recent_sell_through(firm) - 0.55) / 0.45, 0.0, 1.25)
        cash_cover = clamp(self._firm_cash_cover_ratio(firm) / 1.75, 0.0, 1.25)
        growth_pressure = sector_pressure * (0.45 + 0.55 * sell_through) * (0.35 + 0.65 * cash_cover)
        if firm.last_profit < 0.0:
            growth_pressure *= 0.75
        return clamp(growth_pressure, 0.0, 1.5)

    def _baseline_demand(self, spec_key: str, use_current_period: bool = False) -> float:
        cache_key = (spec_key, use_current_period, self.period)
        cached = self._period_baseline_demand_cache.get(cache_key)
        if cached is not None:
            return cached
        structural_demand = self._structural_sector_demand(spec_key)
        if not self.history and not use_current_period and self.period == 0:
            self._period_baseline_demand_cache[cache_key] = structural_demand
            return structural_demand
        if spec_key in ESSENTIAL_SECTOR_KEYS and self._in_startup_grace() and not use_current_period:
            self._period_baseline_demand_cache[cache_key] = structural_demand
            return structural_demand
        observed_demand = self._observed_sector_demand_signal(spec_key, use_current_period=use_current_period)
        if observed_demand > 0.0:
            learning_maturity = self._market_learning_maturity()
            if spec_key in ESSENTIAL_SECTOR_KEYS:
                observed_weight = 0.18 + 0.47 * learning_maturity
                structural_floor = 0.72 - 0.12 * learning_maturity
                baseline = max(
                    structural_floor * structural_demand,
                    observed_weight * observed_demand + (1.0 - observed_weight) * structural_demand,
                )
            else:
                observed_weight = 0.12 + 0.63 * learning_maturity
                structural_floor = 0.45 - 0.20 * learning_maturity
                baseline = max(
                    structural_floor * structural_demand,
                    observed_weight * observed_demand + (1.0 - observed_weight) * structural_demand,
                )
        else:
            baseline = structural_demand
        self._period_baseline_demand_cache[cache_key] = baseline
        return baseline

    def _random_firm_cost_structure(self, spec) -> tuple[float, float, float]:
        # Los bienes esenciales parten con una estructura de costos mas ligera.
        # La idea es reflejar que la produccion basica suele apoyarse en escala,
        # estandarizacion y menor carga de distribucion que los bienes discrecionales.
        input_cost_per_unit = self._draw_firm_input_cost(spec)
        if spec.key == "food":
            transport_range = (0.02, 0.06)
            overhead_range = (0.80, 1.35)
        elif spec.key == "housing":
            transport_range = (0.02, 0.06)
            overhead_range = (0.90, 1.45)
        elif spec.key == "clothing":
            transport_range = (0.02, 0.055)
            overhead_range = (0.85, 1.40)
        elif spec.key == "manufactured":
            transport_range = (0.03, 0.10)
            overhead_range = (1.15, 2.15)
        elif spec.key == "school":
            transport_range = (0.015, 0.05)
            overhead_range = (1.35, 2.40)
        elif spec.key == "university":
            transport_range = (0.015, 0.045)
            overhead_range = (1.55, 2.85)
        else:
            transport_range = (0.04, 0.12)
            overhead_range = (1.20, 2.30)

        transport_cost_per_unit = spec.base_price * self.rng.uniform(*transport_range)
        fixed_overhead = spec.base_wage * self.rng.uniform(*overhead_range)
        return input_cost_per_unit, transport_cost_per_unit, fixed_overhead

    def _draw_firm_input_cost(self, spec) -> float:
        if spec.key == "food":
            input_range = (0.06, 0.14)
        elif spec.key == "housing":
            input_range = (0.07, 0.16)
        elif spec.key == "clothing":
            input_range = (0.065, 0.15)
        elif spec.key == "manufactured":
            input_range = (0.10, 0.22)
        elif spec.key == "school":
            input_range = (0.05, 0.12)
        elif spec.key == "university":
            input_range = (0.05, 0.11)
        else:
            input_range = (0.11, 0.24)
        return spec.base_price * self.rng.uniform(*input_range)

    def _ensure_active_food_input_exemption(self) -> None:
        food_firms = [firm for firm in self.firms if firm.sector == "food"]
        if not food_firms:
            return

        active_food_firms = [firm for firm in food_firms if firm.active]
        if active_food_firms and not any(firm.input_cost_exempt for firm in active_food_firms):
            chosen = self.rng.choice(active_food_firms)
            chosen.input_cost_exempt = True

        for firm in food_firms:
            if firm.input_cost_exempt:
                firm.input_cost_per_unit = 0.0

    def _initial_technology(self, sector_key: str) -> float:
        if sector_key in ESSENTIAL_SECTOR_KEYS:
            baseline = self.config.essential_technology_multiplier
            spread = 0.10
        else:
            baseline = self.config.nonessential_technology_multiplier
            spread = 0.12
        return baseline * self.rng.uniform(1.0 - spread, 1.0 + spread)

    def _initial_demand_elasticity(self, sector_key: str) -> float:
        if sector_key == "food":
            baseline = 0.82
            spread = 0.10
        elif sector_key == "housing":
            baseline = 0.76
            spread = 0.10
        elif sector_key == "clothing":
            baseline = 0.92
            spread = 0.12
        elif sector_key == "manufactured":
            baseline = 1.18
            spread = 0.15
        elif sector_key == "school":
            baseline = 0.96
            spread = 0.12
        elif sector_key == "university":
            baseline = 1.08
            spread = 0.14
        else:
            baseline = 1.32
            spread = 0.18
        return clamp(baseline * self.rng.uniform(1.0 - spread, 1.0 + spread), 0.45, 2.75)

    def _initial_forecast_error_belief(self, sector_key: str) -> float:
        if sector_key in ESSENTIAL_SECTOR_KEYS:
            baseline = 0.16
            spread = 0.07
        elif sector_key == "manufactured":
            baseline = 0.18
            spread = 0.08
        else:
            baseline = 0.20
            spread = 0.09
        return clamp(baseline * self.rng.uniform(1.0 - spread, 1.0 + spread), 0.05, 0.55)

    def _sector_productivity_multiplier(self, sector_key: str) -> float:
        if sector_key in ESSENTIAL_SECTOR_KEYS:
            return self.config.essential_productivity_multiplier
        return self.config.nonessential_productivity_multiplier

    def _entry_productivity_multiplier(self, sector_key: str) -> float:
        baseline = self._sector_productivity_multiplier(sector_key)
        if sector_key in ESSENTIAL_SECTOR_KEYS:
            # Sector frontier productivity can be high, but new firms should not
            # start operating at mature-system efficiency from period 1.
            return 1.0 + 0.08 * max(0.0, baseline - 1.0)
        return baseline

    def _initial_firm_price(self, spec, unit_cost: float) -> float:
        if spec.key in ESSENTIAL_SECTOR_KEYS or self._is_education_sector(spec.key):
            return max(0.1, unit_cost * (1.0 + spec.markup))
        return spec.base_price * self.rng.uniform(0.96, 1.04)

    def _entry_cost_structure_estimate(self, spec) -> tuple[float, float, float]:
        if spec.key == "food":
            return 0.10, 0.04, 1.075
        if spec.key == "housing":
            return 0.115, 0.04, 1.175
        if spec.key == "clothing":
            return 0.1075, 0.0375, 1.125
        if spec.key == "manufactured":
            return 0.16, 0.065, 1.65
        if spec.key == "school":
            return 0.09, 0.03, 1.95
        if spec.key == "university":
            return 0.085, 0.025, 2.20
        return 0.175, 0.08, 1.75

    def _entry_cash_budget(
        self,
        package_scale: float,
        wage_bill: float,
        input_cost: float,
        transport_cost: float,
        fixed_overhead: float,
    ) -> float:
        reserve_periods = clamp(self.config.cash_reserve_periods, 0.75, 1.5)
        operating_cycle_cost = wage_bill + input_cost + transport_cost + fixed_overhead
        minimum_cash_buffer = self.config.startup_firm_cash * max(0.10, package_scale)
        return max(minimum_cash_buffer, operating_cycle_cost * reserve_periods)

    def _entry_package_budgets(
        self,
        spec,
        demand_units: float,
        package_scale: float,
    ) -> tuple[float, float, float]:
        if self._is_education_sector(spec.key):
            return self._education_entry_package_estimate(spec, demand_units, package_scale)
        effective_scale = max(0.05, package_scale)
        effective_demand = max(1.0, demand_units)
        capital_budget = self.config.startup_firm_capital * effective_scale
        inventory_units = max(
            1.0,
            effective_demand * spec.target_inventory_ratio * self.config.startup_inventory_multiplier,
        )
        inventory_budget = inventory_units * spec.base_price
        technology_baseline = (
            self.config.essential_technology_multiplier
            if spec.key in ESSENTIAL_SECTOR_KEYS
            else self.config.nonessential_technology_multiplier
        )
        effective_productivity = max(
            0.1,
            spec.base_productivity
            * self._entry_productivity_multiplier(spec.key)
            * technology_baseline
            * self._capital_efficiency(capital_budget),
        )
        desired_output = effective_demand + inventory_units
        desired_workers = max(1, math.ceil(desired_output / effective_productivity))
        input_ratio, transport_ratio, overhead_multiplier = self._entry_cost_structure_estimate(spec)
        wage_bill = desired_workers * spec.base_wage
        input_cost = effective_demand * spec.base_price * input_ratio
        transport_cost = effective_demand * spec.base_price * transport_ratio
        fixed_overhead = spec.base_wage * overhead_multiplier
        cash_budget = self._entry_cash_budget(
            effective_scale,
            wage_bill,
            input_cost,
            transport_cost,
            fixed_overhead,
        )
        return cash_budget, capital_budget, inventory_budget

    def _restart_funding_need(self, spec, scale: float, demand_units: float | None = None) -> float:
        package_multiplier = self.config.firm_restart_package_multiplier
        baseline_demand = max(
            0.0,
            demand_units if demand_units is not None else self._baseline_demand(spec.key, use_current_period=True),
        )
        effective_scale = scale * package_multiplier
        effective_demand = baseline_demand * effective_scale
        cash_budget, capital_budget, inventory_budget = self._entry_package_budgets(
            spec,
            effective_demand,
            effective_scale,
        )
        return cash_budget + capital_budget + inventory_budget

    def _startup_slot_share(self, sector_key: str) -> float:
        return 1.0 / max(1, self._initial_sector_firm_count(sector_key))

    def _startup_funding_need(
        self,
        spec,
        demand_units: float | None = None,
        package_scale: float = 1.0,
    ) -> float:
        baseline_demand = max(0.0, demand_units if demand_units is not None else self._baseline_demand(spec.key))
        cash_budget, capital_budget, inventory_budget = self._entry_package_budgets(
            spec,
            baseline_demand,
            package_scale,
        )
        return cash_budget + capital_budget + inventory_budget

    def _firm_adaptation_threshold(self, firm: Firm) -> int:
        financial_buffer = self._firm_financial_buffer(firm)
        buffer_ratio = financial_buffer / max(1.0, self.config.startup_firm_capital)
        return max(3, 3 + int(math.log1p(buffer_ratio)))

    def _firm_bankruptcy_limit(self, firm: Firm) -> int:
        financial_buffer = self._firm_financial_buffer(firm)
        buffer_ratio = financial_buffer / max(1.0, self.config.startup_firm_capital)
        return self.config.bankruptcy_streak_limit + 1 + int(math.log1p(buffer_ratio))

    def _firm_financial_buffer(self, firm: Firm) -> float:
        spec = SECTOR_BY_KEY[firm.sector]
        liquidation_inventory = 0.30 * firm.inventory * spec.base_price
        return max(0.0, firm.capital + max(0.0, firm.cash) + liquidation_inventory)

    def _firm_cash_failure_limits(self, firm: Firm) -> tuple[float, float]:
        operating_scale = max(
            20.0,
            firm.last_wage_bill + firm.last_input_cost + firm.last_transport_cost + firm.last_fixed_overhead,
        )
        grace_cash_limit = min(self.config.bankruptcy_cash_threshold, -0.50 * operating_scale)
        critical_cash_limit = min(self.config.critical_cash_threshold, -1.00 * operating_scale)
        return grace_cash_limit, critical_cash_limit

    def _sector_profit_signal(self, sector_key: str) -> float:
        sector_firms = self._sector_firms(sector_key)
        if not sector_firms:
            return 0.0

        margins: list[float] = []
        for firm in sector_firms:
            reference_revenue = max(
                1.0,
                firm.last_revenue,
                firm.last_sales * max(0.1, firm.price),
                firm.last_expected_sales * max(0.1, firm.price),
            )
            margins.append(clamp(firm.last_profit / reference_revenue, -1.0, 1.0))
        return sum(margins) / max(1, len(margins))

    def _sector_entry_opportunity_signal(
        self,
        sector_key: str,
        demand_signal: float,
        entry_gap: float,
    ) -> float:
        gap_ratio = clamp(entry_gap / max(1.0, demand_signal), 0.0, 2.0)
        walkaway_signal = self._sector_public_fragility_signal(sector_key)
        profitability_signal = max(0.0, self._sector_profit_signal(sector_key))
        economy_fragility = self._economy_public_fragility_signal()
        social_fragility = self._social_survival_fragility_signal(sector_key)
        affluence_signal = 0.0
        if self.history:
            last_snapshot = self.history[-1]
            affluence_signal = clamp(
                last_snapshot.family_resources_to_basket_ratio - 1.0,
                0.0,
                1.5,
            )
        if sector_key == "leisure":
            affluence_signal *= 1.25
        elif sector_key == "school":
            affluence_signal *= 1.10
        elif sector_key == "university":
            affluence_signal *= 1.35
        return clamp(
            0.55 * gap_ratio
            + 0.25 * walkaway_signal
            + 0.25 * profitability_signal
            + 0.20 * affluence_signal
            - 0.22 * economy_fragility
            - 0.12 * social_fragility,
            -1.0,
            2.5,
        )

    def _entry_owner_score(
        self,
        owner: Entrepreneur,
        sector_key: str,
        opportunity_signal: float,
        base_restart_cost: float,
    ) -> tuple[float, float]:
        available_surplus = self._owner_total_liquid(owner) - self.config.firm_restart_wealth_threshold
        if available_surplus <= 0.0 or base_restart_cost <= 0.0:
            return float("-inf"), float("inf")

        liquidity_ratio = clamp(available_surplus / base_restart_cost, 0.0, 4.0)
        sector_bias = 0.0
        if sector_key == "leisure":
            sector_bias = 0.10 * max(0.0, owner.entry_appetite - 0.90)
        research_quality = clamp(owner.market_research_skill, 0.40, 1.80)
        perception_noise = self.rng.gauss(
            owner.entry_optimism + sector_bias,
            0.28 / research_quality,
        )
        expected_signal = opportunity_signal + perception_noise
        score = (
            expected_signal * (0.80 + 0.30 * research_quality)
            + 0.22 * liquidity_ratio
            + 0.28 * (owner.entry_appetite - 1.0)
        )
        threshold = 0.82 - 0.22 * (owner.entry_appetite - 1.0) - 0.12 * (research_quality - 1.0)
        return score, threshold

    def _select_entry_owner(
        self,
        sector_key: str,
        demand_signal: float,
        entry_gap: float,
        base_restart_cost: float,
    ) -> Entrepreneur | None:
        opportunity_signal = self._sector_entry_opportunity_signal(
            sector_key,
            demand_signal,
            entry_gap,
        )
        best_owner: Entrepreneur | None = None
        best_score = float("-inf")
        for owner in self.entrepreneurs:
            score, threshold = self._entry_owner_score(
                owner,
                sector_key,
                opportunity_signal,
                base_restart_cost,
            )
            if score < threshold or score <= best_score:
                continue
            best_owner = owner
            best_score = score
        return best_owner

    def _best_peer_firm(self, sector_key: str, exclude_firm_id: int | None = None) -> Firm | None:
        peers = [
            firm
            for firm in self._sector_firms(sector_key)
            if firm.active and firm.id != exclude_firm_id
        ]
        if not peers:
            return None
        return max(
            peers,
            key=lambda firm: (
                firm.last_profit,
                firm.last_revenue,
                firm.last_sales,
                -firm.price,
            ),
        )

    def _copy_peer_behavior(self, firm: Firm, peer: Firm, spec) -> None:
        weight = self.rng.uniform(0.65, 0.85)

        def blend(current: float, target: float, lower: float, upper: float) -> float:
            noisy_target = target * self.rng.uniform(0.96, 1.04)
            return clamp(weight * noisy_target + (1.0 - weight) * current, lower, upper)

        firm.markup_tolerance = blend(firm.markup_tolerance, peer.markup_tolerance, 0.50, 1.60)
        firm.volume_preference = blend(firm.volume_preference, peer.volume_preference, 0.50, 1.70)
        firm.inventory_aversion = blend(firm.inventory_aversion, peer.inventory_aversion, 0.50, 1.70)
        firm.employment_inertia = blend(firm.employment_inertia, peer.employment_inertia, 0.40, 0.95)
        firm.price_aggressiveness = blend(firm.price_aggressiveness, peer.price_aggressiveness, 0.50, 1.70)
        firm.cash_conservatism = blend(firm.cash_conservatism, peer.cash_conservatism, 0.50, 1.70)
        firm.market_share_ambition = blend(firm.market_share_ambition, peer.market_share_ambition, 0.50, 1.70)
        firm.expansion_credit_appetite = blend(
            firm.expansion_credit_appetite,
            peer.expansion_credit_appetite,
            0.55,
            1.75,
        )
        firm.stability_sensitivity = blend(
            firm.stability_sensitivity,
            peer.stability_sensitivity,
            0.55,
            1.75,
        )
        firm.investment_animal_spirits = blend(
            firm.investment_animal_spirits,
            peer.investment_animal_spirits,
            0.55,
            1.75,
        )
        firm.forecast_caution = blend(firm.forecast_caution, peer.forecast_caution, 0.60, 1.85)
        firm.demand_elasticity = blend(firm.demand_elasticity, peer.demand_elasticity, 0.45, 2.75)
        firm.wage_offer = max(
            self._sector_wage_floor(spec.key),
            peer.wage_offer * self.rng.uniform(0.95, 1.05),
        )
        copied_price = self._initial_firm_price(spec, max(0.1, peer.last_unit_cost * self.rng.uniform(0.95, 1.05)))
        firm.price = clamp(
            copied_price,
            spec.base_price * 0.35,
            min(spec.base_price * self.config.price_ceiling_multiplier, firm.price * self._firm_max_price_hike_ratio(firm)),
        )
        if firm.input_cost_exempt and firm.sector == "food":
            firm.input_cost_per_unit = 0.0
        else:
            peer_input_cost = peer.input_cost_per_unit
            if firm.sector == "food" and peer_input_cost <= 0.0:
                peer_input_cost = self._draw_firm_input_cost(spec)
            firm.input_cost_per_unit = max(0.0, peer_input_cost * self.rng.uniform(0.92, 1.08))
        firm.transport_cost_per_unit = max(0.0, peer.transport_cost_per_unit * self.rng.uniform(0.92, 1.08))
        firm.fixed_overhead = max(0.0, peer.fixed_overhead * self.rng.uniform(0.92, 1.08))
        firm.sales_history = [
            max(0.0, sale * self.rng.uniform(0.90, 1.10))
            for sale in self._firm_memory_slice(peer)
        ] or [max(1.0, peer.last_sales)]
        firm.expected_sales_history = [
            max(0.0, sale * self.rng.uniform(0.90, 1.10))
            for sale in (peer.expected_sales_history or [max(1.0, peer.last_expected_sales)])
        ]
        firm.production_history = [
            max(0.0, output * self.rng.uniform(0.90, 1.10))
            for output in (peer.production_history or [max(0.0, peer.last_production)])
        ]
        firm.last_expected_sales = self._smoothed_expected_sales_reference(firm)
        firm.forecast_error_belief = clamp(
            0.75 * firm.forecast_error_belief + 0.25 * peer.forecast_error_belief,
            0.03,
            1.25,
        )
        firm.target_inventory = max(1.0, peer.target_inventory * self.rng.uniform(0.85, 1.15))
        firm.loss_streak = max(0, firm.loss_streak - 1)

    def _mutate_firm_behavior(self, firm: Firm, spec) -> None:
        traits = self._random_firm_behavior_traits(spec)
        drift = self.rng.uniform(0.30, 0.60)

        def mutate(current: float, target: float, lower: float, upper: float) -> float:
            return clamp((1.0 - drift) * current + drift * target, lower, upper)

        firm.markup_tolerance = mutate(firm.markup_tolerance, traits["markup_tolerance"], 0.50, 1.60)
        firm.volume_preference = mutate(firm.volume_preference, traits["volume_preference"], 0.50, 1.70)
        firm.inventory_aversion = mutate(firm.inventory_aversion, traits["inventory_aversion"], 0.50, 1.70)
        firm.employment_inertia = mutate(firm.employment_inertia, traits["employment_inertia"], 0.40, 0.95)
        firm.price_aggressiveness = mutate(firm.price_aggressiveness, traits["price_aggressiveness"], 0.50, 1.70)
        firm.cash_conservatism = mutate(firm.cash_conservatism, traits["cash_conservatism"], 0.50, 1.70)
        firm.market_share_ambition = mutate(firm.market_share_ambition, traits["market_share_ambition"], 0.50, 1.70)
        firm.forecast_caution = mutate(firm.forecast_caution, traits["forecast_caution"], 0.60, 1.85)
        firm.demand_elasticity = clamp(
            0.80 * firm.demand_elasticity + 0.20 * self._initial_demand_elasticity(spec.key),
            0.45,
            2.75,
        )
        firm.wage_offer = max(
            self._sector_wage_floor(spec.key),
            firm.wage_offer * self.rng.uniform(0.94, 1.06),
        )
        mutated_price = firm.price * self.rng.uniform(0.92, 1.08)
        firm.price = clamp(
            mutated_price,
            spec.base_price * 0.35,
            min(spec.base_price * self.config.price_ceiling_multiplier, firm.price * self._firm_max_price_hike_ratio(firm)),
        )
        firm.sales_history = [
            max(0.0, sale * self.rng.uniform(0.85, 1.15))
            for sale in self._firm_memory_slice(firm)
        ] or [max(1.0, firm.last_sales)]
        firm.expected_sales_history = [
            max(0.0, sale * self.rng.uniform(0.85, 1.15))
            for sale in (firm.expected_sales_history or [max(1.0, firm.last_expected_sales)])
        ]
        firm.production_history = [
            max(0.0, output * self.rng.uniform(0.85, 1.15))
            for output in (firm.production_history or [max(0.0, firm.last_production)])
        ]
        firm.last_expected_sales = self._smoothed_expected_sales_reference(firm)
        firm.loss_streak = 0

    def _adapt_losing_firm(self, firm: Firm) -> None:
        spec = SECTOR_BY_KEY[firm.sector]
        peer = self._best_peer_firm(firm.sector, exclude_firm_id=firm.id)
        if peer is not None and self.rng.random() < 0.5:
            self._copy_peer_behavior(firm, peer, spec)
        else:
            self._mutate_firm_behavior(firm, spec)

    def _random_firm_behavior_traits(self, spec) -> dict[str, float]:
        return {
            "markup_tolerance": clamp(self.rng.uniform(0.75, 1.30), 0.50, 1.60),
            "volume_preference": clamp(self.rng.uniform(0.70, 1.35), 0.50, 1.70),
            "inventory_aversion": clamp(self.rng.uniform(0.70, 1.35), 0.50, 1.70),
            "employment_inertia": clamp(self.rng.uniform(0.55, 0.90), 0.40, 0.95),
            "price_aggressiveness": clamp(self.rng.uniform(0.70, 1.40), 0.50, 1.70),
            "cash_conservatism": clamp(self.rng.uniform(0.70, 1.40), 0.50, 1.70),
            "market_share_ambition": clamp(self.rng.uniform(0.70, 1.40), 0.50, 1.70),
            "expansion_credit_appetite": clamp(self.rng.uniform(0.70, 1.45), 0.55, 1.75),
            "stability_sensitivity": clamp(self.rng.uniform(0.70, 1.40), 0.55, 1.75),
            "investment_animal_spirits": clamp(self.rng.uniform(0.70, 1.40), 0.55, 1.75),
            "forecast_caution": clamp(self.rng.uniform(0.75, 1.50), 0.60, 1.85),
        }

    def _price_search_candidates(self, firm: Firm, spec, variable_unit_cost: float, target_price: float) -> list[float]:
        inventory_ratio = firm.inventory / max(1.0, firm.target_inventory)
        cash_cover = firm.cash / max(1.0, firm.last_wage_bill + firm.fixed_overhead + firm.capital * self.config.depreciation_rate)
        rejection_pressure = self._firm_rejection_signal(firm) * self._firm_price_hike_sensitivity(firm)
        affordability_pressure = self._essential_affordability_pressure() if spec.key in ESSENTIAL_SECTOR_KEYS else 0.0
        learning_maturity = self._firm_learning_maturity(firm)
        movement_scale = 0.35 + 0.65 * learning_maturity
        penetration_allowed = (
            spec.key in ESSENTIAL_SECTOR_KEYS
            and inventory_ratio > 1.15
            and cash_cover > 1.05
        )
        if (
            spec.key in ESSENTIAL_SECTOR_KEYS
            and not penetration_allowed
            and affordability_pressure > 0.35
            and inventory_ratio > 1.02
            and cash_cover > 0.95
        ):
            penetration_allowed = True

        if penetration_allowed:
            multipliers = (0.68, 0.76, 0.84, 0.92, 1.00, 1.06, 1.12, 1.20, 1.30)
        elif inventory_ratio > 1.10:
            multipliers = (0.80, 0.88, 0.94, 1.00, 1.06, 1.12, 1.20, 1.30, 1.42)
        else:
            multipliers = (0.84, 0.90, 0.96, 1.00, 1.04, 1.10, 1.18, 1.28, 1.40)

        personal_floor_multiplier = clamp(
            self.config.price_floor_multiplier
            - 0.10 * (firm.price_aggressiveness - 1.0)
            + 0.08 * (firm.cash_conservatism - 1.0),
            0.45,
            0.95,
        )
        if spec.key in ESSENTIAL_SECTOR_KEYS:
            personal_floor_multiplier = clamp(
                personal_floor_multiplier - 0.18 * affordability_pressure,
                0.28,
                0.95,
            )
        floor_price = max(
            variable_unit_cost * (1.01 if penetration_allowed else 1.03),
            spec.base_price * (0.35 if penetration_allowed else personal_floor_multiplier),
        )
        ceiling_price = spec.base_price * self.config.price_ceiling_multiplier
        search_floor_price = floor_price if spec.key in ESSENTIAL_SECTOR_KEYS else min(floor_price, firm.price)
        upward_ceiling = min(ceiling_price, firm.price * self._firm_max_price_hike_ratio(firm))

        adjusted_multipliers: list[float] = []
        for multiplier in multipliers:
            if multiplier > 1.0:
                upward_scale = clamp(1.0 - 0.70 * rejection_pressure, 0.12, 1.0)
                adjusted = 1.0 + (multiplier - 1.0) * upward_scale
            elif multiplier < 1.0:
                downward_scale = clamp(1.0 + 0.55 * rejection_pressure, 1.0, 1.85)
                adjusted = 1.0 - (1.0 - multiplier) * downward_scale
            else:
                adjusted = multiplier
            adjusted_multipliers.append(1.0 + (adjusted - 1.0) * movement_scale)

        candidates = set()
        for multiplier in adjusted_multipliers:
            candidate_price = firm.price * multiplier
            if candidate_price > firm.price:
                candidate_price = min(candidate_price, upward_ceiling)
            candidates.add(clamp(candidate_price, search_floor_price, ceiling_price))
        if rejection_pressure > 0.20:
            candidates.add(clamp(firm.price * (1.0 - 0.12 * rejection_pressure), search_floor_price, ceiling_price))
            candidates.add(clamp(firm.price * (1.0 - 0.22 * rejection_pressure), search_floor_price, ceiling_price))
        if rejection_pressure > 0.45:
            near_cost_price = max(variable_unit_cost * 1.02, firm.last_unit_cost * 1.01)
            candidates.add(clamp(min(near_cost_price, upward_ceiling), search_floor_price, ceiling_price))
        if target_price > firm.price:
            target_price = min(target_price, upward_ceiling)
        candidates.add(clamp(target_price, search_floor_price, ceiling_price))
        return sorted(candidates)

    def _inventory_clearance_discount(self, firm: Firm) -> float:
        target_inventory = max(1.0, firm.target_inventory)
        if firm.inventory <= target_inventory:
            return 0.0

        excess_ratio = (firm.inventory - target_inventory) / target_inventory
        discount = clamp(excess_ratio * (0.85 + 0.40 * firm.inventory_aversion), 0.0, 0.55)
        if firm.sector in ESSENTIAL_SECTOR_KEYS:
            discount = clamp(discount * 1.15, 0.0, 0.45)
        return discount

    def _firm_price_hike_sensitivity(self, firm: Firm) -> float:
        return clamp(
            1.0
            + 0.26 * (firm.inventory_aversion - 1.0)
            + 0.22 * (firm.volume_preference - 1.0)
            + 0.18 * (firm.market_share_ambition - 1.0)
            + 0.12 * (firm.cash_conservatism - 1.0)
            - 0.28 * (firm.price_aggressiveness - 1.0)
            - 0.18 * (firm.markup_tolerance - 1.0),
            0.55,
            1.75,
        )

    def _firm_forecast_uncertainty(self, firm: Firm) -> float:
        return clamp(
            0.55 * firm.forecast_error_belief
            + 0.25 * self._firm_rejection_signal(firm)
            + 0.20 * self._firm_market_fragility_signal(firm),
            0.03,
            1.25,
        )

    def _firm_recent_sales_momentum(self, firm: Firm) -> float:
        recent_window, prior_window = self._firm_recent_prior_sales_windows(firm)
        if len(recent_window) < 3 or not prior_window:
            return 0.0
        prior_avg = sum(prior_window) / len(prior_window)
        recent_avg = sum(recent_window) / len(recent_window)
        return clamp((recent_avg - prior_avg) / max(1.0, prior_avg), -0.5, 0.5)

    def _firm_max_price_hike_ratio(self, firm: Firm) -> float:
        learning_maturity = self._firm_learning_maturity(firm)
        uncertainty = self._firm_forecast_uncertainty(firm)
        caution = clamp(firm.forecast_caution, 0.60, 1.85)
        smoothed_sales = self._smoothed_sales_reference(firm)
        smoothed_expected = self._smoothed_expected_sales_reference(firm)
        smoothed_production = self._smoothed_production_reference(firm)
        demand_realization = smoothed_sales / max(1.0, smoothed_expected)
        sell_through = smoothed_sales / max(1.0, smoothed_production)
        inventory_ratio = firm.inventory / max(1.0, firm.target_inventory)
        sales_momentum = self._firm_recent_sales_momentum(firm)
        readiness = clamp(
            0.40 * clamp((demand_realization - 0.96) / 0.16, 0.0, 1.0)
            + 0.30 * clamp((sell_through - 0.88) / 0.12, 0.0, 1.0)
            + 0.20 * clamp((1.05 - inventory_ratio) / 0.35, 0.0, 1.0)
            + 0.10 * clamp(sales_momentum / 0.10, 0.0, 1.0),
            0.0,
            1.0,
        )
        max_hike = clamp(
            0.01
            + 0.02 * max(0.0, firm.price_aggressiveness - 1.0)
            + 0.05 * readiness
            - 0.03 * max(0.0, caution - 1.0)
            - 0.05 * uncertainty
            - 0.02 * firm.market_fragility_belief,
            0.0,
            0.09,
        )
        max_hike *= 0.30 + 0.70 * learning_maturity
        if firm.sector in ESSENTIAL_SECTOR_KEYS:
            max_hike = min(max_hike, 0.06)
        return 1.0 + max_hike

    def _startup_essential_candidate_prices(
        self,
        firm: Firm,
        variable_unit_cost: float,
    ) -> list[float]:
        # During startup, essential firms should remain cautious, but they still
        # need a viable path out of loss-making underpricing.
        cost_floor = max(
            0.1,
            variable_unit_cost * 1.01,
            firm.last_unit_cost * 1.01,
        )
        base_candidate = max(firm.price, cost_floor)
        cautious_hike = max(
            base_candidate,
            firm.price * self._firm_max_price_hike_ratio(firm),
        )
        return sorted({base_candidate, cautious_hike})

    def _firm_rejection_signal(self, firm: Firm) -> float:
        expected_sales = self._smoothed_expected_sales_reference(firm)
        smoothed_sales = self._smoothed_sales_reference(firm)
        smoothed_production = self._smoothed_production_reference(firm)
        expectation_shortfall = clamp(
            (expected_sales - smoothed_sales) / expected_sales,
            0.0,
            1.5,
        )
        inventory_ratio = firm.inventory / max(1.0, firm.target_inventory)
        inventory_pressure = clamp((inventory_ratio - 1.0) / 1.5, 0.0, 1.0)

        recent_sales_drop = 0.0
        recent_window, prior_window = self._firm_recent_prior_sales_windows(firm)
        if len(recent_window) >= 3 and prior_window:
            prior_avg = sum(prior_window) / len(prior_window)
            recent_avg = sum(recent_window) / len(recent_window)
            recent_sales_drop = clamp(
                (prior_avg - recent_avg) / max(1.0, prior_avg),
                0.0,
                1.5,
            )

        unsold_pressure = 0.0
        if smoothed_production > 0.0:
            unsold_pressure = clamp(
                (smoothed_production - smoothed_sales) / max(1.0, smoothed_production),
                0.0,
                1.0,
            )

        return clamp(
            0.40 * expectation_shortfall
            + 0.25 * inventory_pressure
            + 0.20 * recent_sales_drop
            + 0.15 * unsold_pressure,
            0.0,
            1.5,
        )

    def _sector_public_fragility_signal(self, sector_key: str) -> float:
        sales_units = self._recent_sector_signal_average(
            self._sector_sales_history,
            sector_key,
            fallback=getattr(self, "_last_sector_sales_units", {}).get(sector_key, 0.0),
        )
        revealed_unmet_units = self._recent_sector_signal_average(
            self._sector_revealed_unmet_history,
            sector_key,
            fallback=getattr(self, "_last_sector_revealed_unmet_units", {}).get(sector_key, 0.0),
        )
        sales_history = self._sector_sales_history.get(sector_key, [])
        prior_sales_units = 0.0
        if len(sales_history) >= 2:
            midpoint = max(1, len(sales_history) // 2)
            prior_sales_units = sum(sales_history[:-midpoint]) / max(1, len(sales_history[:-midpoint]))
        elif sales_history:
            prior_sales_units = sales_history[0]

        walkaway_rate = 0.0
        if sales_units > 0.0 or revealed_unmet_units > 0.0:
            walkaway_rate = clamp(
                revealed_unmet_units / max(1.0, sales_units + revealed_unmet_units),
                0.0,
                1.5,
            )

        volume_drop = 0.0
        if prior_sales_units > 0.0:
            volume_drop = clamp(
                (prior_sales_units - sales_units) / prior_sales_units,
                0.0,
                1.5,
            )

        return clamp(0.60 * walkaway_rate + 0.40 * volume_drop, 0.0, 1.5)

    def _economy_public_fragility_signal(self) -> float:
        snapshots = self._recent_snapshots()
        if not snapshots:
            return 0.0

        last_snapshot = snapshots[-1]
        unemployment_stress = clamp(
            (last_snapshot.unemployment_rate - self.config.target_unemployment) / 0.35,
            0.0,
            1.5,
        )
        average_births = sum(snapshot.births for snapshot in snapshots) / len(snapshots)
        average_deaths = sum(snapshot.deaths for snapshot in snapshots) / len(snapshots)
        average_population = sum(snapshot.population for snapshot in snapshots) / len(snapshots)
        demographic_loss = max(0.0, average_deaths - average_births) / max(1, average_population)
        demographic_stress = clamp(
            demographic_loss * self.config.periods_per_year * 4.0,
            0.0,
            1.5,
        )

        population_drop = 0.0
        if len(snapshots) >= 2:
            prior_snapshot = snapshots[0]
            if prior_snapshot.population > 0:
                population_drop = clamp(
                    (prior_snapshot.population - last_snapshot.population) / prior_snapshot.population * 10.0,
                    0.0,
                    1.5,
                )

        return clamp(
            0.50 * unemployment_stress + 0.30 * population_drop + 0.20 * demographic_stress,
            0.0,
            1.5,
        )

    def _social_survival_fragility_signal(self, sector_key: str) -> float:
        snapshots = self._recent_snapshots()
        if not snapshots:
            return 0.0

        last_snapshot = snapshots[-1]
        essential_shortfall = clamp(
            1.0 - sum(snapshot.essential_fulfillment_rate for snapshot in snapshots) / len(snapshots),
            0.0,
            1.5,
        )
        family_stress = clamp(
            sum(snapshot.families_income_below_basket_share for snapshot in snapshots) / len(snapshots),
            0.0,
            1.5,
        )
        average_deaths = sum(snapshot.deaths for snapshot in snapshots) / len(snapshots)
        average_population = sum(snapshot.population for snapshot in snapshots) / len(snapshots)
        mortality_rate = average_deaths / max(1, average_population)
        mortality_stress = clamp(
            mortality_rate * self.config.periods_per_year / 0.15,
            0.0,
            1.5,
        )
        population_drop = 0.0
        if len(snapshots) >= 2:
            prior_snapshot = snapshots[0]
            if prior_snapshot.population > 0:
                population_drop = clamp(
                    (prior_snapshot.population - last_snapshot.population) / prior_snapshot.population * 10.0,
                    0.0,
                    1.5,
                )

        baseline_signal = clamp(
            0.35 * essential_shortfall
            + 0.30 * family_stress
            + 0.20 * mortality_stress
            + 0.15 * population_drop,
            0.0,
            1.5,
        )
        if sector_key in ESSENTIAL_SECTOR_KEYS:
            return baseline_signal
        return clamp(0.45 * baseline_signal, 0.0, 1.2)

    def _firm_market_fragility_signal(self, firm: Firm) -> float:
        social_fragility = self._social_survival_fragility_signal(firm.sector)
        social_weight = 0.30 if firm.sector in ESSENTIAL_SECTOR_KEYS else 0.10
        return clamp(
            (0.50 - social_weight * 0.20) * self._firm_rejection_signal(firm)
            + (0.30 - social_weight * 0.10) * self._sector_public_fragility_signal(firm.sector)
            + (0.20 - social_weight * 0.10) * self._economy_public_fragility_signal()
            + social_weight * social_fragility,
            0.0,
            1.5,
        )

    def _firm_future_market_weight(self, firm: Firm) -> float:
        future_weight = clamp(
            0.20
            + 0.18 * (firm.volume_preference - 1.0)
            + 0.16 * (firm.market_share_ambition - 1.0)
            + 0.12 * (firm.inventory_aversion - 1.0)
            + 0.08 * (firm.cash_conservatism - 1.0)
            - 0.10 * (firm.markup_tolerance - 1.0),
            0.05,
            0.55,
        )
        if firm.sector in ESSENTIAL_SECTOR_KEYS:
            future_weight = clamp(future_weight + 0.06, 0.05, 0.65)
        return future_weight

    def _candidate_market_retention(
        self,
        firm: Firm,
        candidate_price: float,
        reference_price: float,
    ) -> tuple[float, float]:
        hike_sensitivity = self._firm_price_hike_sensitivity(firm)
        social_pressure = self._social_survival_fragility_signal(firm.sector)
        market_pressure = clamp(
            0.45 * firm.market_fragility_belief
            + 0.35 * self._firm_market_fragility_signal(firm)
            + 0.20 * social_pressure,
            0.0,
            1.75,
        )
        price_ratio = max(0.25, candidate_price / max(0.1, reference_price))
        markup_stretch = 0.0
        if firm.last_unit_cost > 0.0:
            markup_stretch = max(0.0, reference_price / max(0.1, firm.last_unit_cost) - 1.0)

        if candidate_price >= reference_price:
            penalty_power = clamp(
                1.15
                + 0.20 * (firm.market_share_ambition - 1.0)
                + 0.12 * (firm.inventory_aversion - 1.0),
                0.85,
                1.60,
            )
            hazard = market_pressure * hike_sensitivity * ((price_ratio - 1.0) ** penalty_power)
            hazard *= 1.0 + 0.18 * clamp(markup_stretch, 0.0, 2.5)
            if firm.sector in ESSENTIAL_SECTOR_KEYS:
                hazard *= 1.0 + 0.35 * social_pressure
            retention = clamp(1.0 - 0.85 * hazard, 0.20, 1.0)
            return retention, clamp(hazard, 0.0, 1.5)

        recovery = market_pressure * clamp(1.0 - price_ratio, 0.0, 0.80)
        recovery *= 0.65 + 0.25 * firm.volume_preference + 0.15 * firm.market_share_ambition
        if firm.sector in ESSENTIAL_SECTOR_KEYS:
            recovery *= 1.0 + 0.20 * social_pressure
        retention = clamp(1.0 + 0.35 * recovery, 0.85, 1.25)
        return retention, 0.0

    def _firm_demand_share_anchor(self, firm: Firm, market_share: float) -> float:
        learned_share = max(0.0, self._smoothed_sales_reference(firm)) / max(
            1.0,
            self._baseline_demand(firm.sector, use_current_period=True),
        )
        if learned_share <= 0.0:
            learned_share = market_share
        learning_maturity = self._firm_learning_maturity(firm)
        learned_weight = 0.30 + 0.50 * learning_maturity
        blended_share = learned_weight * learned_share + (1.0 - learned_weight) * market_share
        return clamp(blended_share, max(0.01, 0.15 * market_share), 0.95)

    def _firm_recent_sell_through(self, firm: Firm) -> float:
        smoothed_sales = self._smoothed_sales_reference(firm)
        expected_reference = max(
            1.0,
            self._smoothed_expected_sales_reference(firm),
            self._smoothed_production_reference(firm),
            smoothed_sales,
        )
        return clamp(smoothed_sales / expected_reference, 0.0, 1.25)

    def _firm_cash_cover_ratio(self, firm: Firm) -> float:
        operating_burn = (
            firm.last_wage_bill
            + firm.last_input_cost
            + firm.last_transport_cost
            + firm.last_fixed_overhead
            + firm.last_interest_cost
        )
        if operating_burn <= 0.0:
            operating_burn = (
                max(1, firm.last_worker_count) * max(0.1, firm.wage_offer)
                + max(0.0, firm.fixed_overhead)
                + max(0.0, firm.capital) * self.config.depreciation_rate
            )
        return firm.cash / max(1.0, operating_burn)

    def _firm_capturable_sales_cap(self, firm: Firm, sector_total_demand: float, market_share: float) -> float:
        sales_memory = max(1.0, self._smoothed_sales_reference(firm))
        if sector_total_demand <= 0.0:
            return sales_memory

        learning_maturity = self._firm_learning_maturity(firm)
        sell_through = self._firm_recent_sell_through(firm)
        cash_cover = clamp(self._firm_cash_cover_ratio(firm), 0.0, 3.0)
        ambition_bonus = clamp(firm.market_share_ambition - 1.0, 0.0, 0.70)
        forecast_penalty = clamp(firm.forecast_error_belief, 0.0, 1.25)
        revealed_shortage = self._sector_revealed_shortage_signal(firm.sector)
        realized_share = sales_memory / max(1.0, sector_total_demand)
        share_gap = max(0.0, market_share - realized_share)
        expansion_allowance = (
            0.02
            + 0.06 * learning_maturity
            + 0.06 * max(0.0, sell_through - 0.70)
            + 0.03 * cash_cover
            + 0.04 * ambition_bonus
            + 0.15 * min(share_gap, 0.20)
            + 0.10 * revealed_shortage * max(0.0, sell_through - 0.55)
            - 0.05 * forecast_penalty
            - 0.08 * max(0.0, 0.85 - sell_through)
        )
        expansion_allowance = clamp(expansion_allowance, 0.01, 0.18)
        capturable_share = clamp(
            realized_share + min(share_gap, expansion_allowance),
            max(0.0025, realized_share),
            min(0.75, max(realized_share + 0.01, market_share)),
        )
        capturable_market_sales = max(1.0, sector_total_demand * capturable_share)

        effective_productivity = max(1.0, self._firm_effective_productivity(firm))
        staffed_capacity = effective_productivity * max(1, firm.last_worker_count)
        proven_scale = max(
            sales_memory,
            min(max(1.0, self._smoothed_production_reference(firm)), max(1.0, staffed_capacity)),
        )
        max_growth_rate = clamp(
            0.18
            + 0.32 * max(0.0, sell_through - 0.65)
            + 0.08 * learning_maturity
            + 0.06 * ambition_bonus
            + 0.05 * cash_cover
            + 0.18 * revealed_shortage * max(0.0, sell_through - 0.55)
            - 0.10 * forecast_penalty,
            0.15,
            0.90,
        )
        growth_limited_sales = max(1.0, proven_scale * (1.0 + max_growth_rate))
        return max(1.0, min(capturable_market_sales, growth_limited_sales))

    def _firm_effective_supply_signal(self, firm: Firm, sector_total_demand: float | None = None) -> float:
        sales_memory = max(0.0, self._smoothed_sales_reference(firm))
        sell_through = self._firm_recent_sell_through(firm)
        proven_sales = max(
            self._smoothed_sales_reference(firm),
            sales_memory,
            self._smoothed_expected_sales_reference(firm) * sell_through,
        )
        if sector_total_demand is None:
            sector_total_demand = max(
                1.0,
                self._baseline_demand(firm.sector, use_current_period=True),
            )
        supply_cap = self._firm_capturable_sales_cap(
            firm,
            sector_total_demand,
            max(0.05, firm.last_market_share),
        )
        return max(0.0, min(proven_sales, supply_cap))

    def _sales_anchor(self, firm: Firm, sector_total_demand: float, market_share: float) -> float:
        sales_memory = self._smoothed_sales_reference(firm)
        capturable_sales_cap = self._firm_capturable_sales_cap(
            firm,
            sector_total_demand,
            self._firm_demand_share_anchor(firm, market_share),
        )
        learning_maturity = self._firm_learning_maturity(firm)
        market_weight = 0.52 - 0.10 * learning_maturity + 0.06 * max(0.0, firm.market_share_ambition - 1.0)
        market_weight = clamp(market_weight, 0.35, 0.60)
        return max(1.0, sales_memory + market_weight * (capturable_sales_cap - sales_memory))

    def _expected_demand_for_price(
        self,
        firm: Firm,
        sector_total_demand: float,
        market_share: float,
        candidate_price: float,
        reference_price: float,
    ) -> float:
        if sector_total_demand <= 0.0:
            return 0.0

        sales_anchor = self._sales_anchor(firm, sector_total_demand, market_share)
        baseline_demand = max(1.0, sales_anchor)
        learning_maturity = self._firm_learning_maturity(firm)
        prior_elasticity = self._sector_demand_elasticity_prior(firm.sector)
        elasticity_floor = prior_elasticity * (0.90 + 0.10 * learning_maturity)
        elasticity = clamp(max(firm.demand_elasticity, elasticity_floor), 0.45, 2.75)
        price_ratio = max(0.25, candidate_price / max(0.1, reference_price))
        rejection_signal = self._firm_market_fragility_signal(firm)
        hike_sensitivity = self._firm_price_hike_sensitivity(firm)
        rejection_pressure = clamp(
            (0.60 * firm.market_fragility_belief + 0.40 * rejection_signal) * hike_sensitivity,
            0.0,
            2.25,
        )
        markup_stretch = 0.0
        if firm.last_unit_cost > 0.0:
            markup_stretch = max(0.0, reference_price / max(0.1, firm.last_unit_cost) - 1.0)

        effective_elasticity = elasticity
        warmup_strength = 1.0 - learning_maturity
        if candidate_price > reference_price:
            effective_elasticity *= 1.0 + 0.55 * rejection_pressure + 0.15 * clamp(markup_stretch, 0.0, 2.5)
            effective_elasticity *= 1.0 + warmup_strength * (0.30 + 0.35 * hike_sensitivity)
        elif candidate_price < reference_price and rejection_pressure > 0.0:
            effective_elasticity *= clamp(1.0 - 0.10 * rejection_pressure, 0.70, 1.0)

        demand = baseline_demand * (price_ratio ** (-effective_elasticity))
        if candidate_price > reference_price and rejection_pressure > 0.0:
            demand *= math.exp(-1.35 * (price_ratio - 1.0) * rejection_pressure)
        elif candidate_price > reference_price and warmup_strength > 0.0:
            demand *= math.exp(-0.95 * (price_ratio - 1.0) * warmup_strength)
        elif candidate_price < reference_price and rejection_pressure > 0.0:
            demand *= 1.0 + 0.18 * (1.0 - price_ratio) * rejection_pressure * (0.55 + 0.45 * learning_maturity)

        if firm.sector in ESSENTIAL_SECTOR_KEYS:
            inventory_ratio = firm.inventory / max(1.0, firm.target_inventory)
            if inventory_ratio > 1.05 and candidate_price < reference_price:
                demand *= 1.0 + 0.12 * clamp(inventory_ratio - 1.0, 0.0, 2.5)

        return clamp(demand, 0.0, sector_total_demand)

    def _conservative_expected_sales(
        self,
        firm: Firm,
        expected_sales: float,
        candidate_price: float,
        reference_price: float,
        market_hazard: float,
    ) -> float:
        learning_maturity = self._firm_learning_maturity(firm)
        warmup_strength = 1.0 - learning_maturity
        uncertainty = self._firm_forecast_uncertainty(firm)
        caution = clamp(firm.forecast_caution, 0.60, 1.85)
        price_change = candidate_price / max(0.1, reference_price) - 1.0
        base_haircut = uncertainty * (0.16 + 0.16 * caution) + 0.05 * warmup_strength
        upward_haircut = max(0.0, price_change) * (0.35 + 0.30 * caution + 0.30 * warmup_strength)
        hazard_haircut = market_hazard * (0.12 + 0.10 * caution)
        downside_relief = max(0.0, -price_change) * (0.06 + 0.04 * firm.volume_preference)
        prudent_share = clamp(
            1.0 - base_haircut - upward_haircut - hazard_haircut + downside_relief,
            0.18,
            1.02,
        )
        return max(0.0, expected_sales * prudent_share)

    def _update_firm_demand_learning(self, firm: Firm) -> None:
        expected_sales = max(0.0, self._smoothed_expected_sales_reference(firm))
        if expected_sales <= 0.0:
            return

        learning_maturity = self._firm_learning_maturity(firm)
        learning_scale = 0.30 + 0.70 * learning_maturity
        prior_elasticity = self._sector_demand_elasticity_prior(firm.sector)
        realization = self._smoothed_sales_reference(firm) / max(1.0, expected_sales)
        inventory_ratio = firm.inventory / max(1.0, firm.target_inventory)
        target_elasticity = firm.demand_elasticity
        hike_sensitivity = self._firm_price_hike_sensitivity(firm)
        fragility_signal = self._firm_market_fragility_signal(firm)
        belief_learning_rate = clamp(
            0.06
            + 0.06 * hike_sensitivity
            + 0.04 * max(0.0, firm.market_share_ambition - 1.0),
            0.06,
            0.18,
        )
        belief_learning_rate *= learning_scale
        firm.market_fragility_belief = clamp(
            (1.0 - belief_learning_rate) * firm.market_fragility_belief
            + belief_learning_rate * fragility_signal,
            0.0,
            1.5,
        )
        rejection_signal = clamp(
            0.55 * self._firm_rejection_signal(firm) + 0.45 * firm.market_fragility_belief,
            0.0,
            1.5,
        )
        markup_stretch = 0.0
        if firm.last_unit_cost > 0.0:
            markup_stretch = max(0.0, firm.price / max(0.1, firm.last_unit_cost) - 1.0)

        if realization < 0.90:
            target_elasticity += 0.05 * (0.90 - realization) * (1.0 + 0.80 * hike_sensitivity)
        elif realization > 1.10:
            target_elasticity -= 0.03 * (realization - 1.10)

        if inventory_ratio > 1.10:
            target_elasticity += 0.03 * (inventory_ratio - 1.0) * (0.85 + 0.55 * hike_sensitivity)
        elif inventory_ratio < 0.85 and realization > 1.0:
            target_elasticity -= 0.02 * (1.0 - inventory_ratio)

        if rejection_signal > 0.25 and markup_stretch > 0.10:
            target_elasticity += 0.04 * rejection_signal * hike_sensitivity + 0.02 * clamp(markup_stretch, 0.0, 2.5)

        learning_rate = clamp(
            0.05
            + 0.08 * rejection_signal * hike_sensitivity
            + 0.03 * max(0.0, 0.90 - realization),
            0.05,
            0.20,
        )
        learning_rate *= learning_scale
        prior_pull = 0.30 * (1.0 - learning_maturity)
        target_elasticity = (1.0 - prior_pull) * target_elasticity + prior_pull * max(target_elasticity, prior_elasticity)
        elasticity_floor = prior_elasticity * (0.85 + 0.15 * learning_maturity)

        firm.demand_elasticity = clamp(
            (1.0 - learning_rate) * firm.demand_elasticity + learning_rate * target_elasticity,
            max(0.45, elasticity_floor),
            2.75,
        )
        forecast_error_target = clamp(
            0.55 * abs(1.0 - realization)
            + 0.20 * max(0.0, inventory_ratio - 1.0)
            + 0.15 * rejection_signal
            + 0.10 * fragility_signal,
            0.03,
            1.25,
        )
        forecast_learning_rate = clamp(
            0.08 + 0.08 * firm.forecast_caution + 0.05 * rejection_signal,
            0.08,
            0.28,
        )
        forecast_learning_rate *= learning_scale
        firm.forecast_error_belief = clamp(
            (1.0 - forecast_learning_rate) * firm.forecast_error_belief
            + forecast_learning_rate * forecast_error_target,
            0.03,
            1.25,
        )

    def _build_households(self) -> list[Household]:
        households: list[Household] = []
        for household_id in range(self.config.households):
            periods_per_year = max(1, self.config.periods_per_year)
            age_draw = self.rng.random()
            if age_draw < 0.25:
                age_years = self.rng.uniform(0.0, self.config.entry_age_years - 0.1)
            elif age_draw < 0.90:
                age_years = self.rng.uniform(self.config.entry_age_years, 45.0)
            else:
                age_years = self.rng.uniform(60.0, self.config.initial_household_age_max_years)
            age_periods = int(round(age_years * periods_per_year))
            households.append(
                Household(
                    id=household_id,
                    sex=self.rng.choice(("F", "M")),
                    savings=self.rng.uniform(
                        self.config.initial_household_savings_min,
                        self.config.initial_household_savings_max,
                    ),
                    reservation_wage=self.rng.uniform(5.7, 7.2),
                    saving_propensity=self.rng.uniform(0.03, 0.16),
                    higher_education_affinity=self.rng.random(),
                    money_trust=self.rng.uniform(0.45, 0.85),
                    consumption_impatience=self.rng.uniform(0.20, 0.80),
                    price_sensitivity=self.rng.uniform(0.6, 1.4),
                    need_scale=self.rng.uniform(0.9, 1.1),
                    sector_preference_weights=self._draw_household_sector_preference_weights(),
                    age_periods=age_periods,
                    partnership_affinity_code=self._draw_partnership_affinity_code(),
                    next_partnership_attempt_period=self._draw_initial_partnership_attempt_period(),
                    fertility_multiplier=self._draw_fertility_multiplier(),
                    desired_children=self._draw_desired_children(),
                )
            )
        return households

    def _build_entrepreneurs(self) -> list[Entrepreneur]:
        entrepreneurs: list[Entrepreneur] = []
        total_firms = sum(self._initial_sector_firm_count(spec.key) for spec in SECTOR_SPECS)
        baseline_firms = max(1, 5 * max(1, self.config.firms_per_sector))
        worker_wealth_pool = sum(household.savings for household in self.households)
        capitalist_share = clamp(self.config.initial_capitalist_wealth_share, 0.05, 0.95)
        capitalist_wealth_pool = worker_wealth_pool * capitalist_share / max(1e-9, 1.0 - capitalist_share)
        capitalist_wealth_pool *= total_firms / baseline_firms
        # Concentrate liquid wealth across owners before firms draw their startup funding.
        wealth_weights = [
            0.35 + self.rng.lognormvariate(0.0, self.config.initial_capitalist_wealth_sigma)
            for _ in range(total_firms)
        ]
        total_weight = sum(wealth_weights)
        if total_weight <= 0.0:
            wealth_weights = [1.0 for _ in range(total_firms)]
            total_weight = float(total_firms)
        for owner_id in range(total_firms):
            wealth_share = wealth_weights[owner_id] / total_weight
            entrepreneurs.append(
                Entrepreneur(
                    id=owner_id,
                    wealth=capitalist_wealth_pool * wealth_share,
                    entry_appetite=self.rng.uniform(0.70, 1.45),
                    market_research_skill=self.rng.uniform(0.60, 1.45),
                    entry_optimism=self.rng.uniform(-0.18, 0.24),
                )
            )
        return entrepreneurs

    def _initial_sector_firm_count(self, sector_key: str) -> int:
        if sector_key == "public_administration":
            return 0
        if sector_key == "school":
            return max(1, self.config.initial_private_school_firms)
        if sector_key == "university":
            return max(1, self.config.initial_private_university_firms)
        return max(1, self.config.firms_per_sector)

    def _build_firms(self) -> list[Firm]:
        firms: list[Firm] = []
        firm_id = 0
        sector_counts = {spec.key: self._initial_sector_firm_count(spec.key) for spec in SECTOR_SPECS}
        startup_demand_by_sector = {
            spec.key: self._baseline_demand(spec.key) * self._startup_slot_share(spec.key)
            for spec in SECTOR_SPECS
        }
        food_input_exempt_slots = {
            index for index in range(sector_counts.get("food", 1)) if self.rng.random() < 0.10
        }
        if not food_input_exempt_slots:
            food_input_exempt_slots.add(self.rng.randrange(sector_counts.get("food", 1)))
        for spec in SECTOR_SPECS:
            sector_count = sector_counts[spec.key]
            startup_slot_share = self._startup_slot_share(spec.key)
            for sector_index in range(sector_count):
                firms.append(
                    self._create_startup_firm(
                        firm_id,
                        spec,
                        input_cost_exempt=spec.key == "food" and sector_index in food_input_exempt_slots,
                        startup_demand=startup_demand_by_sector[spec.key],
                        startup_slot_share=startup_slot_share,
                    )
                )
                firm_id += 1
        self._calibrate_startup_essential_capacity(firms)
        return firms

    def _refresh_firm_startup_state(self, firm: Firm, spec, expected_sales: float | None = None) -> None:
        expected_sales = max(1.0, expected_sales if expected_sales is not None else firm.last_sales)
        effective_productivity = self._firm_effective_productivity(firm)
        if self._is_education_sector(spec.key):
            target_service_units = self._education_service_target_units(firm, expected_sales)
            firm.desired_workers = self._workers_needed_for_units(
                target_service_units,
                effective_productivity,
                productivity_floor=0.25,
            )
            firm.target_inventory = target_service_units
            firm.inventory = min(max(0.0, firm.inventory), target_service_units)
            firm.inventory_batches = []
        else:
            firm.target_inventory = self._firm_target_inventory_units(firm, expected_sales)
            desired_output = self._firm_desired_output_from_expected_sales(firm, expected_sales)
            firm.desired_workers = self._workers_needed_for_units(desired_output, effective_productivity)
            self._ensure_inventory_batches(firm)
        firm.last_worker_count = firm.desired_workers
        firm.last_sales = expected_sales
        firm.last_revenue = expected_sales * firm.price
        firm.last_production = expected_sales
        firm.last_wage_bill = firm.desired_workers * firm.wage_offer
        firm.last_input_cost = expected_sales * firm.input_cost_per_unit
        firm.last_transport_cost = expected_sales * firm.transport_cost_per_unit
        firm.last_fixed_overhead = firm.fixed_overhead
        firm.last_capital_charge = firm.capital * self.config.depreciation_rate
        firm.last_total_cost = (
            firm.last_wage_bill
            + firm.last_wage_bill * self._government_payroll_tax_rate()
            + firm.last_input_cost
            + firm.last_transport_cost
            + firm.last_fixed_overhead
            + firm.last_capital_charge
        )
        firm.last_unit_cost = (
            firm.last_total_cost / max(1.0, expected_sales)
        )
        firm.last_expected_sales = expected_sales
        firm.expected_sales_history = [expected_sales]
        firm.production_history = [firm.last_production]
        firm.last_profit = (
            firm.last_revenue
            - firm.last_total_cost
        )

    def _calibrate_startup_essential_capacity(self, firms: list[Firm]) -> None:
        startup_targets: dict[str, float] = {}
        sector_firms_by_key = {
            sector_key: [firm for firm in firms if firm.sector == sector_key and firm.active]
            for sector_key in ESSENTIAL_SECTOR_KEYS
        }
        for sector_key in ESSENTIAL_SECTOR_KEYS:
            spec = SECTOR_BY_KEY[sector_key]
            sector_firms = sector_firms_by_key[sector_key]
            if not sector_firms:
                continue

            startup_target_units = self._startup_essential_target_units(sector_key)
            startup_targets[sector_key] = startup_target_units
            estimated_capacity = sum(
                self._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                for firm in sector_firms
            )
            if estimated_capacity <= 0.0:
                continue

            boost = clamp(startup_target_units / estimated_capacity, 1.0, 3.0)
            for firm in sector_firms:
                capacity_share = (
                    self._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                ) / max(1e-9, estimated_capacity)
                target_sales = max(1.0, startup_target_units * capacity_share)
                startup_expected_sales = max(
                    1.0,
                    target_sales * clamp(self.config.startup_expected_sales_share, 0.35, 1.0),
                )
                capital_boost = math.sqrt(boost)
                firm.productivity = clamp(firm.productivity * capital_boost, 0.25, 25.0)
                firm.technology = clamp(
                    firm.technology * capital_boost,
                    0.75,
                    self.config.technology_cap,
                )
                firm.capital *= capital_boost
                firm.inventory = max(firm.inventory, self._firm_target_inventory_units(firm, target_sales))
                firm.cash = max(
                    firm.cash,
                    self._entry_cash_budget(
                        1.0,
                        max(1, firm.desired_workers) * firm.wage_offer,
                        target_sales * firm.input_cost_per_unit,
                        target_sales * firm.transport_cost_per_unit,
                        firm.fixed_overhead,
                    ),
                )
                self._refresh_firm_startup_state(firm, spec, target_sales)
                firm.price = self._initial_firm_price(spec, firm.last_unit_cost)
                self._refresh_firm_startup_state(firm, spec, startup_expected_sales)

        eligible_workers = len(
            [household for household in self._active_households() if self._household_labor_capacity(household) > 0.0]
        )
        allowed_essential_workers = max(1.0, eligible_workers * 0.96)
        essential_required_workers = 0.0
        for sector_key in ESSENTIAL_SECTOR_KEYS:
            sector_firms = sector_firms_by_key[sector_key]
            if not sector_firms:
                continue
            sector_target = startup_targets.get(sector_key, self._startup_essential_target_units(sector_key))
            sector_capacity = sum(
                self._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                for firm in sector_firms
            )
            if sector_capacity <= 0.0:
                continue
            for firm in sector_firms:
                firm_capacity = self._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                target_sales = sector_target * (firm_capacity / max(1e-9, sector_capacity))
                essential_required_workers += math.ceil(target_sales / max(0.1, self._firm_effective_productivity(firm)))

        if essential_required_workers > allowed_essential_workers:
            labor_relief_boost = clamp(essential_required_workers / allowed_essential_workers, 1.0, 2.5)
            for sector_key in ESSENTIAL_SECTOR_KEYS:
                spec = SECTOR_BY_KEY[sector_key]
                sector_firms = sector_firms_by_key[sector_key]
                if not sector_firms:
                    continue
                for firm in sector_firms:
                    firm.productivity = clamp(firm.productivity * labor_relief_boost, 0.25, 25.0)
                    firm.technology = clamp(
                        firm.technology * math.sqrt(labor_relief_boost),
                        0.75,
                        self.config.technology_cap,
                    )
                    firm.capital *= math.sqrt(labor_relief_boost)
                sector_target = startup_targets.get(sector_key, self._startup_essential_target_units(sector_key))
                sector_capacity = sum(
                    self._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                    for firm in sector_firms
                )
                if sector_capacity <= 0.0:
                    continue
                for firm in sector_firms:
                    firm_capacity = self._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                    target_sales = sector_target * (firm_capacity / max(1e-9, sector_capacity))
                    self._refresh_firm_startup_state(firm, spec, target_sales)

    def _seed_initial_workforce(self) -> None:
        contract_periods = max(1, self.config.employment_contract_periods)
        eligible_households = [
            household
            for household in self._active_households()
            if self._household_labor_capacity(household) > 0.0 and household.employed_by is None
        ]
        eligible_households.sort(
            key=lambda household: (
                self._household_labor_capacity(household),
                -household.reservation_wage,
                -self._household_age_years(household),
            ),
            reverse=True,
        )
        sector_targets = {
            sector_key: self._startup_essential_target_units(sector_key)
            for sector_key in ESSENTIAL_SECTOR_KEYS
        }
        sector_output = {sector_key: 0.0 for sector_key in ESSENTIAL_SECTOR_KEYS}
        sector_firms = {
            sector_key: [firm for firm in self._sector_firms(sector_key) if firm.active]
            for sector_key in ESSENTIAL_SECTOR_KEYS
        }

        for household in eligible_households:
            best_choice = None
            for sector_key in ESSENTIAL_SECTOR_KEYS:
                if sector_output[sector_key] >= sector_targets[sector_key]:
                    continue
                candidate_firms = [
                    firm
                    for firm in sector_firms[sector_key]
                    if len(firm.workers) < firm.desired_workers and firm.wage_offer >= household.reservation_wage
                ]
                if not candidate_firms:
                    continue
                sector_coverage = sector_output[sector_key] / max(1e-9, sector_targets[sector_key])
                firm = max(
                    candidate_firms,
                    key=lambda candidate: (
                        self._firm_effective_productivity(candidate),
                        candidate.wage_offer,
                        candidate.cash,
                    ),
                )
                marginal_output = self._firm_effective_productivity(firm) * self._household_labor_capacity(household)
                marginal_output *= self._household_skill_multiplier(household, sector_key)
                score = (
                    -sector_coverage,
                    marginal_output / max(1e-9, self._essential_basket_share(sector_key)),
                    firm.wage_offer,
                )
                if best_choice is None or score > best_choice[0]:
                    best_choice = (score, sector_key, firm, marginal_output)

            if best_choice is None:
                continue

            _, sector_key, firm, marginal_output = best_choice
            firm.workers.append(household.id)
            household.employed_by = firm.id
            # Stagger initial renewals so the whole labor market does not renegotiate at once.
            household.employment_tenure = self.rng.randrange(contract_periods)
            sector_output[sector_key] += marginal_output

        for sector_key in ESSENTIAL_SECTOR_KEYS:
            spec = SECTOR_BY_KEY[sector_key]
            for firm in sector_firms[sector_key]:
                startup_sales = max(1.0, firm.last_expected_sales)
                self._refresh_firm_startup_state(firm, spec, startup_sales)
                firm.price = self._initial_firm_price(spec, firm.last_unit_cost)
                self._refresh_firm_startup_state(firm, spec, startup_sales)

    def _create_startup_firm(
        self,
        firm_id: int,
        spec,
        input_cost_exempt: bool = False,
        startup_demand: float | None = None,
        startup_slot_share: float | None = None,
    ) -> Firm:
        owner = self.entrepreneurs[firm_id]
        startup_slot_share = (
            startup_slot_share if startup_slot_share is not None else self._startup_slot_share(spec.key)
        )
        startup_demand = (
            startup_demand
            if startup_demand is not None
            else self._baseline_demand(spec.key) * startup_slot_share
        )
        base_cash_budget, base_capital_budget, base_inventory_budget = self._entry_package_budgets(
            spec,
            startup_demand,
            startup_slot_share,
        )
        startup_need = base_cash_budget + base_capital_budget + base_inventory_budget
        # Fund the startup package from the owner's own wealth so capital is not created for free.
        startup_budget = min(self._owner_total_liquid(owner), startup_need)
        package_funding_ratio = clamp(startup_budget / max(1e-9, startup_need), 0.0, 1.0)
        self._withdraw_owner_liquid(owner, startup_budget)

        living_wage_anchor = self._living_wage_anchor()
        wage_offer = max(
            spec.base_wage * self.rng.uniform(0.96, 1.04),
            living_wage_anchor * self.config.reservation_wage_floor_share,
        )
        if spec.key == "leisure":
            wage_offer = max(wage_offer, living_wage_anchor * (self.config.reservation_wage_floor_share + 0.05))
        elif spec.key == "school":
            wage_offer = max(wage_offer, living_wage_anchor * (self.config.reservation_wage_floor_share + 0.08))
        elif spec.key == "university":
            wage_offer = max(wage_offer, living_wage_anchor * (self.config.reservation_wage_floor_share + 0.16))
        education_blueprint: dict[str, float] | None = None
        if self._is_education_sector(spec.key):
            education_blueprint = self._draw_education_blueprint(spec, startup_demand, startup_slot_share)
            productivity = education_blueprint["students_per_worker"]
        else:
            productivity = (
                spec.base_productivity
                * self._entry_productivity_multiplier(spec.key)
                * self.rng.uniform(0.96, 1.04)
            )
        technology = self._initial_technology(spec.key)
        if education_blueprint is not None:
            capital = education_blueprint["capital_budget"] * package_funding_ratio
            input_cost_per_unit = education_blueprint["input_cost_per_unit"]
            transport_cost_per_unit = education_blueprint["transport_cost_per_unit"]
            fixed_overhead = education_blueprint["fixed_overhead"]
        else:
            capital = base_capital_budget * package_funding_ratio
            input_cost_per_unit, transport_cost_per_unit, fixed_overhead = self._random_firm_cost_structure(spec)
            if input_cost_exempt:
                input_cost_per_unit = 0.0
        inventory_budget = base_inventory_budget * package_funding_ratio
        inventory = inventory_budget / max(0.1, spec.base_price)
        expected_sales = startup_demand * self.rng.uniform(0.9, 1.05) * max(0.10, package_funding_ratio)
        effective_productivity = productivity * technology * self._capital_efficiency(capital)
        if education_blueprint is not None:
            inventory = min(
                inventory,
                self._education_facility_capacity(
                    spec.key,
                    capital,
                    education_blueprint["level_span"],
                ),
            )
            target_service_units = clamp(
                max(1.0, expected_sales * (1.0 + 0.35 * spec.target_inventory_ratio)),
                1.0,
                max(1.0, self._education_facility_capacity(spec.key, capital, education_blueprint["level_span"])),
            )
            desired_workers = max(1, math.ceil(target_service_units / max(0.25, effective_productivity)))
        else:
            desired_output = expected_sales + inventory
            desired_workers = max(1, math.ceil(desired_output / max(0.1, effective_productivity)))
        last_wage_bill = desired_workers * wage_offer
        capital_charge = capital * self.config.depreciation_rate
        last_input_cost = expected_sales * input_cost_per_unit
        last_transport_cost = expected_sales * transport_cost_per_unit
        last_fixed_overhead = fixed_overhead
        last_total_cost = (
            last_wage_bill
            + last_wage_bill * self._government_payroll_tax_rate()
            + last_input_cost
            + last_transport_cost
            + last_fixed_overhead
            + capital_charge
        )
        unit_cost = (
            last_total_cost
            / max(1.0, expected_sales)
        )
        price = self._initial_firm_price(spec, unit_cost)
        last_revenue = expected_sales * price
        last_profit = last_revenue - last_total_cost
        firm = Firm(
            id=firm_id,
            sector=spec.key,
            owner_id=firm_id,
            cash=max(0.0, startup_budget - capital - inventory_budget),
            inventory=inventory,
            capital=capital,
            price=price,
            wage_offer=wage_offer,
            productivity=productivity,
            technology=technology,
            demand_elasticity=self._initial_demand_elasticity(spec.key),
            input_cost_per_unit=input_cost_per_unit,
            input_cost_exempt=input_cost_exempt,
            transport_cost_per_unit=transport_cost_per_unit,
            fixed_overhead=fixed_overhead,
            education_level_span=education_blueprint["level_span"] if education_blueprint is not None else 0.0,
            **self._random_firm_behavior_traits(spec),
            desired_workers=desired_workers,
            target_inventory=inventory,
            sales_this_period=0.0,
            last_worker_count=desired_workers,
            last_sales=expected_sales,
            last_revenue=last_revenue,
            last_production=expected_sales,
            last_profit=last_profit,
            last_wage_bill=last_wage_bill,
            last_input_cost=last_input_cost,
            last_transport_cost=last_transport_cost,
            last_fixed_overhead=last_fixed_overhead,
            last_capital_charge=capital_charge,
            last_total_cost=last_total_cost,
            last_unit_cost=unit_cost,
            last_market_share=0.0,
            sales_history=[expected_sales],
            expected_sales_history=[expected_sales],
            production_history=[expected_sales],
            inventory_batches=[] if self._is_education_sector(spec.key) else [max(0.0, inventory)],
            last_expected_sales=expected_sales,
            market_fragility_belief=clamp(
                0.55 * self._sector_public_fragility_signal(spec.key)
                + 0.30 * self._social_survival_fragility_signal(spec.key)
                + 0.15 * self._economy_public_fragility_signal(),
                0.0,
                1.5,
            ),
            forecast_error_belief=self._initial_forecast_error_belief(spec.key),
            last_technology_investment=0.0,
            last_technology_gain=0.0,
            loss_streak=0,
        )
        self._refresh_firm_startup_state(firm, spec, expected_sales)
        return firm

    def _update_firm_policies(self, last_unemployment: float) -> None:
        living_wage_anchor = self._living_wage_anchor()
        household_income_gap = (
            max(0.0, 1.0 - self.history[-1].family_income_to_basket_ratio)
            if self.history
            else 0.0
        )
        for spec in SECTOR_SPECS:
            sector_firms = self._sector_firms(spec.key)
            if not sector_firms:
                continue

            sector_total_demand = self._baseline_demand(spec.key)
            competitiveness_weights = []
            for firm in sector_firms:
                competitive_price = max(0.1, firm.price + firm.input_cost_per_unit + firm.transport_cost_per_unit)
                competitiveness_weights.append((firm, 1.0 / competitive_price))
            total_competitiveness = sum(weight for _, weight in competitiveness_weights)
            if total_competitiveness <= 0.0:
                total_competitiveness = float(len(sector_firms))

            for firm, competitiveness in competitiveness_weights:
                learning_maturity = self._firm_learning_maturity(firm)
                effective_productivity = self._firm_effective_productivity(firm)
                sell_through = self._firm_recent_sell_through(firm)
                revealed_growth_pressure = self._firm_revealed_growth_pressure(firm)
                previous_output_per_worker = self._smoothed_production_reference(firm) / max(1, firm.last_worker_count)
                productivity_gain_ratio = clamp(
                    (effective_productivity - previous_output_per_worker) / max(0.1, previous_output_per_worker),
                    -0.25,
                    1.0,
                )
                market_share = competitiveness / total_competitiveness
                sales_anchor = self._sales_anchor(firm, sector_total_demand, market_share)
                baseline_demand = max(1.0, sales_anchor)
                preliminary_inventory_target = self._firm_target_inventory_units(firm, baseline_demand)
                preliminary_desired_output = max(0.0, baseline_demand + preliminary_inventory_target - firm.inventory)
                preliminary_desired_workers = max(1, math.ceil(preliminary_desired_output / effective_productivity))
                vacancy_ratio = max(0.0, preliminary_desired_workers - firm.last_worker_count) / max(
                    1, preliminary_desired_workers
                )
                profit_ratio = clamp(firm.last_profit / max(1.0, firm.last_revenue), -0.25, 0.25)
                labor_tightness = self.config.target_unemployment - last_unemployment
                labor_value_per_worker = effective_productivity * max(0.1, firm.price)
                wage_room = clamp(
                    (labor_value_per_worker - firm.wage_offer) / max(1.0, labor_value_per_worker),
                    -0.25,
                    1.0,
                )
                living_wage_gap = (
                    clamp((living_wage_anchor - firm.wage_offer) / max(1.0, living_wage_anchor), -0.40, 1.00)
                    if living_wage_anchor > 0.0
                    else 0.0
                )
                wage_adjustment = (
                    0.12 * labor_tightness
                    + 0.12 * vacancy_ratio
                    + 0.08 * max(0.0, profit_ratio)
                    - 0.07 * max(0.0, -profit_ratio)
                )
                wage_adjustment += (
                    self.config.living_wage_bargaining_weight
                    * max(0.0, living_wage_gap)
                    * (0.50 + 0.50 * max(0.0, wage_room))
                )
                wage_adjustment += 0.08 * max(0.0, productivity_gain_ratio) * max(0.0, wage_room)
                wage_adjustment += self._sector_wage_pressure_bonus(
                    spec.key,
                    vacancy_ratio=vacancy_ratio,
                    labor_tightness=labor_tightness,
                    living_wage_gap=living_wage_gap,
                    wage_room=wage_room,
                )
                if spec.key in ESSENTIAL_SECTOR_KEYS:
                    wage_adjustment += (
                        self.config.essential_wage_bargaining_bonus
                        * household_income_gap
                        * max(0.0, wage_room)
                    )
                    wage_adjustment += 0.05 * max(0.0, productivity_gain_ratio) * max(0.0, living_wage_gap)
                wage_adjustment += (
                    0.05
                    * self.config.firm_revealed_shortage_capacity_weight
                    * revealed_growth_pressure
                    * max(0.0, wage_room)
                )
                profitable_rejection_target = self._profitable_labor_offer_rejection_wage_target(
                    firm,
                    baseline_demand=baseline_demand,
                    effective_productivity=effective_productivity,
                    target_workers=preliminary_desired_workers,
                )
                if profitable_rejection_target is not None:
                    rejection_pressure = clamp(
                        firm.last_labor_offer_rejections / max(1, preliminary_desired_workers),
                        0.0,
                        1.0,
                    )
                    rejection_gap = max(
                        0.0,
                        profitable_rejection_target - firm.wage_offer,
                    ) / max(1.0, firm.wage_offer)
                    wage_adjustment += min(
                        self.config.labor_offer_rejection_response_cap,
                        rejection_gap
                        * self.config.labor_offer_rejection_catchup_share
                        * (0.35 + 0.65 * rejection_pressure),
                    )
                if firm.cash < firm.last_wage_bill * 0.5:
                    wage_adjustment -= 0.04
                wage_adjustment = clamp(wage_adjustment, -0.12, 0.24) * (0.35 + 0.65 * learning_maturity)
                sector_wage_floor = living_wage_anchor * (
                    self.config.reservation_wage_floor_share + self._sector_wage_floor_premium(spec.key)
                )
                adjusted_wage = firm.wage_offer * (1.0 + wage_adjustment)
                firm.wage_offer = max(sector_wage_floor, adjusted_wage)

                variable_unit_cost = (
                    firm.wage_offer / effective_productivity
                    + firm.input_cost_per_unit
                    + firm.transport_cost_per_unit
                )
                fixed_cost = firm.fixed_overhead + firm.capital * self.config.depreciation_rate
                average_unit_cost = variable_unit_cost + fixed_cost / max(1.0, baseline_demand)
                target_price = self._target_price_for_firm(
                    firm,
                    spec,
                    average_unit_cost,
                    variable_unit_cost,
                )
                affordability_pressure = (
                    self._essential_affordability_pressure()
                    if spec.key in ESSENTIAL_SECTOR_KEYS
                    else 0.0
                )
                cost_decline_ratio = (
                    self._firm_cost_decline_ratio(firm, average_unit_cost)
                    if spec.key in ESSENTIAL_SECTOR_KEYS
                    else 0.0
                )
                if self._in_startup_grace() and spec.key in ESSENTIAL_SECTOR_KEYS:
                    candidate_prices = self._startup_essential_candidate_prices(
                        firm,
                        variable_unit_cost,
                    )
                    clearance_discount = 0.0
                else:
                    candidate_prices = self._price_search_candidates(firm, spec, variable_unit_cost, target_price)
                reference_price = max(0.1, firm.price)
                if not self._in_startup_grace() or spec.key not in ESSENTIAL_SECTOR_KEYS:
                    clearance_discount = self._inventory_clearance_discount(firm)

                candidate_records: list[tuple[float, float, float, float, float, float]] = []
                best_profit = float("-inf")
                best_objective = float("-inf")
                for candidate_price in candidate_prices:
                    effective_price = candidate_price * (1.0 - clearance_discount)
                    expected_sales = self._expected_demand_for_price(
                        firm,
                        sector_total_demand,
                        market_share,
                        effective_price,
                        reference_price,
                    )
                    retention, market_hazard = self._candidate_market_retention(
                        firm,
                        effective_price,
                        reference_price,
                    )
                    prudent_sales = self._conservative_expected_sales(
                        firm,
                        expected_sales,
                        effective_price,
                        reference_price,
                        market_hazard,
                    )
                    candidate_profit = self._candidate_total_profit(
                        firm,
                        prudent_sales,
                        effective_price,
                        variable_unit_cost,
                        fixed_cost,
                    )
                    future_sales = clamp(prudent_sales * retention, 0.0, sector_total_demand)
                    future_market_value = future_sales * max(0.0, effective_price - variable_unit_cost)
                    candidate_objective = self._candidate_price_objective(
                        firm,
                        spec,
                        effective_price,
                        prudent_sales,
                        candidate_profit,
                        future_market_value,
                        market_hazard,
                        variable_unit_cost,
                        fixed_cost,
                    )
                    candidate_records.append(
                        (
                            effective_price,
                            prudent_sales,
                            candidate_profit,
                            future_market_value,
                            market_hazard,
                            candidate_objective,
                        )
                    )
                    best_profit = max(best_profit, candidate_profit)
                    best_objective = max(best_objective, candidate_objective)

                penetration_mode = (
                    spec.key in ESSENTIAL_SECTOR_KEYS
                    and firm.inventory / max(1.0, firm.target_inventory) > 1.15
                    and firm.cash / max(1.0, firm.last_wage_bill + fixed_cost) > 1.05
                )
                profit_floor_ratio = clamp(
                    0.95
                    - 0.08 * (firm.markup_tolerance - 1.0)
                    - 0.04 * (firm.market_share_ambition - 1.0)
                    + 0.05 * (firm.cash_conservatism - 1.0),
                    0.78,
                    0.99,
                )
                future_bias = self._firm_future_market_weight(firm) * clamp(
                    firm.market_fragility_belief,
                    0.0,
                    1.5,
                )
                profit_floor_ratio = clamp(profit_floor_ratio - 0.10 * future_bias, 0.65, 0.99)
                forecast_uncertainty = self._firm_forecast_uncertainty(firm)
                profit_floor_ratio = clamp(
                    profit_floor_ratio
                    - 0.05 * max(0.0, firm.forecast_caution - 1.0)
                    - 0.08 * forecast_uncertainty,
                    0.55,
                    0.99,
                )
                if penetration_mode:
                    profit_floor_ratio = clamp(profit_floor_ratio - 0.03 * (firm.volume_preference - 1.0), 0.75, 0.98)
                if spec.key in ESSENTIAL_SECTOR_KEYS:
                    profit_floor_ratio = clamp(
                        profit_floor_ratio
                        - 0.18 * affordability_pressure
                        - 0.22 * cost_decline_ratio
                        - 0.04 * max(0.0, firm.volume_preference - 1.0),
                        0.25,
                        0.95,
                    )
                if best_profit > 0.0:
                    profit_cutoff = best_profit * profit_floor_ratio
                    qualifying_candidates = [
                        record for record in candidate_records if record[2] >= profit_cutoff
                    ]
                else:
                    qualifying_candidates = [
                        record for record in candidate_records if record[2] >= best_profit - 1e-9
                    ]

                if not qualifying_candidates:
                    qualifying_candidates = candidate_records

                objective_floor_ratio = 0.90
                if spec.key in ESSENTIAL_SECTOR_KEYS:
                    objective_floor_ratio = clamp(
                        0.78 - 0.18 * affordability_pressure - 0.10 * cost_decline_ratio,
                        0.45,
                        0.90,
                    )
                objective_cutoff = best_objective * objective_floor_ratio
                if best_objective > 0.0:
                    qualifying_candidates = [
                        record for record in qualifying_candidates if record[5] >= objective_cutoff
                    ] or qualifying_candidates

                min_price = min(record[0] for record in qualifying_candidates)
                max_price = max(record[0] for record in qualifying_candidates)
                min_sales = min(record[1] for record in qualifying_candidates)
                max_sales = max(record[1] for record in qualifying_candidates)
                min_profit = min(record[2] for record in qualifying_candidates)
                max_profit = max(record[2] for record in qualifying_candidates)
                min_future_value = min(record[3] for record in qualifying_candidates)
                max_future_value = max(record[3] for record in qualifying_candidates)
                min_market_hazard = min(record[4] for record in qualifying_candidates)
                max_market_hazard = max(record[4] for record in qualifying_candidates)
                min_objective = min(record[5] for record in qualifying_candidates)
                max_objective = max(record[5] for record in qualifying_candidates)
                price_span = max(1e-9, max_price - min_price)
                sales_span = max(1e-9, max_sales - min_sales)
                profit_span = max(1e-9, max_profit - min_profit)
                future_value_span = max(1e-9, max_future_value - min_future_value)
                market_hazard_span = max(1e-9, max_market_hazard - min_market_hazard)
                objective_span = max(1e-9, max_objective - min_objective)
                volume_weight = 0.85 + 0.30 * firm.volume_preference + 0.20 * firm.market_share_ambition
                profit_weight = 0.85 + 0.25 * firm.cash_conservatism
                price_weight = (
                    0.80
                    + 0.25 * firm.markup_tolerance
                    + 0.30 * firm.forecast_caution
                    + 0.40 * forecast_uncertainty
                )
                future_weight = (
                    0.80
                    + 1.20 * self._firm_future_market_weight(firm)
                    + 0.15 * firm.forecast_caution
                    + 0.20 * forecast_uncertainty
                )
                if spec.key in ESSENTIAL_SECTOR_KEYS:
                    volume_weight += 0.55 * affordability_pressure + 0.30 * cost_decline_ratio
                    price_weight += 0.50 * affordability_pressure + 0.25 * cost_decline_ratio
                    profit_weight = clamp(
                        profit_weight
                        - 0.18 * affordability_pressure
                        - 0.12 * cost_decline_ratio,
                        0.40,
                        5.0,
                    )
                fragility_weight = (
                    0.70
                    + 0.25 * firm.inventory_aversion
                    + 0.20 * firm.cash_conservatism
                    + 0.20 * firm.forecast_caution
                    + 0.25 * forecast_uncertainty
                )

                (
                    chosen_price,
                    best_expected_sales,
                    best_profit,
                    _chosen_future_market_value,
                    _chosen_market_hazard,
                    _chosen_objective,
                ) = max(
                    qualifying_candidates,
                    key=lambda record: (
                        + 1.40 * ((record[5] - min_objective) / objective_span)
                        + profit_weight * ((record[2] - min_profit) / profit_span)
                        + volume_weight * ((record[1] - min_sales) / sales_span)
                        + future_weight * ((record[3] - min_future_value) / future_value_span)
                        - price_weight * ((record[0] - min_price) / price_span),
                        - fragility_weight * ((record[4] - min_market_hazard) / market_hazard_span),
                        record[1],
                        -record[0],
                    ),
                )

                if self._is_education_sector(spec.key):
                    target_inventory = self._education_service_target_units(firm, best_expected_sales)
                else:
                    target_inventory = self._firm_target_inventory_units(firm, best_expected_sales)
                if self._in_startup_grace() and spec.key in ESSENTIAL_SECTOR_KEYS:
                    firm.desired_workers = max(1, len(firm.workers))
                    firm.target_inventory = max(firm.target_inventory, target_inventory, firm.inventory)
                    firm.price = chosen_price
                    firm.last_expected_sales = max(best_expected_sales, firm.last_expected_sales)
                    firm.last_unit_cost = variable_unit_cost + fixed_cost / max(1.0, firm.last_expected_sales)
                    continue
                if self._is_education_sector(spec.key):
                    desired_service_units = target_inventory
                    firm.desired_workers = self._workers_needed_for_units(
                        desired_service_units,
                        effective_productivity,
                        productivity_floor=0.25,
                    )
                else:
                    desired_output = self._firm_desired_output_from_expected_sales(firm, best_expected_sales)
                    firm.desired_workers = self._workers_needed_for_units(desired_output, effective_productivity)
                firm.target_inventory = target_inventory
                firm.price = chosen_price
                firm.last_expected_sales = best_expected_sales
                firm.last_unit_cost = variable_unit_cost + fixed_cost / max(1.0, best_expected_sales)

    def _match_labor(self) -> None:
        firm_order = sorted(
            [firm for firm in self.firms if firm.active],
            key=lambda firm: (
                firm.wage_offer,
                firm.cash,
                -firm.bankruptcy_streak,
            ),
            reverse=True,
        )
        eligible_households = self._eligible_households()

        for household in eligible_households:
            best_firm = None
            best_score = None
            best_rejected_firm = None
            best_rejected_score = None
            age_years = self._household_age_years(household)
            essential_candidates_exist = False
            if age_years < 30.0:
                age_bonus = self.config.young_worker_bonus
            elif age_years < 45.0:
                age_bonus = self.config.young_worker_bonus * 0.6
            elif age_years < self.config.senior_age_years:
                age_bonus = 0.0
            else:
                age_bonus = -self.config.senior_worker_penalty

            if self._in_essential_protection():
                essential_candidates_exist = any(
                    firm.active
                    and firm.sector in ESSENTIAL_SECTOR_KEYS
                    and len(firm.workers) < firm.desired_workers
                    and firm.wage_offer >= household.reservation_wage
                    for firm in firm_order
                )

            for firm in firm_order:
                if len(firm.workers) >= firm.desired_workers:
                    continue
                if self._in_essential_protection() and essential_candidates_exist and firm.sector not in ESSENTIAL_SECTOR_KEYS:
                    continue
                stability = firm.cash / max(1.0, firm.last_wage_bill + firm.price)
                labor_capacity = self._worker_effective_labor_for_sector(household, firm.sector)
                if labor_capacity <= 0.0:
                    continue
                profit_margin = firm.last_profit / max(1.0, firm.last_revenue, firm.last_sales * max(0.1, firm.price))
                viability = clamp(
                    1.0
                    + 0.20 * profit_margin
                    - 0.08 * firm.loss_streak
                    - 0.10 * firm.bankruptcy_streak,
                    0.35,
                    1.15,
                )
                score = (
                    (firm.wage_offer * viability + 0.01 * firm.cash + 0.05 * stability)
                    * labor_capacity
                    + age_bonus
                )
                if firm.wage_offer < household.reservation_wage:
                    if best_rejected_score is None or score > best_rejected_score:
                        best_rejected_score = score
                        best_rejected_firm = firm
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_firm = firm
            if best_firm is not None:
                best_firm.workers.append(household.id)
                household.employed_by = best_firm.id
                household.employment_tenure = 0
            elif best_rejected_firm is not None:
                self._record_labor_offer_rejection(best_rejected_firm, household.reservation_wage)

    def _produce_and_pay_wages(self) -> None:
        for firm in self.firms:
            if not firm.active:
                continue
            workers = firm.workers
            firm.last_worker_count = len(workers)
            effective_labor_units = sum(
                self._worker_effective_labor_for_sector(self.households[worker_id], firm.sector)
                for worker_id in workers
            )
            output_units = self._firm_effective_productivity(firm) * effective_labor_units
            if self._is_education_sector(firm.sector):
                output_units = min(output_units, self._education_firm_capacity(firm))
            firm.last_production = output_units
            if self._is_education_sector(firm.sector):
                # Education services are period capacity, not storable goods.
                firm.inventory = output_units
            else:
                self._add_inventory_units(firm, output_units)
            self._period_production_units += output_units
            if firm.sector in ESSENTIAL_SECTOR_KEYS:
                self._period_essential_production_units += output_units

            wage_bill = firm.wage_offer * len(workers)
            firm.last_wage_bill = wage_bill
            firm.cash -= wage_bill
            self._period_wages += wage_bill
            payroll_tax = wage_bill * self._government_payroll_tax_rate()
            if payroll_tax > 0.0:
                firm.cash -= payroll_tax
                self._record_government_tax_revenue(payroll_tax, "payroll")

            input_cost = output_units * firm.input_cost_per_unit
            transport_cost = (
                output_units
                * firm.transport_cost_per_unit
                * self._public_infrastructure_transport_cost_multiplier()
            )
            fixed_overhead = firm.fixed_overhead
            capital_charge = firm.capital * self.config.depreciation_rate
            firm.last_input_cost = input_cost
            firm.last_transport_cost = transport_cost
            firm.last_fixed_overhead = fixed_overhead
            firm.last_capital_charge = capital_charge
            if output_units > 0.0:
                costing_units = self._firm_costing_units(firm, output_units)
                firm.last_unit_cost = (
                    wage_bill
                    + payroll_tax
                    + input_cost
                    + transport_cost
                    + fixed_overhead
                    + capital_charge
                ) / costing_units
            nonwage_operating_cost = input_cost + transport_cost + fixed_overhead
            firm.cash -= nonwage_operating_cost
            self._queue_sector_payment("manufactured", input_cost + transport_cost)
            self._queue_sector_payment("housing", fixed_overhead)

            for worker_id in workers:
                household = self.households[worker_id]
                labor_tax = min(
                    firm.wage_offer,
                    firm.wage_offer * self._government_labor_tax_rate(firm.wage_offer),
                )
                net_wage = firm.wage_offer - labor_tax
                household.wage_income += net_wage
                household.last_income += net_wage
                if labor_tax > 0.0:
                    self._record_government_tax_revenue(labor_tax, "labor")

            self._cash_before_sales[firm.id] = firm.cash

        self._pay_public_administration_wages()
        self._flush_pending_sector_payments()

    def _family_budget_profile(self, members: list[Household]) -> tuple[float, float, float, int]:
        adult_members = [
            member
            for member in members
            if self._household_age_years(member) >= self.config.entry_age_years
        ]
        decision_members = adult_members if adult_members else members
        price_sensitivity = sum(member.price_sensitivity for member in decision_members) / max(1, len(decision_members))
        saving_propensity = sum(member.saving_propensity for member in decision_members) / max(1, len(decision_members))
        consumption_multiplier = sum(
            self._household_consumption_multiplier(member) for member in members
        ) / max(1, len(members))
        return price_sensitivity, saving_propensity, consumption_multiplier, len(adult_members)

    def _family_savings_rate(
        self,
        members: list[Household],
        *,
        family_cash: float,
        family_basic_basket_cost: float,
        family_price_sensitivity: float,
        family_saving_propensity: float,
        family_consumption_multiplier: float,
        inflation_pressure: float,
    ) -> float:
        adult_members = [
            member
            for member in members
            if self._household_age_years(member) >= self.config.entry_age_years
        ]
        decision_members = adult_members if adult_members else members
        if not decision_members:
            return 0.0

        family_money_trust = sum(member.money_trust for member in decision_members) / max(1, len(decision_members))
        family_consumption_impatience = (
            sum(member.consumption_impatience for member in decision_members) / max(1, len(decision_members))
        )
        employment_stability = sum(
            1.0 for member in decision_members if member.employed_by is not None
        ) / max(1.0, len(decision_members))
        deprivation_pressure = sum(
            clamp(
                0.45 * member.deprivation_streak / max(1.0, float(self.config.starvation_death_periods))
                + 0.35 * member.severe_hunger_streak / max(1.0, float(self.config.starvation_death_periods))
                + 0.20 * member.health_fragility,
                0.0,
                1.0,
            )
            for member in decision_members
        ) / max(1, len(decision_members))
        residual_after_floor = max(0.0, family_cash - family_basic_basket_cost)
        residual_share = clamp(residual_after_floor / max(1.0, family_basic_basket_cost), 0.0, 2.0)
        desired_cushion_months = clamp(
            1.0
            + 2.1 * family_saving_propensity
            + 1.1 * (1.0 - family_money_trust)
            + 0.8 * (1.0 - employment_stability)
            + 0.9 * deprivation_pressure,
            1.0,
            6.0,
        )
        desired_cushion = family_basic_basket_cost * desired_cushion_months
        cushion_gap = clamp((desired_cushion - family_cash) / max(1.0, desired_cushion), 0.0, 1.0)
        cushion_surplus = clamp((family_cash - desired_cushion) / max(1.0, desired_cushion), 0.0, 2.0)
        liquidity_stress = clamp(1.0 - family_cash / max(1.0, family_basic_basket_cost), 0.0, 1.0)
        insecurity = clamp(
            0.40 * deprivation_pressure
            + 0.35 * (1.0 - employment_stability)
            + 0.25 * liquidity_stress,
            0.0,
            1.0,
        )
        macro_trust = clamp(
            1.0 - 2.5 * max(0.0, inflation_pressure) - 0.15 * max(0.0, family_price_sensitivity - 1.0),
            0.0,
            1.0,
        )
        trust = clamp(0.70 * family_money_trust + 0.30 * macro_trust, 0.0, 1.0)
        consumption_desire = clamp(
            0.70 * family_consumption_impatience + 0.30 * family_consumption_multiplier,
            0.0,
            1.0,
        )
        score = (
            -2.20
            + 0.35 * residual_share
            + 0.85 * cushion_gap
            + 0.65 * insecurity
            + 0.30 * (1.0 - trust)
            + 0.12 * family_saving_propensity
            - 1.00 * consumption_desire
            - 1.10 * cushion_surplus
        )
        savings_rate = 1.0 / (1.0 + math.exp(-score))
        if family_cash >= desired_cushion:
            post_cushion_scale = clamp(
                0.16
                + 0.14 * insecurity
                + 0.10 * (1.0 - trust)
                - 0.10 * cushion_surplus,
                0.02,
                0.22,
            )
            savings_rate *= post_cushion_scale
        return clamp(savings_rate, 0.0, 0.45)

    def _advance_household_education(self, household: Household) -> None:
        if self._is_school_age(household):
            school_coverage = self._household_sector_coverage(household, "school")
            household.school_years_completed = clamp(
                household.school_years_completed + school_coverage / max(1, self.config.periods_per_year),
                0.0,
                self.config.school_years_required,
            )
        elif self._household_age_years(household) >= self.config.entry_age_years:
            household.school_years_completed = min(
                self.config.school_years_required,
                household.school_years_completed,
            )

        if self._is_university_age(household) and self._household_has_school_credential(household):
            university_coverage = self._household_sector_coverage(household, "university")
            household.university_years_completed = clamp(
                household.university_years_completed + university_coverage / max(1, self.config.periods_per_year),
                0.0,
                self.config.university_years_required,
            )
        elif self._household_age_years(household) >= self.config.university_age_max_years:
            household.university_years_completed = min(
                self.config.university_years_required,
                household.university_years_completed,
            )

    def _update_household_post_consumption_state(
        self,
        member: Household,
        *,
        allocated_units: dict[str, float],
        target_units_by_sector: dict[str, float] | None = None,
        coverages_by_sector: dict[str, float] | None = None,
        per_adult_savings: float,
        family_remaining_cash: float,
        family_basic_basket_cost: float,
        inflation_pressure: float,
    ) -> None:
        member.wage_income = 0.0
        member.last_consumption = allocated_units.copy()
        self._advance_household_education(member)
        if self._household_age_years(member) >= self.config.entry_age_years:
            member.savings = per_adult_savings
        else:
            member.savings = 0.0

        if target_units_by_sector is None:
            target_units_by_sector = {
                spec.key: self._household_sector_desired_units(member, spec.key)
                for spec in SECTOR_SPECS
            }
        if coverages_by_sector is None:
            coverages_by_sector = {
                sector_key: self._coverage_from_units(
                    allocated_units.get(sector_key, 0.0),
                    target_units_by_sector.get(sector_key, 0.0),
                )
                for sector_key in (spec.key for spec in SECTOR_SPECS)
            }
        self._update_public_education_support_persistence(member, coverages_by_sector)

        food_coverage = coverages_by_sector["food"]
        housing_coverage = coverages_by_sector["housing"]
        clothing_coverage = coverages_by_sector["clothing"]
        subsistence_threshold = self._food_subsistence_coverage_ratio()
        severe_threshold = self._food_severe_hunger_coverage_ratio()

        if food_coverage < subsistence_threshold:
            member.deprivation_streak += 1
        else:
            member.deprivation_streak = 0

        if food_coverage < severe_threshold:
            member.severe_hunger_streak += 1
        else:
            member.severe_hunger_streak = 0

        if housing_coverage < 0.80:
            member.housing_deprivation_streak += 1
        else:
            member.housing_deprivation_streak = 0

        if clothing_coverage < 0.75:
            member.clothing_deprivation_streak += 1
        else:
            member.clothing_deprivation_streak = 0

        severe_food_gap = clamp((severe_threshold - food_coverage) / max(1e-9, severe_threshold), 0.0, 1.5)
        subsistence_food_gap = clamp((subsistence_threshold - food_coverage) / max(1e-9, subsistence_threshold), 0.0, 1.5)
        housing_gap = clamp((0.80 - housing_coverage) / 0.80, 0.0, 1.25)
        clothing_gap = clamp((0.75 - clothing_coverage) / 0.75, 0.0, 1.25)
        fragility_shock = (
            0.45 * subsistence_food_gap
            + 0.25 * severe_food_gap
            + 0.18 * housing_gap
            + 0.12 * clothing_gap
        )
        recovery = 0.0
        if food_coverage >= 1.0:
            recovery += 0.10
        if housing_coverage >= 0.90:
            recovery += 0.05
        if clothing_coverage >= 0.90:
            recovery += 0.03
        member.health_fragility = clamp(0.82 * member.health_fragility + fragility_shock - recovery, 0.0, 3.0)

        trust_signal = clamp(
            0.60 * (1.0 - 2.5 * max(0.0, inflation_pressure))
            + 0.20 * food_coverage
            + 0.10 * min(housing_coverage, 1.0)
            + 0.10 * clamp(family_remaining_cash / max(1.0, family_basic_basket_cost), 0.0, 1.5) / 1.5,
            0.0,
            1.0,
        )
        member.money_trust = clamp(0.85 * member.money_trust + 0.15 * trust_signal, 0.0, 1.0)
        member.last_perceived_utility = self._household_perceived_utility_from_inputs(
            member,
            coverages_by_sector=coverages_by_sector,
            target_units_by_sector=target_units_by_sector,
            family_remaining_cash=family_remaining_cash,
            family_basic_basket_cost=family_basic_basket_cost,
        )

    def _update_public_education_support_persistence(
        self,
        household: Household,
        coverages_by_sector: dict[str, float],
    ) -> None:
        school_signal = 1.0 if coverages_by_sector.get("school", 0.0) >= 0.60 else 0.0
        university_signal = 1.0 if coverages_by_sector.get("university", 0.0) >= 0.55 else 0.0
        household.public_school_support_persistence = clamp(
            0.60 * household.public_school_support_persistence + 0.40 * school_signal,
            0.0,
            1.0,
        )
        household.public_university_support_persistence = clamp(
            0.60 * household.public_university_support_persistence + 0.40 * university_signal,
            0.0,
            1.0,
        )

    def _government_education_budget_base(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        return self._government_structural_budget_anchor()

    def _consume_households(self) -> None:
        discretionary_firms = [firm for key in DISCRETIONARY_SECTOR_KEYS for firm in self._sector_firms(key)]
        discretionary_price_pressure = 1.0
        if discretionary_firms:
            discretionary_price_pressure = sum(
                firm.price / SECTOR_BY_KEY[firm.sector].base_price for firm in discretionary_firms
            ) / len(discretionary_firms)
        public_school_budget_remaining = 0.0
        public_university_budget_remaining = 0.0
        if self.config.government_enabled:
            education_budget_base = self._government_education_budget_base()
            public_school_budget_remaining = max(0.0, education_budget_base * self.config.public_school_budget_share)
            public_university_budget_remaining = max(
                0.0,
                education_budget_base * self.config.public_university_budget_share,
            )

        inflation_pressure = 0.0
        if self.history:
            current_price_index = self.history[-1].price_index
            prior_price_index = self.history[-2].price_index if len(self.history) > 1 else current_price_index
            if prior_price_index > 0.0:
                inflation_pressure = clamp((current_price_index / prior_price_index) - 1.0, 0.0, 0.35)

        self._ensure_period_essential_household_arrays()

        for members in self._family_groups_consumption_order():
            family_members = [member for member in members if member.alive]
            if not family_members:
                continue

            family_row_indices: list[int] = []
            member_target_units_by_sector: dict[int, dict[str, float]] = {}
            public_education_target_by_sector = {"school": 0.0, "university": 0.0}
            family_baseline_demand: np.ndarray | None = None
            if self._period_essential_desired_units_matrix is not None:
                family_row_indices = [
                    self._period_household_row_index_cache.get(member.id, -1)
                    for member in family_members
                ]
                if family_row_indices and all(index >= 0 for index in family_row_indices):
                    family_baseline_demand = self._period_essential_desired_units_matrix[
                        np.asarray(family_row_indices, dtype=np.int64)
                    ].sum(axis=0)

            family_price_sensitivity, family_saving_propensity, family_consumption_multiplier, adult_count = (
                self._family_budget_profile(family_members)
            )
            family_cash = sum(self._household_cash_balance(member) for member in family_members)
            family_last_available_cash = family_cash / max(1, adult_count if adult_count > 0 else len(family_members))
            spending_log = {spec.key: 0.0 for spec in SECTOR_SPECS}
            purchased_units_by_sector = {spec.key: 0.0 for spec in SECTOR_SPECS}
            self._period_worker_cash_available += family_cash

            for member in family_members:
                if adult_count > 0:
                    member.last_available_cash = (
                        family_last_available_cash if self._household_age_years(member) >= self.config.entry_age_years else 0.0
                    )
                else:
                    member.last_available_cash = family_last_available_cash

            for member_index, member in enumerate(family_members):
                target_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
                if family_baseline_demand is not None and family_row_indices:
                    baseline_row = self._period_essential_desired_units_matrix[family_row_indices[member_index]]
                    for sector_key in ARRAY_BACKED_SECTOR_KEYS:
                        target_units[sector_key] = float(
                            baseline_row[ARRAY_BACKED_SECTOR_INDEX[sector_key]]
                        )
                else:
                    for sector_key in ARRAY_BACKED_SECTOR_KEYS:
                        target_units[sector_key] = self._household_sector_desired_units(member, sector_key)
                target_units["school"] = self._household_sector_desired_units(member, "school")
                target_units["university"] = self._household_sector_desired_units(member, "university")
                member_target_units_by_sector[member.id] = target_units
                if self.config.government_enabled:
                    for sector_key in MERIT_SECTOR_KEYS:
                        public_education_target_by_sector[sector_key] += self._public_education_target_units(
                            member,
                            sector_key,
                            target_units[sector_key],
                        )

            if self._period_essential_budget_vector is not None and family_row_indices and all(index >= 0 for index in family_row_indices):
                family_basic_basket_cost = float(
                    self._period_essential_budget_vector[np.asarray(family_row_indices, dtype=np.int64)].sum()
                )
            else:
                family_basic_basket_cost = sum(self._essential_budget(member) for member in family_members)

            if family_cash <= 0.0:
                allocated_units_by_member = self._allocate_family_consumption_units(
                    family_members,
                    purchased_units_by_sector,
                    target_units_by_member=member_target_units_by_sector,
                )
                for member in family_members:
                    allocated_units = allocated_units_by_member.get(member.id, spending_log)
                    target_units = member_target_units_by_sector.get(member.id, {})
                    coverages = {
                        sector_key: self._coverage_from_units(
                            allocated_units.get(sector_key, 0.0),
                            target_units.get(sector_key, 0.0),
                        )
                        for sector_key in target_units
                    }
                    self._update_household_post_consumption_state(
                        member,
                        allocated_units=allocated_units,
                        target_units_by_sector=target_units,
                        coverages_by_sector=coverages,
                        per_adult_savings=0.0,
                        family_remaining_cash=0.0,
                        family_basic_basket_cost=family_basic_basket_cost,
                        inflation_pressure=inflation_pressure,
                    )
                continue

            cash = family_cash
            essential_target_units = 0.0
            essential_units_bought = 0.0
            baseline_essential_units_bought = 0.0
            essential_target_by_sector: dict[str, float] = {}

            for sector_index, sector_key in enumerate(ESSENTIAL_SECTOR_KEYS):
                if family_baseline_demand is not None:
                    desired_units = float(family_baseline_demand[sector_index])
                else:
                    desired_units = sum(
                        member_target_units_by_sector[member.id][sector_key]
                        for member in family_members
                    )
                essential_target_by_sector[sector_key] = desired_units
                essential_target_units += desired_units
                self._period_essential_demand_units += desired_units
                self._period_potential_demand_units += desired_units
                self._period_sector_potential_demand_units[sector_key] += desired_units
                self._period_sector_budget_demand_units[sector_key] += desired_units
                cash, units_bought = self._purchase_from_sector(
                    family_price_sensitivity,
                    sector_key,
                    desired_units,
                    cash,
                    spending_log,
                )
                essential_units_bought += units_bought
                baseline_essential_units_bought += units_bought
                purchased_units_by_sector[sector_key] += units_bought
                self._period_essential_sales_units += units_bought

            merit_target_by_sector: dict[str, float] = {}
            for sector_key in MERIT_SECTOR_KEYS:
                desired_units = sum(
                    member_target_units_by_sector[member.id][sector_key]
                    for member in family_members
                )
                private_target_units = max(
                    0.0,
                    desired_units - public_education_target_by_sector.get(sector_key, 0.0),
                )
                merit_target_by_sector[sector_key] = private_target_units
                if private_target_units <= 0.0:
                    continue
                self._period_potential_demand_units += private_target_units
                self._period_sector_potential_demand_units[sector_key] += private_target_units
                self._period_sector_budget_demand_units[sector_key] += private_target_units
                cash, units_bought = self._purchase_from_sector(
                    family_price_sensitivity,
                    sector_key,
                    private_target_units,
                    cash,
                    spending_log,
                )
                purchased_units_by_sector[sector_key] += units_bought

            target_savings_rate = self._family_savings_rate(
                family_members,
                family_cash=cash,
                family_basic_basket_cost=family_basic_basket_cost,
                family_price_sensitivity=family_price_sensitivity,
                family_saving_propensity=family_saving_propensity,
                family_consumption_multiplier=family_consumption_multiplier,
                inflation_pressure=inflation_pressure,
            )
            target_savings = cash * target_savings_rate
            discretionary_budget_neutral = max(0.0, cash - target_savings)
            discretionary_budget_neutral *= family_consumption_multiplier
            discretionary_budget_effective = discretionary_budget_neutral * clamp(
                1.0 - family_price_sensitivity * max(0.0, discretionary_price_pressure - 1.0) * 0.35,
                0.25,
                1.0,
            )

            extra_essential_spent = 0.0
            if discretionary_budget_effective > 0.0 and cash > 0.0:
                essential_coverage = essential_units_bought / essential_target_units if essential_target_units > 0.0 else 1.0
                extra_essential_budget = min(
                    cash,
                    discretionary_budget_effective * self._essential_extra_budget_share(essential_coverage),
                )
                if extra_essential_budget > 0.0:
                    essential_weights: dict[str, float] = {}
                    for sector_key in ESSENTIAL_SECTOR_KEYS:
                        target_units = max(1e-9, essential_target_by_sector.get(sector_key, 0.0))
                        sector_coverage = purchased_units_by_sector[sector_key] / target_units
                        average_preference = sum(
                            self._household_sector_preference(member, sector_key) for member in family_members
                        ) / max(1, len(family_members))
                        essential_weights[sector_key] = average_preference * self._essential_marginal_utility(sector_coverage)
                    total_essential_weight = sum(essential_weights.values())
                    if total_essential_weight > 0.0:
                        for sector_key in ESSENTIAL_SECTOR_KEYS:
                            if cash <= 0.0:
                                continue
                            sector_share = essential_weights[sector_key] / total_essential_weight
                            intended_spend = extra_essential_budget * sector_share
                            average_price = self._average_sector_price(sector_key)
                            remaining_gap_units = self._extra_essential_gap_units(
                                essential_target_by_sector.get(sector_key, 0.0),
                                purchased_units_by_sector[sector_key],
                            )
                            desired_units = min(
                                intended_spend / max(0.1, average_price),
                                remaining_gap_units,
                            )
                            if desired_units <= 0.0:
                                continue
                            self._period_potential_demand_units += desired_units
                            self._period_sector_potential_demand_units[sector_key] += desired_units
                            self._period_sector_budget_demand_units[sector_key] += desired_units
                            cash_before_purchase = cash
                            cash, units_bought = self._purchase_from_sector(
                                family_price_sensitivity,
                                sector_key,
                                desired_units,
                                cash,
                                spending_log,
                            )
                            spent = max(0.0, cash_before_purchase - cash)
                            extra_essential_spent += spent
                            essential_units_bought += units_bought
                            purchased_units_by_sector[sector_key] += units_bought

            effective_nonessential_budget = max(0.0, discretionary_budget_effective - extra_essential_spent)
            nonessential_budget_factor = (
                effective_nonessential_budget / discretionary_budget_effective
                if discretionary_budget_effective > 0.0
                else 0.0
            )
            neutral_nonessential_budget = discretionary_budget_neutral * nonessential_budget_factor

            if neutral_nonessential_budget > 0.0:
                sector_preference_units: dict[str, float] = {}
                for sector_key in PURE_DISCRETIONARY_SECTOR_KEYS:
                    if family_baseline_demand is not None and sector_key in ARRAY_BACKED_SECTOR_KEYS:
                        preference_units = float(
                            family_baseline_demand[ARRAY_BACKED_SECTOR_INDEX[sector_key]]
                        )
                    else:
                        preference_units = sum(
                            member_target_units_by_sector[member.id][sector_key]
                            for member in family_members
                        )
                    sector_preference_units[sector_key] = preference_units
                total_preference_units = sum(sector_preference_units.values())
                if total_preference_units <= 0.0:
                    sector_preference_units = {key: 1.0 for key in PURE_DISCRETIONARY_SECTOR_KEYS}
                essential_coverage = self._family_essential_coverage_ratio(
                    essential_target_by_sector,
                    purchased_units_by_sector,
                )
                cash = self._purchase_discretionary_bundle(
                    sector_keys=list(PURE_DISCRETIONARY_SECTOR_KEYS),
                    sector_preference_units=sector_preference_units,
                    essential_coverage=essential_coverage,
                    family_remaining_cash=cash,
                    family_basic_basket_cost=family_basic_basket_cost,
                    family_price_sensitivity=family_price_sensitivity,
                    budget_neutral=neutral_nonessential_budget,
                    budget_effective=effective_nonessential_budget,
                    cash=cash,
                    spending_log=spending_log,
                    purchased_units_by_sector=purchased_units_by_sector,
                )

            for sector_key in ("school", "university"):
                if not self.config.government_enabled:
                    continue
                public_budget_remaining = (
                    public_school_budget_remaining if sector_key == "school" else public_university_budget_remaining
                )
                if public_budget_remaining <= 0.0:
                    continue
                total_target_units = sum(
                    member_target_units_by_sector[member.id][sector_key] for member in family_members
                )
                public_target_units = public_education_target_by_sector[sector_key]
                unmet_units = max(
                    0.0,
                    max(
                        public_target_units,
                        total_target_units - purchased_units_by_sector.get(sector_key, 0.0),
                    ),
                )
                if unmet_units <= 0.0:
                    continue
                budget_cap = min(public_budget_remaining, unmet_units * self._average_sector_price(sector_key))
                if budget_cap <= 0.0:
                    continue
                financing_gap = max(0.0, budget_cap - self.government.treasury_cash)
                if financing_gap > 0.0:
                    self._issue_government_bonds(financing_gap)
                budget_cap = min(budget_cap, self.government.treasury_cash)
                if budget_cap <= 0.0:
                    continue
                government_remaining_cash = budget_cap
                government_units_bought = 0.0
                if self._sector_firms(sector_key):
                    government_remaining_cash, government_units_bought = self._purchase_from_sector(
                        max(0.1, family_price_sensitivity * 0.85),
                        sector_key,
                        unmet_units,
                        budget_cap,
                        spending_log,
                    )
                remaining_unmet_units = max(0.0, unmet_units - government_units_bought)
                direct_public_units = 0.0
                if government_remaining_cash > 0.0 and remaining_unmet_units > 0.0:
                    affordable_public_units = self._public_education_service_units(sector_key, government_remaining_cash)
                    direct_public_units = min(remaining_unmet_units, affordable_public_units)
                    if direct_public_units >= affordable_public_units - 1e-9:
                        direct_public_spending = government_remaining_cash
                    else:
                        direct_public_spending = (
                            direct_public_units
                            * max(0.1, self._average_sector_price(sector_key))
                            / max(0.1, self.config.government_spending_efficiency)
                        )
                    government_remaining_cash = max(0.0, government_remaining_cash - direct_public_spending)
                    government_units_bought += direct_public_units
                    self._period_sales_units += direct_public_units
                    self._period_sales_revenue += direct_public_spending
                    self._period_sector_sales_units[sector_key] += direct_public_units
                    self._distribute_public_education_service_income(sector_key, direct_public_spending)
                government_spent = max(0.0, budget_cap - government_remaining_cash)
                if government_spent <= 0.0:
                    continue
                self.government.treasury_cash = max(0.0, self.government.treasury_cash - government_spent)
                self._period_government_education_spending += government_spent
                self.government.education_spending_this_period += government_spent
                if sector_key == "school":
                    public_school_budget_remaining = max(0.0, public_school_budget_remaining - government_spent)
                    self._period_government_school_spending += government_spent
                    self._period_government_school_units += government_units_bought
                    self.government.school_public_spending_this_period += government_spent
                else:
                    public_university_budget_remaining = max(0.0, public_university_budget_remaining - government_spent)
                    self._period_government_university_spending += government_spent
                    self._period_government_university_units += government_units_bought
                    self.government.university_public_spending_this_period += government_spent
                purchased_units_by_sector[sector_key] += government_units_bought

            family_remaining_cash = max(0.0, cash)
            unmet_basic_essentials = max(0.0, essential_target_units - baseline_essential_units_bought)
            involuntary_retained_cash = (
                family_remaining_cash
                if unmet_basic_essentials > 1e-9 and family_remaining_cash > 0.0
                else 0.0
            )
            voluntary_saved_cash = max(0.0, family_remaining_cash - involuntary_retained_cash)
            self._period_worker_cash_saved += family_remaining_cash
            self._period_worker_voluntary_saved += voluntary_saved_cash
            self._period_worker_involuntary_retained += involuntary_retained_cash
            per_adult_savings = family_remaining_cash / max(1, adult_count)
            allocated_units_by_member = self._allocate_family_consumption_units(
                family_members,
                purchased_units_by_sector,
                target_units_by_member=member_target_units_by_sector,
            )

            for member in family_members:
                allocated_units = allocated_units_by_member.get(member.id, spending_log)
                target_units = member_target_units_by_sector.get(member.id, {})
                coverages = {
                    sector_key: self._coverage_from_units(
                        allocated_units.get(sector_key, 0.0),
                        target_units.get(sector_key, 0.0),
                    )
                    for sector_key in target_units
                }
                self._update_household_post_consumption_state(
                    member,
                    allocated_units=allocated_units,
                    target_units_by_sector=target_units,
                    coverages_by_sector=coverages,
                    per_adult_savings=per_adult_savings,
                    family_remaining_cash=family_remaining_cash,
                    family_basic_basket_cost=family_basic_basket_cost,
                    inflation_pressure=inflation_pressure,
                )

            self._update_family_child_desires(
                family_members,
                essential_target_by_sector=essential_target_by_sector,
                purchased_units_by_sector=purchased_units_by_sector,
                family_remaining_cash=family_remaining_cash,
            )

        self._period_household_summary_cache = None
        self._period_living_wage_anchor_cache = None

    def _consume_entrepreneur(self, owner: Entrepreneur, budget: float) -> float:
        budget = max(0.0, budget)
        available_cash = owner.wealth + owner.vault_cash
        if budget <= 0.0 or available_cash <= 0.0:
            return 0.0

        spendable = min(budget, available_cash)
        if spendable <= 0.0:
            return 0.0

        withdraw_from_wealth = min(owner.wealth, spendable)
        owner.wealth -= withdraw_from_wealth

        withdraw_from_vault = max(0.0, spendable - withdraw_from_wealth)
        if withdraw_from_vault > 0.0:
            withdraw_from_vault = min(owner.vault_cash, withdraw_from_vault)
            owner.vault_cash -= withdraw_from_vault

        cash = spendable

        spending_log = {spec.key: 0.0 for spec in SECTOR_SPECS}
        price_sensitivity = clamp(0.85 + 0.20 * (1.0 - owner.consumption_propensity), 0.55, 1.30)
        sector_weights = {
            "food": 0.24,
            "housing": 0.16,
            "clothing": 0.10,
            "manufactured": 0.20,
            "leisure": 0.14,
            "school": 0.08,
            "university": 0.08,
        }

        for sector_key in ("food", "housing", "clothing", "manufactured", "leisure", "school", "university"):
            if cash <= 0.0:
                break
            average_price = self._average_sector_price(sector_key)
            desired_units = cash * sector_weights[sector_key] / max(0.1, average_price)
            cash, _ = self._purchase_from_sector(
                price_sensitivity,
                sector_key,
                desired_units,
                cash,
                spending_log,
            )

        unspent_cash = max(0.0, cash)
        if unspent_cash > 0.0:
            owner.wealth += unspent_cash

        spent = spendable - unspent_cash
        if spent > 0.0:
            self._period_business_cost_recycled += spent
            self._period_entrepreneur_spending += spent
            self._period_business_cost_to_owners += spent
        return spent

    def _consume_entrepreneurs(self) -> None:
        for owner in self.entrepreneurs:
            if not owner.active:
                continue
            liquid_wealth = owner.wealth + owner.vault_cash
            if liquid_wealth <= 0.0:
                continue
            spend_budget = liquid_wealth * clamp(owner.consumption_propensity, 0.0, 2.0) * max(
                0.0, self.config.entrepreneur_consumption_share
            )
            if spend_budget <= 0.0:
                continue
            self._consume_entrepreneur(owner, spend_budget)

    def _firm_debt_service_due(self, firm: Firm) -> tuple[float, float]:
        if not firm.active or firm.loan_balance <= 1e-9:
            return 0.0, 0.0
        interest_due = firm.loan_balance * self._loan_rate_for_firm(firm)
        principal_share = max(0.0, self.config.firm_loan_principal_payment_share)
        if firm.loan_restructure_grace_periods > 0:
            principal_share = 0.0
        principal_due = firm.loan_balance * principal_share
        return interest_due, min(firm.loan_balance, principal_due)

    def _restructure_firm_loan(self, firm: Firm) -> None:
        if firm.loan_balance <= 1e-9 or firm.loan_restructure_count > 0:
            return
        bank = self._bank_for_id(self._bank_id_for_firm(firm))
        haircut = firm.loan_balance * max(0.0, self.config.firm_loan_restructure_haircut_share)
        firm.loan_balance = max(0.0, firm.loan_balance - haircut)
        firm.loan_restructure_count += 1
        firm.loan_restructure_grace_periods = max(0, self.config.firm_loan_restructure_grace_periods)
        firm.loan_delinquency_periods = 0
        self._period_bank_loan_restructures += 1
        self._period_firm_loan_restructures += 1
        if bank is not None and haircut > 0.0:
            bank.profits -= haircut
            self._period_bank_writeoffs += haircut

    def _service_firm_loans(self, firm: Firm) -> float:
        if not firm.active or firm.loan_balance <= 1e-9:
            firm.last_interest_cost = 0.0
            firm.loan_restructure_count = 0
            firm.loan_default_flag = False
            return 0.0

        interest_due, principal_due = self._firm_debt_service_due(firm)
        total_due = interest_due + principal_due
        firm.last_interest_cost = interest_due
        if total_due <= 1e-9:
            firm.loan_delinquency_periods = 0
            firm.loan_default_flag = False
            return interest_due

        operating_buffer = max(
            0.0,
            (
                firm.last_wage_bill
                + firm.last_input_cost
                + firm.last_transport_cost
                + firm.last_fixed_overhead
            )
            * max(0.0, self.config.firm_debt_service_operating_buffer_share),
        )
        available_cash = max(0.0, firm.cash - operating_buffer)
        payment = min(total_due, available_cash)
        if payment > 0.0:
            firm.cash -= payment

        bank = self._bank_for_id(self._bank_id_for_firm(firm))
        firm.loan_balance, _, _, _, shortfall = self._apply_loan_payment(
            balance=firm.loan_balance,
            interest_due=interest_due,
            principal_due=principal_due,
            payment=payment,
            bank=bank,
        )
        if firm.loan_balance <= 1e-9:
            firm.loan_balance = 0.0
            firm.loan_delinquency_periods = 0
            firm.loan_restructure_count = 0
            firm.loan_restructure_grace_periods = 0
            firm.credit_exclusion_periods = 0
            firm.loan_default_flag = False
            return interest_due

        if shortfall > 1e-6:
            firm.loan_delinquency_periods += 1
        else:
            firm.loan_delinquency_periods = max(0, firm.loan_delinquency_periods - 1)

        if (
            firm.loan_delinquency_periods >= self.config.firm_loan_restructure_delinquency
            and firm.loan_restructure_count == 0
        ):
            self._restructure_firm_loan(firm)
            firm.loan_default_flag = False
        elif firm.loan_delinquency_periods >= self.config.firm_loan_default_delinquency:
            firm.loan_default_flag = True
        else:
            firm.loan_default_flag = False
        return interest_due

    def _settle_firm_default_on_exit(self, firm: Firm) -> None:
        if firm.loan_balance <= 1e-9:
            firm.loan_balance = 0.0
            firm.loan_default_flag = False
            return
        bank = self._bank_for_id(self._bank_id_for_firm(firm))
        remaining_balance = max(0.0, firm.loan_balance)
        cash_recovery = min(remaining_balance, max(0.0, firm.cash))
        if cash_recovery > 0.0:
            firm.cash -= cash_recovery
            remaining_balance -= cash_recovery
        if firm.loan_default_flag or firm.loan_delinquency_periods > 0 or remaining_balance > 0.0:
            self._period_firm_loan_defaults += 1
        if remaining_balance > 0.0:
            self._period_bank_writeoffs += remaining_balance
            if bank is not None:
                bank.profits -= remaining_balance
        firm.loan_balance = 0.0
        firm.loan_delinquency_periods = 0
        firm.loan_restructure_grace_periods = 0
        firm.credit_exclusion_periods = max(
            firm.credit_exclusion_periods,
            self.config.firm_default_credit_cooldown_periods,
        )
        firm.loan_default_flag = False

    def _settle_firms(self) -> None:
        planned_dividends: list[tuple[Firm, Entrepreneur, float]] = []
        planned_investments: list[tuple[Firm, float, float, float]] = []

        for firm in self.firms:
            if not firm.active:
                continue
            revenue = firm.cash - self._cash_before_sales.get(firm.id, firm.cash)
            firm.last_revenue = revenue
            firm.last_sales = firm.sales_this_period
            firm.sales_history.append(firm.last_sales)
            firm.expected_sales_history.append(max(0.0, firm.last_expected_sales))
            firm.production_history.append(max(0.0, firm.last_production))
            max_history = self._firm_market_memory_periods()
            if len(firm.sales_history) > max_history:
                del firm.sales_history[:-max_history]
            if len(firm.expected_sales_history) > max_history:
                del firm.expected_sales_history[:-max_history]
            if len(firm.production_history) > max_history:
                del firm.production_history[:-max_history]

            depreciation = firm.last_capital_charge
            firm.capital = max(0.0, firm.capital - depreciation)
            firm.last_total_cost = (
                firm.last_wage_bill
                + firm.last_wage_bill * self._government_payroll_tax_rate()
                + firm.last_input_cost
                + firm.last_transport_cost
                + firm.last_fixed_overhead
                + depreciation
            )
            carry_cost, waste_cost = self._apply_inventory_carry_and_waste(firm)
            firm.last_total_cost += carry_cost + waste_cost
            if carry_cost > 0.0:
                firm.cash -= carry_cost
                self._queue_sector_payment("housing", carry_cost)
            interest_cost = self._service_firm_loans(firm)
            firm.last_total_cost += interest_cost
            pre_tax_profit = revenue - firm.last_total_cost
            corporate_tax = 0.0
            if pre_tax_profit > 0.0 and self.config.government_enabled:
                corporate_tax_rate = self._government_corporate_tax_rate(
                    pre_tax_profit / max(1.0, revenue)
                )
                corporate_tax = min(firm.cash, pre_tax_profit * corporate_tax_rate)
                if corporate_tax > 0.0:
                    firm.cash -= corporate_tax
                    self._record_government_tax_revenue(corporate_tax, "corporate")
            profit = pre_tax_profit - corporate_tax
            firm.last_profit = profit
            self._period_profit += profit

            if profit > 0.0 and firm.cash > 0.0:
                owner = self.entrepreneurs[firm.owner_id]
                sell_through = self._firm_recent_sell_through(firm)
                revealed_growth_pressure = self._firm_revealed_growth_pressure(
                    firm,
                    use_current_period=True,
                )
                cash_reserve = max(
                    20.0,
                    firm.last_wage_bill * self.config.cash_reserve_periods,
                )
                excess_cash_ratio = clamp(
                    (firm.cash - cash_reserve) / max(1.0, cash_reserve),
                    0.0,
                    3.0,
                )
                macro_stability = self._economy_investment_stability()
                investment_confidence = self._firm_investment_confidence(
                    firm,
                    macro_stability=macro_stability,
                )
                loan_rate = max(0.0, self._loan_rate_for_firm(firm))
                neutral_rate = max(1e-6, self.config.firm_interest_rate_neutral)
                interest_drag = clamp((loan_rate - neutral_rate) / neutral_rate, -0.50, 2.0)
                dividend_cap = max(0.0, firm.cash - 0.75 * cash_reserve)
                effective_payout_ratio = clamp(
                    self.config.payout_ratio
                    + 0.12 * max(0.0, profit / max(1.0, revenue))
                    + 0.10 * excess_cash_ratio
                    - self.config.firm_revealed_shortage_investment_weight * 0.10 * revealed_growth_pressure
                    - 0.05 * max(0.0, firm.cash_conservatism - 1.0),
                    0.18,
                    0.60,
                )
                dividend = min(profit * effective_payout_ratio, dividend_cap)
                if dividend > 0.0:
                    planned_dividends.append((firm, owner, dividend))

                investment_cap = max(0.0, firm.cash - dividend - 0.75 * cash_reserve)
                effective_investment_rate = clamp(
                    self.config.investment_rate
                    + 0.14 * max(0.0, profit / max(1.0, revenue))
                    + 0.10 * excess_cash_ratio
                    + 0.06 * max(0.0, firm.market_share_ambition - 1.0)
                    + self.config.firm_macro_stability_investment_weight * (investment_confidence - 1.0)
                    - self.config.firm_investment_interest_sensitivity * max(0.0, interest_drag),
                    0.15,
                    0.60,
                )
                effective_investment_rate = clamp(
                    effective_investment_rate
                    + self.config.firm_revealed_shortage_investment_weight
                    * (
                        0.14 * revealed_growth_pressure
                        + 0.06 * max(0.0, sell_through - 0.70)
                    ),
                    0.15,
                    0.75,
                )
                investment = min(profit * effective_investment_rate, investment_cap)
                if investment > 0.0:
                    tech_share = clamp(
                        self.rng.uniform(
                            self.config.technology_investment_share_min,
                            self.config.technology_investment_share_max,
                        ),
                        self.config.technology_investment_share_min,
                        self.config.technology_investment_share_max,
                    )
                    if firm.sector in ESSENTIAL_SECTOR_KEYS:
                        tech_share = clamp(tech_share + 0.05, 0.05, 0.60)
                    tech_share = clamp(
                        tech_share
                        + self.config.firm_revealed_shortage_investment_weight * 0.08 * revealed_growth_pressure,
                        0.05,
                        0.70,
                    )
                    technology_investment = investment * tech_share
                    capital_investment = investment - technology_investment
                    planned_investments.append(
                        (
                            firm,
                            investment,
                            capital_investment,
                            technology_investment,
                        )
                    )

        for firm, owner, dividend in planned_dividends:
            gross_dividend = min(dividend, max(0.0, firm.cash))
            if gross_dividend <= 0.0:
                continue
            dividend_tax = gross_dividend * self._government_dividend_tax_rate(gross_dividend)
            dividend_tax = min(gross_dividend, dividend_tax)
            net_dividend = gross_dividend - dividend_tax
            firm.cash -= gross_dividend
            if dividend_tax > 0.0:
                self._record_government_tax_revenue(dividend_tax, "dividend")
            owner_bank_share = max(
                0.0,
                1.0 - self.config.entrepreneur_vault_share,
            )
            owner.wealth += net_dividend * owner_bank_share
            owner.vault_cash += net_dividend * max(0.0, self.config.entrepreneur_vault_share)
            self._period_dividends_paid += net_dividend

        self._consume_entrepreneurs()
        self._collect_government_wealth_tax()

        total_industrial_investment = 0.0
        investment_knowledge_multiplier = self._investment_knowledge_multiplier()
        for firm, investment, capital_investment, technology_investment in planned_investments:
            effective_investment = min(investment, max(0.0, firm.cash))
            if effective_investment <= 0.0:
                continue
            capital_share = capital_investment / investment if investment > 0.0 else 0.0
            actual_capital_investment = effective_investment * capital_share * investment_knowledge_multiplier
            actual_technology_investment = (
                effective_investment * (1.0 - capital_share) * investment_knowledge_multiplier
            )
            firm.cash -= effective_investment
            firm.capital += actual_capital_investment
            firm.technology = clamp(
                firm.technology * (1.0 - self.config.technology_depreciation_rate),
                0.75,
                self.config.technology_cap,
            )
            technology_boost = self.rng.uniform(
                self.config.technology_gain_min,
                self.config.technology_gain_max,
            )
            technology_gain = technology_boost * investment_knowledge_multiplier * actual_technology_investment / max(
                1.0, firm.last_wage_bill + firm.last_fixed_overhead + 1.0
            )
            firm.technology = clamp(
                firm.technology * (1.0 + technology_gain),
                0.75,
                self.config.technology_cap,
            )
            firm.last_technology_investment = actual_technology_investment
            firm.last_technology_gain = technology_gain
            self._period_investment_spending += effective_investment
            total_industrial_investment += effective_investment

        self._distribute_payment_to_sector(
            "manufactured",
            total_industrial_investment,
            book_profit=True,
        )

        for firm in self.firms:
            if not firm.active:
                continue
            if firm.last_profit < 0.0:
                firm.loss_streak += 1
                if firm.loss_streak >= self._firm_adaptation_threshold(firm):
                    self._adapt_losing_firm(firm)
            else:
                firm.loss_streak = 0

            if firm.last_production > 0.0 and firm.price < firm.last_unit_cost:
                utilization_ratio = firm.last_production / max(1.0, firm.last_expected_sales)
                if utilization_ratio < 0.35 and firm.cash >= self.config.bankruptcy_cash_threshold:
                    firm.bankruptcy_streak += 1
                else:
                    # If the firm is still moving product and keeping cash above the hard threshold,
                    # treat underpricing as a recoverable pricing error instead of immediate failure.
                    firm.bankruptcy_streak = max(0, firm.bankruptcy_streak - 1)
            elif firm.last_profit < 0.0 or firm.cash < self.config.bankruptcy_cash_threshold:
                firm.bankruptcy_streak += 1
            else:
                firm.bankruptcy_streak = 0

            firm.age += 1
            self._update_firm_demand_learning(firm)

        sector_sales_totals: dict[str, float] = {spec.key: 0.0 for spec in SECTOR_SPECS}
        for firm in self.firms:
            if firm.active:
                sector_sales_totals[firm.sector] += firm.last_sales
        for firm in self.firms:
            if not firm.active:
                firm.last_market_share = 0.0
                continue
            total_sales = sector_sales_totals.get(firm.sector, 0.0)
            firm.last_market_share = firm.last_sales / total_sales if total_sales > 0.0 else 0.0

    def _resolve_bankruptcy_and_entry(self) -> None:
        for firm in self.firms:
            if not firm.active:
                continue
            dynamic_bankruptcy_limit = self._firm_bankruptcy_limit(firm)
            grace_cash_limit, critical_cash_limit = self._firm_cash_failure_limits(firm)
            should_fail = (
                firm.cash < critical_cash_limit
                or firm.loan_default_flag
                or (
                    firm.age >= self.config.bankruptcy_grace_period
                    and (
                        firm.cash < grace_cash_limit
                        or firm.bankruptcy_streak >= dynamic_bankruptcy_limit
                    )
                )
            )
            if not should_fail:
                continue

            self._bankruptcies += 1
            self._settle_firm_default_on_exit(firm)
            owner = self.entrepreneurs[firm.owner_id]
            owner.wealth += firm.cash
            if firm.cash > 0.0:
                self._period_bankruptcy_cash_recoveries += firm.cash

            firm.active = False
            for worker_id in firm.workers:
                self._release_household_from_employment(self.households[worker_id])
            firm.workers.clear()
            firm.cash = 0.0
            firm.inventory = 0.0
            firm.inventory_batches.clear()
            firm.capital = 0.0
            firm.education_level_span = 0.0
            firm.desired_workers = 0
            firm.target_inventory = 0.0
            firm.sales_this_period = 0.0
            firm.last_sales = 0.0
            firm.last_revenue = 0.0
            firm.last_production = 0.0
            firm.last_profit = 0.0
            firm.last_wage_bill = 0.0
            firm.last_input_cost = 0.0
            firm.last_transport_cost = 0.0
            firm.last_fixed_overhead = 0.0
            firm.last_capital_charge = 0.0
            firm.last_inventory_carry_cost = 0.0
            firm.last_inventory_waste_cost = 0.0
            firm.last_unit_cost = 0.0
            firm.last_market_share = 0.0
            firm.sales_history.clear()
            firm.expected_sales_history.clear()
            firm.production_history.clear()
            firm.last_expected_sales = 0.0
            firm.market_fragility_belief = 0.0
            firm.forecast_error_belief = 0.0
            firm.last_technology_investment = 0.0
            firm.last_technology_gain = 0.0
            firm.last_interest_cost = 0.0
            firm.labor_offer_rejections = 0
            firm.labor_offer_rejection_wage_floor = 0.0
            firm.last_labor_offer_rejections = 0
            firm.last_labor_offer_rejection_wage_floor = 0.0
            firm.technology = 0.0
            firm.demand_elasticity = 0.0
            firm.loan_restructure_count = 0
            firm.credit_exclusion_periods = max(
                firm.credit_exclusion_periods,
                self.config.firm_default_credit_cooldown_periods,
            )
            firm.loan_default_flag = False
            firm.bankruptcy_streak = 0
            firm.loss_streak = 0

        self._refresh_period_sector_caches()
        if self.config.replacement_enabled:
            self._attempt_endogenous_sector_entry()
            self._refresh_period_sector_caches()
        self._ensure_active_food_input_exemption()

    def _append_inactive_firm_slot(self, sector_key: str) -> Firm:
        spec = SECTOR_BY_KEY[sector_key]
        firm = Firm(
            id=self._next_firm_id,
            sector=sector_key,
            owner_id=0,
            cash=0.0,
            inventory=0.0,
            capital=0.0,
            price=spec.base_price,
            wage_offer=spec.base_wage,
            productivity=spec.base_productivity,
            active=False,
            demand_elasticity=self._initial_demand_elasticity(sector_key),
            **self._random_firm_behavior_traits(spec),
        )
        self._next_firm_id += 1
        self.firms.append(firm)
        self.firm_by_id[firm.id] = firm
        self.firms_by_sector.setdefault(sector_key, []).append(firm)
        return firm

    def _attempt_endogenous_sector_entry(self) -> None:
        for spec in SECTOR_SPECS:
            if self._in_essential_protection() and spec.key not in ESSENTIAL_SECTOR_KEYS:
                last_essential_fulfillment = self.history[-1].essential_fulfillment_rate if self.history else 0.0
                if last_essential_fulfillment < 0.95:
                    continue
            revealed_unmet_units = self._period_sector_revealed_unmet_units.get(spec.key, 0.0)
            revealed_entry_pressure = self._sector_revealed_expansion_pressure(
                spec.key,
                use_current_period=True,
            )
            demand_signal = self._baseline_demand(spec.key, use_current_period=True)
            if spec.key in ("school", "university"):
                prior_demand_signal = self._baseline_demand(spec.key, use_current_period=False)
                demand_signal = 0.55 * demand_signal + 0.45 * prior_demand_signal
            observed_market_signal = self._observed_sector_demand_signal(spec.key, use_current_period=True)
            demand_signal = max(
                demand_signal,
                observed_market_signal
                * (
                    1.0
                    + self.config.firm_revealed_shortage_entry_weight
                    * 0.12
                    * revealed_entry_pressure
                ),
            )
            if demand_signal <= 0.0:
                continue
            active_supply = sum(
                self._firm_effective_supply_signal(firm, demand_signal)
                for firm in self._sector_firms(spec.key)
            )
            if spec.key in ("school", "university"):
                active_supply += self._public_education_supply_signal(spec.key, use_current_period=False)
            supply_coverage_target = clamp(
                0.85
                - 0.15
                * self.config.firm_revealed_shortage_entry_weight
                * revealed_entry_pressure,
                0.55,
                0.85,
            )
            entry_gap = demand_signal - supply_coverage_target * active_supply
            entry_gap += (
                self.config.firm_revealed_shortage_entry_weight
                * 0.35
                * revealed_unmet_units
            )
            if entry_gap <= 1.0:
                continue
            gap_ratio = entry_gap / max(1.0, demand_signal)
            max_new_entries = 1
            if gap_ratio >= 0.30:
                max_new_entries = 2
            if gap_ratio >= 0.60:
                max_new_entries = 3
            for _ in range(max_new_entries):
                revealed_unmet_units = self._period_sector_revealed_unmet_units.get(spec.key, 0.0)
                revealed_entry_pressure = self._sector_revealed_expansion_pressure(
                    spec.key,
                    use_current_period=True,
                )
                demand_signal = self._baseline_demand(spec.key, use_current_period=True)
                if spec.key in ("school", "university"):
                    prior_demand_signal = self._baseline_demand(spec.key, use_current_period=False)
                    demand_signal = 0.55 * demand_signal + 0.45 * prior_demand_signal
                observed_market_signal = self._observed_sector_demand_signal(spec.key, use_current_period=True)
                demand_signal = max(
                    demand_signal,
                    observed_market_signal
                    * (
                        1.0
                        + self.config.firm_revealed_shortage_entry_weight
                        * 0.12
                        * revealed_entry_pressure
                    ),
                )
                if demand_signal <= 0.0:
                    break
                active_supply = sum(
                    self._firm_effective_supply_signal(firm, demand_signal)
                    for firm in self._sector_firms(spec.key)
                )
                if spec.key in ("school", "university"):
                    active_supply += self._public_education_supply_signal(spec.key, use_current_period=False)
                supply_coverage_target = clamp(
                    0.85
                    - 0.15
                    * self.config.firm_revealed_shortage_entry_weight
                    * revealed_entry_pressure,
                    0.55,
                    0.85,
                )
                entry_gap = demand_signal - supply_coverage_target * active_supply
                entry_gap += (
                    self.config.firm_revealed_shortage_entry_weight
                    * 0.35
                    * revealed_unmet_units
                )
                if entry_gap <= 1.0:
                    break
                inactive_firms = [firm for firm in self.firms_by_sector.get(spec.key, []) if not firm.active]
                entry_firm = (
                    min(inactive_firms, key=lambda firm: firm.id)
                    if inactive_firms
                    else self._append_inactive_firm_slot(spec.key)
                )
                if not self._restart_firm(
                    entry_firm,
                    demand_signal=demand_signal,
                    entry_gap=entry_gap,
                ):
                    break

    def _restart_firm(
        self,
        firm: Firm,
        demand_signal: float | None = None,
        entry_gap: float | None = None,
    ) -> bool:
        spec = SECTOR_BY_KEY[firm.sector]
        target_demand = max(
            0.0,
            demand_signal if demand_signal is not None else self._baseline_demand(spec.key, use_current_period=True),
        )
        if target_demand <= 0.0:
            return False

        base_restart_cost = self._restart_funding_need(spec, 1.0, demand_units=target_demand)
        if base_restart_cost <= 0.0:
            return False

        revealed_entry_pressure = self._sector_revealed_expansion_pressure(
            spec.key,
            use_current_period=True,
        )
        perceived_gap = max(0.0, entry_gap if entry_gap is not None else target_demand)
        perceived_gap *= (
            1.0
            + self.config.firm_revealed_shortage_entry_weight
            * 0.20
            * revealed_entry_pressure
        )
        owner = self._select_entry_owner(
            spec.key,
            target_demand,
            perceived_gap,
            base_restart_cost,
        )
        if owner is None:
            return False

        available_surplus = self._owner_total_liquid(owner) - self.config.firm_restart_wealth_threshold
        if available_surplus <= 0.0:
            return False

        scale = available_surplus / base_restart_cost
        if scale < self.config.firm_restart_min_scale:
            return False
        scale *= (
            1.0
            + self.config.firm_revealed_shortage_entry_weight
            * 0.10
            * revealed_entry_pressure
        )
        scale = clamp(scale, self.config.firm_restart_min_scale, self.config.firm_restart_max_scale)
        restart_cost = min(
            available_surplus,
            self._restart_funding_need(spec, scale, demand_units=target_demand),
        )
        self._withdraw_owner_liquid(owner, restart_cost)
        package_multiplier = self.config.firm_restart_package_multiplier * (
            1.0
            + self.config.firm_revealed_shortage_entry_weight
            * 0.20
            * revealed_entry_pressure
        )
        effective_scale = scale * package_multiplier
        effective_demand = target_demand * effective_scale
        base_cash_budget, base_capital_budget, base_inventory_budget = self._entry_package_budgets(
            spec,
            effective_demand,
            effective_scale,
        )
        base_package_cost = max(1e-9, base_cash_budget + base_capital_budget + base_inventory_budget)
        package_funding_ratio = clamp(restart_cost / base_package_cost, 0.0, 1.0)
        startup_capital = base_capital_budget * package_funding_ratio
        startup_inventory_budget = base_inventory_budget * package_funding_ratio
        startup_cash = max(0.0, restart_cost - startup_capital - startup_inventory_budget)
        startup_inventory_units = startup_inventory_budget / max(0.1, spec.base_price)
        industrial_procurement_spending = startup_capital + startup_inventory_budget
        self._period_startup_fixed_capital_formation += startup_capital
        self._period_startup_inventory_investment += startup_inventory_budget

        firm.active = True
        firm.age = 0
        firm.owner_id = owner.id
        firm.cash = startup_cash
        firm.capital = startup_capital
        living_wage_anchor = self._living_wage_anchor()
        firm.wage_offer = max(
            spec.base_wage * self.rng.uniform(0.96, 1.04),
            living_wage_anchor * (
                self.config.reservation_wage_floor_share + self._sector_wage_floor_premium(spec.key)
            ),
        )
        firm.productivity = (
            spec.base_productivity
            * self._entry_productivity_multiplier(spec.key)
            * self.rng.uniform(0.96, 1.04)
        )
        education_blueprint: dict[str, float] | None = None
        if self._is_education_sector(spec.key):
            education_blueprint = self._draw_education_blueprint(spec, effective_demand, effective_scale)
            firm.productivity = education_blueprint["students_per_worker"]
        firm.technology = self._initial_technology(spec.key)
        firm.demand_elasticity = self._initial_demand_elasticity(spec.key)
        if education_blueprint is not None:
            firm.capital = education_blueprint["capital_budget"] * package_funding_ratio
            firm.input_cost_per_unit = education_blueprint["input_cost_per_unit"]
            firm.transport_cost_per_unit = education_blueprint["transport_cost_per_unit"]
            firm.fixed_overhead = education_blueprint["fixed_overhead"]
            firm.education_level_span = education_blueprint["level_span"]
        else:
            firm.input_cost_per_unit, firm.transport_cost_per_unit, firm.fixed_overhead = self._random_firm_cost_structure(spec)
            if firm.input_cost_exempt and spec.key == "food":
                firm.input_cost_per_unit = 0.0
            firm.education_level_span = 0.0
        firm.inventory = max(0.0, startup_inventory_units)
        firm.inventory_batches = [] if education_blueprint is not None else [max(0.0, firm.inventory)]

        expected_sales = effective_demand * self.rng.uniform(0.9, 1.05) * max(0.10, package_funding_ratio)
        firm.last_worker_count = 0
        firm.sales_this_period = 0.0
        effective_productivity = firm.productivity * firm.technology * self._capital_efficiency(firm.capital)
        if education_blueprint is not None:
            firm.inventory = min(
                firm.inventory,
                self._education_firm_capacity(firm),
            )
            firm.inventory_batches = []
            target_service_units = self._education_service_target_units(firm, expected_sales)
            desired_workers = self._workers_needed_for_units(
                target_service_units,
                effective_productivity,
                productivity_floor=0.25,
            )
        else:
            desired_output = self._firm_desired_output_from_expected_sales(firm, expected_sales)
            desired_workers = self._workers_needed_for_units(desired_output, effective_productivity)
        last_wage_bill = desired_workers * firm.wage_offer
        capital_charge = firm.capital * self.config.depreciation_rate
        last_input_cost = expected_sales * firm.input_cost_per_unit
        last_transport_cost = expected_sales * firm.transport_cost_per_unit
        last_total_cost = (
            last_wage_bill
            + last_wage_bill * self._government_payroll_tax_rate()
            + last_input_cost
            + last_transport_cost
            + firm.fixed_overhead
            + capital_charge
        )
        unit_cost = last_total_cost / max(1.0, expected_sales)
        traits = self._random_firm_behavior_traits(spec)
        firm.markup_tolerance = traits["markup_tolerance"]
        firm.volume_preference = traits["volume_preference"]
        firm.inventory_aversion = traits["inventory_aversion"]
        firm.employment_inertia = traits["employment_inertia"]
        firm.price_aggressiveness = traits["price_aggressiveness"]
        firm.cash_conservatism = traits["cash_conservatism"]
        firm.market_share_ambition = traits["market_share_ambition"]
        firm.expansion_credit_appetite = traits["expansion_credit_appetite"]
        firm.stability_sensitivity = traits["stability_sensitivity"]
        firm.investment_animal_spirits = traits["investment_animal_spirits"]
        firm.forecast_caution = traits["forecast_caution"]
        firm.price = self._initial_firm_price(spec, unit_cost)
        self._refresh_firm_startup_state(firm, spec, expected_sales)
        firm.last_market_share = 0.0
        firm.sales_history = [expected_sales]
        firm.expected_sales_history = [expected_sales]
        firm.production_history = [firm.last_production]
        firm.market_fragility_belief = clamp(
            0.45 * self._sector_public_fragility_signal(spec.key)
            + 0.35 * self._social_survival_fragility_signal(spec.key)
            + 0.20 * self._economy_public_fragility_signal(),
            0.0,
            1.5,
        )
        firm.forecast_error_belief = self._initial_forecast_error_belief(spec.key)
        firm.last_technology_investment = 0.0
        firm.last_technology_gain = 0.0
        firm.last_total_cost = last_total_cost
        firm.last_interest_cost = 0.0
        firm.loan_balance = 0.0
        firm.loan_delinquency_periods = 0
        firm.loan_restructure_count = 0
        firm.loan_restructure_grace_periods = 0
        firm.credit_exclusion_periods = 0
        firm.loan_default_flag = False
        firm.loss_streak = 0
        self._distribute_payment_to_sector(
            "manufactured",
            industrial_procurement_spending,
            book_profit=True,
        )
        return True

    def _build_snapshot(self) -> PeriodSnapshot:
        periods_per_year = max(1, self.config.periods_per_year)
        year = ((self.period - 1) // periods_per_year) + 1
        period_in_year = ((self.period - 1) % periods_per_year) + 1
        (
            population,
            women,
            men,
            fertile_women,
            children,
            adults,
            seniors,
            labor_force,
            employment_rate,
            unemployment_rate,
            average_age,
        ) = self._population_metrics()
        active_households = self._active_households()
        children_with_guardian = sum(
            1
            for household in active_households
            if self._household_age_years(household) < self.config.entry_age_years
            and any(
                parent is not None and parent.alive
                for parent in (
                    self._household_by_id(household.guardian_id),
                    self._household_by_id(household.mother_id),
                    self._household_by_id(household.father_id),
                )
            )
        )
        orphans = sum(
            1
            for household in active_households
            if self._household_age_years(household) < self.config.entry_age_years
            and (
                all(
                    parent is None or not parent.alive
                    for parent in (
                        self._household_by_id(household.guardian_id),
                        self._household_by_id(household.mother_id),
                        self._household_by_id(household.father_id),
                    )
                )
            )
        )
        (
            family_units,
            average_family_income,
            average_family_resources,
            average_family_basic_basket_cost,
            family_income_to_basket_ratio,
            family_resources_to_basket_ratio,
            families_income_below_basket_share,
            families_resources_below_basket_share,
        ) = self._family_economic_metrics()
        (
            fertile_families,
            fertile_families_with_births,
            fertile_capable_families,
            fertile_capable_families_low_desire_no_birth,
            fertile_capable_families_with_births,
        ) = self._family_reproductive_metrics()
        (
            fertile_capable_women,
            fertile_capable_women_low_desire_no_birth,
            fertile_capable_women_with_births,
        ) = self._fertile_women_reproductive_metrics()
        total_household_share = sum(spec.household_demand_share for spec in SECTOR_SPECS)
        price_index = sum(
            (self._average_sector_price(spec.key) / spec.base_price) * spec.household_demand_share
            for spec in SECTOR_SPECS
        ) / max(1e-9, total_household_share)
        active_essential_firms = [
            firm for firm in self.firms if firm.active and firm.sector in ESSENTIAL_SECTOR_KEYS
        ]
        total_capital_stock = sum(firm.capital for firm in self.firms if firm.active)
        total_inventory_units = sum(firm.inventory for firm in self.firms if firm.active)
        essential_inventory_units = sum(max(0.0, firm.inventory) for firm in active_essential_firms)
        essential_target_inventory_units = sum(max(0.0, firm.target_inventory) for firm in active_essential_firms)
        essential_expected_sales_units = sum(max(0.0, firm.last_expected_sales) for firm in active_essential_firms)
        essential_total_sales_units = sum(
            max(0.0, self._period_sector_sales_units.get(sector_key, 0.0))
            for sector_key in ESSENTIAL_SECTOR_KEYS
        )
        essential_government_sales_units = max(
            0.0,
            essential_total_sales_units - self._period_essential_sales_units,
        )
        total_inventory_book_value = self._current_inventory_book_value()
        goods_monetary_mass = self._current_goods_monetary_mass()
        total_liquid_money = self._current_total_liquid_money()
        worker_bank_deposits = sum(household.savings for household in active_households)
        worker_credit_outstanding = sum(max(0.0, household.loan_balance) for household in active_households)
        household_delinquent_loans = sum(
            max(0.0, household.loan_balance)
            for household in active_households
            if household.loan_delinquency_periods > 0
        )
        household_nonperforming_loans = sum(
            max(0.0, household.loan_balance)
            for household in active_households
            if household.loan_delinquency_periods >= self.config.household_loan_restructure_delinquency
        )
        capitalist_bank_deposits = sum(entrepreneur.wealth for entrepreneur in self.entrepreneurs)
        capitalist_vault_cash = sum(entrepreneur.vault_cash for entrepreneur in self.entrepreneurs)
        capitalist_firm_cash = sum(firm.cash for firm in self.firms if firm.active)
        capitalist_credit_outstanding = sum(max(0.0, firm.loan_balance) for firm in self.firms if firm.active)
        firm_delinquent_loans = sum(
            max(0.0, firm.loan_balance)
            for firm in self.firms
            if firm.active and firm.loan_delinquency_periods > 0
        )
        firm_nonperforming_loans = sum(
            max(0.0, firm.loan_balance)
            for firm in self.firms
            if firm.active and firm.loan_delinquency_periods >= self.config.firm_loan_restructure_delinquency
        )
        capitalist_productive_capital = sum(firm.capital for firm in self.firms if firm.active)
        capitalist_inventory_value = sum(firm.inventory * firm.price for firm in self.firms if firm.active)
        capitalist_controlled_assets = (
            capitalist_bank_deposits
            + capitalist_vault_cash
            + capitalist_firm_cash
            + capitalist_productive_capital
            + capitalist_inventory_value
        )
        capitalist_liquid_assets = capitalist_bank_deposits + capitalist_vault_cash + capitalist_firm_cash
        worker_liquid_assets = sum(household.savings + household.wage_income for household in active_households)
        capitalist_asset_share = capitalist_controlled_assets / max(
            1e-9,
            capitalist_controlled_assets + worker_bank_deposits,
        )
        capitalist_liquid_share = capitalist_liquid_assets / max(
            1e-9,
            capitalist_liquid_assets + worker_liquid_assets,
        )
        worker_liquid_share = 1.0 - capitalist_liquid_share
        gdp_nominal = self._period_sales_revenue + self._period_investment_spending
        prior_inventory_book_value = (
            self.history[-1].total_inventory_book_value
            if self.history
            else self._startup_inventory_book_value
        )
        worker_consumption_spending = max(
            0.0,
            self._period_worker_cash_available - self._period_worker_cash_saved,
        )
        household_final_consumption = worker_consumption_spending + self._period_entrepreneur_spending
        government_final_consumption = (
            self._period_government_procurement_spending
            + self._period_government_education_spending
            + self._period_government_public_administration_spending
        )
        gross_fixed_capital_formation = (
            self._period_investment_spending
            + self._period_startup_fixed_capital_formation
            + self._period_government_public_capital_formation
        )
        change_in_inventories = (
            total_inventory_book_value - prior_inventory_book_value + self._period_startup_inventory_investment
        )
        valuables_acquisition = 0.0
        gross_capital_formation = gross_fixed_capital_formation + change_in_inventories + valuables_acquisition
        exports = 0.0
        imports = 0.0
        net_exports = exports - imports
        gdp_expenditure_sna = (
            household_final_consumption
            + government_final_consumption
            + gross_capital_formation
            + net_exports
        )
        gdp_expenditure_gap = gdp_nominal - gdp_expenditure_sna
        gdp_deflator = self._gdp_deflator_estimate(
            gdp_nominal=gdp_nominal,
            household_final_consumption=household_final_consumption,
            government_final_consumption=government_final_consumption,
            gross_fixed_capital_formation=gross_fixed_capital_formation,
            change_in_inventories=change_in_inventories,
            net_exports=net_exports,
            price_index=price_index,
            government_procurement_spending=self._period_government_procurement_spending,
            government_school_spending=self._period_government_school_spending,
            government_university_spending=self._period_government_university_spending,
            government_public_administration_spending=self._period_government_public_administration_spending,
        )
        gdp_per_capita = gdp_nominal / max(1, population)
        wage_earner_incomes = sorted(
            household.last_income
            for household in active_households
            if household.last_income > 0.0
        )
        median_wage = (
            wage_earner_incomes[len(wage_earner_incomes) // 2]
            if wage_earner_incomes
            else 0.0
        )
        demand_fulfillment_rate = self._period_sales_units / max(1.0, self._period_potential_demand_units)
        essential_fulfillment_rate = self._period_essential_sales_units / max(1.0, self._period_essential_demand_units)
        money_velocity = gdp_nominal / max(1e-9, total_liquid_money)
        central_bank_target_money_supply = self._period_central_bank_target_money_supply
        central_bank_monetary_gap_share = (
            (central_bank_target_money_supply - total_liquid_money)
            / max(1e-9, central_bank_target_money_supply)
        )
        gdp_denom = max(1e-9, gdp_nominal)
        retained_profit = max(
            0.0,
            self._period_profit - self._period_dividends_paid - self._period_investment_spending,
        )
        active_banks = [bank for bank in self.banks if bank.active]
        bank_deposits_by_id = {bank.id: 0.0 for bank in active_banks}
        bank_loans_households_by_id = {bank.id: 0.0 for bank in active_banks}
        bank_loans_firms_by_id = {bank.id: 0.0 for bank in active_banks}
        for household in active_households:
            bank_id = self._bank_id_for_household(household)
            if bank_id in bank_deposits_by_id:
                bank_deposits_by_id[bank_id] += max(0.0, household.savings)
                bank_loans_households_by_id[bank_id] += max(0.0, household.loan_balance)
        for owner in self.entrepreneurs:
            bank_id = self._bank_id_for_entrepreneur(owner)
            if bank_id in bank_deposits_by_id:
                bank_deposits_by_id[bank_id] += max(0.0, owner.wealth)
        for firm in self.firms:
            if not firm.active:
                continue
            bank_id = self._bank_id_for_firm(firm)
            if bank_id in bank_deposits_by_id:
                bank_deposits_by_id[bank_id] += max(0.0, firm.cash)
                bank_loans_firms_by_id[bank_id] += max(0.0, firm.loan_balance)
        if self.government.bank_id in bank_deposits_by_id:
            bank_deposits_by_id[self.government.bank_id] += max(0.0, self.government.treasury_cash)
        average_bank_deposit_rate = sum(bank.deposit_rate for bank in active_banks) / max(1, len(active_banks))
        average_bank_loan_rate = sum(bank.loan_rate for bank in active_banks) / max(1, len(active_banks))
        average_bank_reserve_ratio = sum(bank.reserve_ratio for bank in active_banks) / max(1, len(active_banks))
        total_bank_deposits = sum(bank_deposits_by_id.values())
        total_bank_reserves = sum(bank.reserves for bank in active_banks)
        total_bank_loans_households = sum(bank_loans_households_by_id.values())
        total_bank_loans_firms = sum(bank_loans_firms_by_id.values())
        total_bank_nonperforming_loans = household_nonperforming_loans + firm_nonperforming_loans
        total_bank_bond_holdings = sum(bank.bond_holdings for bank in active_banks)
        total_central_bank_borrowing = sum(bank.central_bank_borrowing for bank in active_banks)
        total_bank_assets = (
            total_bank_reserves
            + total_bank_loans_households
            + total_bank_loans_firms
            + total_bank_bond_holdings
        )
        total_bank_liabilities = total_bank_deposits + total_central_bank_borrowing
        bank_equity = total_bank_assets - total_bank_liabilities
        total_required_reserves = sum(
            bank.reserve_ratio * bank_deposits_by_id.get(bank.id, 0.0) for bank in active_banks
        )
        insolvent_banks = sum(
            1
            for bank in active_banks
            if (
                bank.reserves
                + bank_loans_households_by_id.get(bank.id, 0.0)
                + bank_loans_firms_by_id.get(bank.id, 0.0)
                + bank.bond_holdings
            )
            < bank_deposits_by_id.get(bank.id, 0.0) + bank.central_bank_borrowing
        )
        insolvent_share = max(
            insolvent_banks / max(1, len(active_banks)),
            self._period_bank_insolvent_share_signal,
        )
        undercapitalized_banks = sum(
            1
            for bank in active_banks
            if self._bank_capital_ratio(bank) < self._bank_warning_capital_ratio()
        )
        undercapitalized_share = max(
            undercapitalized_banks / max(1, len(active_banks)),
            self._period_bank_undercapitalized_share_signal,
        )
        bank_capital_ratio = bank_equity / max(1e-9, total_bank_assets)
        bank_asset_liability_ratio = total_bank_assets / max(1e-9, total_bank_liabilities)
        bank_reserve_coverage_ratio = (
            total_bank_reserves / max(1e-9, total_required_reserves) if total_required_reserves > 0.0 else 1.0
        )
        bank_liquidity_ratio = (
            (total_bank_reserves + total_bank_bond_holdings) / max(1e-9, total_bank_deposits)
        )
        bank_loan_to_deposit_ratio = (
            (total_bank_loans_households + total_bank_loans_firms) / max(1e-9, total_bank_deposits)
        )
        household_credit_creation = self._period_household_credit_issued
        firm_credit_creation = self._period_firm_credit_issued
        commercial_bank_credit_creation = household_credit_creation + firm_credit_creation
        labor_force_households = [
            household
            for household in active_households
            if self._household_labor_capacity(household) > 0.0
        ]
        average_worker_savings = sum(household.savings for household in labor_force_households) / max(
            1, len(labor_force_households)
        )
        food_sufficient_count = 0
        food_subsistence_count = 0
        food_acute_hunger_count = 0
        food_severe_hunger_count = 0
        people_full_essential_coverage = 0
        total_food_meals = 0.0
        total_health_fragility = 0.0
        total_perceived_utility = 0.0
        for household in active_households:
            meals = self._household_food_meals_consumed(household)
            total_food_meals += meals
            total_health_fragility += household.health_fragility
            total_perceived_utility += household.last_perceived_utility
            sufficient_meals = self._household_food_sufficient_meals(household)
            subsistence_meals = self._household_food_subsistence_meals(household)
            severe_meals = self._household_food_severe_hunger_meals(household)
            if all(self._household_sector_coverage(household, sector_key) >= 1.0 for sector_key in ESSENTIAL_SECTOR_KEYS):
                people_full_essential_coverage += 1
            if meals >= sufficient_meals:
                food_sufficient_count += 1
            elif meals >= subsistence_meals:
                food_subsistence_count += 1
            elif meals >= severe_meals:
                food_acute_hunger_count += 1
            else:
                food_severe_hunger_count += 1
        average_food_meals_per_person = total_food_meals / max(1, population)
        average_health_fragility = total_health_fragility / max(1, population)
        average_perceived_utility = total_perceived_utility / max(1, population)
        full_essential_coverage_share = people_full_essential_coverage / max(1, population)
        adult_households = [
            household
            for household in active_households
            if self._household_age_years(household) >= self.config.entry_age_years
        ]
        (
            _family_income_ratio_by_household,
            family_resources_ratio_by_household,
            family_resources_below_basket_by_household,
        ) = self._family_status_maps()
        school_age_population = sum(1 for household in active_households if self._is_school_age(household))
        university_age_population = sum(
            1
            for household in active_households
            if self._is_university_age(household)
            and not self._household_has_university_credential(household)
            and self._household_sector_desired_units(household, "university") > 0.0
        )
        school_students = sum(
            1
            for household in active_households
            if self._is_school_age(household)
            and self._household_sector_desired_units(household, "school") > 0.0
            and self._household_sector_coverage(household, "school") >= 0.5
        )
        university_students = sum(
            1
            for household in active_households
            if self._is_university_age(household)
            and self._household_sector_desired_units(household, "university") > 0.0
            and self._household_sector_coverage(household, "university") >= 0.5
        )
        school_completion_share = sum(
            1 for household in adult_households if self._household_has_school_credential(household)
        ) / max(1, len(adult_households))
        university_completion_share = sum(
            1 for household in adult_households if self._household_has_university_credential(household)
        ) / max(1, len(adult_households))
        school_labor_share = sum(
            1 for household in labor_force_households if self._household_has_school_credential(household)
        ) / max(1, len(labor_force_households))
        skilled_labor_share = sum(
            1 for household in labor_force_households if self._household_has_university_credential(household)
        ) / max(1, len(labor_force_households))
        low_resource_school_age_population = sum(
            1
            for household in active_households
            if self._is_school_age(household)
            and family_resources_below_basket_by_household.get(household.id, False)
        )
        low_resource_university_age_population = sum(
            1
            for household in active_households
            if self._is_university_age(household)
            and not self._household_has_university_credential(household)
            and self._household_sector_desired_units(household, "university") > 0.0
            and family_resources_below_basket_by_household.get(household.id, False)
        )
        low_resource_school_students = sum(
            1
            for household in active_households
            if self._is_school_age(household)
            and self._household_sector_desired_units(household, "school") > 0.0
            and self._household_sector_coverage(household, "school") >= 0.5
            and family_resources_below_basket_by_household.get(household.id, False)
        )
        low_resource_university_students = sum(
            1
            for household in active_households
            if self._is_university_age(household)
            and self._household_sector_desired_units(household, "university") > 0.0
            and self._household_sector_coverage(household, "university") >= 0.5
            and family_resources_below_basket_by_household.get(household.id, False)
        )

        def _average_last_income(households: list[Household]) -> float:
            if not households:
                return 0.0
            return sum(max(0.0, household.last_income) for household in households) / len(households)

        def _premium(numerator_group: list[Household], denominator_group: list[Household]) -> float:
            numerator = _average_last_income(numerator_group)
            denominator = _average_last_income(denominator_group)
            if numerator <= 0.0 and denominator <= 0.0:
                return 1.0
            if denominator <= 0.0:
                return 1.0 + numerator
            return numerator / denominator

        educated_labor_force = [
            household for household in labor_force_households if self._household_has_school_credential(household)
        ]
        uneducated_labor_force = [
            household for household in labor_force_households if not self._household_has_school_credential(household)
        ]
        university_labor_force = [
            household for household in labor_force_households if self._household_has_university_credential(household)
        ]
        nonuniversity_labor_force = [
            household for household in labor_force_households if not self._household_has_university_credential(household)
        ]
        school_income_premium = _premium(educated_labor_force, uneducated_labor_force)
        university_income_premium = _premium(university_labor_force, nonuniversity_labor_force)
        poverty_rate_without_university = sum(
            1
            for household in adult_households
            if not self._household_has_university_credential(household)
            and family_resources_ratio_by_household.get(household.id, 0.0) < 1.0
        ) / max(
            1,
            sum(1 for household in adult_households if not self._household_has_university_credential(household)),
        )
        poverty_rate_with_university = sum(
            1
            for household in adult_households
            if self._household_has_university_credential(household)
            and family_resources_ratio_by_household.get(household.id, 0.0) < 1.0
        ) / max(
            1,
            sum(1 for household in adult_households if self._household_has_university_credential(household)),
        )
        tracked_origin_adults = [
            household for household in adult_households if household.origin_record_period >= 0
        ]
        low_resource_origin_adults = [
            household for household in tracked_origin_adults if household.low_resource_origin
        ]
        low_resource_origin_upwardly_mobile = [
            household
            for household in low_resource_origin_adults
            if family_resources_ratio_by_household.get(household.id, 0.0) >= 1.0
        ]
        low_resource_origin_university_adults = [
            household
            for household in low_resource_origin_adults
            if self._household_has_university_credential(household)
        ]
        low_resource_origin_nonuniversity_adults = [
            household
            for household in low_resource_origin_adults
            if not self._household_has_university_credential(household)
        ]
        low_resource_origin_university_upwardly_mobile = [
            household
            for household in low_resource_origin_university_adults
            if family_resources_ratio_by_household.get(household.id, 0.0) >= 1.0
        ]
        low_resource_origin_nonuniversity_upwardly_mobile = [
            household
            for household in low_resource_origin_nonuniversity_adults
            if family_resources_ratio_by_household.get(household.id, 0.0) >= 1.0
        ]
        skilled_firms = [
            firm for firm in self.firms if firm.active and firm.sector in QUALIFIED_SECTOR_KEYS
        ]
        public_administration_desired_workers = self._public_administration_target_workers()
        public_administration_filled_workers = sum(
            1 for household in active_households if household.employed_by == PUBLIC_ADMINISTRATION_EMPLOYER_ID
        )
        total_desired_workers = (
            sum(max(0, firm.desired_workers) for firm in self.firms if firm.active)
            + public_administration_desired_workers
        )
        skilled_desired_workers = (
            sum(max(0, firm.desired_workers) for firm in skilled_firms)
            + public_administration_desired_workers
        )
        skilled_filled_workers = (
            sum(len(firm.workers) for firm in skilled_firms)
            + public_administration_filled_workers
        )
        active_school_firms = sum(1 for firm in self.firms if firm.active and firm.sector == "school")
        active_university_firms = sum(1 for firm in self.firms if firm.active and firm.sector == "university")

        return PeriodSnapshot(
            period=self.period,
            year=year,
            period_in_year=period_in_year,
            population=population,
            women=women,
            men=men,
            fertile_women=fertile_women,
            fertile_capable_women=fertile_capable_women,
            fertile_capable_women_low_desire_no_birth=fertile_capable_women_low_desire_no_birth,
            fertile_capable_women_with_births=fertile_capable_women_with_births,
            fertile_families=fertile_families,
            fertile_families_with_births=fertile_families_with_births,
            fertile_capable_families=fertile_capable_families,
            fertile_capable_families_low_desire_no_birth=fertile_capable_families_low_desire_no_birth,
            fertile_capable_families_with_births=fertile_capable_families_with_births,
            children=children,
            adults=adults,
            seniors=seniors,
            labor_force=labor_force,
            employment_rate=employment_rate,
            unemployment_rate=unemployment_rate,
            children_with_guardian=children_with_guardian,
            orphans=orphans,
            family_units=family_units,
            average_family_income=average_family_income,
            average_family_resources=average_family_resources,
            average_family_basic_basket_cost=average_family_basic_basket_cost,
            family_income_to_basket_ratio=family_income_to_basket_ratio,
            family_resources_to_basket_ratio=family_resources_to_basket_ratio,
            families_income_below_basket_share=families_income_below_basket_share,
            families_resources_below_basket_share=families_resources_below_basket_share,
            total_wages=self._period_wages,
            median_wage=median_wage,
            total_sales_units=self._period_sales_units,
            potential_demand_units=self._period_potential_demand_units,
            demand_fulfillment_rate=demand_fulfillment_rate,
            essential_demand_units=self._period_essential_demand_units,
            essential_production_units=self._period_essential_production_units,
            essential_sales_units=self._period_essential_sales_units,
            essential_total_sales_units=essential_total_sales_units,
            essential_government_sales_units=essential_government_sales_units,
            essential_inventory_units=essential_inventory_units,
            essential_target_inventory_units=essential_target_inventory_units,
            essential_expected_sales_units=essential_expected_sales_units,
            essential_fulfillment_rate=essential_fulfillment_rate,
            people_full_essential_coverage=people_full_essential_coverage,
            full_essential_coverage_share=full_essential_coverage_share,
            average_food_meals_per_person=average_food_meals_per_person,
            food_sufficient_share=food_sufficient_count / max(1, population),
            food_subsistence_share=food_subsistence_count / max(1, population),
            food_acute_hunger_share=food_acute_hunger_count / max(1, population),
            food_severe_hunger_share=food_severe_hunger_count / max(1, population),
            average_health_fragility=average_health_fragility,
            average_perceived_utility=average_perceived_utility,
            school_age_population=school_age_population,
            university_age_population=university_age_population,
            school_students=school_students,
            university_students=university_students,
            school_enrollment_share=school_students / max(1, school_age_population),
            university_enrollment_share=university_students / max(1, university_age_population),
            school_completion_share=school_completion_share,
            university_completion_share=university_completion_share,
            school_labor_share=school_labor_share,
            skilled_labor_share=skilled_labor_share,
            low_resource_school_enrollment_share=low_resource_school_students
            / max(1, low_resource_school_age_population),
            low_resource_university_enrollment_share=low_resource_university_students
            / max(1, low_resource_university_age_population),
            low_resource_university_student_share=low_resource_university_students / max(1, university_students),
            school_income_premium=school_income_premium,
            university_income_premium=university_income_premium,
            poverty_rate_without_university=poverty_rate_without_university,
            poverty_rate_with_university=poverty_rate_with_university,
            tracked_origin_adults=len(tracked_origin_adults),
            low_resource_origin_adults=len(low_resource_origin_adults),
            low_resource_origin_upward_mobility_share=len(low_resource_origin_upwardly_mobile)
            / max(1, len(low_resource_origin_adults)),
            low_resource_origin_university_completion_share=len(low_resource_origin_university_adults)
            / max(1, len(low_resource_origin_adults)),
            low_resource_origin_university_upward_mobility_share=len(low_resource_origin_university_upwardly_mobile)
            / max(1, len(low_resource_origin_university_adults)),
            low_resource_origin_nonuniversity_upward_mobility_share=len(
                low_resource_origin_nonuniversity_upwardly_mobile
            )
            / max(1, len(low_resource_origin_nonuniversity_adults)),
            skilled_job_demand_share=skilled_desired_workers / max(1, total_desired_workers),
            skilled_job_fill_rate=skilled_filled_workers / max(1, skilled_desired_workers),
            skilled_labor_supply_to_demand_ratio=len(university_labor_force) / max(1, skilled_desired_workers),
            total_sales_revenue=self._period_sales_revenue,
            total_production_units=self._period_production_units,
            period_investment_spending=self._period_investment_spending,
            startup_fixed_capital_formation=self._period_startup_fixed_capital_formation,
            startup_inventory_investment=self._period_startup_inventory_investment,
            business_cost_recycled=self._period_business_cost_recycled,
            business_cost_to_firms=self._period_business_cost_to_firms,
            business_cost_to_households=self._period_business_cost_to_households,
            business_cost_to_owners=self._period_business_cost_to_owners,
            inheritance_transfers=self._period_inheritance_transfers,
            bankruptcy_cash_recoveries=self._period_bankruptcy_cash_recoveries,
            gdp_nominal=gdp_nominal,
            gdp_per_capita=gdp_per_capita,
            total_capital_stock=total_capital_stock,
            total_inventory_units=total_inventory_units,
            total_profit=self._period_profit,
            active_firms=sum(1 for firm in self.firms if firm.active),
            active_school_firms=active_school_firms,
            active_university_firms=active_university_firms,
            bankruptcies=self._bankruptcies,
            births=self._period_births,
            deaths=self._period_deaths,
            average_age=average_age,
            average_worker_savings=average_worker_savings,
            worker_cash_available=self._period_worker_cash_available,
            worker_cash_saved=self._period_worker_cash_saved,
            worker_voluntary_saved=self._period_worker_voluntary_saved,
            worker_involuntary_retained=self._period_worker_involuntary_retained,
            worker_bank_deposits=worker_bank_deposits,
            worker_credit_outstanding=worker_credit_outstanding,
            gini_household_savings=gini([household.savings for household in active_households]),
            gini_owner_wealth=gini([entrepreneur.wealth + entrepreneur.vault_cash for entrepreneur in self.entrepreneurs]),
            capitalist_bank_deposits=capitalist_bank_deposits,
            capitalist_vault_cash=capitalist_vault_cash,
            capitalist_firm_cash=capitalist_firm_cash,
            capitalist_credit_outstanding=capitalist_credit_outstanding,
            capitalist_productive_capital=capitalist_productive_capital,
            capitalist_inventory_value=capitalist_inventory_value,
            capitalist_controlled_assets=capitalist_controlled_assets,
            capitalist_asset_share=capitalist_asset_share,
            capitalist_liquid_share=capitalist_liquid_share,
            worker_liquid_share=worker_liquid_share,
            goods_monetary_mass=goods_monetary_mass,
            price_index=price_index,
            gdp_deflator=gdp_deflator,
            government_treasury_cash=self.government.treasury_cash,
            government_debt_outstanding=self.government.debt_outstanding,
            government_tax_revenue=self._period_government_tax_revenue,
            government_labor_tax_revenue=self._period_government_labor_tax_revenue,
            government_payroll_tax_revenue=self._period_government_payroll_tax_revenue,
            government_corporate_tax_revenue=self._period_government_corporate_tax_revenue,
            government_dividend_tax_revenue=self._period_government_dividend_tax_revenue,
            government_wealth_tax_revenue=self._period_government_wealth_tax_revenue,
            government_transfers=self._period_government_transfers,
            government_unemployment_support=self._period_government_unemployment_support,
            government_child_allowance=self._period_government_child_allowance,
            government_basic_support=self._period_government_basic_support,
            government_procurement_spending=self._period_government_procurement_spending,
            government_education_spending=self._period_government_education_spending,
            government_school_spending=self._period_government_school_spending,
            government_university_spending=self._period_government_university_spending,
            government_school_units=self._period_government_school_units,
            government_university_units=self._period_government_university_units,
            school_average_price=self._average_sector_price("school"),
            university_average_price=self._average_sector_price("university"),
            government_public_administration_spending=self._period_government_public_administration_spending,
            government_infrastructure_spending=self._period_government_infrastructure_spending,
            government_bond_issuance=self._period_government_bond_issuance,
            government_deficit=self._period_government_deficit,
            government_surplus=self._period_government_surplus,
            recession_flag=self._period_recession_flag,
            recession_intensity=self._period_recession_intensity,
            government_countercyclical_support_multiplier=(
                self._period_government_countercyclical_support_multiplier
            ),
            government_countercyclical_procurement_multiplier=(
                self._period_government_countercyclical_procurement_multiplier
            ),
            government_countercyclical_spending=self._period_government_countercyclical_spending,
            total_inventory_book_value=total_inventory_book_value,
            household_final_consumption=household_final_consumption,
            government_final_consumption=government_final_consumption,
            gross_fixed_capital_formation=gross_fixed_capital_formation,
            change_in_inventories=change_in_inventories,
            valuables_acquisition=valuables_acquisition,
            gross_capital_formation=gross_capital_formation,
            exports=exports,
            imports=imports,
            net_exports=net_exports,
            gdp_expenditure_sna=gdp_expenditure_sna,
            gdp_expenditure_gap=gdp_expenditure_gap,
            labor_share_gdp=self._period_wages / gdp_denom,
            profit_share_gdp=self._period_profit / gdp_denom,
            investment_share_gdp=gross_fixed_capital_formation / gdp_denom,
            capitalist_consumption_share_gdp=self._period_entrepreneur_spending / gdp_denom,
            government_spending_share_gdp=(
                self._period_government_transfers
                + self._period_government_procurement_spending
                + self._period_government_education_spending
                + self._period_government_public_administration_spending
                + self._period_government_infrastructure_spending
            ) / gdp_denom,
            dividend_share_gdp=self._period_dividends_paid / gdp_denom,
            retained_profit_share_gdp=retained_profit / gdp_denom,
            firm_expansion_credit_creation=self._period_firm_expansion_credit_issued,
            investment_knowledge_multiplier=self._investment_knowledge_multiplier(),
            central_bank_money_supply=self.central_bank.money_supply,
            central_bank_target_money_supply=self._period_central_bank_target_money_supply,
            central_bank_policy_rate=self.central_bank.policy_rate,
            central_bank_issuance=self._period_central_bank_issuance,
            cumulative_central_bank_issuance=self.central_bank.cumulative_issuance,
            central_bank_monetary_gap_share=central_bank_monetary_gap_share,
            average_bank_reserve_ratio=average_bank_reserve_ratio,
            household_credit_creation=household_credit_creation,
            firm_credit_creation=firm_credit_creation,
            commercial_bank_credit_creation=commercial_bank_credit_creation,
            average_bank_deposit_rate=average_bank_deposit_rate,
            average_bank_loan_rate=average_bank_loan_rate,
            total_bank_deposits=total_bank_deposits,
            total_bank_reserves=total_bank_reserves,
            total_bank_loans_households=total_bank_loans_households,
            total_bank_loans_firms=total_bank_loans_firms,
            household_delinquent_loans=household_delinquent_loans,
            firm_delinquent_loans=firm_delinquent_loans,
            bank_nonperforming_loans=total_bank_nonperforming_loans,
            total_bank_bond_holdings=total_bank_bond_holdings,
            total_bank_assets=total_bank_assets,
            total_bank_liabilities=total_bank_liabilities,
            bank_equity=bank_equity,
            bank_writeoffs=self._period_bank_writeoffs,
            bank_loan_restructures=self._period_bank_loan_restructures,
            household_loan_defaults=self._period_household_loan_defaults,
            firm_loan_defaults=self._period_firm_loan_defaults,
            household_loan_restructures=self._period_household_loan_restructures,
            firm_loan_restructures=self._period_firm_loan_restructures,
            bank_recapitalization=self._period_bank_recapitalization,
            bank_resolution_events=self._period_bank_resolution_events,
            bank_undercapitalized_share=undercapitalized_share,
            bank_capital_ratio=bank_capital_ratio,
            bank_asset_liability_ratio=bank_asset_liability_ratio,
            bank_reserve_coverage_ratio=bank_reserve_coverage_ratio,
            bank_liquidity_ratio=bank_liquidity_ratio,
            bank_loan_to_deposit_ratio=bank_loan_to_deposit_ratio,
            bank_insolvent_share=insolvent_share,
            money_velocity=money_velocity,
            total_liquid_money=total_liquid_money,
            total_household_savings=sum(household.savings for household in active_households),
            public_capital_stock=self.government.public_capital_stock,
            public_infrastructure_productivity_multiplier=self._public_infrastructure_productivity_multiplier(),
            public_infrastructure_transport_cost_multiplier=self._public_infrastructure_transport_cost_multiplier(),
        )

    def _build_firm_period_snapshots(self) -> list[FirmPeriodSnapshot]:
        periods_per_year = max(1, self.config.periods_per_year)
        year = ((self.period - 1) // periods_per_year) + 1
        period_in_year = ((self.period - 1) % periods_per_year) + 1
        snapshots: list[FirmPeriodSnapshot] = []
        for firm in self.firms:
            operated_this_period = firm.age > 0
            snapshots.append(
                FirmPeriodSnapshot(
                    period=self.period,
                    year=year,
                    period_in_year=period_in_year,
                    firm_id=firm.id,
                    sector=firm.sector,
                    active=firm.active,
                    workers=len(firm.workers),
                    desired_workers=firm.desired_workers,
                    vacancies=max(0, firm.desired_workers - len(firm.workers)),
                    price=firm.price,
                    wage_offer=firm.wage_offer,
                    cash=firm.cash,
                    capital=firm.capital,
                    inventory=firm.inventory,
                    productivity=firm.productivity,
                    input_cost_per_unit=firm.input_cost_per_unit,
                    transport_cost_per_unit=firm.transport_cost_per_unit,
                    fixed_overhead=firm.fixed_overhead,
                    capital_charge=firm.last_capital_charge,
                    unit_cost=firm.last_unit_cost,
                    markup_tolerance=firm.markup_tolerance,
                    volume_preference=firm.volume_preference,
                    inventory_aversion=firm.inventory_aversion,
                    employment_inertia=firm.employment_inertia,
                    price_aggressiveness=firm.price_aggressiveness,
                    cash_conservatism=firm.cash_conservatism,
                    market_share_ambition=firm.market_share_ambition,
                    demand_elasticity=firm.demand_elasticity,
                    forecast_caution=firm.forecast_caution,
                    learning_maturity=self._firm_learning_maturity(firm),
                    technology=firm.technology,
                    technology_investment=firm.last_technology_investment,
                    technology_gain=firm.last_technology_gain,
                    sales=firm.last_sales if operated_this_period else 0.0,
                    expected_sales=firm.last_expected_sales,
                    revenue=firm.last_revenue if operated_this_period else 0.0,
                    production=firm.last_production if operated_this_period else 0.0,
                    profit=firm.last_profit if operated_this_period else 0.0,
                    total_cost=firm.last_total_cost if operated_this_period else 0.0,
                    loss_streak=firm.loss_streak,
                    market_share=firm.last_market_share if operated_this_period else 0.0,
                    market_fragility_belief=firm.market_fragility_belief,
                    forecast_error_belief=firm.forecast_error_belief,
                    target_inventory=firm.target_inventory,
                    age=firm.age,
                )
            )
        return snapshots


def run_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    return EconomySimulation(config=config).run()
