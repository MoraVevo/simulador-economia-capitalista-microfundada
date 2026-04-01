from __future__ import annotations

import bisect
import heapq
import math
import random

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


class EconomySimulation:
    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.rng = random.Random(self.config.seed)
        self.period = 0
        self._period_active_households_cache: list[Household] | None = None
        self._period_active_firms_by_sector_cache: dict[str, list[Firm]] | None = None
        self._period_average_sector_price_cache: dict[str, float] | None = None
        self._period_household_age_years_cache: dict[int, tuple[int, float]] = {}
        self._period_household_desired_units_cache: dict[tuple[int, int, str], float] = {}
        self._period_essential_budget_cache: dict[int, tuple[int, float]] = {}
        self._period_family_groups_cache: dict[int, list[Household]] | None = None
        self.history: list[PeriodSnapshot] = []
        self.firm_history: list[FirmPeriodSnapshot] = []
        self.households = self._build_households()
        self._startup_structural_demand_cache = self._compute_structural_demand_map(
            [household for household in self.households if household.alive]
        )
        startup_population = sum(1 for household in self.households if household.alive)
        self._startup_essential_target_cache = {
            sector_key: max(
                self._startup_structural_demand_cache.get(sector_key, 0.0),
                startup_population * self._essential_basket_share(sector_key),
            )
            * 1.10
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
        self._assign_initial_guardians()
        self._seed_initial_workforce()
        self._startup_structural_demand_cache = None
        self._startup_essential_target_cache = None
        self._startup_goods_monetary_mass = self._current_goods_monetary_mass()

        self._cash_before_sales: dict[int, float] = {}
        self._period_wages = 0.0
        self._period_sales_units = 0.0
        self._period_potential_demand_units = 0.0
        self._period_essential_demand_units = 0.0
        self._period_essential_sales_units = 0.0
        self._period_sales_revenue = 0.0
        self._period_production_units = 0.0
        self._period_investment_spending = 0.0
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
        self._period_entrepreneur_spending = 0.0
        self._period_dividends_paid = 0.0
        self._period_government_tax_revenue = 0.0
        self._period_government_corporate_tax_revenue = 0.0
        self._period_government_dividend_tax_revenue = 0.0
        self._period_government_wealth_tax_revenue = 0.0
        self._period_government_transfers = 0.0
        self._period_government_unemployment_support = 0.0
        self._period_government_child_allowance = 0.0
        self._period_government_basic_support = 0.0
        self._period_government_procurement_spending = 0.0
        self._period_government_bond_issuance = 0.0
        self._period_government_deficit = 0.0
        self._period_government_surplus = 0.0
        self._bankruptcies = 0
        self._period_births = 0
        self._period_deaths = 0
        self._last_sector_potential_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._last_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._prior_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._last_sector_budget_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_potential_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_budget_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}

    def run(self) -> SimulationResult:
        while self.period < self.config.periods:
            self.step()
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
        self._apply_bank_credit_policy()
        self._produce_and_pay_wages()
        self._apply_government_household_support()
        self._consume_households()
        self._apply_government_essential_procurement()
        self._settle_firms()
        self._finalize_government_period()
        self._resolve_bankruptcy_and_entry()
        self._refresh_period_sector_caches()
        _, _, _, _, _, _, _, _, _, current_unemployment, _ = self._population_metrics()
        self._apply_demography(current_unemployment)
        self._refresh_period_household_caches()
        self._refresh_period_family_cache()

        snapshot = self._build_snapshot()
        self._roll_forward_sector_demand_signals()
        self.history.append(snapshot)
        self.firm_history.extend(self._build_firm_period_snapshots())
        return snapshot

    def _reset_period_counters(self) -> None:
        self._cash_before_sales = {}
        self._period_wages = 0.0
        self._period_sales_units = 0.0
        self._period_potential_demand_units = 0.0
        self._period_essential_demand_units = 0.0
        self._period_essential_sales_units = 0.0
        self._period_sales_revenue = 0.0
        self._period_production_units = 0.0
        self._period_investment_spending = 0.0
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
        self._period_entrepreneur_spending = 0.0
        self._period_dividends_paid = 0.0
        self._period_government_tax_revenue = 0.0
        self._period_government_corporate_tax_revenue = 0.0
        self._period_government_dividend_tax_revenue = 0.0
        self._period_government_wealth_tax_revenue = 0.0
        self._period_government_transfers = 0.0
        self._period_government_unemployment_support = 0.0
        self._period_government_child_allowance = 0.0
        self._period_government_basic_support = 0.0
        self._period_government_procurement_spending = 0.0
        self._period_government_bond_issuance = 0.0
        self._period_government_deficit = 0.0
        self._period_government_surplus = 0.0
        self._bankruptcies = 0
        self._period_births = 0
        self._period_deaths = 0
        self._period_sector_potential_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_budget_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_household_desired_units_cache = {}
        self._period_essential_budget_cache = {}
        self.government.tax_revenue_this_period = 0.0
        self.government.corporate_tax_revenue = 0.0
        self.government.dividend_tax_revenue = 0.0
        self.government.wealth_tax_revenue = 0.0
        self.government.transfers_this_period = 0.0
        self.government.unemployment_support_this_period = 0.0
        self.government.child_allowance_this_period = 0.0
        self.government.basic_support_this_period = 0.0
        self.government.procurement_spending_this_period = 0.0
        self.government.bond_issuance_this_period = 0.0
        self.government.deficit_this_period = 0.0
        self.government.surplus_this_period = 0.0
        for bank in self.banks:
            bank.interest_income = 0.0
            bank.interest_expense = 0.0
            bank.profits = 0.0
        for firm in self.firms:
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

    def _refresh_period_sector_caches(self) -> None:
        active_firms_by_sector = {
            spec.key: [firm for firm in self.firms_by_sector.get(spec.key, []) if firm.active]
            for spec in SECTOR_SPECS
        }
        self._period_active_firms_by_sector_cache = active_firms_by_sector
        self._period_average_sector_price_cache = {
            sector_key: (
                sum(firm.price for firm in firms) / len(firms)
                if firms
                else SECTOR_BY_KEY[sector_key].base_price
            )
            for sector_key, firms in active_firms_by_sector.items()
        }
        self._period_essential_budget_cache = {}

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

    def _reset_household_labor_state(self) -> None:
        for household in self.households:
            household.wage_income = 0.0
            household.last_income = 0.0
            household.last_available_cash = 0.0
            household.last_consumption = {spec.key: 0.0 for spec in SECTOR_SPECS}

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
            family_income = sum(member.last_income for member in members)
            income_cover = family_income / max(1e-9, family_basket_cost)
            family_stress = clamp(1.0 - income_cover, 0.0, 1.0)
            reservation_target = max(
                living_wage_anchor * floor_share,
                reproduction_wage * (0.92 + 0.18 * family_stress),
            )

            for member in earning_members:
                current_wage = 0.0
                if member.employed_by is not None and member.employed_by in self.firm_by_id:
                    current_wage = self.firm_by_id[member.employed_by].wage_offer
                blended_target = (1.0 - adjustment_speed) * member.reservation_wage + adjustment_speed * reservation_target
                member.reservation_wage = clamp(
                    max(current_wage * 0.92, blended_target),
                    0.0,
                    living_wage_anchor * 3.0,
                )

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

    def _best_available_wage_offer(
        self,
        current_firm_id: int,
    ) -> float | None:
        best_wage: float | None = None
        for firm in self.firms:
            if not firm.active or firm.id == current_firm_id:
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
            firm = self.firm_by_id.get(household.employed_by)
            if firm is None or not firm.active:
                self._release_household_from_employment(household)
                continue
            household.employment_tenure += 1
            if household.employment_tenure >= contract_periods:
                current_capacity = self._firm_hiring_capacity(firm)
                if len(firm.workers) <= current_capacity:
                    best_wage = self._best_available_wage_offer(firm.id)
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
            return list(cached)
        return [household for household in self.households if household.alive]

    def _sector_firms(self, sector_key: str, active_only: bool = True) -> list[Firm]:
        firms_by_sector = getattr(self, "firms_by_sector", None)
        if firms_by_sector is None:
            return []
        firms = firms_by_sector.get(sector_key, [])
        if active_only:
            cached = self._period_active_firms_by_sector_cache
            if cached is not None:
                return list(cached.get(sector_key, []))
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
        firms = [firm for firm in self._sector_firms(sector_key) if firm.inventory > 0.0 and firm.price > 0.0]
        if not firms or desired_units <= 0.0 or cash <= 0.0:
            return cash, 0.0

        elasticity = 1.0 + 0.35 * price_sensitivity
        weighted_firms: list[tuple[Firm, float]] = []
        for firm in firms:
            competitiveness = 1.0 / max(0.1, firm.price) ** elasticity
            competitiveness *= 1.0 + min(0.25, firm.inventory / max(1.0, desired_units))
            weighted_firms.append((firm, competitiveness))

        total_weight = sum(weight for _, weight in weighted_firms)
        if total_weight <= 0.0:
            return cash, 0.0

        bought_total = 0.0
        ranked_firms = sorted(weighted_firms, key=lambda item: item[0].price)
        remaining_desired_units = desired_units
        for index, (firm, weight) in enumerate(ranked_firms):
            if cash <= 0.0 or remaining_desired_units <= 0.0:
                break
            remaining_weight = sum(
                candidate_weight
                for candidate_firm, candidate_weight in ranked_firms[index:]
                if candidate_firm.inventory > 0.0
            )
            if remaining_weight <= 0.0:
                break
            target_units = remaining_desired_units * (weight / remaining_weight)
            affordable_units = cash / firm.price
            units_bought = min(target_units, affordable_units, firm.inventory)
            if units_bought <= 0.0:
                continue
            spend = units_bought * firm.price
            cash -= spend
            firm.inventory -= units_bought
            firm.cash += spend
            firm.sales_this_period += units_bought
            self._period_sales_units += units_bought
            self._period_sales_revenue += spend
            self._period_sector_sales_units[sector_key] += units_bought
            spending_log[sector_key] += units_bought
            bought_total += units_bought
            remaining_desired_units -= units_bought

        return cash, bought_total

    def _draw_desired_children(self) -> int:
        choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        weights = [12, 16, 18, 16, 12, 10, 7, 4, 2, 2, 1]
        return self.rng.choices(choices, weights=weights, k=1)[0]

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
        members = self._family_groups().get(root_id, [household])
        return [member for member in members if member.alive]

    def _projected_family_labor_income(self, members: list[Household]) -> float:
        total_income = 0.0
        for member in members:
            projected_income = max(0.0, member.last_income)
            if member.employed_by is not None:
                firm = self.firm_by_id.get(member.employed_by)
                if firm is not None and firm.active:
                    projected_income = max(projected_income, firm.wage_offer)
            total_income += projected_income
        return total_income

    def _household_creditworthy(self, borrower: Household, amount: float, bank: CommercialBank) -> bool:
        amount = max(0.0, amount)
        if amount <= 0.0:
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

        expected_revenue = max(
            0.0,
            firm.last_expected_sales * max(0.0, firm.price),
            firm.last_revenue,
            firm.last_sales * max(0.0, firm.price),
        )
        operating_cost = max(0.0, firm.last_wage_bill + firm.last_input_cost + firm.last_transport_cost + firm.last_fixed_overhead)
        operating_surplus = max(0.0, expected_revenue - operating_cost)
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

        if projected_debt_to_revenue > max(0.5, self.config.bank_firm_max_debt_to_revenue):
            return False
        if projected_interest_cost > 0.0 and interest_coverage < max(0.5, self.config.bank_firm_min_interest_coverage):
            return False
        if operating_surplus <= 0.0 and firm.cash < 0.75 * max(1.0, operating_cost):
            return False
        return True

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

    def _estimate_firm_credit_requests(self) -> list[tuple[Firm, float]]:
        requests: list[tuple[Firm, float]] = []
        for firm in self.firms:
            if not firm.active:
                continue
            working_capital_target = (
                max(firm.last_wage_bill, firm.desired_workers * firm.wage_offer)
                + firm.last_input_cost
                + firm.last_transport_cost
                + firm.last_fixed_overhead
                + firm.capital * self.config.depreciation_rate
            )
            buffer_target = 0.30 * max(1.0, firm.last_expected_sales * firm.price)
            loan_need = max(0.0, working_capital_target + buffer_target - firm.cash)
            if loan_need > 0.0:
                requests.append((firm, loan_need))
        return requests

    def _estimate_credit_demand_by_bank(self) -> dict[int, float]:
        demand_by_bank = {bank.id: 0.0 for bank in self.banks}
        for borrower, amount in self._estimate_household_credit_requests():
            demand_by_bank[self._bank_id_for_household(borrower)] += amount
        for firm, amount in self._estimate_firm_credit_requests():
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

    def _create_household_credit(self, borrower: Household, amount: float, bank: CommercialBank) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return 0.0
        borrower.savings += amount
        borrower.loan_balance += amount
        bank.loans_households += amount
        self._period_household_credit_issued += amount
        return amount

    def _create_firm_credit(self, firm: Firm, amount: float, bank: CommercialBank) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return 0.0
        firm.cash += amount
        firm.loan_balance += amount
        bank.loans_firms += amount
        self._period_firm_credit_issued += amount
        return amount

    def _apply_bank_credit_policy(self) -> None:
        if not self.banks or not self.config.central_bank_enabled:
            return

        self._refresh_bank_balance_sheets()
        household_requests = self._estimate_household_credit_requests()
        firm_requests = self._estimate_firm_credit_requests()
        requests_by_bank: dict[int, list[tuple[str, Household | Firm, float]]] = {
            bank.id: [] for bank in self.banks
        }

        for borrower, amount in household_requests:
            requests_by_bank[self._bank_id_for_household(borrower)].append(("household", borrower, amount))
        for firm, amount in firm_requests:
            requests_by_bank[self._bank_id_for_firm(firm)].append(("firm", firm, amount))

        for bank in self.banks:
            bank_requests = requests_by_bank.get(bank.id, [])
            if not bank_requests:
                continue

            approved_requests: list[tuple[str, Household | Firm, float]] = []
            for request_type, borrower, amount in bank_requests:
                approved = (
                    self._household_creditworthy(borrower, amount, bank)
                    if request_type == "household"
                    else self._firm_creditworthy(borrower, amount, bank)
                )
                if approved:
                    approved_requests.append((request_type, borrower, amount))

            total_request = sum(amount for _, _, amount in approved_requests)
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

            loan_pool = min(total_request, capacity)
            if loan_pool <= 0.0:
                continue

            allocation_scale = loan_pool / total_request
            for request_type, borrower, amount in approved_requests:
                granted = amount * allocation_scale
                if granted <= 0.0:
                    continue
                if request_type == "household":
                    self._create_household_credit(borrower, granted, bank)
                else:
                    self._create_firm_credit(borrower, granted, bank)

            reserve_requirement = self._bank_reserve_requirement(bank)
            excess_reserves = max(0.0, bank.reserves - reserve_requirement)
            if excess_reserves > 0.0:
                bond_purchase = excess_reserves * clamp(self.config.bank_bond_allocation_share, 0.0, 1.0)
                bond_purchase = min(bond_purchase, excess_reserves)
                bank.reserves -= bond_purchase
                bank.bond_holdings += bond_purchase

        self._refresh_bank_balance_sheets()
        self._reconcile_bank_reserves()
        self.central_bank.money_supply = self._current_total_liquid_money()
        self._update_bank_interest_rates()

    def _current_price_index_estimate(self) -> float:
        return sum(
            (self._average_sector_price(spec.key) / spec.base_price) * spec.household_demand_share
            for spec in SECTOR_SPECS
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

    def _update_bank_interest_rates(self) -> None:
        target_period_inflation = self._annual_to_period_growth(self.config.central_bank_target_annual_inflation)
        observed_inflation = 0.0
        essential_gap = 0.0
        unemployment_gap = 0.0
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
            - 0.15 * essential_gap,
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
                loan_floor + 0.025 * reserve_stress + 0.040 * discount_stress - 0.010 * liquidity_relief,
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
            prior_goods_mass = max(1e-9, self._startup_goods_monetary_mass)
        else:
            previous_goods_mass = current_goods_mass
            prior_goods_mass = max(1e-9, self._startup_goods_monetary_mass)

        goods_growth = max(0.0, (previous_goods_mass - prior_goods_mass) / prior_goods_mass)
        target_growth = goods_growth * pass_through
        return current_money_supply * (1.0 + target_growth)

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

    def _record_government_tax_revenue(self, amount: float, revenue_type: str) -> float:
        amount = max(0.0, amount)
        if amount <= 0.0:
            return 0.0
        self.government.treasury_cash += amount
        self.government.tax_revenue_this_period += amount
        self._period_government_tax_revenue += amount
        if revenue_type == "corporate":
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

    def _plan_government_household_support(
        self,
    ) -> list[tuple[Household, float, float, float]]:
        if not self.config.government_enabled:
            return []
        living_wage_anchor = self._living_wage_anchor()
        if living_wage_anchor <= 0.0:
            return []
        spending_scale = max(0.0, self.config.government_spending_scale)

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
        return total_paid

    def _apply_government_essential_procurement(self) -> float:
        if not self.config.government_enabled:
            return 0.0
        population = len(self._active_households())
        if population <= 0:
            return 0.0

        spending_scale = max(0.0, self.config.government_spending_scale)
        spending_efficiency = clamp(self.config.government_spending_efficiency, 0.0, 1.0)
        procurement_targets: list[tuple[str, float, float]] = []
        total_budget_needed = 0.0
        for sector_key in ESSENTIAL_SECTOR_KEYS:
            minimum_required_units = population * self._essential_basket_share(sector_key)
            private_units_sold = self._period_sector_sales_units.get(sector_key, 0.0)
            unmet_units = max(0.0, minimum_required_units - private_units_sold)
            target_units = unmet_units * max(0.0, self.config.government_procurement_gap_share) * spending_scale
            if target_units <= 0.0:
                continue
            average_price = self._average_sector_price(sector_key)
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
        return total_spent

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

    def _finalize_government_period(self) -> None:
        spending = self._period_government_transfers + self._period_government_procurement_spending
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
            self._household_labor_capacity(self.households[worker_id]) for worker_id in firm.workers
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
            issuance_gap = max(0.0, target_money_supply - current_money_supply)
            estimated_credit_demand = sum(self._estimate_credit_demand_by_bank().values())
            issuance = min(issuance_gap, estimated_credit_demand)
            if issuance > 0.0:
                reserve_injection = issuance * max(0.0, self.config.reserve_ratio)
                self._distribute_reserve_injection_to_banks(reserve_injection, self._estimate_credit_demand_by_bank())
                current_money_supply += issuance
        else:
            target_money_supply = self._target_money_supply_fisher()
            issuance_gap = max(0.0, target_money_supply - current_money_supply)
            estimated_credit_demand = sum(self._estimate_credit_demand_by_bank().values())
            issuance = min(issuance_gap, estimated_credit_demand)
            if issuance > 0.0:
                reserve_injection = issuance * max(0.0, self.config.reserve_ratio)
                self._distribute_reserve_injection_to_banks(reserve_injection, self._estimate_credit_demand_by_bank())
                current_money_supply += issuance
        if issuance > 0.0:
            self.central_bank.cumulative_issuance += issuance
        self.central_bank.money_supply = current_money_supply
        self.central_bank.target_money_supply = target_money_supply
        self.central_bank.issuance_this_period = issuance
        self._period_central_bank_issuance = issuance
        self._period_central_bank_target_money_supply = target_money_supply
        self._update_bank_interest_rates()

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
        }

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

    def _household_sector_desired_units(self, household: Household, sector_key: str) -> float:
        cache_key = (household.id, household.age_periods, sector_key)
        cached = self._period_household_desired_units_cache.get(cache_key)
        if cached is not None:
            return cached
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

    def _essential_affordability_pressure(self) -> float:
        if not self.history or len(self.history) < max(1, self.config.startup_grace_periods):
            return 0.0
        last_snapshot = self.history[-1]
        basket_gap = max(0.0, 1.0 - last_snapshot.family_income_to_basket_ratio)
        essential_gap = max(0.0, 1.0 - last_snapshot.essential_fulfillment_rate)
        return clamp(0.65 * basket_gap + 0.35 * essential_gap, 0.0, 1.0)

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
        inventory_pressure = max(0.0, firm.inventory - firm.target_inventory)
        inventory_relief_units = min(prudent_sales, inventory_pressure)
        unit_contribution = max(0.0, effective_price - variable_unit_cost)
        affordability_pressure = self._essential_affordability_pressure() if spec.key in ESSENTIAL_SECTOR_KEYS else 0.0

        # Rational objective: current operating profit plus expected retained market value
        # and the benefit of clearing costly excess inventory, minus fragility risk.
        future_weight = (
            0.65
            + 0.90 * self._firm_future_market_weight(firm)
            + 0.20 * max(0.0, firm.market_share_ambition - 1.0)
            + (0.45 * affordability_pressure if spec.key in ESSENTIAL_SECTOR_KEYS else 0.0)
        )
        inventory_relief_value = inventory_relief_units * max(0.0, unit_contribution + 0.15 * variable_unit_cost)
        hazard_penalty = market_hazard * (
            fixed_cost * (0.80 + 0.30 * firm.cash_conservatism)
            + 0.30 * future_market_value
        )
        return candidate_profit + future_weight * future_market_value + inventory_relief_value - hazard_penalty

    def _compute_structural_demand_map(
        self,
        active_households: list[Household] | None = None,
    ) -> dict[str, float]:
        households = active_households
        if households is None:
            households = [household for household in self.households if household.alive]

        totals = {spec.key: 0.0 for spec in SECTOR_SPECS}
        discretionary_scale = (
            self.config.nonessential_demand_multiplier
            * sum(SECTOR_BY_KEY[key].household_demand_share for key in DISCRETIONARY_SECTOR_KEYS)
        )
        for household in households:
            self._ensure_household_demand_shares(household)
            base_units = household.need_scale * self._household_consumption_multiplier(household)
            for sector_key in ESSENTIAL_SECTOR_KEYS:
                totals[sector_key] += base_units * household.essential_shares.get(sector_key, 0.0)
            for sector_key in DISCRETIONARY_SECTOR_KEYS:
                totals[sector_key] += (
                    base_units
                    * discretionary_scale
                    * household.discretionary_shares.get(sector_key, 0.0)
                )
        return totals

    def _structural_sector_demand(self, spec_key: str) -> float:
        if self._startup_structural_demand_cache is not None:
            return self._startup_structural_demand_cache.get(spec_key, 0.0)
        return sum(
            self._household_sector_desired_units(household, spec_key)
            for household in self._active_households()
        )

    def _essential_basket_share(self, sector_key: str) -> float:
        total_essential_need = sum(SECTOR_BY_KEY[key].essential_need for key in ESSENTIAL_SECTOR_KEYS)
        return SECTOR_BY_KEY[sector_key].essential_need / max(1e-9, total_essential_need)

    def _startup_essential_target_units(self, sector_key: str) -> float:
        if self._startup_essential_target_cache is not None:
            return self._startup_essential_target_cache.get(sector_key, 0.0)
        structural_demand = self._structural_sector_demand(sector_key)
        survival_floor = len(self._active_households()) * self._essential_basket_share(sector_key)
        return max(structural_demand, survival_floor) * 1.10

    def _essential_budget(self, household: Household) -> float:
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
        if partner_id is None:
            return
        partner = self._household_by_id(partner_id)
        if partner is not None and partner.partner_id == household.id:
            partner.partner_id = None

    def _adult_partnership_candidates(self, sex: str) -> list[Household]:
        return [
            household
            for household in self._active_households()
            if household.sex == sex
            and household.partner_id is None
            and self.config.entry_age_years <= self._household_age_years(household) <= self.config.fertile_age_max_years
        ]

    def _pair_score(self, first: Household, second: Household) -> float:
        age_gap = abs(self._household_age_years(first) - self._household_age_years(second))
        savings_gap = abs(first.savings - second.savings)
        desire_bonus = (first.desired_children + second.desired_children) / 2.0
        return desire_bonus + 1.5 / (1.0 + age_gap) + 1.0 / (1.0 + savings_gap)

    def _refresh_family_links(self) -> None:
        active_households = self._active_households()
        age_lookup = {
            household.id: self._household_age_years(household)
            for household in active_households
        }

        for household in active_households:
            partner = self._household_by_id(household.partner_id) if household.partner_id is not None else None
            if partner is not None and partner.alive and partner.partner_id == household.id:
                continue
            if household.partner_id is not None:
                household.partner_id = None

        males = [
            household
            for household in active_households
            if household.sex == "M"
            and household.partner_id is None
            and self.config.entry_age_years <= age_lookup[household.id] <= self.config.fertile_age_max_years
        ]
        females = [
            household
            for household in active_households
            if household.sex == "F"
            and household.partner_id is None
            and self.config.entry_age_years <= age_lookup[household.id] <= self.config.fertile_age_max_years
        ]
        if not males or not females:
            return

        males.sort(key=lambda household: (age_lookup[household.id], -household.savings, household.id))
        females.sort(key=lambda household: (age_lookup[household.id], -household.savings, household.id))
        female_ages = [age_lookup[household.id] for household in females]

        for male in males:
            if male.partner_id is not None:
                continue

            male_age = age_lookup[male.id]
            lower_index = bisect.bisect_left(female_ages, male_age - 14.0)
            upper_index = bisect.bisect_right(female_ages, male_age + 14.0)
            best_female = None
            best_score = None
            for female in females[lower_index:upper_index]:
                if female.partner_id is not None:
                    continue
                female_age = age_lookup[female.id]
                age_gap = abs(male_age - female_age)
                if male.desired_children == 0 and female.desired_children == 0:
                    continue
                savings_gap = abs(male.savings - female.savings)
                desire_bonus = (male.desired_children + female.desired_children) / 2.0
                score = desire_bonus + 1.5 / (1.0 + age_gap) + 1.0 / (1.0 + savings_gap)
                if best_score is None or score > best_score:
                    best_score = score
                    best_female = female

            if best_female is None:
                continue
            if best_score is not None and best_score < 0.9:
                continue

            male.partner_id = best_female.id
            best_female.partner_id = male.id
            family_desire = round((male.desired_children + best_female.desired_children) / 2.0)
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
        age_years = self._household_age_years(household)
        if age_years < self.config.entry_age_years:
            return 0.0
        if age_years < self.config.senior_age_years:
            base_capacity = 1.0
        elif age_years >= self.config.max_age_years:
            return 0.0
        else:
            retirement_span = max(1.0, self.config.retirement_age_years - self.config.senior_age_years)
            progress = clamp((age_years - self.config.senior_age_years) / retirement_span, 0.0, 1.0)
            base_capacity = clamp(
                1.0 - progress * (1.0 - self.config.senior_productivity_floor),
                self.config.senior_productivity_floor,
                1.0,
            )
        fragility_penalty = clamp(0.12 * household.health_fragility, 0.0, 0.45)
        housing_penalty = clamp(0.03 * household.housing_deprivation_streak, 0.0, 0.15)
        clothing_penalty = clamp(0.02 * household.clothing_deprivation_streak, 0.0, 0.10)
        return clamp(base_capacity * (1.0 - fragility_penalty - housing_penalty - clothing_penalty), 0.0, 1.0)

    def _in_startup_grace(self) -> bool:
        return self.period <= max(0, self.config.startup_grace_periods)

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

    def _sector_demand_elasticity_prior(self, sector_key: str) -> float:
        if sector_key == "food":
            return 0.82
        if sector_key == "housing":
            return 0.76
        if sector_key == "clothing":
            return 0.92
        if sector_key == "manufactured":
            return 1.18
        return 1.32

    def _smoothed_sales_reference(self, firm: Firm) -> float:
        if not firm.sales_history:
            return max(1.0, firm.last_sales, firm.last_expected_sales)
        recent_window = firm.sales_history[-4:]
        long_window = firm.sales_history[-8:] if len(firm.sales_history) >= 8 else firm.sales_history
        recent_average = sum(recent_window) / len(recent_window)
        long_average = sum(long_window) / len(long_window)
        return max(1.0, 0.60 * recent_average + 0.40 * long_average)

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

    def _household_sector_coverage(self, household: Household, sector_key: str) -> float:
        target_units = self._household_sector_desired_units(household, sector_key)
        if target_units <= 0.0:
            return 1.0
        consumed_units = household.last_consumption.get(sector_key, 0.0)
        return clamp(consumed_units / target_units, 0.0, 3.0)

    def _household_food_meals_consumed(self, household: Household) -> float:
        return self._household_sector_coverage(household, "food") * self._household_food_sufficient_meals(household)

    def _food_subsistence_coverage_ratio(self) -> float:
        return self.config.food_meals_per_day_subsistence / max(1e-9, self.config.food_meals_per_day_sufficient)

    def _food_severe_hunger_coverage_ratio(self) -> float:
        return self.config.food_meals_per_day_severe / max(1e-9, self.config.food_meals_per_day_sufficient)

    def _allocate_family_consumption_units(
        self,
        family_members: list[Household],
        purchased_units_by_sector: dict[str, float],
    ) -> dict[int, dict[str, float]]:
        allocation = {
            member.id: {spec.key: 0.0 for spec in SECTOR_SPECS}
            for member in family_members
        }
        for spec in SECTOR_SPECS:
            sector_key = spec.key
            total_target = sum(
                self._household_sector_desired_units(member, sector_key)
                for member in family_members
            )
            units_bought = max(0.0, purchased_units_by_sector.get(sector_key, 0.0))
            if units_bought <= 0.0 or total_target <= 0.0:
                continue
            for member in family_members:
                member_target = self._household_sector_desired_units(member, sector_key)
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
            return {root_id: list(members) for root_id, members in cached.items()}

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

    def _family_economic_metrics(self) -> tuple[int, float, float, float, float, float, float, float]:
        groups = self._family_groups()
        if not groups:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        family_incomes: list[float] = []
        family_resource_values: list[float] = []
        family_baskets: list[float] = []
        income_below_count = 0
        resources_below_count = 0

        for members in groups.values():
            basket_cost = sum(self._essential_budget(member) for member in members)
            family_income = sum(member.last_income for member in members)
            family_resource_total = sum(member.last_available_cash for member in members)
            family_incomes.append(family_income)
            family_resource_values.append(family_resource_total)
            family_baskets.append(basket_cost)
            if family_income < basket_cost:
                income_below_count += 1
            if family_resource_total < basket_cost:
                resources_below_count += 1

        family_units = len(groups)
        average_family_income = sum(family_incomes) / family_units
        average_family_resources = sum(family_resource_values) / family_units
        average_family_basic_basket_cost = sum(family_baskets) / family_units
        family_income_to_basket_ratio = average_family_income / max(1e-9, average_family_basic_basket_cost)
        family_resources_to_basket_ratio = average_family_resources / max(1e-9, average_family_basic_basket_cost)
        families_income_below_basket_share = income_below_count / family_units
        families_resources_below_basket_share = resources_below_count / family_units
        return (
            family_units,
            average_family_income,
            average_family_resources,
            average_family_basic_basket_cost,
            family_income_to_basket_ratio,
            family_resources_to_basket_ratio,
            families_income_below_basket_share,
            families_resources_below_basket_share,
        )

    def _family_reproductive_metrics(self) -> tuple[int, int, int, int, int]:
        fertile_families = 0
        fertile_families_with_births = 0
        fertile_capable_families = 0
        fertile_capable_families_low_desire_no_birth = 0
        fertile_capable_families_with_births = 0
        for members in self._family_groups().values():
            fertile_mothers = [
                member
                for member in members
                if member.sex == "F"
                and self._household_age_years(member) >= self.config.entry_age_years
                and self._is_fertile(member)
            ]
            if not fertile_mothers:
                continue
            fertile_families += 1
            birth_happened = any(member.last_birth_period == self.period for member in fertile_mothers)
            if birth_happened:
                fertile_families_with_births += 1

            family_basket_cost = sum(self._essential_budget(member) for member in members)
            family_resources = sum(member.last_available_cash for member in members)
            economically_capable = family_resources >= family_basket_cost
            if not economically_capable:
                continue

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
        return (
            fertile_families,
            fertile_families_with_births,
            fertile_capable_families,
            fertile_capable_families_low_desire_no_birth,
            fertile_capable_families_with_births,
        )

    def _fertile_women_reproductive_metrics(self) -> tuple[int, int, int]:
        fertile_capable_women = 0
        fertile_capable_women_low_desire_no_birth = 0
        fertile_capable_women_with_births = 0

        for members in self._family_groups().values():
            family_basket_cost = sum(self._essential_budget(member) for member in members)
            family_resources = sum(member.last_available_cash for member in members)
            economically_capable = family_resources >= family_basket_cost
            if not economically_capable:
                continue

            for member in members:
                if member.sex != "F":
                    continue
                if self._household_age_years(member) < self.config.entry_age_years:
                    continue
                if not self._is_fertile(member):
                    continue

                birth_happened = member.last_birth_period == self.period
                months_since_last_birth = self.period - member.last_birth_period
                spacing_ready = (
                    months_since_last_birth >= self.config.birth_interval_periods
                    or birth_happened
                )
                if not spacing_ready:
                    continue

                fertile_capable_women += 1
                if birth_happened:
                    fertile_capable_women_with_births += 1

                low_desire_no_birth = (
                    not birth_happened
                    and months_since_last_birth >= self.config.birth_interval_periods
                    and member.children_count >= max(0, member.desired_children)
                    and member.last_birth_period < self.period
                )
                if low_desire_no_birth:
                    fertile_capable_women_low_desire_no_birth += 1

        return (
            fertile_capable_women,
            fertile_capable_women_low_desire_no_birth,
            fertile_capable_women_with_births,
        )

    def _living_wage_anchor(self) -> float:
        active_households = self._active_households()
        labor_force = sum(
            1
            for household in active_households
            if self._household_labor_capacity(household) > 0.0
        )
        if labor_force <= 0:
            return 0.0
        total_essential_basket = sum(self._essential_budget(household) for household in active_households)
        return total_essential_basket / labor_force

    def _labor_force_participant_ids(self) -> set[int]:
        participant_ids: set[int] = set()
        family_groups = self._family_groups()
        living_wage_anchor = max(1.0, self._living_wage_anchor())

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
                age_years = self._household_age_years(member)
                family_income += max(0.0, member.last_income)
                family_liquid += max(0.0, self._household_cash_balance(member))
                family_basket += max(0.0, self._essential_budget(member))
                if age_years < self.config.entry_age_years:
                    dependent_children += 1
                    continue
                if self._household_labor_capacity(member) <= 0.0:
                    continue
                if member.employed_by is not None:
                    participant_ids.add(member.id)
                    employed_participants += 1
                if age_years < self.config.senior_age_years:
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
        active_households = self._active_households()
        population = len(active_households)
        women = 0
        men = 0
        fertile_women = 0
        children = 0
        adults = 0
        seniors = 0
        labor_force = 0
        employed = 0
        labor_force_participants = self._labor_force_participant_ids()
        for household in active_households:
            if household.sex == "F":
                women += 1
                if self._is_fertile(household):
                    fertile_women += 1
            else:
                men += 1
            age_years = self._household_age_years(household)
            if age_years < self.config.entry_age_years:
                children += 1
            elif age_years < self.config.senior_age_years:
                adults += 1
            else:
                seniors += 1
            if household.id in labor_force_participants:
                labor_force += 1
                if household.employed_by is not None:
                    employed += 1
        employment_rate = employed / max(1, labor_force)
        unemployment_rate = 1.0 - employment_rate if labor_force > 0 else 0.0
        average_age = (
            sum(household.age_periods for household in active_households) / max(1, population)
            / max(1, self.config.periods_per_year)
        )
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
                saving_propensity=self.rng.uniform(0.05, 0.22),
                money_trust=self.rng.uniform(0.40, 0.75),
                consumption_impatience=self.rng.uniform(0.25, 0.85),
                price_sensitivity=self.rng.uniform(0.7, 1.3),
                need_scale=self.rng.uniform(0.9, 1.05),
                sector_preference_weights=self._draw_household_sector_preference_weights(),
                age_periods=0,
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
                0.03,
                0.0015
                * household.severe_hunger_streak
                * vulnerability_multiplier,
            )
        elif household.deprivation_streak > 0:
            period_probability += min(
                0.05,
                self.config.period_food_subsistence_death_risk
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

        economically_capable_by_household: dict[int, bool] = {}
        for members in self._family_groups().values():
            family_basket_cost = sum(self._essential_budget(member) for member in members)
            family_resources = sum(member.last_available_cash for member in members)
            economically_capable = family_resources >= family_basket_cost
            for member in members:
                economically_capable_by_household[member.id] = economically_capable

        for household in active_households:
            previous_age_years = self._household_age_years(household)
            household.age_periods += 1
            current_age_years = self._household_age_years(household)
            if previous_age_years < self.config.entry_age_years <= current_age_years:
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
            birth_probability = self._annual_to_period_probability(annual_birth_rate * prosperity)
            remaining_desire = max(0, mother.desired_children - mother.children_count)
            family_bias = clamp(0.35 + 0.15 * remaining_desire, 0.35, 1.75)
            if self.rng.random() < clamp(birth_probability * family_bias, 0.0, 0.95):
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

            months_since_last_birth = max(0, self.period - member.last_birth_period)
            if stability >= 0.80 and family_remaining_cash >= 0.20 * family_basic_basket_cost:
                time_factor = 1.0 + max(0.0, months_since_last_birth - self.config.birth_interval_periods) / max(
                    1.0, float(self.config.birth_interval_periods)
                )
                member.child_desire_pressure = clamp(
                    member.child_desire_pressure + (0.10 + 0.10 * stability) * time_factor,
                    0.0,
                    3.0,
                )
            elif stability >= 0.60 and months_since_last_birth >= self.config.birth_interval_periods:
                member.child_desire_pressure = clamp(member.child_desire_pressure + 0.05 * stability, 0.0, 3.0)
            else:
                member.child_desire_pressure = max(0.0, member.child_desire_pressure - 0.03)

            while member.child_desire_pressure >= 1.0 and member.desired_children < 12:
                member.child_desire_pressure -= 1.0
                member.desired_children += 1

            partner = self._household_by_id(member.partner_id) if member.partner_id is not None else None
            if partner is not None and partner.alive and self._household_age_years(partner) >= self.config.entry_age_years:
                if partner.desired_children < member.desired_children:
                    partner.desired_children = member.desired_children
                partner.child_desire_pressure = max(partner.child_desire_pressure, member.child_desire_pressure)

    def _capital_efficiency(self, capital: float) -> float:
        return 1.0 + math.log1p(max(0.0, capital) / self.config.capital_scale)

    def _firm_effective_productivity(self, firm: Firm) -> float:
        return max(0.1, firm.productivity * firm.technology * self._capital_efficiency(firm.capital))

    def _firm_costing_units(self, firm: Firm, realized_output: float) -> float:
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
            budget_units = self._period_sector_budget_demand_units.get(spec_key, 0.0)
            potential_units = self._period_sector_potential_demand_units.get(spec_key, 0.0)
        else:
            budget_units = self._last_sector_budget_demand_units.get(spec_key, 0.0)
            potential_units = self._last_sector_potential_demand_units.get(spec_key, 0.0)
        # This signal is meant to capture sector-wide solvent desire, not realized
        # sales. Otherwise one firm's bad pricing would incorrectly shrink the
        # perceived size of the whole market.
        return max(0.0, 0.80 * budget_units + 0.20 * potential_units)

    def _baseline_demand(self, spec_key: str, use_current_period: bool = False) -> float:
        structural_demand = self._structural_sector_demand(spec_key)
        if not self.history and not use_current_period and self.period == 0:
            return structural_demand
        if spec_key in ESSENTIAL_SECTOR_KEYS and self._in_startup_grace() and not use_current_period:
            return structural_demand
        observed_demand = self._observed_sector_demand_signal(spec_key, use_current_period=use_current_period)
        if observed_demand > 0.0:
            learning_maturity = self._market_learning_maturity()
            if spec_key in ESSENTIAL_SECTOR_KEYS:
                observed_weight = 0.18 + 0.47 * learning_maturity
                structural_floor = 0.72 - 0.12 * learning_maturity
                return max(
                    structural_floor * structural_demand,
                    observed_weight * observed_demand + (1.0 - observed_weight) * structural_demand,
                )
            observed_weight = 0.12 + 0.63 * learning_maturity
            structural_floor = 0.45 - 0.20 * learning_maturity
            return max(
                structural_floor * structural_demand,
                observed_weight * observed_demand + (1.0 - observed_weight) * structural_demand,
            )
        return structural_demand

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
        if spec.key in ESSENTIAL_SECTOR_KEYS:
            return max(0.1, unit_cost)
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

    def _startup_slot_share(self) -> float:
        return 1.0 / max(1, self.config.firms_per_sector)

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

    def _best_entry_owner(self) -> Entrepreneur | None:
        threshold = self.config.firm_restart_wealth_threshold
        candidates = [
            owner
            for owner in self.entrepreneurs
            if self._owner_total_liquid(owner) - threshold > 0.0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda owner: self._owner_total_liquid(owner))

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
        firm.forecast_caution = blend(firm.forecast_caution, peer.forecast_caution, 0.60, 1.85)
        firm.demand_elasticity = blend(firm.demand_elasticity, peer.demand_elasticity, 0.45, 2.75)
        firm.wage_offer = max(0.0, peer.wage_offer * self.rng.uniform(0.95, 1.05))
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
            for sale in peer.sales_history[-6:]
        ] or [max(1.0, peer.last_sales)]
        firm.last_expected_sales = max(1.0, sum(firm.sales_history) / len(firm.sales_history))
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
        firm.wage_offer = max(0.0, firm.wage_offer * self.rng.uniform(0.94, 1.06))
        mutated_price = firm.price * self.rng.uniform(0.92, 1.08)
        firm.price = clamp(
            mutated_price,
            spec.base_price * 0.35,
            min(spec.base_price * self.config.price_ceiling_multiplier, firm.price * self._firm_max_price_hike_ratio(firm)),
        )
        firm.sales_history = [
            max(0.0, sale * self.rng.uniform(0.85, 1.15))
            for sale in firm.sales_history[-6:]
        ] or [max(1.0, firm.last_sales)]
        firm.last_expected_sales = max(1.0, sum(firm.sales_history) / len(firm.sales_history))
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
        search_floor_price = min(floor_price, firm.price)
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
        if len(firm.sales_history) < 4:
            return 0.0
        recent_window = firm.sales_history[-3:]
        prior_window = firm.sales_history[-6:-3] or firm.sales_history[:-3]
        if not prior_window:
            return 0.0
        prior_avg = sum(prior_window) / len(prior_window)
        recent_avg = sum(recent_window) / len(recent_window)
        return clamp((recent_avg - prior_avg) / max(1.0, prior_avg), -0.5, 0.5)

    def _firm_max_price_hike_ratio(self, firm: Firm) -> float:
        learning_maturity = self._firm_learning_maturity(firm)
        uncertainty = self._firm_forecast_uncertainty(firm)
        caution = clamp(firm.forecast_caution, 0.60, 1.85)
        demand_realization = firm.last_sales / max(1.0, firm.last_expected_sales)
        sell_through = firm.last_sales / max(1.0, firm.last_production)
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

    def _firm_rejection_signal(self, firm: Firm) -> float:
        expected_sales = max(1.0, firm.last_expected_sales)
        expectation_shortfall = clamp(
            (expected_sales - firm.last_sales) / expected_sales,
            0.0,
            1.5,
        )
        inventory_ratio = firm.inventory / max(1.0, firm.target_inventory)
        inventory_pressure = clamp((inventory_ratio - 1.0) / 1.5, 0.0, 1.0)

        recent_sales_drop = 0.0
        if len(firm.sales_history) >= 4:
            recent_window = firm.sales_history[-3:]
            prior_window = firm.sales_history[-6:-3] or firm.sales_history[:-3]
            if prior_window:
                prior_avg = sum(prior_window) / len(prior_window)
                recent_avg = sum(recent_window) / len(recent_window)
                recent_sales_drop = clamp(
                    (prior_avg - recent_avg) / max(1.0, prior_avg),
                    0.0,
                    1.5,
                )

        unsold_pressure = 0.0
        if firm.last_production > 0.0:
            unsold_pressure = clamp(
                (firm.last_production - firm.last_sales) / max(1.0, firm.last_production),
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
        last_budget_units = getattr(self, "_last_sector_budget_demand_units", {})
        last_sales_units = getattr(self, "_last_sector_sales_units", {})
        prior_sales_units_map = getattr(self, "_prior_sector_sales_units", {})
        budget_units = last_budget_units.get(sector_key, 0.0)
        sales_units = last_sales_units.get(sector_key, 0.0)
        prior_sales_units = prior_sales_units_map.get(sector_key, 0.0)

        walkaway_rate = 0.0
        if budget_units > 0.0:
            walkaway_rate = clamp((budget_units - sales_units) / budget_units, 0.0, 1.5)

        volume_drop = 0.0
        if prior_sales_units > 0.0:
            volume_drop = clamp(
                (prior_sales_units - sales_units) / prior_sales_units,
                0.0,
                1.5,
            )

        return clamp(0.60 * walkaway_rate + 0.40 * volume_drop, 0.0, 1.5)

    def _economy_public_fragility_signal(self) -> float:
        if not self.history:
            return 0.0

        last_snapshot = self.history[-1]
        unemployment_stress = clamp(
            (last_snapshot.unemployment_rate - self.config.target_unemployment) / 0.35,
            0.0,
            1.5,
        )
        demographic_loss = max(0.0, last_snapshot.deaths - last_snapshot.births) / max(1, last_snapshot.population)
        demographic_stress = clamp(
            demographic_loss * self.config.periods_per_year * 4.0,
            0.0,
            1.5,
        )

        population_drop = 0.0
        if len(self.history) >= 2:
            prior_snapshot = self.history[-2]
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
        if not self.history:
            return 0.0

        last_snapshot = self.history[-1]
        essential_shortfall = clamp(1.0 - last_snapshot.essential_fulfillment_rate, 0.0, 1.5)
        family_stress = clamp(last_snapshot.families_income_below_basket_share, 0.0, 1.5)
        mortality_rate = last_snapshot.deaths / max(1, last_snapshot.population)
        mortality_stress = clamp(
            mortality_rate * self.config.periods_per_year / 0.15,
            0.0,
            1.5,
        )
        population_drop = 0.0
        if len(self.history) >= 2:
            prior_snapshot = self.history[-2]
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
        if firm.last_market_share > 0.0:
            learned_share = firm.last_market_share
        else:
            learned_share = market_share
        learning_maturity = self._firm_learning_maturity(firm)
        learned_weight = 0.30 + 0.50 * learning_maturity
        blended_share = learned_weight * learned_share + (1.0 - learned_weight) * market_share
        return clamp(blended_share, max(0.01, 0.15 * market_share), 0.95)

    def _sales_anchor(self, firm: Firm, sector_total_demand: float, market_share: float) -> float:
        share_anchor = self._firm_demand_share_anchor(firm, market_share)
        market_anchor = max(1.0, sector_total_demand * share_anchor)
        sales_memory = self._smoothed_sales_reference(firm)
        learning_maturity = self._firm_learning_maturity(firm)
        market_weight = 0.85 - 0.20 * learning_maturity
        return max(1.0, market_weight * market_anchor + (1.0 - market_weight) * sales_memory)

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
        expected_sales = max(0.0, firm.last_expected_sales)
        if expected_sales <= 0.0:
            return

        learning_maturity = self._firm_learning_maturity(firm)
        learning_scale = 0.30 + 0.70 * learning_maturity
        prior_elasticity = self._sector_demand_elasticity_prior(firm.sector)
        realization = firm.last_sales / max(1.0, expected_sales)
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
                    saving_propensity=self.rng.uniform(0.08, 0.28),
                    money_trust=self.rng.uniform(0.45, 0.85),
                    consumption_impatience=self.rng.uniform(0.20, 0.80),
                    price_sensitivity=self.rng.uniform(0.6, 1.4),
                    need_scale=self.rng.uniform(0.9, 1.1),
                    sector_preference_weights=self._draw_household_sector_preference_weights(),
                    age_periods=age_periods,
                    desired_children=self._draw_desired_children(),
                )
            )
        return households

    def _build_entrepreneurs(self) -> list[Entrepreneur]:
        entrepreneurs: list[Entrepreneur] = []
        total_firms = len(SECTOR_SPECS) * max(1, self.config.firms_per_sector)
        worker_wealth_pool = sum(household.savings for household in self.households)
        capitalist_share = clamp(self.config.initial_capitalist_wealth_share, 0.05, 0.95)
        capitalist_wealth_pool = worker_wealth_pool * capitalist_share / max(1e-9, 1.0 - capitalist_share)
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
                )
            )
        return entrepreneurs

    def _build_firms(self) -> list[Firm]:
        firms: list[Firm] = []
        firm_id = 0
        sector_count = max(1, self.config.firms_per_sector)
        startup_slot_share = self._startup_slot_share()
        startup_demand_by_sector = {
            spec.key: self._baseline_demand(spec.key) * startup_slot_share
            for spec in SECTOR_SPECS
        }
        food_input_exempt_slots = {
            index for index in range(sector_count) if self.rng.random() < 0.10
        }
        if not food_input_exempt_slots:
            food_input_exempt_slots.add(self.rng.randrange(sector_count))
        for spec in SECTOR_SPECS:
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
        desired_output = expected_sales + firm.inventory
        firm.desired_workers = max(1, math.ceil(desired_output / max(0.1, effective_productivity)))
        firm.target_inventory = firm.inventory
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
            + firm.last_input_cost
            + firm.last_transport_cost
            + firm.last_fixed_overhead
            + firm.last_capital_charge
        )
        firm.last_unit_cost = (
            firm.last_total_cost / max(1.0, expected_sales)
        )
        firm.last_expected_sales = expected_sales
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
                capital_boost = math.sqrt(boost)
                firm.productivity = clamp(firm.productivity * capital_boost, 0.25, 25.0)
                firm.technology = clamp(
                    firm.technology * capital_boost,
                    0.75,
                    self.config.technology_cap,
                )
                firm.capital *= capital_boost
                firm.inventory = max(firm.inventory, target_sales * spec.target_inventory_ratio)
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
            household.employment_tenure = 0
            sector_output[sector_key] += marginal_output

    def _create_startup_firm(
        self,
        firm_id: int,
        spec,
        input_cost_exempt: bool = False,
        startup_demand: float | None = None,
        startup_slot_share: float | None = None,
    ) -> Firm:
        owner = self.entrepreneurs[firm_id]
        startup_slot_share = startup_slot_share if startup_slot_share is not None else self._startup_slot_share()
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
        productivity = (
            spec.base_productivity
            * self._entry_productivity_multiplier(spec.key)
            * self.rng.uniform(0.96, 1.04)
        )
        technology = self._initial_technology(spec.key)
        capital = base_capital_budget * package_funding_ratio
        input_cost_per_unit, transport_cost_per_unit, fixed_overhead = self._random_firm_cost_structure(spec)
        if input_cost_exempt:
            input_cost_per_unit = 0.0
        inventory_budget = base_inventory_budget * package_funding_ratio
        inventory = inventory_budget / max(0.1, spec.base_price)
        expected_sales = startup_demand * self.rng.uniform(0.9, 1.05) * max(0.10, package_funding_ratio)
        effective_productivity = productivity * technology * self._capital_efficiency(capital)
        desired_output = expected_sales + inventory
        desired_workers = max(1, math.ceil(desired_output / max(0.1, effective_productivity)))
        last_wage_bill = desired_workers * wage_offer
        capital_charge = capital * self.config.depreciation_rate
        last_input_cost = expected_sales * input_cost_per_unit
        last_transport_cost = expected_sales * transport_cost_per_unit
        last_fixed_overhead = fixed_overhead
        last_total_cost = (
            last_wage_bill
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
                previous_output_per_worker = firm.last_production / max(1, firm.last_worker_count)
                productivity_gain_ratio = clamp(
                    (effective_productivity - previous_output_per_worker) / max(0.1, previous_output_per_worker),
                    -0.25,
                    1.0,
                )
                market_share = competitiveness / total_competitiveness
                sales_anchor = self._sales_anchor(firm, sector_total_demand, market_share)
                baseline_demand = max(1.0, sales_anchor)
                inventory_target_multiplier = clamp(
                    1.10 - 0.18 * (firm.inventory_aversion - 1.0),
                    0.70,
                    1.25,
                )
                preliminary_inventory_target = max(
                    1.0,
                    baseline_demand * spec.target_inventory_ratio * inventory_target_multiplier,
                )
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
                if spec.key in ESSENTIAL_SECTOR_KEYS:
                    wage_adjustment += (
                        self.config.essential_wage_bargaining_bonus
                        * household_income_gap
                        * max(0.0, wage_room)
                    )
                    wage_adjustment += 0.05 * max(0.0, productivity_gain_ratio) * max(0.0, living_wage_gap)
                if firm.cash < firm.last_wage_bill * 0.5:
                    wage_adjustment -= 0.04
                wage_adjustment = clamp(wage_adjustment, -0.12, 0.24) * (0.35 + 0.65 * learning_maturity)
                firm.wage_offer = max(0.0, firm.wage_offer * (1.0 + wage_adjustment))

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
                    candidate_prices = [
                        min(
                            max(0.1, firm.last_unit_cost),
                            firm.price * self._firm_max_price_hike_ratio(firm),
                        )
                    ]
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
                    candidate_profit = (effective_price - variable_unit_cost) * expected_sales - fixed_cost
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
                    candidate_profit = (effective_price - variable_unit_cost) * prudent_sales - fixed_cost
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

                target_inventory = max(1.0, best_expected_sales * spec.target_inventory_ratio)
                if self._in_startup_grace() and spec.key in ESSENTIAL_SECTOR_KEYS:
                    firm.desired_workers = max(1, len(firm.workers))
                    firm.target_inventory = max(firm.target_inventory, target_inventory, firm.inventory)
                    firm.price = chosen_price
                    firm.last_expected_sales = max(best_expected_sales, firm.last_expected_sales)
                    firm.last_unit_cost = variable_unit_cost + fixed_cost / max(1.0, firm.last_expected_sales)
                    continue
                desired_output = max(0.0, best_expected_sales + target_inventory - firm.inventory)
                firm.desired_workers = max(1, math.ceil(desired_output / effective_productivity))
                firm.desired_workers = max(
                    1,
                    round(firm.employment_inertia * firm.last_worker_count + (1.0 - firm.employment_inertia) * firm.desired_workers),
                )
                if firm.last_worker_count > 0:
                    worker_adjustment_limit = 0.05 + 0.25 * learning_maturity
                    min_workers = max(1, math.floor(firm.last_worker_count * (1.0 - worker_adjustment_limit)))
                    max_workers = max(1, math.ceil(firm.last_worker_count * (1.0 + worker_adjustment_limit)))
                    firm.desired_workers = max(min_workers, min(max_workers, firm.desired_workers))
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
            age_years = self._household_age_years(household)
            if age_years < 30.0:
                age_bonus = self.config.young_worker_bonus
            elif age_years < 45.0:
                age_bonus = self.config.young_worker_bonus * 0.6
            elif age_years < self.config.senior_age_years:
                age_bonus = 0.0
            else:
                age_bonus = -self.config.senior_worker_penalty

            for firm in firm_order:
                if len(firm.workers) >= firm.desired_workers:
                    continue
                if firm.wage_offer < household.reservation_wage:
                    continue
                stability = firm.cash / max(1.0, firm.last_wage_bill + firm.price)
                labor_capacity = self._household_labor_capacity(household)
                score = (firm.wage_offer + 0.01 * firm.cash + 0.05 * stability) * labor_capacity + age_bonus
                if best_score is None or score > best_score:
                    best_score = score
                    best_firm = firm
            if best_firm is not None:
                best_firm.workers.append(household.id)
                household.employed_by = best_firm.id
                household.employment_tenure = 0

    def _produce_and_pay_wages(self) -> None:
        for firm in self.firms:
            if not firm.active:
                continue
            workers = firm.workers
            firm.last_worker_count = len(workers)
            effective_labor_units = sum(
                self._household_labor_capacity(self.households[worker_id]) for worker_id in workers
            )
            output_units = self._firm_effective_productivity(firm) * effective_labor_units
            firm.last_production = output_units
            firm.inventory += output_units
            self._period_production_units += output_units

            wage_bill = firm.wage_offer * len(workers)
            firm.last_wage_bill = wage_bill
            firm.cash -= wage_bill
            self._period_wages += wage_bill

            input_cost = output_units * firm.input_cost_per_unit
            transport_cost = output_units * firm.transport_cost_per_unit
            fixed_overhead = firm.fixed_overhead
            capital_charge = firm.capital * self.config.depreciation_rate
            firm.last_input_cost = input_cost
            firm.last_transport_cost = transport_cost
            firm.last_fixed_overhead = fixed_overhead
            firm.last_capital_charge = capital_charge
            if output_units > 0.0:
                costing_units = self._firm_costing_units(firm, output_units)
                firm.last_unit_cost = (
                    wage_bill + input_cost + transport_cost + fixed_overhead + capital_charge
                ) / costing_units
            nonwage_operating_cost = input_cost + transport_cost + fixed_overhead
            firm.cash -= nonwage_operating_cost
            self._queue_sector_payment("manufactured", input_cost + transport_cost)
            self._queue_sector_payment("housing", fixed_overhead)

            for worker_id in workers:
                household = self.households[worker_id]
                household.wage_income += firm.wage_offer
                household.last_income += firm.wage_offer

            self._cash_before_sales[firm.id] = firm.cash

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
            1.0 + 1.4 * family_saving_propensity + 0.7 * (1.0 - family_money_trust),
            0.75,
            4.0,
        )
        desired_cushion = family_basic_basket_cost * desired_cushion_months
        cushion_gap = clamp((desired_cushion - family_cash) / max(1.0, desired_cushion), 0.0, 1.0)
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
            -0.90
            + 1.20 * residual_share
            + 1.00 * cushion_gap
            + 0.85 * insecurity
            + 0.70 * (1.0 - trust)
            + 0.35 * family_saving_propensity
            - 1.05 * consumption_desire
        )
        savings_rate = 1.0 / (1.0 + math.exp(-score))
        return clamp(savings_rate, 0.0, 0.95)

    def _update_household_post_consumption_state(
        self,
        member: Household,
        *,
        allocated_units: dict[str, float],
        per_adult_savings: float,
        family_remaining_cash: float,
        family_basic_basket_cost: float,
        inflation_pressure: float,
    ) -> None:
        member.wage_income = 0.0
        member.last_consumption = allocated_units.copy()
        if self._household_age_years(member) >= self.config.entry_age_years:
            member.savings = per_adult_savings
        else:
            member.savings = 0.0

        food_coverage = self._household_sector_coverage(member, "food")
        housing_coverage = self._household_sector_coverage(member, "housing")
        clothing_coverage = self._household_sector_coverage(member, "clothing")
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

    def _consume_households(self) -> None:
        discretionary_firms = [firm for key in DISCRETIONARY_SECTOR_KEYS for firm in self._sector_firms(key)]
        discretionary_price_pressure = 1.0
        if discretionary_firms:
            discretionary_price_pressure = sum(
                firm.price / SECTOR_BY_KEY[firm.sector].base_price for firm in discretionary_firms
            ) / len(discretionary_firms)

        inflation_pressure = 0.0
        if self.history:
            current_price_index = self.history[-1].price_index
            prior_price_index = self.history[-2].price_index if len(self.history) > 1 else current_price_index
            if prior_price_index > 0.0:
                inflation_pressure = clamp((current_price_index / prior_price_index) - 1.0, 0.0, 0.35)

        for members in self._family_groups().values():
            family_members = [member for member in members if member.alive]
            if not family_members:
                continue

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

            family_basic_basket_cost = sum(self._essential_budget(member) for member in family_members)

            if family_cash <= 0.0:
                allocated_units_by_member = self._allocate_family_consumption_units(
                    family_members,
                    purchased_units_by_sector,
                )
                for member in family_members:
                    self._update_household_post_consumption_state(
                        member,
                        allocated_units=allocated_units_by_member.get(member.id, spending_log),
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

            for sector_key in ESSENTIAL_SECTOR_KEYS:
                desired_units = sum(
                    self._household_sector_desired_units(member, sector_key)
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
                            desired_units = intended_spend / max(0.1, average_price)
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
                total_preference_units = 0.0
                for sector_key in DISCRETIONARY_SECTOR_KEYS:
                    preference_units = sum(
                        self._household_sector_desired_units(member, sector_key)
                        for member in family_members
                    )
                    sector_preference_units[sector_key] = preference_units
                    total_preference_units += preference_units
                if total_preference_units <= 0.0:
                    total_preference_units = float(len(DISCRETIONARY_SECTOR_KEYS))
                    sector_preference_units = {key: 1.0 for key in DISCRETIONARY_SECTOR_KEYS}

                for sector_key in DISCRETIONARY_SECTOR_KEYS:
                    if cash <= 0.0:
                        continue
                    spec = SECTOR_BY_KEY[sector_key]
                    share = sector_preference_units[sector_key] / total_preference_units
                    neutral_spend = neutral_nonessential_budget * share
                    intended_spend = effective_nonessential_budget * share
                    average_price = self._average_sector_price(sector_key)
                    desired_units = intended_spend / max(0.1, average_price)
                    desired_units_neutral = neutral_spend / max(0.1, spec.base_price)
                    self._period_potential_demand_units += desired_units
                    self._period_sector_potential_demand_units[sector_key] += desired_units
                    self._period_sector_budget_demand_units[sector_key] += desired_units_neutral
                    sector_firms = self._sector_firms(sector_key)
                    if not sector_firms:
                        continue
                    cash, _ = self._purchase_from_sector(
                        family_price_sensitivity,
                        sector_key,
                        desired_units,
                        cash,
                        spending_log,
                    )

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
            )

            for member in family_members:
                self._update_household_post_consumption_state(
                    member,
                    allocated_units=allocated_units_by_member.get(member.id, spending_log),
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
            "food": 0.28,
            "housing": 0.18,
            "clothing": 0.12,
            "manufactured": 0.24,
            "leisure": 0.18,
        }

        for sector_key in ("food", "housing", "clothing", "manufactured", "leisure"):
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
            if len(firm.sales_history) > 6:
                del firm.sales_history[:-6]

            depreciation = firm.last_capital_charge
            firm.capital = max(0.0, firm.capital - depreciation)
            firm.last_total_cost = (
                firm.last_wage_bill
                + firm.last_input_cost
                + firm.last_transport_cost
                + firm.last_fixed_overhead
                + depreciation
            )
            pre_tax_profit = (
                revenue
                - firm.last_total_cost
            )
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
                cash_reserve = max(
                    20.0,
                    firm.last_wage_bill * self.config.cash_reserve_periods,
                )
                excess_cash_ratio = clamp(
                    (firm.cash - cash_reserve) / max(1.0, cash_reserve),
                    0.0,
                    3.0,
                )
                dividend_cap = max(0.0, firm.cash - 0.75 * cash_reserve)
                effective_payout_ratio = clamp(
                    self.config.payout_ratio
                    + 0.12 * max(0.0, profit / max(1.0, revenue))
                    + 0.10 * excess_cash_ratio
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
                    + 0.06 * max(0.0, firm.market_share_ambition - 1.0),
                    0.15,
                    0.60,
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
        for firm, investment, capital_investment, technology_investment in planned_investments:
            effective_investment = min(investment, max(0.0, firm.cash))
            if effective_investment <= 0.0:
                continue
            capital_share = capital_investment / investment if investment > 0.0 else 0.0
            actual_capital_investment = effective_investment * capital_share
            actual_technology_investment = effective_investment - actual_capital_investment
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
            technology_gain = technology_boost * actual_technology_investment / max(
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
                    firm.bankruptcy_streak = max(firm.bankruptcy_streak, self._firm_bankruptcy_limit(firm))
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
            firm.capital = 0.0
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
            firm.last_unit_cost = 0.0
            firm.last_market_share = 0.0
            firm.sales_history.clear()
            firm.last_expected_sales = 0.0
            firm.market_fragility_belief = 0.0
            firm.forecast_error_belief = 0.0
            firm.last_technology_investment = 0.0
            firm.last_technology_gain = 0.0
            firm.technology = 0.0
            firm.demand_elasticity = 0.0
            firm.bankruptcy_streak = 0
            firm.loss_streak = 0

        self._refresh_period_sector_caches()
        if self.config.replacement_enabled:
            self._attempt_endogenous_sector_entry()
            self._refresh_period_sector_caches()
        self._ensure_active_food_input_exemption()

    def _attempt_endogenous_sector_entry(self) -> None:
        for spec in SECTOR_SPECS:
            inactive_firms = [firm for firm in self.firms_by_sector.get(spec.key, []) if not firm.active]
            if not inactive_firms:
                continue
            demand_signal = self._baseline_demand(spec.key, use_current_period=True)
            if demand_signal <= 0.0:
                continue
            active_supply = sum(
                max(0.0, firm.last_expected_sales, firm.last_sales)
                for firm in self._sector_firms(spec.key)
            )
            entry_gap = demand_signal - 0.85 * active_supply
            if entry_gap <= 1.0:
                continue
            entry_firm = min(inactive_firms, key=lambda firm: firm.id)
            self._restart_firm(entry_firm, demand_signal=entry_gap)

    def _restart_firm(self, firm: Firm, demand_signal: float | None = None) -> bool:
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

        owner = self._best_entry_owner()
        if owner is None:
            return False

        available_surplus = self._owner_total_liquid(owner) - self.config.firm_restart_wealth_threshold
        if available_surplus <= 0.0:
            return False

        scale = available_surplus / base_restart_cost
        if scale < self.config.firm_restart_min_scale:
            return False
        scale = clamp(scale, self.config.firm_restart_min_scale, self.config.firm_restart_max_scale)
        restart_cost = min(
            available_surplus,
            self._restart_funding_need(spec, scale, demand_units=target_demand),
        )
        self._withdraw_owner_liquid(owner, restart_cost)
        package_multiplier = self.config.firm_restart_package_multiplier
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

        firm.active = True
        firm.age = 0
        firm.owner_id = owner.id
        firm.cash = startup_cash
        firm.capital = startup_capital
        firm.wage_offer = spec.base_wage * self.rng.uniform(0.96, 1.04)
        firm.productivity = (
            spec.base_productivity
            * self._entry_productivity_multiplier(spec.key)
            * self.rng.uniform(0.96, 1.04)
        )
        firm.technology = self._initial_technology(spec.key)
        firm.demand_elasticity = self._initial_demand_elasticity(spec.key)
        firm.input_cost_per_unit, firm.transport_cost_per_unit, firm.fixed_overhead = self._random_firm_cost_structure(spec)
        if firm.input_cost_exempt and spec.key == "food":
            firm.input_cost_per_unit = 0.0
        firm.inventory = max(0.0, startup_inventory_units)

        expected_sales = effective_demand * self.rng.uniform(0.9, 1.05) * max(0.10, package_funding_ratio)
        firm.last_worker_count = 0
        firm.sales_this_period = 0.0
        effective_productivity = firm.productivity * firm.technology * self._capital_efficiency(firm.capital)
        desired_output = expected_sales + firm.inventory
        desired_workers = max(1, math.ceil(desired_output / max(0.1, effective_productivity)))
        last_wage_bill = desired_workers * firm.wage_offer
        capital_charge = firm.capital * self.config.depreciation_rate
        last_input_cost = expected_sales * firm.input_cost_per_unit
        last_transport_cost = expected_sales * firm.transport_cost_per_unit
        last_total_cost = (
            last_wage_bill
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
        firm.forecast_caution = traits["forecast_caution"]
        firm.price = self._initial_firm_price(spec, unit_cost)
        self._refresh_firm_startup_state(firm, spec, expected_sales)
        firm.last_market_share = 0.0
        firm.sales_history = [expected_sales]
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
        price_index = sum(
            (self._average_sector_price(spec.key) / spec.base_price) * spec.household_demand_share
            for spec in SECTOR_SPECS
        )
        total_capital_stock = sum(firm.capital for firm in self.firms if firm.active)
        total_inventory_units = sum(firm.inventory for firm in self.firms if firm.active)
        goods_monetary_mass = self._current_goods_monetary_mass()
        total_liquid_money = self._current_total_liquid_money()
        worker_bank_deposits = sum(household.savings for household in active_households)
        worker_credit_outstanding = sum(max(0.0, household.loan_balance) for household in active_households)
        capitalist_bank_deposits = sum(entrepreneur.wealth for entrepreneur in self.entrepreneurs)
        capitalist_vault_cash = sum(entrepreneur.vault_cash for entrepreneur in self.entrepreneurs)
        capitalist_firm_cash = sum(firm.cash for firm in self.firms if firm.active)
        capitalist_credit_outstanding = sum(max(0.0, firm.loan_balance) for firm in self.firms if firm.active)
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
        gdp_per_capita = gdp_nominal / max(1, population)
        demand_fulfillment_rate = self._period_sales_units / max(1.0, self._period_potential_demand_units)
        essential_fulfillment_rate = self._period_essential_sales_units / max(1.0, self._period_essential_demand_units)
        money_velocity = gdp_nominal / max(1e-9, total_liquid_money)
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
        total_bank_deposits = sum(bank_deposits_by_id.values())
        total_bank_reserves = sum(bank.reserves for bank in active_banks)
        total_bank_loans_households = sum(bank_loans_households_by_id.values())
        total_bank_loans_firms = sum(bank_loans_firms_by_id.values())
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
        total_food_meals = 0.0
        total_health_fragility = 0.0
        for household in active_households:
            meals = self._household_food_meals_consumed(household)
            total_food_meals += meals
            total_health_fragility += household.health_fragility
            sufficient_meals = self._household_food_sufficient_meals(household)
            subsistence_meals = self._household_food_subsistence_meals(household)
            severe_meals = self._household_food_severe_hunger_meals(household)
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
            total_sales_units=self._period_sales_units,
            potential_demand_units=self._period_potential_demand_units,
            demand_fulfillment_rate=demand_fulfillment_rate,
            essential_demand_units=self._period_essential_demand_units,
            essential_sales_units=self._period_essential_sales_units,
            essential_fulfillment_rate=essential_fulfillment_rate,
            average_food_meals_per_person=average_food_meals_per_person,
            food_sufficient_share=food_sufficient_count / max(1, population),
            food_subsistence_share=food_subsistence_count / max(1, population),
            food_acute_hunger_share=food_acute_hunger_count / max(1, population),
            food_severe_hunger_share=food_severe_hunger_count / max(1, population),
            average_health_fragility=average_health_fragility,
            total_sales_revenue=self._period_sales_revenue,
            total_production_units=self._period_production_units,
            period_investment_spending=self._period_investment_spending,
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
            government_treasury_cash=self.government.treasury_cash,
            government_debt_outstanding=self.government.debt_outstanding,
            government_tax_revenue=self._period_government_tax_revenue,
            government_corporate_tax_revenue=self._period_government_corporate_tax_revenue,
            government_dividend_tax_revenue=self._period_government_dividend_tax_revenue,
            government_wealth_tax_revenue=self._period_government_wealth_tax_revenue,
            government_transfers=self._period_government_transfers,
            government_unemployment_support=self._period_government_unemployment_support,
            government_child_allowance=self._period_government_child_allowance,
            government_basic_support=self._period_government_basic_support,
            government_procurement_spending=self._period_government_procurement_spending,
            government_bond_issuance=self._period_government_bond_issuance,
            government_deficit=self._period_government_deficit,
            government_surplus=self._period_government_surplus,
            labor_share_gdp=self._period_wages / gdp_denom,
            profit_share_gdp=self._period_profit / gdp_denom,
            investment_share_gdp=self._period_investment_spending / gdp_denom,
            capitalist_consumption_share_gdp=self._period_entrepreneur_spending / gdp_denom,
            government_spending_share_gdp=(
                self._period_government_transfers + self._period_government_procurement_spending
            ) / gdp_denom,
            dividend_share_gdp=self._period_dividends_paid / gdp_denom,
            retained_profit_share_gdp=retained_profit / gdp_denom,
            central_bank_money_supply=self.central_bank.money_supply,
            central_bank_target_money_supply=self._period_central_bank_target_money_supply,
            central_bank_policy_rate=self.central_bank.policy_rate,
            central_bank_issuance=self._period_central_bank_issuance,
            cumulative_central_bank_issuance=self.central_bank.cumulative_issuance,
            household_credit_creation=household_credit_creation,
            firm_credit_creation=firm_credit_creation,
            commercial_bank_credit_creation=commercial_bank_credit_creation,
            average_bank_deposit_rate=average_bank_deposit_rate,
            average_bank_loan_rate=average_bank_loan_rate,
            total_bank_deposits=total_bank_deposits,
            total_bank_reserves=total_bank_reserves,
            total_bank_loans_households=total_bank_loans_households,
            total_bank_loans_firms=total_bank_loans_firms,
            total_bank_bond_holdings=total_bank_bond_holdings,
            total_bank_assets=total_bank_assets,
            total_bank_liabilities=total_bank_liabilities,
            bank_equity=bank_equity,
            bank_capital_ratio=bank_capital_ratio,
            bank_asset_liability_ratio=bank_asset_liability_ratio,
            bank_reserve_coverage_ratio=bank_reserve_coverage_ratio,
            bank_liquidity_ratio=bank_liquidity_ratio,
            bank_loan_to_deposit_ratio=bank_loan_to_deposit_ratio,
            bank_insolvent_share=insolvent_banks / max(1, len(active_banks)),
            money_velocity=money_velocity,
            total_liquid_money=total_liquid_money,
            total_household_savings=sum(household.savings for household in active_households),
        )

    def _build_firm_period_snapshots(self) -> list[FirmPeriodSnapshot]:
        periods_per_year = max(1, self.config.periods_per_year)
        year = ((self.period - 1) // periods_per_year) + 1
        period_in_year = ((self.period - 1) % periods_per_year) + 1
        snapshots: list[FirmPeriodSnapshot] = []
        for firm in self.firms:
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
                    sales=firm.last_sales,
                    expected_sales=firm.last_expected_sales,
                    revenue=firm.last_revenue,
                    production=firm.last_production,
                    profit=firm.last_profit,
                    total_cost=firm.last_total_cost,
                    loss_streak=firm.loss_streak,
                    market_share=firm.last_market_share,
                    market_fragility_belief=firm.market_fragility_belief,
                    forecast_error_belief=firm.forecast_error_belief,
                    target_inventory=firm.target_inventory,
                    age=firm.age,
                )
            )
        return snapshots


def run_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    return EconomySimulation(config=config).run()
