from __future__ import annotations

import math
import random

from .domain import (
    DISCRETIONARY_SECTOR_KEYS,
    ESSENTIAL_SECTOR_KEYS,
    SECTOR_BY_KEY,
    SECTOR_SPECS,
    Entrepreneur,
    Firm,
    FirmPeriodSnapshot,
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
        self.history: list[PeriodSnapshot] = []
        self.firm_history: list[FirmPeriodSnapshot] = []
        self.households = self._build_households()
        self.entrepreneurs = self._build_entrepreneurs()
        self.firms = self._build_firms()
        self.firm_by_id = {firm.id: firm for firm in self.firms}
        self.firms_by_sector: dict[str, list[Firm]] = {spec.key: [] for spec in SECTOR_SPECS}
        for firm in self.firms:
            self.firms_by_sector[firm.sector].append(firm)
        self._next_household_id = len(self.households)
        self._assign_initial_guardians()

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
        )

    def step(self) -> PeriodSnapshot:
        self.period += 1
        self._reset_period_counters()
        self._reset_household_labor_state()
        self._age_employment_contracts()
        self._refresh_family_links()

        last_unemployment = self.history[-1].unemployment_rate if self.history else 0.12
        self._update_firm_policies(last_unemployment)
        self._match_labor()
        self._produce_and_pay_wages()
        self._consume_households()
        self._settle_firms()
        self._resolve_bankruptcy_and_entry()
        _, _, _, _, _, _, _, _, _, current_unemployment, _ = self._population_metrics()
        self._apply_demography(current_unemployment)

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
        self._bankruptcies = 0
        self._period_births = 0
        self._period_deaths = 0
        self._period_sector_potential_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_sales_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        self._period_sector_budget_demand_units = {spec.key: 0.0 for spec in SECTOR_SPECS}
        for firm in self.firms:
            firm.sales_this_period = 0.0
            firm.last_technology_investment = 0.0
            firm.last_technology_gain = 0.0

    def _reset_household_labor_state(self) -> None:
        for household in self.households:
            household.wage_income = 0.0
            household.last_income = 0.0
            household.last_available_cash = 0.0
            household.last_consumption = {spec.key: 0.0 for spec in SECTOR_SPECS}

    def _release_household_from_employment(self, household: Household) -> None:
        firm_id = household.employed_by
        if firm_id is not None:
            firm = self.firm_by_id.get(firm_id)
            if firm is not None:
                firm.workers = [worker_id for worker_id in firm.workers if worker_id != household.id]
        household.employed_by = None
        household.employment_tenure = 0

    def _firm_hiring_capacity(self, firm: Firm, max_employees_per_firm: int) -> int:
        return min(firm.desired_workers, max_employees_per_firm)

    def _best_available_wage_offer(
        self,
        current_firm_id: int,
        max_employees_per_firm: int,
    ) -> float | None:
        best_wage: float | None = None
        for firm in self.firms:
            if not firm.active or firm.id == current_firm_id:
                continue
            if len(firm.workers) >= self._firm_hiring_capacity(firm, max_employees_per_firm):
                continue
            if best_wage is None or firm.wage_offer > best_wage:
                best_wage = firm.wage_offer
        return best_wage

    def _age_employment_contracts(self) -> None:
        contract_periods = max(1, self.config.employment_contract_periods)
        active_households = self._active_households()
        max_employees_per_firm = max(
            1,
            math.ceil(len(active_households) * self.config.max_firm_employment_share),
        )
        for household in active_households:
            if household.employed_by is None:
                continue
            firm = self.firm_by_id.get(household.employed_by)
            if firm is None or not firm.active:
                self._release_household_from_employment(household)
                continue
            household.employment_tenure += 1
            if household.employment_tenure >= contract_periods:
                current_capacity = self._firm_hiring_capacity(firm, max_employees_per_firm)
                if len(firm.workers) <= current_capacity:
                    best_wage = self._best_available_wage_offer(firm.id, max_employees_per_firm)
                    if best_wage is None or best_wage <= firm.wage_offer:
                        household.employment_tenure = 0
                        continue
                self._release_household_from_employment(household)

    def _active_households(self) -> list[Household]:
        return [household for household in self.households if household.alive]

    def _sector_firms(self, sector_key: str, active_only: bool = True) -> list[Firm]:
        firms = self.firms_by_sector.get(sector_key, [])
        if active_only:
            return [firm for firm in firms if firm.active]
        return list(firms)

    def _average_sector_price(self, sector_key: str) -> float:
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
        for firm, weight in sorted(weighted_firms, key=lambda item: item[0].price):
            if cash <= 0.0 or desired_units <= 0.0:
                break
            target_units = desired_units * (weight / total_weight)
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

    def _draw_household_sector_preference_weights(self) -> dict[str, float]:
        return {
            "food": self.rng.uniform(0.90, 1.12),
            "housing": self.rng.uniform(0.88, 1.10),
            "clothing": self.rng.uniform(0.75, 1.25),
            "manufactured": self.rng.uniform(0.45, 1.75),
            "leisure": self.rng.uniform(0.35, 1.85),
        }

    def _household_sector_preference(self, household: Household, sector_key: str) -> float:
        return max(0.2, household.sector_preference_weights.get(sector_key, 1.0))

    def _household_essential_share(self, household: Household, sector_key: str) -> float:
        spec = SECTOR_BY_KEY[sector_key]
        total_weight = sum(
            SECTOR_BY_KEY[key].essential_need * self._household_sector_preference(household, key)
            for key in ESSENTIAL_SECTOR_KEYS
        )
        if total_weight <= 0.0:
            total_weight = sum(SECTOR_BY_KEY[key].essential_need for key in ESSENTIAL_SECTOR_KEYS)
        weighted_need = spec.essential_need * self._household_sector_preference(household, sector_key)
        return weighted_need / max(1e-9, total_weight)

    def _household_discretionary_share(self, household: Household, sector_key: str) -> float:
        spec = SECTOR_BY_KEY[sector_key]
        total_weight = sum(
            SECTOR_BY_KEY[key].household_demand_share * self._household_sector_preference(household, key)
            for key in DISCRETIONARY_SECTOR_KEYS
        )
        if total_weight <= 0.0:
            total_weight = sum(SECTOR_BY_KEY[key].household_demand_share for key in DISCRETIONARY_SECTOR_KEYS)
        weighted_preference = spec.household_demand_share * self._household_sector_preference(household, sector_key)
        return weighted_preference / max(1e-9, total_weight)

    def _household_sector_desired_units(self, household: Household, sector_key: str) -> float:
        base_units = household.need_scale * self._household_consumption_multiplier(household)
        if sector_key in ESSENTIAL_SECTOR_KEYS:
            return base_units * self._household_essential_share(household, sector_key)
        discretionary_scale = (
            self.config.nonessential_demand_multiplier
            * sum(SECTOR_BY_KEY[key].household_demand_share for key in DISCRETIONARY_SECTOR_KEYS)
        )
        return base_units * discretionary_scale * self._household_discretionary_share(household, sector_key)

    def _structural_sector_demand(self, spec_key: str) -> float:
        return sum(
            self._household_sector_desired_units(household, spec_key)
            for household in self._active_households()
        )

    def _essential_budget(self, household: Household) -> float:
        budget = 0.0
        for sector_key in ESSENTIAL_SECTOR_KEYS:
            price = self._average_sector_price(sector_key)
            desired_units = self._household_sector_desired_units(household, sector_key)
            budget += desired_units * price
        return budget

    def _select_guardian_for_child(self, child: Household, exclude_id: int | None = None) -> Household | None:
        candidates = [
            household
            for household in self._active_households()
            if self._household_age_years(household) >= self.config.entry_age_years
            and household.id != child.id
            and household.id != exclude_id
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda household: (
                household.dependent_children,
                -self._household_cash_balance(household),
                self._household_age_years(household),
                household.id,
            )
        )
        return candidates[0]

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
        for household in self._active_households():
            partner = self._household_by_id(household.partner_id) if household.partner_id is not None else None
            if partner is not None and partner.alive and partner.partner_id == household.id:
                continue
            if household.partner_id is not None:
                household.partner_id = None

        males = self._adult_partnership_candidates("M")
        females = self._adult_partnership_candidates("F")
        if not males or not females:
            return

        males.sort(key=lambda household: (self._household_age_years(household), -household.savings, household.id))
        females.sort(key=lambda household: (self._household_age_years(household), -household.savings, household.id))

        for male in males:
            if male.partner_id is not None:
                continue

            best_female = None
            best_score = None
            for female in females:
                if female.partner_id is not None:
                    continue
                age_gap = abs(self._household_age_years(male) - self._household_age_years(female))
                if age_gap > 14.0:
                    continue
                if male.desired_children == 0 and female.desired_children == 0:
                    continue
                score = self._pair_score(male, female)
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
                male.last_birth_period = self.period - 24
            if best_female.last_birth_period < 0:
                best_female.last_birth_period = self.period - 24

    def _assign_initial_guardians(self) -> None:
        children = [household for household in self.households if self._household_age_years(household) < self.config.entry_age_years]
        for child in children:
            guardian = self._select_guardian_for_child(child)
            if guardian is not None:
                self._assign_guardian(child, guardian)
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
        return [
            household
            for household in self._active_households()
            if self._household_age_years(household) >= self.config.entry_age_years
            and self._household_labor_capacity(household) > 0.0
            and household.employed_by is None
        ]

    def _household_labor_capacity(self, household: Household) -> float:
        age_years = self._household_age_years(household)
        if age_years < self.config.entry_age_years:
            return 0.0
        if age_years < self.config.senior_age_years:
            return 1.0
        if age_years >= self.config.max_age_years:
            return 0.0

        retirement_span = max(1.0, self.config.retirement_age_years - self.config.senior_age_years)
        progress = clamp((age_years - self.config.senior_age_years) / retirement_span, 0.0, 1.0)
        return clamp(
            1.0 - progress * (1.0 - self.config.senior_productivity_floor),
            self.config.senior_productivity_floor,
            1.0,
        )

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
        groups: dict[int, list[Household]] = {}
        for household in self._active_households():
            root_id = (
                self._family_root_for_child(household)
                if self._household_age_years(household) < self.config.entry_age_years
                else self._family_root_for_adult(household)
            )
            groups.setdefault(root_id, []).append(household)
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
            if self._household_labor_capacity(household) > 0.0:
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
        return household.age_periods / max(1, self.config.periods_per_year)

    def _is_fertile(self, household: Household) -> bool:
        age_years = self._household_age_years(household)
        return self.config.fertile_age_min_years <= age_years <= self.config.fertile_age_max_years

    def _birth_household(self, mother: Household, father: Household) -> None:
        parent_ids = {mother.id, father.id}
        guardian = mother if mother.savings + mother.wage_income >= father.savings + father.wage_income else father
        self.households.append(
            Household(
                id=self._next_household_id,
                sex=self.rng.choice(("F", "M")),
                savings=self.rng.uniform(
                    self.config.newborn_savings_min,
                    self.config.newborn_savings_max,
                ),
                reservation_wage=self.rng.uniform(
                    self.config.newborn_reservation_wage_min,
                    self.config.newborn_reservation_wage_max,
                ),
                saving_propensity=self.rng.uniform(0.05, 0.22),
                price_sensitivity=self.rng.uniform(0.7, 1.3),
                need_scale=self.rng.uniform(0.9, 1.05),
                sector_preference_weights=self._draw_household_sector_preference_weights(),
                age_periods=0,
                guardian_id=guardian.id,
                mother_id=mother.id,
                father_id=father.id,
                desired_children=self._draw_desired_children(),
            )
        )
        for parent_id in parent_ids:
            parent = self._household_by_id(parent_id)
            if parent is not None:
                parent.dependent_children += 1
        mother.children_count += 1
        father.children_count += 1
        mother.last_birth_period = self.period
        father.last_birth_period = self.period
        self._next_household_id += 1
        self._period_births += 1

    def _household_death_probability(self, household: Household, unemployment_rate: float, average_savings: float) -> float:
        age_years = self._household_age_years(household)
        if age_years >= self.config.max_age_years:
            return 1.0

        age_span = max(1.0, self.config.max_age_years - self.config.fertile_age_min_years)
        age_pressure = clamp((age_years - self.config.fertile_age_min_years) / age_span, 0.0, 1.0)
        annual_probability = self.config.annual_base_death_rate + self.config.annual_senior_death_rate * (age_pressure ** 3)

        if household.deprivation_streak >= self.config.starvation_death_periods:
            annual_probability += 0.35

        if household.savings < 1.0:
            annual_probability += 0.01

        liquidity_buffer = household.savings + household.wage_income
        liquidity_stress = liquidity_buffer < self._essential_budget(household) * 0.35
        if (
            household.employed_by is None
            and self._household_labor_capacity(household) > 0.0
            and liquidity_stress
            and unemployment_rate > self.config.target_unemployment
        ):
            annual_probability += min(0.03, (unemployment_rate - self.config.target_unemployment) * 0.15)

        if average_savings < 8.0:
            annual_probability += 0.01

        return self._annual_to_period_probability(annual_probability)

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
        birth_probability = self._annual_to_period_probability(self.config.annual_birth_rate * prosperity)

        for household in active_households:
            previous_age_years = self._household_age_years(household)
            household.age_periods += 1
            current_age_years = self._household_age_years(household)
            if previous_age_years < self.config.entry_age_years <= current_age_years:
                self._release_guardian_dependency(household)
            death_probability = self._household_death_probability(household, unemployment_rate, average_savings)
            if self.rng.random() < death_probability:
                self._clear_partner_link(household)
                if current_age_years < self.config.entry_age_years:
                    self._release_guardian_dependency(household)
                self._release_household_from_employment(household)
                household.alive = False
                household.wage_income = 0.0
                household.savings = 0.0
                household.deprivation_streak = 0
                self._period_deaths += 1

        self._reassign_orphans()
        fertile_mothers = [
            household
            for household in self._active_households()
            if household.sex == "F"
            and household.partner_id is not None
            and self._household_age_years(household) <= self.config.fertile_age_max_years
            and self._is_fertile(household)
            and household.children_count < max(0, household.desired_children)
            and (self.period - household.last_birth_period) >= 24
        ]
        for mother in fertile_mothers:
            father = self._household_by_id(mother.partner_id) if mother.partner_id is not None else None
            if father is None or not father.alive:
                continue
            remaining_desire = max(0, mother.desired_children - mother.children_count)
            family_bias = clamp(0.35 + 0.15 * remaining_desire, 0.35, 1.75)
            if self.rng.random() < clamp(birth_probability * family_bias, 0.0, 0.95):
                self._birth_household(mother, father)

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
        observed_demand = self._observed_sector_demand_signal(spec_key, use_current_period=use_current_period)
        if observed_demand > 0.0:
            if spec_key in ESSENTIAL_SECTOR_KEYS:
                return max(0.60 * structural_demand, 0.65 * observed_demand + 0.35 * structural_demand)
            return max(0.25 * structural_demand, 0.75 * observed_demand + 0.25 * structural_demand)
        return structural_demand

    def _random_firm_cost_structure(self, spec) -> tuple[float, float, float]:
        # Los bienes esenciales parten con una estructura de costos mas ligera.
        # La idea es reflejar que la produccion basica suele apoyarse en escala,
        # estandarizacion y menor carga de distribucion que los bienes discrecionales.
        if spec.key == "food":
            input_range = (0.06, 0.14)
            transport_range = (0.02, 0.06)
            overhead_range = (0.80, 1.35)
        elif spec.key == "housing":
            input_range = (0.07, 0.16)
            transport_range = (0.02, 0.06)
            overhead_range = (0.90, 1.45)
        elif spec.key == "clothing":
            input_range = (0.065, 0.15)
            transport_range = (0.02, 0.055)
            overhead_range = (0.85, 1.40)
        elif spec.key == "manufactured":
            input_range = (0.10, 0.22)
            transport_range = (0.03, 0.10)
            overhead_range = (1.15, 2.15)
        else:
            input_range = (0.11, 0.24)
            transport_range = (0.04, 0.12)
            overhead_range = (1.20, 2.30)

        input_cost_per_unit = spec.base_price * self.rng.uniform(*input_range)
        transport_cost_per_unit = spec.base_price * self.rng.uniform(*transport_range)
        fixed_overhead = spec.base_wage * self.rng.uniform(*overhead_range)
        return input_cost_per_unit, transport_cost_per_unit, fixed_overhead

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

    def _restart_funding_need(self, spec, scale: float, demand_units: float | None = None) -> float:
        package_multiplier = self.config.firm_restart_package_multiplier
        baseline_demand = max(
            0.0,
            demand_units if demand_units is not None else self._baseline_demand(spec.key, use_current_period=True),
        )
        inventory_units = max(
            1.0,
            baseline_demand
            * spec.target_inventory_ratio
            * self.config.startup_inventory_multiplier
            * scale
            * package_multiplier,
        )
        inventory_funding = inventory_units * spec.base_price * 0.25
        return (self.config.startup_firm_cash + self.config.startup_firm_capital) * scale * package_multiplier + inventory_funding

    def _startup_slot_share(self) -> float:
        return 1.0 / max(1, self.config.firms_per_sector)

    def _startup_funding_need(
        self,
        spec,
        demand_units: float | None = None,
        package_scale: float = 1.0,
    ) -> float:
        baseline_demand = max(0.0, demand_units if demand_units is not None else self._baseline_demand(spec.key))
        inventory_units = max(
            1.0,
            baseline_demand * spec.target_inventory_ratio * self.config.startup_inventory_multiplier,
        )
        inventory_funding = inventory_units * spec.base_price
        return (
            (self.config.startup_firm_cash + self.config.startup_firm_capital) * max(0.05, package_scale)
            + inventory_funding
        )

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

    def _best_entry_owner(self) -> Entrepreneur | None:
        threshold = self.config.firm_restart_wealth_threshold
        candidates = [
            owner
            for owner in self.entrepreneurs
            if owner.wealth - threshold > 0.0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda owner: owner.wealth)

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
        firm.demand_elasticity = blend(firm.demand_elasticity, peer.demand_elasticity, 0.45, 2.75)
        firm.wage_offer = clamp(
            peer.wage_offer * self.rng.uniform(0.95, 1.05),
            spec.base_wage * self.config.wage_floor_multiplier,
            spec.base_wage * self.config.wage_ceiling_multiplier,
        )
        firm.price = self._initial_firm_price(spec, max(0.1, peer.last_unit_cost * self.rng.uniform(0.95, 1.05)))
        firm.input_cost_per_unit = max(0.0, peer.input_cost_per_unit * self.rng.uniform(0.92, 1.08))
        firm.transport_cost_per_unit = max(0.0, peer.transport_cost_per_unit * self.rng.uniform(0.92, 1.08))
        firm.fixed_overhead = max(0.0, peer.fixed_overhead * self.rng.uniform(0.92, 1.08))
        firm.sales_history = [
            max(0.0, sale * self.rng.uniform(0.90, 1.10))
            for sale in peer.sales_history[-6:]
        ] or [max(1.0, peer.last_sales)]
        firm.last_expected_sales = max(1.0, sum(firm.sales_history) / len(firm.sales_history))
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
        firm.demand_elasticity = clamp(
            0.80 * firm.demand_elasticity + 0.20 * self._initial_demand_elasticity(spec.key),
            0.45,
            2.75,
        )
        firm.wage_offer = clamp(
            firm.wage_offer * self.rng.uniform(0.94, 1.06),
            spec.base_wage * self.config.wage_floor_multiplier,
            spec.base_wage * self.config.wage_ceiling_multiplier,
        )
        firm.price = clamp(
            firm.price * self.rng.uniform(0.92, 1.08),
            spec.base_price * 0.35,
            spec.base_price * self.config.price_ceiling_multiplier,
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
        }

    def _price_search_candidates(self, firm: Firm, spec, variable_unit_cost: float, target_price: float) -> list[float]:
        inventory_ratio = firm.inventory / max(1.0, firm.target_inventory)
        cash_cover = firm.cash / max(1.0, firm.last_wage_bill + firm.fixed_overhead + firm.capital * self.config.depreciation_rate)
        rejection_pressure = self._firm_rejection_signal(firm) * self._firm_price_hike_sensitivity(firm)
        penetration_allowed = (
            spec.key in ESSENTIAL_SECTOR_KEYS
            and inventory_ratio > 1.15
            and cash_cover > 1.05
        )

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
        floor_price = max(
            variable_unit_cost * (1.01 if penetration_allowed else 1.03),
            spec.base_price * (0.35 if penetration_allowed else personal_floor_multiplier),
        )
        ceiling_price = spec.base_price * self.config.price_ceiling_multiplier

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
            adjusted_multipliers.append(adjusted)

        candidates = {
            clamp(firm.price * multiplier, floor_price, ceiling_price)
            for multiplier in adjusted_multipliers
        }
        if rejection_pressure > 0.20:
            candidates.add(clamp(firm.price * (1.0 - 0.12 * rejection_pressure), floor_price, ceiling_price))
            candidates.add(clamp(firm.price * (1.0 - 0.22 * rejection_pressure), floor_price, ceiling_price))
        if rejection_pressure > 0.45:
            near_cost_price = max(variable_unit_cost * 1.02, firm.last_unit_cost * 1.01)
            candidates.add(clamp(near_cost_price, floor_price, ceiling_price))
        candidates.add(clamp(target_price, floor_price, ceiling_price))
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
        family_stress = clamp(last_snapshot.families_resources_below_basket_share, 0.0, 1.5)
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
        blended_share = 0.65 * learned_share + 0.35 * market_share
        return clamp(blended_share, max(0.01, 0.15 * market_share), 0.95)

    def _sales_anchor(self, firm: Firm, sector_total_demand: float, market_share: float) -> float:
        share_anchor = self._firm_demand_share_anchor(firm, market_share)
        return max(1.0, sector_total_demand * share_anchor)

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
        elasticity = clamp(firm.demand_elasticity, 0.45, 2.75)
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
        if candidate_price > reference_price:
            effective_elasticity *= 1.0 + 0.55 * rejection_pressure + 0.15 * clamp(markup_stretch, 0.0, 2.5)
        elif candidate_price < reference_price and rejection_pressure > 0.0:
            effective_elasticity *= clamp(1.0 - 0.10 * rejection_pressure, 0.70, 1.0)

        demand = baseline_demand * (price_ratio ** (-effective_elasticity))
        if candidate_price > reference_price and rejection_pressure > 0.0:
            demand *= math.exp(-1.35 * (price_ratio - 1.0) * rejection_pressure)
        elif candidate_price < reference_price and rejection_pressure > 0.0:
            demand *= 1.0 + 0.18 * (1.0 - price_ratio) * rejection_pressure

        if firm.sector in ESSENTIAL_SECTOR_KEYS:
            inventory_ratio = firm.inventory / max(1.0, firm.target_inventory)
            if inventory_ratio > 1.05 and candidate_price < reference_price:
                demand *= 1.0 + 0.12 * clamp(inventory_ratio - 1.0, 0.0, 2.5)

        return clamp(demand, 0.0, sector_total_demand)

    def _update_firm_demand_learning(self, firm: Firm) -> None:
        expected_sales = max(0.0, firm.last_expected_sales)
        if expected_sales <= 0.0:
            return

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

        firm.demand_elasticity = clamp(
            (1.0 - learning_rate) * firm.demand_elasticity + learning_rate * target_elasticity,
            0.45,
            2.75,
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
        for spec in SECTOR_SPECS:
            for _ in range(max(1, self.config.firms_per_sector)):
                firms.append(self._create_startup_firm(firm_id, spec))
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
        for sector_key in ESSENTIAL_SECTOR_KEYS:
            sector_firms = [firm for firm in firms if firm.sector == sector_key and firm.active]
            if not sector_firms:
                continue

            target_units = self._structural_sector_demand(sector_key)
            estimated_capacity = sum(
                self._firm_effective_productivity(firm) * max(1.0, firm.desired_workers)
                for firm in sector_firms
            )
            if estimated_capacity <= 0.0:
                continue

            # Boost only the missing part of the startup gap so essentials begin near coverage,
            # but still remain exposed to later market dynamics.
            boost = clamp(math.sqrt(target_units / estimated_capacity), 1.0, 2.5)
            if boost <= 1.0:
                continue

            for firm in sector_firms:
                firm.productivity = clamp(firm.productivity * boost, 0.25, 25.0)
                firm.technology = clamp(
                    firm.technology * boost,
                    0.75,
                    self.config.technology_cap,
                )
                firm.cash *= boost
                firm.capital *= boost
                firm.inventory *= boost
                self._refresh_firm_startup_state(firm, spec, firm.last_sales * boost)

    def _create_startup_firm(self, firm_id: int, spec) -> Firm:
        owner = self.entrepreneurs[firm_id]
        startup_slot_share = self._startup_slot_share()
        startup_demand = self._baseline_demand(spec.key) * startup_slot_share
        startup_need = self._startup_funding_need(
            spec,
            demand_units=startup_demand,
            package_scale=startup_slot_share,
        )
        # Fund the startup package from the owner's own wealth so capital is not created for free.
        startup_budget = min(owner.wealth, startup_need)
        startup_scale = startup_budget / max(1e-9, startup_need)
        owner.wealth = max(0.0, owner.wealth - startup_budget)

        wage_offer = spec.base_wage * self.rng.uniform(0.96, 1.04)
        productivity = (
            spec.base_productivity
            * self._entry_productivity_multiplier(spec.key)
            * self.rng.uniform(0.96, 1.04)
        )
        technology = self._initial_technology(spec.key)
        capital = (
            self.config.startup_firm_capital
            * startup_slot_share
            * startup_scale
            * self.rng.uniform(0.9, 1.1)
        )
        input_cost_per_unit, transport_cost_per_unit, fixed_overhead = self._random_firm_cost_structure(spec)
        inventory = (
            startup_demand
            * spec.target_inventory_ratio
            * self.config.startup_inventory_multiplier
            * startup_scale
        )
        expected_sales = startup_demand * self.rng.uniform(0.9, 1.05) * startup_scale
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
            cash=(
                self.config.startup_firm_cash
                * startup_slot_share
                * startup_scale
                * self.rng.uniform(0.9, 1.1)
            ),
            inventory=inventory,
            capital=capital,
            price=price,
            wage_offer=wage_offer,
            productivity=productivity,
            technology=technology,
            demand_elasticity=self._initial_demand_elasticity(spec.key),
            input_cost_per_unit=input_cost_per_unit,
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
            last_technology_investment=0.0,
            last_technology_gain=0.0,
            loss_streak=0,
        )
        self._refresh_firm_startup_state(firm, spec, expected_sales)
        return firm

    def _update_firm_policies(self, last_unemployment: float) -> None:
        living_wage_anchor = self._living_wage_anchor()
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
                effective_productivity = self._firm_effective_productivity(firm)
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
                wage_adjustment = (
                    0.12 * labor_tightness
                    + 0.12 * vacancy_ratio
                    + 0.08 * max(0.0, profit_ratio)
                    - 0.07 * max(0.0, -profit_ratio)
                )
                if firm.cash < firm.last_wage_bill * 0.5:
                    wage_adjustment -= 0.04
                firm.wage_offer = clamp(
                    firm.wage_offer * (1.0 + wage_adjustment),
                    spec.base_wage * self.config.wage_floor_multiplier,
                    spec.base_wage * self.config.wage_ceiling_multiplier,
                )
                if (
                    living_wage_anchor > 0.0
                    and firm.cash > 0.75 * max(1.0, firm.last_wage_bill)
                    and (
                        spec.key in ESSENTIAL_SECTOR_KEYS
                        or preliminary_desired_workers > firm.last_worker_count
                    )
                ):
                    living_wage_floor = living_wage_anchor * (
                        0.90 if spec.key in ESSENTIAL_SECTOR_KEYS else 0.85
                    )
                    firm.wage_offer = clamp(
                        max(firm.wage_offer, living_wage_floor),
                        spec.base_wage * self.config.wage_floor_multiplier,
                        spec.base_wage * self.config.wage_ceiling_multiplier,
                    )

                variable_unit_cost = (
                    firm.wage_offer / effective_productivity
                    + firm.input_cost_per_unit
                    + firm.transport_cost_per_unit
                )
                fixed_cost = firm.fixed_overhead + firm.capital * self.config.depreciation_rate
                average_unit_cost = variable_unit_cost + fixed_cost / max(1.0, baseline_demand)
                target_margin = spec.markup * clamp(
                    1.0 - 0.18 * (firm.markup_tolerance - 1.0),
                    0.70,
                    1.30,
                )
                target_price = average_unit_cost * (1.0 + target_margin)
                if self.period == 1 and spec.key in ESSENTIAL_SECTOR_KEYS:
                    candidate_prices = [max(0.1, firm.last_unit_cost)]
                    clearance_discount = 0.0
                else:
                    candidate_prices = self._price_search_candidates(firm, spec, variable_unit_cost, target_price)
                reference_price = max(0.1, firm.price)
                if self.period != 1 or spec.key not in ESSENTIAL_SECTOR_KEYS:
                    clearance_discount = self._inventory_clearance_discount(firm)

                candidate_records: list[tuple[float, float, float, float, float]] = []
                best_profit = float("-inf")
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
                    future_sales = clamp(expected_sales * retention, 0.0, sector_total_demand)
                    future_market_value = future_sales * max(0.0, effective_price - variable_unit_cost)
                    candidate_records.append(
                        (
                            effective_price,
                            expected_sales,
                            candidate_profit,
                            future_market_value,
                            market_hazard,
                        )
                    )
                    best_profit = max(best_profit, candidate_profit)

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
                if penetration_mode:
                    profit_floor_ratio = clamp(profit_floor_ratio - 0.03 * (firm.volume_preference - 1.0), 0.75, 0.98)
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
                price_span = max(1e-9, max_price - min_price)
                sales_span = max(1e-9, max_sales - min_sales)
                profit_span = max(1e-9, max_profit - min_profit)
                future_value_span = max(1e-9, max_future_value - min_future_value)
                market_hazard_span = max(1e-9, max_market_hazard - min_market_hazard)
                volume_weight = 0.85 + 0.30 * firm.volume_preference + 0.20 * firm.market_share_ambition
                profit_weight = 0.85 + 0.25 * firm.cash_conservatism
                price_weight = 0.80 + 0.25 * firm.markup_tolerance
                future_weight = 0.80 + 1.20 * self._firm_future_market_weight(firm)
                fragility_weight = 0.70 + 0.25 * firm.inventory_aversion + 0.20 * firm.cash_conservatism

                (
                    chosen_price,
                    best_expected_sales,
                    best_profit,
                    _chosen_future_market_value,
                    _chosen_market_hazard,
                ) = max(
                    qualifying_candidates,
                    key=lambda record: (
                        profit_weight * ((record[2] - min_profit) / profit_span)
                        + volume_weight * ((record[1] - min_sales) / sales_span)
                        + future_weight * ((record[3] - min_future_value) / future_value_span)
                        - price_weight * ((record[0] - min_price) / price_span),
                        - fragility_weight * ((record[4] - min_market_hazard) / market_hazard_span),
                        record[1],
                        -record[0],
                    ),
                )

                target_inventory = max(1.0, best_expected_sales * spec.target_inventory_ratio)
                desired_output = max(0.0, best_expected_sales + target_inventory - firm.inventory)
                firm.desired_workers = max(1, math.ceil(desired_output / effective_productivity))
                firm.desired_workers = max(
                    1,
                    round(firm.employment_inertia * firm.last_worker_count + (1.0 - firm.employment_inertia) * firm.desired_workers),
                )
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
        max_employees_per_firm = max(1, math.ceil(len(eligible_households) * self.config.max_firm_employment_share))

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
                if len(firm.workers) >= min(firm.desired_workers, max_employees_per_firm):
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
            firm.cash -= input_cost + transport_cost + fixed_overhead

            for worker_id in workers:
                household = self.households[worker_id]
                household.wage_income += firm.wage_offer
                household.last_income += firm.wage_offer

            self._cash_before_sales[firm.id] = firm.cash

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

    def _consume_households(self) -> None:
        discretionary_firms = [firm for key in DISCRETIONARY_SECTOR_KEYS for firm in self._sector_firms(key)]
        discretionary_price_pressure = 1.0
        if discretionary_firms:
            discretionary_price_pressure = sum(
                firm.price / SECTOR_BY_KEY[firm.sector].base_price for firm in discretionary_firms
            ) / len(discretionary_firms)

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

            for member in family_members:
                if adult_count > 0:
                    member.last_available_cash = (
                        family_last_available_cash if self._household_age_years(member) >= self.config.entry_age_years else 0.0
                    )
                else:
                    member.last_available_cash = family_last_available_cash

            if family_cash <= 0.0:
                for member in family_members:
                    member.savings = 0.0
                    member.wage_income = 0.0
                    member.last_consumption = spending_log.copy()
                    member.deprivation_streak += 1
                continue

            cash = family_cash
            essential_target_units = 0.0
            essential_units_bought = 0.0

            for sector_key in ESSENTIAL_SECTOR_KEYS:
                desired_units = sum(
                    self._household_sector_desired_units(member, sector_key)
                    for member in family_members
                )
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
                self._period_essential_sales_units += units_bought

            target_savings = family_cash * family_saving_propensity
            discretionary_budget_neutral = max(0.0, cash - target_savings)
            discretionary_budget_neutral *= family_consumption_multiplier
            discretionary_budget_effective = discretionary_budget_neutral * clamp(
                1.0 - family_price_sensitivity * max(0.0, discretionary_price_pressure - 1.0) * 0.35,
                0.25,
                1.0,
            )

            if discretionary_budget_neutral > 0.0:
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
                    neutral_spend = discretionary_budget_neutral * share
                    intended_spend = discretionary_budget_effective * share
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
            per_adult_savings = family_remaining_cash / max(1, adult_count)
            family_covered_essentials = (
                essential_units_bought / essential_target_units
                if essential_target_units > 0.0
                else 1.0
            )

            for member in family_members:
                member.wage_income = 0.0
                member.last_consumption = spending_log.copy()
                if self._household_age_years(member) >= self.config.entry_age_years:
                    member.savings = per_adult_savings
                else:
                    member.savings = 0.0

                if essential_target_units > 0.0 and family_covered_essentials < self.config.essential_sustenance_fraction:
                    member.deprivation_streak += 1
                else:
                    member.deprivation_streak = 0

    def _settle_firms(self) -> None:
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
            profit = (
                revenue
                - firm.last_total_cost
            )
            firm.last_profit = profit
            self._period_profit += profit

            if profit < 0.0:
                firm.loss_streak += 1
                if firm.loss_streak >= self._firm_adaptation_threshold(firm):
                    self._adapt_losing_firm(firm)
            else:
                firm.loss_streak = 0

            if profit > 0.0 and firm.cash > 0.0:
                owner = self.entrepreneurs[firm.owner_id]
                cash_reserve = max(
                    20.0,
                    firm.last_wage_bill * self.config.cash_reserve_periods,
                )
                dividend_cap = max(0.0, firm.cash - cash_reserve)
                dividend = min(profit * self.config.payout_ratio, dividend_cap)
                if dividend > 0.0:
                    firm.cash -= dividend
                    owner.wealth += dividend

                investment_cap = max(0.0, firm.cash - cash_reserve)
                investment = min(profit * self.config.investment_rate, investment_cap)
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
                    firm.cash -= investment
                    firm.capital += capital_investment
                    firm.technology = clamp(
                        firm.technology * (1.0 - self.config.technology_depreciation_rate),
                        0.75,
                        self.config.technology_cap,
                    )
                    technology_boost = self.rng.uniform(
                        self.config.technology_gain_min,
                        self.config.technology_gain_max,
                    )
                    technology_gain = technology_boost * technology_investment / max(
                        1.0, firm.last_wage_bill + firm.last_fixed_overhead + 1.0
                    )
                    firm.technology = clamp(
                        firm.technology * (1.0 + technology_gain),
                        0.75,
                        self.config.technology_cap,
                    )
                    firm.last_technology_investment = technology_investment
                    firm.last_technology_gain = technology_gain
                    self._period_investment_spending += investment

            if firm.last_production > 0.0 and firm.price < firm.last_unit_cost:
                utilization_ratio = firm.last_production / max(1.0, firm.last_expected_sales)
                if utilization_ratio < 0.35 and firm.cash >= self.config.bankruptcy_cash_threshold:
                    firm.bankruptcy_streak += 1
                else:
                    firm.bankruptcy_streak = max(firm.bankruptcy_streak, self._firm_bankruptcy_limit(firm))
            elif profit < 0.0 or firm.cash < self.config.bankruptcy_cash_threshold:
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
            should_fail = (
                firm.cash < self.config.critical_cash_threshold
                or (
                    firm.age >= self.config.bankruptcy_grace_period
                    and (
                        firm.cash < self.config.bankruptcy_cash_threshold
                        or firm.bankruptcy_streak >= dynamic_bankruptcy_limit
                    )
                )
            )
            if not should_fail:
                continue

            self._bankruptcies += 1
            owner = self.entrepreneurs[firm.owner_id]
            spec = SECTOR_BY_KEY[firm.sector]
            gross_loss = max(0.0, -firm.cash) + 0.25 * firm.capital + 0.10 * firm.last_wage_bill
            liquidation_value = max(0.0, firm.cash) + 0.35 * firm.capital + 0.25 * firm.inventory * spec.base_price
            net_loss = max(0.0, gross_loss - liquidation_value)
            owner.wealth = max(0.0, owner.wealth - net_loss)

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
            firm.last_technology_investment = 0.0
            firm.last_technology_gain = 0.0
            firm.technology = 0.0
            firm.demand_elasticity = 0.0
            firm.bankruptcy_streak = 0
            firm.loss_streak = 0

        if self.config.replacement_enabled:
            self._attempt_endogenous_sector_entry()

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

        available_surplus = owner.wealth - self.config.firm_restart_wealth_threshold
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
        owner.wealth = max(0.0, owner.wealth - restart_cost)
        package_multiplier = self.config.firm_restart_package_multiplier
        effective_scale = scale * package_multiplier

        firm.active = True
        firm.age = 0
        firm.owner_id = owner.id
        firm.cash = self.config.startup_firm_cash * effective_scale * self.rng.uniform(0.9, 1.1)
        firm.capital = self.config.startup_firm_capital * effective_scale * self.rng.uniform(0.9, 1.1)
        firm.wage_offer = spec.base_wage * self.rng.uniform(0.96, 1.04)
        firm.productivity = (
            spec.base_productivity
            * self._entry_productivity_multiplier(spec.key)
            * self.rng.uniform(0.96, 1.04)
        )
        firm.technology = self._initial_technology(spec.key)
        firm.demand_elasticity = self._initial_demand_elasticity(spec.key)
        firm.input_cost_per_unit, firm.transport_cost_per_unit, firm.fixed_overhead = self._random_firm_cost_structure(spec)
        firm.inventory = max(
            1.0,
            target_demand
            * spec.target_inventory_ratio
            * self.config.startup_inventory_multiplier
            * effective_scale,
        )

        expected_sales = target_demand * self.rng.uniform(0.9, 1.05) * effective_scale
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
        firm.last_technology_investment = 0.0
        firm.last_technology_gain = 0.0
        firm.last_total_cost = last_total_cost
        firm.loss_streak = 0
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
        price_index = sum(
            (self._average_sector_price(spec.key) / spec.base_price) * spec.household_demand_share
            for spec in SECTOR_SPECS
        )
        total_capital_stock = sum(firm.capital for firm in self.firms if firm.active)
        total_inventory_units = sum(firm.inventory for firm in self.firms if firm.active)
        capitalist_controlled_assets = (
            sum(entrepreneur.wealth for entrepreneur in self.entrepreneurs)
            + sum(firm.cash for firm in self.firms if firm.active)
            + sum(firm.capital for firm in self.firms if firm.active)
            + sum(firm.inventory * firm.price for firm in self.firms if firm.active)
        )
        capitalist_liquid_assets = (
            sum(entrepreneur.wealth for entrepreneur in self.entrepreneurs)
            + sum(firm.cash for firm in self.firms if firm.active)
        )
        capitalist_asset_share = capitalist_controlled_assets / max(
            1e-9,
            capitalist_controlled_assets + sum(household.savings for household in active_households),
        )
        capitalist_liquid_share = capitalist_liquid_assets / max(
            1e-9,
            capitalist_liquid_assets + sum(household.savings for household in active_households),
        )
        gdp_nominal = self._period_sales_revenue + self._period_investment_spending
        gdp_per_capita = gdp_nominal / max(1, population)
        demand_fulfillment_rate = self._period_sales_units / max(1.0, self._period_potential_demand_units)
        essential_fulfillment_rate = self._period_essential_sales_units / max(1.0, self._period_essential_demand_units)
        labor_force_households = [
            household
            for household in active_households
            if self._household_labor_capacity(household) > 0.0
        ]
        average_worker_savings = sum(household.savings for household in labor_force_households) / max(
            1, len(labor_force_households)
        )

        return PeriodSnapshot(
            period=self.period,
            year=year,
            period_in_year=period_in_year,
            population=population,
            women=women,
            men=men,
            fertile_women=fertile_women,
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
            total_sales_revenue=self._period_sales_revenue,
            total_production_units=self._period_production_units,
            period_investment_spending=self._period_investment_spending,
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
            gini_household_savings=gini([household.savings for household in active_households]),
            gini_owner_wealth=gini([entrepreneur.wealth for entrepreneur in self.entrepreneurs]),
            capitalist_controlled_assets=capitalist_controlled_assets,
            capitalist_asset_share=capitalist_liquid_share,
            price_index=price_index,
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
                    technology=firm.technology,
                    technology_investment=firm.last_technology_investment,
                    technology_gain=firm.last_technology_gain,
                    sales=firm.last_sales,
                    revenue=firm.last_revenue,
                    production=firm.last_production,
                    profit=firm.last_profit,
                    total_cost=firm.last_total_cost,
                    loss_streak=firm.loss_streak,
                    market_share=firm.last_market_share,
                    target_inventory=firm.target_inventory,
                    age=firm.age,
                )
            )
        return snapshots


def run_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    return EconomySimulation(config=config).run()
