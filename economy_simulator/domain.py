from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SectorSpec:
    key: str
    name: str
    base_price: float
    base_wage: float
    base_productivity: float
    household_demand_share: float
    essential_need: float
    discretionary_weight: float
    target_inventory_ratio: float
    markup: float


SECTOR_SPECS: tuple[SectorSpec, ...] = (
    SectorSpec(
        key="food",
        name="Basic food",
        base_price=8.0,
        base_wage=7.0,
        base_productivity=0.95,
        household_demand_share=0.35,
        essential_need=0.35,
        discretionary_weight=0.0,
        target_inventory_ratio=0.22,
        markup=0.16,
    ),
    SectorSpec(
        key="housing",
        name="Housing and essential services",
        base_price=10.0,
        base_wage=7.8,
        base_productivity=0.85,
        household_demand_share=0.25,
        essential_need=0.25,
        discretionary_weight=0.0,
        target_inventory_ratio=0.20,
        markup=0.18,
    ),
    SectorSpec(
        key="clothing",
        name="Clothing and hygiene",
        base_price=7.5,
        base_wage=7.1,
        base_productivity=0.90,
        household_demand_share=0.15,
        essential_need=0.15,
        discretionary_weight=0.0,
        target_inventory_ratio=0.20,
        markup=0.16,
    ),
    SectorSpec(
        key="manufactured",
        name="Non-essential manufactured goods",
        base_price=9.0,
        base_wage=7.4,
        base_productivity=0.92,
        household_demand_share=0.15,
        essential_need=0.0,
        discretionary_weight=0.60,
        target_inventory_ratio=0.24,
        markup=0.24,
    ),
    SectorSpec(
        key="leisure",
        name="Leisure / entertainment / simple technology",
        base_price=8.2,
        base_wage=7.0,
        base_productivity=0.88,
        household_demand_share=0.10,
        essential_need=0.0,
        discretionary_weight=0.40,
        target_inventory_ratio=0.24,
        markup=0.23,
    ),
)

SECTOR_BY_KEY = {spec.key: spec for spec in SECTOR_SPECS}
ESSENTIAL_SECTOR_KEYS = ("food", "housing", "clothing")
DISCRETIONARY_SECTOR_KEYS = ("manufactured", "leisure")


@dataclass
class Household:
    id: int
    sex: str
    savings: float
    reservation_wage: float
    saving_propensity: float
    price_sensitivity: float
    need_scale: float
    sector_preference_weights: dict[str, float]
    age_periods: int
    employed_by: Optional[int] = None
    guardian_id: Optional[int] = None
    partner_id: Optional[int] = None
    mother_id: Optional[int] = None
    father_id: Optional[int] = None
    children_count: int = 0
    desired_children: int = 0
    last_birth_period: int = -999
    dependent_children: int = 0
    employment_tenure: int = 0
    wage_income: float = 0.0
    last_income: float = 0.0
    last_available_cash: float = 0.0
    alive: bool = True
    deprivation_streak: int = 0
    last_consumption: dict[str, float] = field(default_factory=dict)


@dataclass
class Entrepreneur:
    id: int
    wealth: float
    active: bool = True


@dataclass
class Firm:
    id: int
    sector: str
    owner_id: int
    cash: float
    inventory: float
    capital: float
    price: float
    wage_offer: float
    productivity: float
    technology: float = 1.0
    demand_elasticity: float = 1.0
    input_cost_per_unit: float = 0.0
    transport_cost_per_unit: float = 0.0
    fixed_overhead: float = 0.0
    markup_tolerance: float = 1.0
    volume_preference: float = 1.0
    inventory_aversion: float = 1.0
    employment_inertia: float = 0.75
    price_aggressiveness: float = 1.0
    cash_conservatism: float = 1.0
    market_share_ambition: float = 1.0
    active: bool = True
    age: int = 0
    desired_workers: int = 0
    workers: list[int] = field(default_factory=list)
    target_inventory: float = 0.0
    sales_this_period: float = 0.0
    sales_history: list[float] = field(default_factory=list)
    last_worker_count: int = 0
    last_sales: float = 0.0
    last_revenue: float = 0.0
    last_production: float = 0.0
    last_profit: float = 0.0
    last_total_cost: float = 0.0
    last_wage_bill: float = 0.0
    last_input_cost: float = 0.0
    last_transport_cost: float = 0.0
    last_fixed_overhead: float = 0.0
    last_capital_charge: float = 0.0
    last_unit_cost: float = 0.0
    last_market_share: float = 0.0
    last_expected_sales: float = 0.0
    market_fragility_belief: float = 0.0
    last_technology_investment: float = 0.0
    last_technology_gain: float = 0.0
    bankruptcy_streak: int = 0
    loss_streak: int = 0


@dataclass(frozen=True)
class FirmPeriodSnapshot:
    period: int
    year: int
    period_in_year: int
    firm_id: int
    sector: str
    active: bool
    workers: int
    desired_workers: int
    vacancies: int
    price: float
    wage_offer: float
    cash: float
    capital: float
    inventory: float
    productivity: float
    input_cost_per_unit: float
    transport_cost_per_unit: float
    fixed_overhead: float
    capital_charge: float
    unit_cost: float
    markup_tolerance: float
    volume_preference: float
    inventory_aversion: float
    employment_inertia: float
    price_aggressiveness: float
    cash_conservatism: float
    market_share_ambition: float
    technology: float
    technology_investment: float
    technology_gain: float
    sales: float
    revenue: float
    production: float
    profit: float
    total_cost: float
    loss_streak: int
    market_share: float
    target_inventory: float
    age: int


@dataclass(frozen=True)
class SimulationConfig:
    periods: int = 120
    households: int = 10000
    seed: int = 7
    periods_per_year: int = 12
    firms_per_sector: int = 20
    target_unemployment: float = 0.08
    capital_scale: float = 350.0
    depreciation_rate: float = 0.04
    wage_floor_multiplier: float = 0.80
    wage_ceiling_multiplier: float = 1.8
    price_floor_multiplier: float = 0.55
    price_ceiling_multiplier: float = 6.0
    payout_ratio: float = 0.20
    investment_rate: float = 0.15
    cash_reserve_periods: float = 1.0
    bankruptcy_cash_threshold: float = -50.0
    bankruptcy_streak_limit: int = 5
    bankruptcy_grace_period: int = 2
    critical_cash_threshold: float = -100.0
    startup_owner_wealth: float = 150.0
    startup_firm_cash: float = 180.0
    startup_firm_capital: float = 80.0
    startup_inventory_multiplier: float = 0.75
    firm_restart_package_multiplier: float = 0.1
    firm_restart_wealth_threshold: float = 1.0
    firm_restart_min_scale: float = 0.01
    firm_restart_max_scale: float = 3.0
    employment_contract_periods: int = 6
    essential_productivity_multiplier: float = 1.00
    nonessential_productivity_multiplier: float = 1.08
    essential_technology_multiplier: float = 1.18
    nonessential_technology_multiplier: float = 1.08
    nonessential_demand_multiplier: float = 0.30
    technology_investment_share_min: float = 0.10
    technology_investment_share_max: float = 0.35
    technology_gain_min: float = 0.02
    technology_gain_max: float = 0.08
    technology_depreciation_rate: float = 0.004
    technology_cap: float = 4.0
    max_firm_employment_share: float = 0.40
    initial_household_age_min_years: float = 0.0
    initial_household_age_max_years: float = 75.0
    entry_age_years: float = 18.0
    fertile_age_min_years: float = 18.0
    fertile_age_max_years: float = 40.0
    senior_age_years: float = 70.0
    retirement_age_years: float = 80.0
    max_age_years: float = 85.0
    annual_birth_rate: float = 0.15
    annual_base_death_rate: float = 0.005
    annual_senior_death_rate: float = 0.08
    starvation_death_periods: int = 3
    essential_sustenance_fraction: float = 0.70
    child_consumption_multiplier: float = 0.55
    senior_consumption_multiplier: float = 0.90
    senior_productivity_floor: float = 0.35
    young_worker_bonus: float = 0.08
    senior_worker_penalty: float = 0.12
    newborn_savings_min: float = 0.0
    newborn_savings_max: float = 12.0
    newborn_reservation_wage_min: float = 5.4
    newborn_reservation_wage_max: float = 6.8
    initial_household_savings_min: float = 0.0
    initial_household_savings_max: float = 40.0
    initial_capitalist_wealth_share: float = 0.50
    initial_capitalist_wealth_sigma: float = 0.65
    initial_owner_wealth_min: float = 120.0
    initial_owner_wealth_max: float = 220.0
    replacement_enabled: bool = True


@dataclass
class PeriodSnapshot:
    period: int
    year: int
    period_in_year: int
    population: int
    women: int
    men: int
    fertile_women: int
    children: int
    adults: int
    seniors: int
    labor_force: int
    employment_rate: float
    unemployment_rate: float
    children_with_guardian: int
    orphans: int
    family_units: int
    average_family_income: float
    average_family_resources: float
    average_family_basic_basket_cost: float
    family_income_to_basket_ratio: float
    family_resources_to_basket_ratio: float
    families_income_below_basket_share: float
    families_resources_below_basket_share: float
    total_wages: float
    total_sales_units: float
    potential_demand_units: float
    demand_fulfillment_rate: float
    essential_demand_units: float
    essential_sales_units: float
    essential_fulfillment_rate: float
    total_sales_revenue: float
    total_production_units: float
    period_investment_spending: float
    gdp_nominal: float
    gdp_per_capita: float
    total_capital_stock: float
    total_inventory_units: float
    total_profit: float
    active_firms: int
    bankruptcies: int
    births: int
    deaths: int
    average_age: float
    average_worker_savings: float
    gini_household_savings: float
    gini_owner_wealth: float
    capitalist_controlled_assets: float
    capitalist_asset_share: float
    price_index: float
    total_household_savings: float


@dataclass
class SimulationResult:
    config: SimulationConfig
    history: list[PeriodSnapshot]
    firm_history: list[FirmPeriodSnapshot]
    households: list[Household]
    entrepreneurs: list[Entrepreneur]
    firms: list[Firm]
