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
        name="Manufacturing / industrial goods",
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
        base_wage=8.2,
        base_productivity=0.88,
        household_demand_share=0.10,
        essential_need=0.0,
        discretionary_weight=0.40,
        target_inventory_ratio=0.24,
        markup=0.23,
    ),
    SectorSpec(
        key="school",
        name="School / basic education",
        base_price=13.5,
        base_wage=9.1,
        base_productivity=0.84,
        household_demand_share=0.08,
        essential_need=0.0,
        discretionary_weight=0.58,
        target_inventory_ratio=0.08,
        markup=0.18,
    ),
    SectorSpec(
        key="university",
        name="University / advanced education",
        base_price=18.5,
        base_wage=11.2,
        base_productivity=0.80,
        household_demand_share=0.07,
        essential_need=0.0,
        discretionary_weight=0.66,
        target_inventory_ratio=0.07,
        markup=0.20,
    ),
)

SECTOR_BY_KEY = {spec.key: spec for spec in SECTOR_SPECS}
ESSENTIAL_SECTOR_KEYS = ("food", "housing", "clothing")
DISCRETIONARY_SECTOR_KEYS = ("manufactured", "leisure", "school", "university")


@dataclass
class Household:
    id: int
    sex: str
    savings: float
    reservation_wage: float
    saving_propensity: float
    higher_education_affinity: float
    money_trust: float
    consumption_impatience: float
    price_sensitivity: float
    need_scale: float
    sector_preference_weights: dict[str, float]
    age_periods: int
    essential_shares: dict[str, float] = field(default_factory=dict)
    discretionary_shares: dict[str, float] = field(default_factory=dict)
    bank_id: int = 0
    loan_balance: float = 0.0
    employed_by: Optional[int] = None
    guardian_id: Optional[int] = None
    partner_id: Optional[int] = None
    mother_id: Optional[int] = None
    father_id: Optional[int] = None
    children_count: int = 0
    desired_children: int = 0
    child_desire_pressure: float = 0.0
    last_birth_period: int = -999
    dependent_children: int = 0
    employment_tenure: int = 0
    wage_income: float = 0.0
    last_income: float = 0.0
    last_available_cash: float = 0.0
    alive: bool = True
    deprivation_streak: int = 0
    severe_hunger_streak: int = 0
    housing_deprivation_streak: int = 0
    clothing_deprivation_streak: int = 0
    health_fragility: float = 0.0
    last_consumption: dict[str, float] = field(default_factory=dict)
    last_perceived_utility: float = 0.0
    school_years_completed: float = 0.0
    university_years_completed: float = 0.0
    origin_record_period: int = -1
    low_resource_origin: bool = False
    origin_family_income_to_basket_ratio: float = 0.0
    origin_family_resources_to_basket_ratio: float = 0.0


@dataclass
class Entrepreneur:
    id: int
    wealth: float
    bank_id: int = 0
    vault_cash: float = 0.0
    consumption_propensity: float = 0.0
    entry_appetite: float = 1.0
    market_research_skill: float = 1.0
    entry_optimism: float = 0.0
    loan_balance: float = 0.0
    active: bool = True


@dataclass
class CentralBank:
    money_supply: float
    target_money_supply: float
    policy_rate: float = 0.0
    issuance_this_period: float = 0.0
    cumulative_issuance: float = 0.0


@dataclass
class CommercialBank:
    id: int
    name: str
    reserve_ratio: float = 0.20
    deposits: float = 0.0
    reserves: float = 0.0
    loans_households: float = 0.0
    loans_firms: float = 0.0
    bond_holdings: float = 0.0
    central_bank_borrowing: float = 0.0
    deposit_rate: float = 0.0
    loan_rate: float = 0.0
    bond_yield: float = 0.0
    interest_income: float = 0.0
    interest_expense: float = 0.0
    profits: float = 0.0
    active: bool = True


@dataclass
class Government:
    treasury_cash: float = 0.0
    bank_id: int = 0
    debt_outstanding: float = 0.0
    tax_revenue_this_period: float = 0.0
    corporate_tax_revenue: float = 0.0
    dividend_tax_revenue: float = 0.0
    wealth_tax_revenue: float = 0.0
    transfers_this_period: float = 0.0
    unemployment_support_this_period: float = 0.0
    child_allowance_this_period: float = 0.0
    basic_support_this_period: float = 0.0
    procurement_spending_this_period: float = 0.0
    education_spending_this_period: float = 0.0
    school_public_spending_this_period: float = 0.0
    university_public_spending_this_period: float = 0.0
    bond_issuance_this_period: float = 0.0
    deficit_this_period: float = 0.0
    surplus_this_period: float = 0.0
    cumulative_deficit: float = 0.0


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
    input_cost_exempt: bool = False
    transport_cost_per_unit: float = 0.0
    fixed_overhead: float = 0.0
    markup_tolerance: float = 1.0
    volume_preference: float = 1.0
    inventory_aversion: float = 1.0
    employment_inertia: float = 0.75
    price_aggressiveness: float = 1.0
    cash_conservatism: float = 1.0
    market_share_ambition: float = 1.0
    forecast_caution: float = 1.0
    active: bool = True
    bank_id: int = 0
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
    forecast_error_belief: float = 0.15
    last_technology_investment: float = 0.0
    last_technology_gain: float = 0.0
    loan_balance: float = 0.0
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
    demand_elasticity: float
    forecast_caution: float
    learning_maturity: float
    technology: float
    technology_investment: float
    technology_gain: float
    sales: float
    expected_sales: float
    revenue: float
    production: float
    profit: float
    total_cost: float
    loss_streak: int
    market_share: float
    market_fragility_belief: float
    forecast_error_belief: float
    target_inventory: float
    age: int


@dataclass(frozen=True)
class SimulationConfig:
    periods: int = 120
    households: int = 10000
    seed: int = 7
    periods_per_year: int = 12
    firms_per_sector: int = 40
    commercial_banks: int = 3
    target_unemployment: float = 0.08
    capital_scale: float = 350.0
    depreciation_rate: float = 0.04
    wage_floor_multiplier: float = 0.80
    wage_ceiling_multiplier: float = 1.8
    price_floor_multiplier: float = 0.55
    price_ceiling_multiplier: float = 6.0
    payout_ratio: float = 0.32
    investment_rate: float = 0.30
    cash_reserve_periods: float = 0.75
    bankruptcy_cash_threshold: float = -50.0
    bankruptcy_streak_limit: int = 5
    bankruptcy_grace_period: int = 2
    critical_cash_threshold: float = -100.0
    startup_owner_wealth: float = 150.0
    startup_firm_cash: float = 180.0
    startup_firm_capital: float = 80.0
    startup_inventory_multiplier: float = 0.75
    startup_expected_sales_share: float = 0.65
    firm_restart_package_multiplier: float = 0.1
    firm_restart_wealth_threshold: float = 1.0
    firm_restart_min_scale: float = 0.01
    firm_restart_max_scale: float = 3.0
    employment_contract_periods: int = 12
    essential_protection_periods: int = 24
    startup_essential_supply_buffer: float = 1.35
    startup_clothing_supply_multiplier: float = 2.0
    essential_productivity_multiplier: float = 1.00
    nonessential_productivity_multiplier: float = 1.08
    essential_technology_multiplier: float = 1.18
    nonessential_technology_multiplier: float = 1.08
    nonessential_demand_multiplier: float = 0.30
    extra_essential_coverage_cap: float = 1.10
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
    school_age_min_years: float = 6.0
    school_age_max_years: float = 18.0
    university_age_min_years: float = 18.0
    university_age_max_years: float = 30.0
    school_years_required: float = 12.0
    university_years_required: float = 4.0
    initial_school_completion_share: float = 0.88
    initial_university_completion_share: float = 0.25
    startup_grace_periods: int = 2
    firm_learning_warmup_periods: int = 18
    birth_interval_periods: int = 9
    annual_birth_rate: float = 0.15
    annual_birth_rate_capable_single: float = 0.10
    annual_birth_rate_capable_partnered: float = 0.50
    annual_birth_rate_noncapable: float = 0.03
    period_days: int = 31
    food_meals_per_day_sufficient: float = 3.0
    food_meals_per_day_subsistence: float = 2.0
    food_meals_per_day_severe: float = 1.0
    annual_base_death_rate: float = 0.005
    annual_senior_death_rate: float = 0.08
    period_base_death_probability: float = 0.0004
    period_senior_death_probability: float = 0.0065
    period_food_subsistence_death_risk: float = 0.002
    period_severe_hunger_death_risk: float = 0.014
    period_health_fragility_death_risk: float = 0.0035
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
    central_bank_enabled: bool = True
    central_bank_rule: str = "goods_growth"
    central_bank_target_velocity: float = 0.20
    central_bank_target_annual_inflation: float = 0.04
    central_bank_max_issue_share: float = 0.05
    central_bank_goods_growth_pass_through: float = 1.0
    central_bank_policy_rate_base: float = 0.02
    central_bank_policy_rate_floor: float = 0.00
    central_bank_policy_rate_ceiling: float = 0.12
    central_bank_productivity_dividend_share: float = 1.0
    reserve_ratio: float = 0.20
    bank_initial_reserve_multiplier: float = 1.00
    bank_bond_allocation_share: float = 0.50
    bank_min_capital_ratio: float = 0.08
    bank_bond_risk_weight: float = 0.20
    bank_loan_rate: float = 0.03
    bank_bond_yield: float = 0.02
    bank_deposit_rate_share: float = 0.55
    bank_household_max_debt_to_income: float = 8.0
    bank_household_max_interest_share: float = 0.35
    bank_firm_min_interest_coverage: float = 1.15
    bank_firm_max_debt_to_revenue: float = 6.0
    central_bank_discount_window_spread: float = 0.05
    entrepreneur_consumption_share: float = 0.20
    entrepreneur_vault_share: float = 0.02
    reservation_wage_adjustment_speed: float = 0.85
    reservation_wage_floor_share: float = 1.00
    living_wage_bargaining_weight: float = 0.45
    essential_wage_bargaining_bonus: float = 0.20
    government_enabled: bool = True
    government_corporate_tax_rate_low: float = 0.10
    government_corporate_tax_rate_mid: float = 0.18
    government_corporate_tax_rate_high: float = 0.28
    government_corporate_tax_margin_mid: float = 0.08
    government_corporate_tax_margin_high: float = 0.18
    government_dividend_tax_rate_low: float = 0.04
    government_dividend_tax_rate_mid: float = 0.10
    government_dividend_tax_rate_high: float = 0.18
    government_dividend_bracket_low: float = 2.0
    government_dividend_bracket_high: float = 8.0
    government_wealth_tax_rate: float = 0.002
    government_wealth_tax_threshold_multiple: float = 24.0
    government_unemployment_benefit_share: float = 0.30
    government_child_allowance_share: float = 0.10
    government_basic_support_gap_share: float = 0.35
    government_procurement_gap_share: float = 0.30
    public_school_budget_share: float = 0.015
    public_university_budget_share: float = 0.015
    government_procurement_price_sensitivity: float = 0.85
    government_spending_scale: float = 1.00
    government_spending_efficiency: float = 0.95
    government_countercyclical_enabled: bool = True
    government_recession_unemployment_buffer: float = 0.03
    government_recession_output_gap_threshold: float = 0.05
    government_recession_lookback_periods: int = 3
    government_countercyclical_transfer_weight: float = 0.70
    government_countercyclical_procurement_weight: float = 0.85
    government_countercyclical_support_multiplier_max: float = 2.00
    government_countercyclical_procurement_multiplier_max: float = 2.40
    initial_private_school_firms: int = 10
    initial_private_university_firms: int = 1
    track_firm_history: bool = False


@dataclass
class PeriodSnapshot:
    period: int
    year: int
    period_in_year: int
    population: int
    women: int
    men: int
    fertile_women: int
    fertile_capable_women: int
    fertile_capable_women_low_desire_no_birth: int
    fertile_capable_women_with_births: int
    fertile_families: int
    fertile_families_with_births: int
    fertile_capable_families: int
    fertile_capable_families_low_desire_no_birth: int
    fertile_capable_families_with_births: int
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
    essential_production_units: float
    essential_sales_units: float
    essential_fulfillment_rate: float
    people_full_essential_coverage: int
    full_essential_coverage_share: float
    average_food_meals_per_person: float
    food_sufficient_share: float
    food_subsistence_share: float
    food_acute_hunger_share: float
    food_severe_hunger_share: float
    average_health_fragility: float
    average_perceived_utility: float
    school_age_population: int
    university_age_population: int
    school_students: int
    university_students: int
    school_enrollment_share: float
    university_enrollment_share: float
    school_completion_share: float
    university_completion_share: float
    school_labor_share: float
    skilled_labor_share: float
    low_resource_school_enrollment_share: float
    low_resource_university_enrollment_share: float
    low_resource_university_student_share: float
    school_income_premium: float
    university_income_premium: float
    poverty_rate_without_university: float
    poverty_rate_with_university: float
    tracked_origin_adults: int
    low_resource_origin_adults: int
    low_resource_origin_upward_mobility_share: float
    low_resource_origin_university_completion_share: float
    low_resource_origin_university_upward_mobility_share: float
    low_resource_origin_nonuniversity_upward_mobility_share: float
    skilled_job_demand_share: float
    skilled_job_fill_rate: float
    skilled_labor_supply_to_demand_ratio: float
    total_sales_revenue: float
    total_production_units: float
    period_investment_spending: float
    startup_fixed_capital_formation: float
    startup_inventory_investment: float
    business_cost_recycled: float
    business_cost_to_firms: float
    business_cost_to_households: float
    business_cost_to_owners: float
    inheritance_transfers: float
    bankruptcy_cash_recoveries: float
    gdp_nominal: float
    gdp_per_capita: float
    total_capital_stock: float
    total_inventory_units: float
    total_profit: float
    active_firms: int
    active_school_firms: int
    active_university_firms: int
    bankruptcies: int
    births: int
    deaths: int
    average_age: float
    average_worker_savings: float
    worker_cash_available: float
    worker_cash_saved: float
    worker_voluntary_saved: float
    worker_involuntary_retained: float
    worker_bank_deposits: float
    worker_credit_outstanding: float
    gini_household_savings: float
    gini_owner_wealth: float
    capitalist_bank_deposits: float
    capitalist_vault_cash: float
    capitalist_firm_cash: float
    capitalist_credit_outstanding: float
    capitalist_productive_capital: float
    capitalist_inventory_value: float
    capitalist_controlled_assets: float
    capitalist_asset_share: float
    capitalist_liquid_share: float
    worker_liquid_share: float
    goods_monetary_mass: float
    price_index: float
    government_treasury_cash: float
    government_debt_outstanding: float
    government_tax_revenue: float
    government_corporate_tax_revenue: float
    government_dividend_tax_revenue: float
    government_wealth_tax_revenue: float
    government_transfers: float
    government_unemployment_support: float
    government_child_allowance: float
    government_basic_support: float
    government_procurement_spending: float
    government_education_spending: float
    government_school_spending: float
    government_university_spending: float
    government_bond_issuance: float
    government_deficit: float
    government_surplus: float
    recession_flag: float
    recession_intensity: float
    government_countercyclical_support_multiplier: float
    government_countercyclical_procurement_multiplier: float
    government_countercyclical_spending: float
    total_inventory_book_value: float
    household_final_consumption: float
    government_final_consumption: float
    gross_fixed_capital_formation: float
    change_in_inventories: float
    valuables_acquisition: float
    gross_capital_formation: float
    exports: float
    imports: float
    net_exports: float
    gdp_expenditure_sna: float
    gdp_expenditure_gap: float
    labor_share_gdp: float
    profit_share_gdp: float
    investment_share_gdp: float
    capitalist_consumption_share_gdp: float
    government_spending_share_gdp: float
    dividend_share_gdp: float
    retained_profit_share_gdp: float
    central_bank_money_supply: float
    central_bank_target_money_supply: float
    central_bank_policy_rate: float
    central_bank_issuance: float
    cumulative_central_bank_issuance: float
    household_credit_creation: float
    firm_credit_creation: float
    commercial_bank_credit_creation: float
    average_bank_deposit_rate: float
    average_bank_loan_rate: float
    total_bank_deposits: float
    total_bank_reserves: float
    total_bank_loans_households: float
    total_bank_loans_firms: float
    total_bank_bond_holdings: float
    total_bank_assets: float
    total_bank_liabilities: float
    bank_equity: float
    bank_recapitalization: float
    bank_resolution_events: int
    bank_undercapitalized_share: float
    bank_capital_ratio: float
    bank_asset_liability_ratio: float
    bank_reserve_coverage_ratio: float
    bank_liquidity_ratio: float
    bank_loan_to_deposit_ratio: float
    bank_insolvent_share: float
    money_velocity: float
    total_liquid_money: float
    total_household_savings: float


@dataclass
class SimulationResult:
    config: SimulationConfig
    history: list[PeriodSnapshot]
    firm_history: list[FirmPeriodSnapshot]
    households: list[Household]
    entrepreneurs: list[Entrepreneur]
    firms: list[Firm]
    central_bank: CentralBank
    banks: list[CommercialBank]
    government: Government
