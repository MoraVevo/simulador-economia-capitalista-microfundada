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
        base_price=160.0,
        base_wage=18.0,
        base_productivity=0.30,
        household_demand_share=0.0,
        essential_need=0.0,
        discretionary_weight=0.0,
        target_inventory_ratio=0.0,
        markup=0.35,
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
        base_productivity=0.88,
        household_demand_share=0.10,
        essential_need=0.0,
        discretionary_weight=0.82,
        target_inventory_ratio=0.12,
        markup=0.18,
    ),
    SectorSpec(
        key="university",
        name="University / advanced education",
        base_price=18.5,
        base_wage=11.2,
        base_productivity=0.84,
        household_demand_share=0.08,
        essential_need=0.0,
        discretionary_weight=0.90,
        target_inventory_ratio=0.10,
        markup=0.20,
    ),
    SectorSpec(
        key="public_administration",
        name="Public administration / state services",
        base_price=14.0,
        base_wage=9.8,
        base_productivity=0.98,
        household_demand_share=0.0,
        essential_need=0.0,
        discretionary_weight=0.0,
        target_inventory_ratio=0.0,
        markup=0.0,
    ),
)

SECTOR_BY_KEY = {spec.key: spec for spec in SECTOR_SPECS}
ESSENTIAL_SECTOR_KEYS = ("food", "housing", "clothing")
DISCRETIONARY_SECTOR_KEYS = ("leisure", "school", "university")


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
    partnership_affinity_code: int = 0
    next_partnership_attempt_period: int = 0
    fertility_multiplier: float = 1.0
    essential_shares: dict[str, float] = field(default_factory=dict)
    discretionary_shares: dict[str, float] = field(default_factory=dict)
    bank_id: int = 0
    loan_balance: float = 0.0
    loan_delinquency_periods: int = 0
    loan_restructure_count: int = 0
    loan_restructure_grace_periods: int = 0
    credit_exclusion_periods: int = 0
    employed_by: Optional[int] = None
    contract_wage: float = 0.0
    pending_employer_id: Optional[int] = None
    pending_contract_wage: float = 0.0
    guardian_id: Optional[int] = None
    partner_id: Optional[int] = None
    partnership_start_period: int = -999
    mother_id: Optional[int] = None
    father_id: Optional[int] = None
    children_count: int = 0
    desired_children: int = 0
    child_desire_pressure: float = 0.0
    last_birth_period: int = -999
    dependent_children: int = 0
    employment_tenure: int = 0
    unemployment_duration: int = 0
    job_change_aversion: float = 0.0
    reservation_wage_distress_sensitivity: float = 1.0
    employment_insecurity_memory: float = 0.0
    wage_income: float = 0.0
    last_income: float = 0.0
    previous_income: float = 0.0
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
    public_school_support_persistence: float = 0.0
    public_university_support_persistence: float = 0.0
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
    public_capital_stock: float = 0.0
    tax_revenue_this_period: float = 0.0
    labor_tax_revenue: float = 0.0
    payroll_tax_revenue: float = 0.0
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
    public_administration_spending_this_period: float = 0.0
    infrastructure_spending_this_period: float = 0.0
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
    education_level_span: float = 0.0
    markup_tolerance: float = 1.0
    volume_preference: float = 1.0
    inventory_aversion: float = 1.0
    employment_inertia: float = 0.75
    price_aggressiveness: float = 1.0
    cash_conservatism: float = 1.0
    market_share_ambition: float = 1.0
    expansion_credit_appetite: float = 1.0
    stability_sensitivity: float = 1.0
    investment_animal_spirits: float = 1.0
    forecast_caution: float = 1.0
    stockout_perception_bias: float = 1.0
    stockout_sensitivity: float = 1.0
    stockout_patience: float = 1.0
    active: bool = True
    bank_id: int = 0
    age: int = 0
    desired_workers: int = 0
    workers: list[int] = field(default_factory=list)
    installed_worker_capacity: float = 0.0
    target_inventory: float = 0.0
    sales_this_period: float = 0.0
    stockout_rejections_this_period: float = 0.0
    competitive_demand_rejections_this_period: float = 0.0
    capacity_shortage_rejections_this_period: float = 0.0
    capital_goods_sales_this_period: float = 0.0
    sales_history: list[float] = field(default_factory=list)
    observed_demand_history: list[float] = field(default_factory=list)
    stockout_rejection_history: list[float] = field(default_factory=list)
    expected_sales_history: list[float] = field(default_factory=list)
    production_history: list[float] = field(default_factory=list)
    inventory_batches: list[float] = field(default_factory=list)
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
    last_inventory_carry_cost: float = 0.0
    last_inventory_waste_cost: float = 0.0
    last_severance_cost: float = 0.0
    last_effective_marginal_unit_cost: float = 0.0
    last_unit_cost: float = 0.0
    last_market_share: float = 0.0
    last_expected_sales: float = 0.0
    last_stockout_rejections: float = 0.0
    last_competitive_demand_rejections: float = 0.0
    last_capacity_shortage_rejections: float = 0.0
    last_observed_demand: float = 0.0
    last_stockout_pressure: float = 0.0
    market_fragility_belief: float = 0.0
    forecast_error_belief: float = 0.15
    last_capital_investment: float = 0.0
    last_industrial_investment_spending: float = 0.0
    last_investment_goods_units: float = 0.0
    last_productivity_goods_spending: float = 0.0
    last_capacity_goods_spending: float = 0.0
    last_capacity_gain_workers: float = 0.0
    last_unfilled_investment_budget: float = 0.0
    last_investment_decision_reason: str = "sin_decision_registrada"
    last_technology_investment: float = 0.0
    last_technology_gain: float = 0.0
    last_interest_cost: float = 0.0
    labor_offer_rejections: int = 0
    labor_offer_rejection_wage_floor: float = 0.0
    last_labor_offer_rejections: int = 0
    last_labor_offer_rejection_wage_floor: float = 0.0
    vacancy_duration: int = 0
    last_vacancy_duration: int = 0
    loan_balance: float = 0.0
    loan_delinquency_periods: int = 0
    loan_restructure_count: int = 0
    loan_restructure_grace_periods: int = 0
    credit_exclusion_periods: int = 0
    loan_default_flag: bool = False
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
    starting_workers: int
    workers: int
    desired_workers: int
    vacancies: int
    vacancy_duration: int
    worker_exits: int
    worker_quits: int
    worker_dismissals: int
    exited_workers_reemployed: int
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
    capital_efficiency_percent: float
    technology_level_percent: float
    effective_worker_productivity_capacity: float
    installed_production_capacity_units: float
    capacity_utilization_rate: float
    effective_marginal_unit_cost: float
    unit_cost: float
    markup_tolerance: float
    volume_preference: float
    inventory_aversion: float
    employment_inertia: float
    price_aggressiveness: float
    cash_conservatism: float
    market_share_ambition: float
    investment_animal_spirits: float
    demand_elasticity: float
    forecast_caution: float
    learning_maturity: float
    technology: float
    capital_investment: float
    technology_investment: float
    technology_gain: float
    industrial_investment_spending: float
    investment_goods_units: float
    productivity_goods_spending: float
    capacity_goods_spending: float
    capacity_gain_workers: float
    unfilled_investment_budget: float
    investment_decision_reason: str
    stockout_rejected_units: float
    competitive_demand_rejected_units: float
    capacity_shortage_rejected_units: float
    observed_demand_units: float
    stockout_pressure: float
    sales: float
    expected_sales: float
    revenue: float
    production: float
    profit: float
    payroll_total: float
    severance_total: float
    total_cost: float
    loss_streak: int
    market_share: float
    market_fragility_belief: float
    forecast_error_belief: float
    target_inventory: float
    age: int


@dataclass(frozen=True)
class BankPeriodSnapshot:
    period: int
    year: int
    period_in_year: int
    bank_id: int
    bank_name: str
    active: bool
    deposits: float
    reserves: float
    legal_reserve_ratio: float
    required_reserves: float
    excess_reserves: float
    effective_reserve_ratio: float
    reserve_coverage_ratio: float
    loans_households: float
    loans_firms: float
    total_loans: float
    bond_holdings: float
    central_bank_borrowing: float
    liquid_assets: float
    total_assets: float
    total_liabilities: float
    equity: float
    capital_ratio: float
    leverage_ratio: float
    loan_to_deposit_ratio: float
    credit_deployment_ratio: float
    funds_deployed_ratio: float
    liquidity_ratio: float
    deposit_rate: float
    loan_rate: float
    net_interest_margin: float
    interest_income: float
    interest_expense: float
    profits: float
    credit_market_share: float
    deposit_market_share: float


@dataclass(frozen=True)
class FamilyPeriodSnapshot:
    period: int
    year: int
    period_in_year: int
    family_id: int
    family_members: int
    adult_members: int
    labor_capable_members: int
    employed_members: int
    total_basic_basket_cost_including_school: float
    private_school_basket_cost: float
    total_family_income: float
    family_employment_rate: float
    family_cash_available: float
    family_period_income_cash: float
    period_income_covers_basic_basket: bool
    family_start_savings_cash: float
    family_cash_spent: float
    family_income_spent_cash: float
    family_savings_spent_cash: float
    family_period_net_saving_cash: float
    family_voluntary_saved_cash: float
    family_involuntary_retained_cash: float
    family_expected_salary: float
    family_accepted_salary: float
    basic_goods_coverage_percent: float
    basic_goods_shortfall_reason: str
    marginal_propensity_to_spend: float
    marginal_propensity_to_save: float
    necessary_essential_demand_units: float
    essential_offer_units: float
    necessary_demand_to_offer_ratio: float


@dataclass(frozen=True)
class SimulationConfig:
    periods: int = 120
    households: int = 5000
    seed: int = 7
    periods_per_year: int = 12
    firms_per_sector: int = 20
    initial_manufacturing_firms: int = 2
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
    firm_interest_rate_neutral: float = 0.035
    firm_investment_interest_sensitivity: float = 0.12
    firm_macro_stability_investment_weight: float = 0.16
    firm_expansion_credit_interest_sensitivity: float = 0.38
    firm_expansion_credit_max_revenue_share: float = 0.30
    firm_investment_gross_revenue_share: float = 0.20
    firm_capital_goods_price_multiplier: float = 1.0
    manufactured_capital_goods_capacity_share: float = 1.0
    manufactured_minimum_active_output_units: float = 1.0
    manufactured_minimum_capacity_buffer: float = 1.15
    manufactured_full_capacity_workers: float = 20.0
    firm_capital_goods_quality_floor: float = 0.65
    firm_capital_goods_quality_ceiling: float = 1.40
    firm_capital_goods_labor_required_min: float = 5.0
    firm_capital_goods_labor_required_max: float = 18.0
    firm_capital_goods_supported_workers_min: float = 4.0
    firm_capital_goods_supported_workers_max: float = 16.0
    firm_workforce_skill_investment_weight: float = 0.22
    firm_installation_capacity_multiplier: float = 2.00
    firm_installation_capacity_binding: bool = True
    firm_labor_full_efficiency_capital: float = 2.5
    firm_labor_diminishing_absorption: float = 0.85
    firm_investment_knowledge_floor: float = 0.90
    firm_investment_knowledge_ceiling: float = 1.18
    firm_investment_knowledge_university_weight: float = 0.65
    firm_investment_knowledge_skill_weight: float = 0.35
    bankruptcy_cash_threshold: float = -50.0
    bankruptcy_streak_limit: int = 5
    bankruptcy_grace_period: int = 2
    critical_cash_threshold: float = -100.0
    startup_owner_wealth: float = 150.0
    startup_firm_cash: float = 180.0
    startup_firm_capital: float = 80.0
    startup_inventory_multiplier: float = 0.75
    startup_liquid_asset_buffer_share: float = 0.20
    startup_expected_sales_share: float = 0.65
    inventory_carry_cost_share: float = 0.012
    firm_stockout_perception_deviation: float = 0.35
    firm_stockout_expectation_weight: float = 0.45
    firm_stockout_inventory_buffer_weight: float = 0.35
    firm_restart_package_multiplier: float = 0.1
    firm_restart_wealth_threshold: float = 1.0
    firm_restart_min_scale: float = 0.01
    firm_restart_max_scale: float = 3.0
    employment_contract_periods: int = 12
    contract_notice_periods: int = 2
    severance_months_per_year: float = 1.0
    severance_max_months: float = 12.0
    severance_layoff_payback_periods: int = 6
    severance_cash_stress_payroll_share: float = 0.50
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
    adult_school_catchup_age_max_years: float = 30.0
    adult_university_catchup_age_max_years: float = 40.0
    school_years_required: float = 12.0
    university_years_required: float = 4.0
    school_students_per_classroom: float = 26.0
    university_students_per_classroom: float = 18.0
    school_classroom_capital_cost: float = 6.0
    university_classroom_capital_cost: float = 10.0
    school_support_staff_ratio: float = 0.32
    university_support_staff_ratio: float = 0.45
    initial_school_completion_share: float = 0.88
    initial_university_completion_share: float = 0.25
    startup_grace_periods: int = 2
    firm_learning_warmup_periods: int = 18
    price_adjustment_inertia: float = 0.78
    price_adjustment_min_history: int = 4
    inventory_shelf_life_months: int = 6
    birth_interval_periods: int = 9
    partnership_affinity_buckets: int = 20
    partnership_base_match_probability: float = 0.08
    partnership_retry_periods: int = 4
    partnership_age_gap_neutral_years: float = 5.0
    partnership_age_gap_soft_cap_years: float = 10.0
    partnership_age_gap_soft_penalty: float = 0.80
    partnership_age_gap_hard_penalty: float = 0.06
    birth_capable_resource_ratio_min: float = 1.35
    partnered_birth_ramp_years: float = 5.0
    partnered_birth_ramp_floor: float = 0.30
    fertility_heterogeneity_max: float = 0.20
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
    central_bank_rule: str = "inflation_targeting"
    central_bank_target_velocity: float = 0.20
    central_bank_target_annual_inflation: float = 0.04
    central_bank_max_issue_share: float = 0.05
    central_bank_goods_growth_pass_through: float = 1.0
    central_bank_inflation_gap_liquidity_weight: float = 1.25
    central_bank_unemployment_gap_liquidity_weight: float = 0.035
    central_bank_demand_gap_liquidity_weight: float = 0.015
    central_bank_credit_accommodation_share: float = 0.55
    central_bank_monetary_gap_rate_weight: float = 0.20
    central_bank_omo_response_share: float = 0.60
    central_bank_dynamic_reserve_ratio_enabled: bool = True
    central_bank_reserve_ratio_floor: float = 0.08
    central_bank_reserve_ratio_ceiling: float = 0.30
    central_bank_reserve_ratio_gap_sensitivity: float = 0.08
    central_bank_reserve_ratio_adjustment_speed: float = 0.35
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
    household_loan_principal_payment_share: float = 0.015
    household_debt_service_protected_basket_share: float = 0.65
    household_default_protected_basket_share: float = 0.35
    household_loan_restructure_delinquency: int = 3
    household_loan_default_delinquency: int = 6
    household_loan_restructure_grace_periods: int = 6
    household_loan_restructure_haircut_share: float = 0.08
    household_default_credit_cooldown_periods: int = 18
    firm_loan_principal_payment_share: float = 0.02
    firm_debt_service_operating_buffer_share: float = 0.45
    firm_loan_restructure_delinquency: int = 2
    firm_loan_default_delinquency: int = 4
    firm_loan_restructure_grace_periods: int = 4
    firm_loan_restructure_haircut_share: float = 0.10
    firm_default_credit_cooldown_periods: int = 24
    central_bank_discount_window_spread: float = 0.05
    entrepreneur_consumption_share: float = 0.20
    entrepreneur_vault_share: float = 0.02
    reservation_wage_adjustment_speed: float = 0.45
    reservation_wage_reference_cushion_months: float = 2.5
    reservation_wage_liquidity_discount_max: float = 0.22
    reservation_wage_liquidity_premium_max: float = 0.10
    reservation_wage_unemployment_discount_max: float = 0.18
    reservation_wage_unemployment_pressure_periods: int = 6
    reservation_wage_distress_sensitivity_min: float = 0.45
    reservation_wage_distress_sensitivity_max: float = 1.75
    labor_offer_rejection_catchup_share: float = 0.50
    labor_offer_rejection_response_cap: float = 0.10
    firm_market_memory_years: float = 3.0
    firm_revealed_shortage_capacity_weight: float = 0.35
    firm_revealed_shortage_investment_weight: float = 0.30
    firm_revealed_shortage_entry_weight: float = 0.65
    firm_costing_scale_floor_share: float = 0.55
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
    government_labor_tax_rate_low: float = 0.03
    government_labor_tax_rate_mid: float = 0.08
    government_labor_tax_rate_high: float = 0.14
    government_labor_tax_bracket_low: float = 1.1
    government_labor_tax_bracket_high: float = 2.1
    government_payroll_tax_rate: float = 0.08
    government_wealth_tax_rate: float = 0.002
    government_wealth_tax_threshold_multiple: float = 24.0
    government_unemployment_benefit_share: float = 0.30
    government_child_allowance_share: float = 0.10
    government_basic_support_gap_share: float = 0.35
    government_procurement_gap_share: float = 0.30
    public_school_budget_share: float = 0.015
    public_university_budget_share: float = 0.015
    public_school_min_target_units: float = 0.65
    public_university_min_target_units: float = 0.32
    public_school_support_package_share: float = 0.95
    public_university_support_package_share: float = 0.88
    public_education_low_resource_priority_bonus: float = 0.18
    public_school_support_continuity_bonus: float = 0.15
    public_university_support_continuity_bonus: float = 0.28
    adult_school_catchup_target_units: float = 0.30
    adult_university_catchup_target_units: float = 0.20
    government_structural_procurement_budget_share: float = 0.00
    public_administration_budget_share: float = 0.06
    government_infrastructure_budget_share: float = 0.015
    public_administration_wage_premium: float = 0.10
    public_administration_payroll_share: float = 0.65
    public_administration_employment_floor_share: float = 0.015
    public_administration_employment_state_size_sensitivity: float = 0.16
    public_administration_employment_cap_share: float = 0.06
    government_final_consumption_floor_share_gdp: float = 0.10
    government_procurement_price_sensitivity: float = 0.85
    government_spending_scale: float = 1.00
    government_spending_efficiency: float = 0.95
    government_structural_deficit_tolerance: float = 0.25
    government_public_capital_depreciation_rate: float = 0.0025
    government_public_capital_productivity_gain: float = 0.035
    government_public_capital_transport_gain: float = 0.060
    government_public_capital_scale: float = 600.0
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
    track_family_history: bool = False
    track_bank_history: bool = False


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
    median_wage: float
    total_sales_units: float
    potential_demand_units: float
    demand_fulfillment_rate: float
    essential_demand_units: float
    essential_production_units: float
    essential_sales_units: float
    essential_total_sales_units: float
    essential_government_sales_units: float
    essential_inventory_units: float
    essential_target_inventory_units: float
    essential_expected_sales_units: float
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
    school_students: float
    university_students: float
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
    gdp_deflator: float
    government_treasury_cash: float
    government_debt_outstanding: float
    government_tax_revenue: float
    government_labor_tax_revenue: float
    government_payroll_tax_revenue: float
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
    government_school_units: float
    government_university_units: float
    school_average_price: float
    university_average_price: float
    government_public_administration_spending: float
    government_infrastructure_spending: float
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
    firm_expansion_credit_creation: float
    investment_knowledge_multiplier: float
    central_bank_money_supply: float
    central_bank_target_money_supply: float
    central_bank_policy_rate: float
    central_bank_issuance: float
    cumulative_central_bank_issuance: float
    central_bank_monetary_gap_share: float
    average_bank_reserve_ratio: float
    household_credit_creation: float
    firm_credit_creation: float
    commercial_bank_credit_creation: float
    average_bank_deposit_rate: float
    average_bank_loan_rate: float
    total_bank_deposits: float
    total_bank_reserves: float
    total_bank_loans_households: float
    total_bank_loans_firms: float
    household_delinquent_loans: float
    firm_delinquent_loans: float
    bank_nonperforming_loans: float
    total_bank_bond_holdings: float
    total_bank_assets: float
    total_bank_liabilities: float
    bank_equity: float
    bank_writeoffs: float
    bank_loan_restructures: int
    household_loan_defaults: int
    firm_loan_defaults: int
    household_loan_restructures: int
    firm_loan_restructures: int
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
    public_capital_stock: float
    public_infrastructure_productivity_multiplier: float
    public_infrastructure_transport_cost_multiplier: float


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
    family_history: list[FamilyPeriodSnapshot] = field(default_factory=list)
    bank_history: list[BankPeriodSnapshot] = field(default_factory=list)
