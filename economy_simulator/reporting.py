from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .domain import (
    BankPeriodSnapshot,
    FirmPeriodSnapshot,
    PeriodSnapshot,
    SECTOR_BY_KEY,
    SECTOR_SPECS,
    SimulationResult,
)


CORE_HISTORY_COLUMNS = [
    "period",
    "year",
    "period_in_year",
    "population",
    "fertile_women",
    "births",
    "deaths",
    "labor_force",
    "employment_count",
    "unemployment_rate",
    "family_income_to_basket_ratio",
    "gdp_nominal",
    "real_gdp_nominal",
    "potential_gdp_nominal",
    "output_gap_share",
    "cpi",
    "gdp_deflator",
    "inflation_rate",
    "gdp_growth",
    "population_growth",
    "average_wage",
    "median_wage",
    "real_average_wage",
    "essential_demand_units",
    "essential_production_units",
    "essential_sales_units",
    "essential_total_sales_units",
    "essential_government_sales_units",
    "essential_inventory_units",
    "essential_target_inventory_units",
    "essential_expected_sales_units",
    "people_full_essential_coverage",
    "full_essential_coverage_share",
    "average_food_meals_per_person",
    "bank_equity",
    "bank_writeoffs",
    "bank_nonperforming_loan_share",
    "bank_capital_ratio",
    "bank_insolvent_share",
    "bank_undercapitalized_share",
    "central_bank_money_supply",
    "central_bank_target_money_supply",
    "central_bank_policy_rate",
    "central_bank_issuance",
    "central_bank_monetary_gap_share",
    "average_bank_reserve_ratio",
    "government_tax_revenue",
    "government_labor_tax_revenue",
    "government_payroll_tax_revenue",
    "government_total_spending",
    "government_deficit",
    "government_debt_outstanding",
    "government_school_units",
    "government_university_units",
    "school_average_price",
    "university_average_price",
    "government_school_unit_cost",
    "government_university_unit_cost",
    "government_school_unit_cost_ratio_private_price",
    "government_university_unit_cost_ratio_private_price",
    "recession_flag",
    "recession_intensity",
    "government_countercyclical_spending",
    "government_countercyclical_support_multiplier",
    "government_countercyclical_procurement_multiplier",
    "household_final_consumption_share_gdp",
    "government_final_consumption_share_gdp",
    "government_infrastructure_spending_share_gdp",
    "government_spending_share_gdp",
    "government_tax_burden_gdp",
    "government_labor_tax_burden_gdp",
    "government_payroll_tax_burden_gdp",
    "gross_capital_formation_share_gdp",
    "firm_expansion_credit_creation",
    "investment_knowledge_multiplier",
    "public_capital_stock",
    "net_exports_share_gdp",
    "gdp_expenditure_gap_share_gdp",
    "government_deficit_share_gdp",
    "school_enrollment_share",
    "children_studying_ratio",
    "university_enrollment_share",
    "school_completion_share",
    "adults_with_school_credential_ratio",
    "university_completion_share",
    "adults_with_university_credential_ratio",
    "low_resource_school_enrollment_share",
    "low_resource_university_enrollment_share",
    "low_resource_university_student_share",
    "low_resource_origin_upward_mobility_share",
    "low_resource_origin_university_completion_share",
    "poor_origin_university_mobility_lift",
    "school_income_premium",
    "university_income_premium",
    "poverty_rate_without_university",
    "poverty_rate_with_university",
    "skilled_job_fill_rate",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, pd.NA)


def _append_derived_columns(frame: pd.DataFrame, derived_columns: dict[str, pd.Series]) -> pd.DataFrame:
    if not derived_columns:
        return frame.copy()
    enriched = frame.copy()
    for column, values in derived_columns.items():
        enriched[column] = values
    return enriched.copy()


def history_frame(
    history: list[PeriodSnapshot],
    periods_per_year: int = 12,
    target_unemployment: float = 0.08,
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records([asdict(snapshot) for snapshot in history])
    if frame.empty:
        return frame

    periods_per_year = max(1, periods_per_year)
    target_employment_rate = max(1e-9, 1.0 - target_unemployment)
    capitalist_liquid_wealth = (
        frame["capitalist_bank_deposits"] + frame["capitalist_vault_cash"] + frame["capitalist_firm_cash"]
    )
    capitalist_augmented_assets = (
        capitalist_liquid_wealth
        + frame["capitalist_productive_capital"]
        + frame["capitalist_inventory_value"]
    )
    worker_augmented_asset_denominator = frame["worker_bank_deposits"] + capitalist_augmented_assets
    employment_count = frame["labor_force"] * frame["employment_rate"]
    gdp_deflator = frame["gdp_deflator"] if "gdp_deflator" in frame.columns else frame["price_index"]
    real_gdp_nominal = _safe_ratio(frame["gdp_nominal"], gdp_deflator)
    average_wage = _safe_ratio(frame["total_wages"], employment_count)
    potential_multiplier = (target_employment_rate / frame["employment_rate"].clip(lower=1e-9)).clip(lower=1.0)
    potential_real_gdp = real_gdp_nominal * potential_multiplier
    potential_gdp_nominal = potential_real_gdp * gdp_deflator
    derived_columns = {
        "cpi": frame["price_index"],
        "gdp_deflator": gdp_deflator,
        "inflation_rate": gdp_deflator.pct_change(),
        "cpi_inflation_rate": frame["price_index"].pct_change(),
        "gdp_growth": frame["gdp_nominal"].pct_change(),
        "gdp_per_capita_growth": frame["gdp_per_capita"].pct_change(),
        "population_growth": frame["population"].pct_change(),
        "perceived_utility_growth": frame["average_perceived_utility"].pct_change(),
        "birth_death_balance": frame["births"] - frame["deaths"],
        "birth_rate": _safe_ratio(frame["births"], frame["population"]),
        "death_rate": _safe_ratio(frame["deaths"], frame["population"]),
        "child_share": _safe_ratio(frame["children"], frame["population"]),
        "adult_share": _safe_ratio(frame["adults"], frame["population"]),
        "senior_share": _safe_ratio(frame["seniors"], frame["population"]),
        "female_share": _safe_ratio(frame["women"], frame["population"]),
        "male_share": _safe_ratio(frame["men"], frame["population"]),
        "fertile_women_share": _safe_ratio(frame["fertile_women"], frame["women"]),
        "fertile_women_birth_rate": _safe_ratio(frame["births"], frame["fertile_women"]),
        "fertile_capable_women_birth_rate": _safe_ratio(
            frame["fertile_capable_women_with_births"],
            frame["fertile_capable_women"],
        ),
        "fertile_capable_women_low_desire_share": _safe_ratio(
            frame["fertile_capable_women_low_desire_no_birth"],
            frame["fertile_capable_women"],
        ),
        "fertile_family_birth_rate": _safe_ratio(
            frame["fertile_families_with_births"],
            frame["fertile_families"],
        ),
        "fertile_capable_family_birth_rate": _safe_ratio(
            frame["fertile_capable_families_with_births"],
            frame["fertile_capable_families"],
        ),
        "fertile_capable_family_low_desire_share": _safe_ratio(
            frame["fertile_capable_families_low_desire_no_birth"],
            frame["fertile_capable_families"],
        ),
        "dependency_ratio": _safe_ratio(frame["children"] + frame["seniors"], frame["adults"]),
        "real_gdp_nominal": real_gdp_nominal,
        "employment_count": employment_count,
        "average_wage": average_wage,
        "median_wage": frame["median_wage"],
        "real_average_wage": _safe_ratio(average_wage, frame["price_index"]),
        "potential_real_gdp": potential_real_gdp,
        "potential_gdp_nominal": potential_gdp_nominal,
        "output_gap_share": _safe_ratio(potential_real_gdp - real_gdp_nominal, potential_real_gdp),
        "capital_growth": frame["total_capital_stock"].pct_change(),
        "inventory_growth": frame["total_inventory_units"].pct_change(),
        "worker_savings_rate": _safe_ratio(frame["worker_voluntary_saved"], frame["worker_cash_available"]),
        "worker_involuntary_retention_rate": _safe_ratio(
            frame["worker_involuntary_retained"],
            frame["worker_cash_available"],
        ),
        "worker_consumption_spending": (
            frame["worker_cash_available"] - frame["worker_cash_saved"]
        ).clip(lower=0.0),
        "worker_consumption_share_gdp": _safe_ratio(
            (
                frame["worker_cash_available"] - frame["worker_cash_saved"]
            ).clip(lower=0.0),
            frame["gdp_nominal"],
        ),
        "education_poverty_gap": (
            frame["poverty_rate_without_university"] - frame["poverty_rate_with_university"]
        ),
        "poor_origin_university_mobility_lift": (
            frame["low_resource_origin_university_upward_mobility_share"]
            - frame["low_resource_origin_nonuniversity_upward_mobility_share"]
        ),
        "household_final_consumption_share_gdp": _safe_ratio(
            frame["household_final_consumption"],
            frame["gdp_nominal"],
        ),
        "government_final_consumption_share_gdp": _safe_ratio(
            frame["government_final_consumption"],
            frame["gdp_nominal"],
        ),
        "gross_fixed_capital_formation_share_gdp": _safe_ratio(
            frame["gross_fixed_capital_formation"],
            frame["gdp_nominal"],
        ),
        "change_in_inventories_share_gdp": _safe_ratio(
            frame["change_in_inventories"],
            frame["gdp_nominal"],
        ),
        "gross_capital_formation_share_gdp": _safe_ratio(
            frame["gross_capital_formation"],
            frame["gdp_nominal"],
        ),
        "government_education_spending_share_gdp": _safe_ratio(
            frame["government_education_spending"],
            frame["gdp_nominal"],
        ),
        "government_public_administration_spending_share_gdp": _safe_ratio(
            frame["government_public_administration_spending"],
            frame["gdp_nominal"],
        ),
        "government_infrastructure_spending_share_gdp": _safe_ratio(
            frame["government_infrastructure_spending"],
            frame["gdp_nominal"],
        ),
        "government_school_spending_share_gdp": _safe_ratio(
            frame["government_school_spending"],
            frame["gdp_nominal"],
        ),
        "government_university_spending_share_gdp": _safe_ratio(
            frame["government_university_spending"],
            frame["gdp_nominal"],
        ),
        "government_school_unit_cost": _safe_ratio(
            frame["government_school_spending"],
            frame["government_school_units"],
        ),
        "government_university_unit_cost": _safe_ratio(
            frame["government_university_spending"],
            frame["government_university_units"],
        ),
        "government_school_unit_cost_ratio_private_price": _safe_ratio(
            _safe_ratio(frame["government_school_spending"], frame["government_school_units"]),
            frame["school_average_price"],
        ),
        "government_university_unit_cost_ratio_private_price": _safe_ratio(
            _safe_ratio(frame["government_university_spending"], frame["government_university_units"]),
            frame["university_average_price"],
        ),
        "children_studying_ratio": frame["school_enrollment_share"],
        "adults_with_school_credential_ratio": frame["school_completion_share"],
        "adults_with_university_credential_ratio": frame["university_completion_share"],
        "net_exports_share_gdp": _safe_ratio(
            frame["net_exports"],
            frame["gdp_nominal"],
        ),
        "gdp_expenditure_gap_share_gdp": _safe_ratio(
            frame["gdp_expenditure_gap"],
            frame["gdp_nominal"],
        ),
        "capitalist_liquid_wealth": capitalist_liquid_wealth,
        "capitalist_augmented_assets": capitalist_augmented_assets,
        "worker_net_financial_position": frame["worker_bank_deposits"] - frame["worker_credit_outstanding"],
        "capitalist_net_financial_position": capitalist_augmented_assets - frame["capitalist_credit_outstanding"],
        "worker_augmented_asset_share": _safe_ratio(frame["worker_bank_deposits"], worker_augmented_asset_denominator),
        "total_bank_loans": frame["total_bank_loans_households"] + frame["total_bank_loans_firms"],
        "bank_nonperforming_loan_share": _safe_ratio(
            frame["bank_nonperforming_loans"],
            frame["total_bank_loans_households"] + frame["total_bank_loans_firms"],
        ),
        "household_delinquent_loan_share": _safe_ratio(
            frame["household_delinquent_loans"],
            frame["total_bank_loans_households"],
        ),
        "firm_delinquent_loan_share": _safe_ratio(
            frame["firm_delinquent_loans"],
            frame["total_bank_loans_firms"],
        ),
        "government_total_spending": (
            frame["government_transfers"]
            + frame["government_procurement_spending"]
            + frame["government_education_spending"]
            + frame["government_public_administration_spending"]
            + frame["government_infrastructure_spending"]
        ),
        "government_tax_burden_gdp": _safe_ratio(frame["government_tax_revenue"], frame["gdp_nominal"]),
        "government_labor_tax_burden_gdp": _safe_ratio(
            frame["government_labor_tax_revenue"],
            frame["gdp_nominal"],
        ),
        "government_payroll_tax_burden_gdp": _safe_ratio(
            frame["government_payroll_tax_revenue"],
            frame["gdp_nominal"],
        ),
        "government_corporate_tax_burden_gdp": _safe_ratio(
            frame["government_corporate_tax_revenue"],
            frame["gdp_nominal"],
        ),
        "government_dividend_tax_burden_gdp": _safe_ratio(
            frame["government_dividend_tax_revenue"],
            frame["gdp_nominal"],
        ),
        "government_wealth_tax_burden_gdp": _safe_ratio(
            frame["government_wealth_tax_revenue"],
            frame["gdp_nominal"],
        ),
        "government_deficit_share_gdp": _safe_ratio(frame["government_deficit"], frame["gdp_nominal"]),
        "commercial_bank_credit_creation_share_money": _safe_ratio(
            frame["commercial_bank_credit_creation"],
            frame["central_bank_money_supply"],
        ),
    }
    history = _append_derived_columns(frame, derived_columns)
    history["capitalist_augmented_asset_share"] = 1.0 - history["worker_augmented_asset_share"]
    history["real_gdp_growth"] = history["real_gdp_nominal"].pct_change()
    return history.copy()


def core_history_frame(
    history: list[PeriodSnapshot],
    periods_per_year: int = 12,
    target_unemployment: float = 0.08,
) -> pd.DataFrame:
    frame = history_frame(
        history,
        periods_per_year=periods_per_year,
        target_unemployment=target_unemployment,
    )
    if frame.empty:
        return frame
    available_columns = [column for column in CORE_HISTORY_COLUMNS if column in frame.columns]
    return frame[available_columns].copy()


def annual_frame(history_frame: pd.DataFrame, target_unemployment: float = 0.08) -> pd.DataFrame:
    if history_frame.empty:
        return history_frame.copy()

    history_frame = history_frame.copy()
    annual = (
        history_frame.groupby("year")
        .agg(
            population=("population", "mean"),
            women=("women", "mean"),
            men=("men", "mean"),
            fertile_women=("fertile_women", "mean"),
            fertile_capable_women=("fertile_capable_women", "mean"),
            fertile_capable_women_low_desire_no_birth=("fertile_capable_women_low_desire_no_birth", "mean"),
            fertile_capable_women_with_births=("fertile_capable_women_with_births", "mean"),
            children=("children", "mean"),
            adults=("adults", "mean"),
            seniors=("seniors", "mean"),
            labor_force=("labor_force", "mean"),
            family_units=("family_units", "mean"),
            end_population=("population", "last"),
            end_women=("women", "last"),
            end_men=("men", "last"),
            end_fertile_women=("fertile_women", "last"),
            end_fertile_capable_women=("fertile_capable_women", "last"),
            end_children=("children", "last"),
            end_adults=("adults", "last"),
            end_seniors=("seniors", "last"),
            end_labor_force=("labor_force", "last"),
            children_with_guardian=("children_with_guardian", "last"),
            orphans=("orphans", "last"),
            average_age=("average_age", "mean"),
            average_family_income=("average_family_income", "mean"),
            average_family_resources=("average_family_resources", "mean"),
            average_family_basic_basket_cost=("average_family_basic_basket_cost", "mean"),
            family_income_to_basket_ratio=("family_income_to_basket_ratio", "mean"),
            family_resources_to_basket_ratio=("family_resources_to_basket_ratio", "mean"),
            families_income_below_basket_share=("families_income_below_basket_share", "mean"),
            families_resources_below_basket_share=("families_resources_below_basket_share", "mean"),
            births=("births", "sum"),
            deaths=("deaths", "sum"),
            gdp_nominal=("gdp_nominal", "sum"),
            real_gdp_nominal=("real_gdp_nominal", "sum"),
            potential_real_gdp=("potential_real_gdp", "sum"),
            gdp_per_capita_monthly=("gdp_per_capita", "mean"),
            total_wages=("total_wages", "sum"),
            total_sales_units=("total_sales_units", "sum"),
            potential_demand_units=("potential_demand_units", "sum"),
            demand_fulfillment_rate=("demand_fulfillment_rate", "mean"),
            essential_demand_units=("essential_demand_units", "sum"),
            essential_production_units=("essential_production_units", "sum"),
            essential_sales_units=("essential_sales_units", "sum"),
            essential_total_sales_units=("essential_total_sales_units", "sum"),
            essential_government_sales_units=("essential_government_sales_units", "sum"),
            essential_inventory_units=("essential_inventory_units", "mean"),
            essential_target_inventory_units=("essential_target_inventory_units", "mean"),
            essential_expected_sales_units=("essential_expected_sales_units", "sum"),
            essential_fulfillment_rate=("essential_fulfillment_rate", "mean"),
            average_food_meals_per_person=("average_food_meals_per_person", "mean"),
            food_sufficient_share=("food_sufficient_share", "mean"),
            food_subsistence_share=("food_subsistence_share", "mean"),
            food_acute_hunger_share=("food_acute_hunger_share", "mean"),
            food_severe_hunger_share=("food_severe_hunger_share", "mean"),
            average_health_fragility=("average_health_fragility", "mean"),
            average_perceived_utility=("average_perceived_utility", "mean"),
            school_age_population=("school_age_population", "mean"),
            university_age_population=("university_age_population", "mean"),
            school_students=("school_students", "mean"),
            university_students=("university_students", "mean"),
            school_enrollment_share=("school_enrollment_share", "mean"),
            university_enrollment_share=("university_enrollment_share", "mean"),
            school_completion_share=("school_completion_share", "mean"),
            university_completion_share=("university_completion_share", "mean"),
            school_labor_share=("school_labor_share", "mean"),
            skilled_labor_share=("skilled_labor_share", "mean"),
            low_resource_school_enrollment_share=("low_resource_school_enrollment_share", "mean"),
            low_resource_university_enrollment_share=("low_resource_university_enrollment_share", "mean"),
            low_resource_university_student_share=("low_resource_university_student_share", "mean"),
            school_income_premium=("school_income_premium", "mean"),
            university_income_premium=("university_income_premium", "mean"),
            poverty_rate_without_university=("poverty_rate_without_university", "mean"),
            poverty_rate_with_university=("poverty_rate_with_university", "mean"),
            tracked_origin_adults=("tracked_origin_adults", "last"),
            low_resource_origin_adults=("low_resource_origin_adults", "last"),
            low_resource_origin_upward_mobility_share=("low_resource_origin_upward_mobility_share", "mean"),
            low_resource_origin_university_completion_share=("low_resource_origin_university_completion_share", "mean"),
            low_resource_origin_university_upward_mobility_share=(
                "low_resource_origin_university_upward_mobility_share",
                "mean",
            ),
            low_resource_origin_nonuniversity_upward_mobility_share=(
                "low_resource_origin_nonuniversity_upward_mobility_share",
                "mean",
            ),
            skilled_job_demand_share=("skilled_job_demand_share", "mean"),
            skilled_job_fill_rate=("skilled_job_fill_rate", "mean"),
            skilled_labor_supply_to_demand_ratio=("skilled_labor_supply_to_demand_ratio", "mean"),
            total_sales_revenue=("total_sales_revenue", "sum"),
            total_production_units=("total_production_units", "sum"),
            period_investment_spending=("period_investment_spending", "sum"),
            startup_fixed_capital_formation=("startup_fixed_capital_formation", "sum"),
            startup_inventory_investment=("startup_inventory_investment", "sum"),
            business_cost_recycled=("business_cost_recycled", "sum"),
            business_cost_to_firms=("business_cost_to_firms", "sum"),
            business_cost_to_households=("business_cost_to_households", "sum"),
            business_cost_to_owners=("business_cost_to_owners", "sum"),
            inheritance_transfers=("inheritance_transfers", "sum"),
            bankruptcy_cash_recoveries=("bankruptcy_cash_recoveries", "sum"),
            total_profit=("total_profit", "sum"),
            active_school_firms=("active_school_firms", "last"),
            active_university_firms=("active_university_firms", "last"),
            avg_employment_rate=("employment_rate", "mean"),
            avg_unemployment_rate=("unemployment_rate", "mean"),
            average_worker_savings=("average_worker_savings", "mean"),
            worker_cash_available=("worker_cash_available", "sum"),
            worker_cash_saved=("worker_cash_saved", "sum"),
            worker_voluntary_saved=("worker_voluntary_saved", "sum"),
            worker_involuntary_retained=("worker_involuntary_retained", "sum"),
            worker_bank_deposits=("worker_bank_deposits", "last"),
            worker_credit_outstanding=("worker_credit_outstanding", "last"),
            goods_monetary_mass=("goods_monetary_mass", "last"),
            end_price_index=("price_index", "last"),
            end_gdp_deflator=("gdp_deflator", "last"),
            gini_household_savings=("gini_household_savings", "last"),
            gini_owner_wealth=("gini_owner_wealth", "last"),
            capitalist_bank_deposits=("capitalist_bank_deposits", "last"),
            capitalist_vault_cash=("capitalist_vault_cash", "last"),
            capitalist_firm_cash=("capitalist_firm_cash", "last"),
            capitalist_credit_outstanding=("capitalist_credit_outstanding", "last"),
            capitalist_productive_capital=("capitalist_productive_capital", "last"),
            capitalist_inventory_value=("capitalist_inventory_value", "last"),
            capitalist_controlled_assets=("capitalist_controlled_assets", "last"),
            capitalist_asset_share=("capitalist_asset_share", "last"),
            capitalist_liquid_share=("capitalist_liquid_share", "last"),
            worker_liquid_share=("worker_liquid_share", "last"),
            government_treasury_cash=("government_treasury_cash", "last"),
            government_debt_outstanding=("government_debt_outstanding", "last"),
            government_tax_revenue=("government_tax_revenue", "sum"),
            government_labor_tax_revenue=("government_labor_tax_revenue", "sum"),
            government_payroll_tax_revenue=("government_payroll_tax_revenue", "sum"),
            government_corporate_tax_revenue=("government_corporate_tax_revenue", "sum"),
            government_dividend_tax_revenue=("government_dividend_tax_revenue", "sum"),
            government_wealth_tax_revenue=("government_wealth_tax_revenue", "sum"),
            government_transfers=("government_transfers", "sum"),
            government_unemployment_support=("government_unemployment_support", "sum"),
            government_child_allowance=("government_child_allowance", "sum"),
            government_basic_support=("government_basic_support", "sum"),
            government_procurement_spending=("government_procurement_spending", "sum"),
            government_education_spending=("government_education_spending", "sum"),
            government_school_spending=("government_school_spending", "sum"),
            government_university_spending=("government_university_spending", "sum"),
            government_school_units=("government_school_units", "sum"),
            government_university_units=("government_university_units", "sum"),
            school_average_price=("school_average_price", "mean"),
            university_average_price=("university_average_price", "mean"),
            government_public_administration_spending=("government_public_administration_spending", "sum"),
            government_infrastructure_spending=("government_infrastructure_spending", "sum"),
            government_bond_issuance=("government_bond_issuance", "sum"),
            government_deficit=("government_deficit", "sum"),
            government_surplus=("government_surplus", "sum"),
            recession_flag=("recession_flag", "mean"),
            recession_intensity=("recession_intensity", "mean"),
            government_countercyclical_spending=("government_countercyclical_spending", "sum"),
            government_countercyclical_support_multiplier=(
                "government_countercyclical_support_multiplier",
                "mean",
            ),
            government_countercyclical_procurement_multiplier=(
                "government_countercyclical_procurement_multiplier",
                "mean",
            ),
            total_inventory_book_value=("total_inventory_book_value", "last"),
            household_final_consumption=("household_final_consumption", "sum"),
            government_final_consumption=("government_final_consumption", "sum"),
            gross_fixed_capital_formation=("gross_fixed_capital_formation", "sum"),
            change_in_inventories=("change_in_inventories", "sum"),
            valuables_acquisition=("valuables_acquisition", "sum"),
            gross_capital_formation=("gross_capital_formation", "sum"),
            exports=("exports", "sum"),
            imports=("imports", "sum"),
            net_exports=("net_exports", "sum"),
            gdp_expenditure_sna=("gdp_expenditure_sna", "sum"),
            gdp_expenditure_gap=("gdp_expenditure_gap", "sum"),
            labor_share_gdp=("labor_share_gdp", "mean"),
            profit_share_gdp=("profit_share_gdp", "mean"),
            investment_share_gdp=("investment_share_gdp", "mean"),
            capitalist_consumption_share_gdp=("capitalist_consumption_share_gdp", "mean"),
            government_spending_share_gdp=("government_spending_share_gdp", "mean"),
            dividend_share_gdp=("dividend_share_gdp", "mean"),
            retained_profit_share_gdp=("retained_profit_share_gdp", "mean"),
            firm_expansion_credit_creation=("firm_expansion_credit_creation", "sum"),
            investment_knowledge_multiplier=("investment_knowledge_multiplier", "mean"),
            central_bank_money_supply=("central_bank_money_supply", "last"),
            central_bank_target_money_supply=("central_bank_target_money_supply", "last"),
            central_bank_policy_rate=("central_bank_policy_rate", "last"),
            central_bank_issuance=("central_bank_issuance", "sum"),
            cumulative_central_bank_issuance=("cumulative_central_bank_issuance", "last"),
            central_bank_monetary_gap_share=("central_bank_monetary_gap_share", "mean"),
            household_credit_creation=("household_credit_creation", "sum"),
            firm_credit_creation=("firm_credit_creation", "sum"),
            commercial_bank_credit_creation=("commercial_bank_credit_creation", "sum"),
            average_bank_deposit_rate=("average_bank_deposit_rate", "mean"),
            average_bank_loan_rate=("average_bank_loan_rate", "mean"),
            average_bank_reserve_ratio=("average_bank_reserve_ratio", "mean"),
            total_bank_deposits=("total_bank_deposits", "last"),
            total_bank_reserves=("total_bank_reserves", "last"),
            total_bank_loans_households=("total_bank_loans_households", "last"),
            total_bank_loans_firms=("total_bank_loans_firms", "last"),
            total_bank_bond_holdings=("total_bank_bond_holdings", "last"),
            total_bank_assets=("total_bank_assets", "last"),
            total_bank_liabilities=("total_bank_liabilities", "last"),
            bank_equity=("bank_equity", "last"),
            bank_recapitalization=("bank_recapitalization", "sum"),
            bank_resolution_events=("bank_resolution_events", "sum"),
            bank_capital_ratio=("bank_capital_ratio", "mean"),
            bank_asset_liability_ratio=("bank_asset_liability_ratio", "mean"),
            bank_reserve_coverage_ratio=("bank_reserve_coverage_ratio", "mean"),
            bank_liquidity_ratio=("bank_liquidity_ratio", "mean"),
            bank_loan_to_deposit_ratio=("bank_loan_to_deposit_ratio", "mean"),
            bank_undercapitalized_share=("bank_undercapitalized_share", "mean"),
            bank_insolvent_share=("bank_insolvent_share", "mean"),
            money_velocity=("money_velocity", "mean"),
            total_bankruptcies=("bankruptcies", "sum"),
            total_capital_stock=("total_capital_stock", "last"),
            total_inventory_units=("total_inventory_units", "last"),
            total_liquid_money=("total_liquid_money", "last"),
            total_household_savings=("total_household_savings", "last"),
            public_capital_stock=("public_capital_stock", "last"),
            public_infrastructure_productivity_multiplier=(
                "public_infrastructure_productivity_multiplier",
                "last",
            ),
            public_infrastructure_transport_cost_multiplier=(
                "public_infrastructure_transport_cost_multiplier",
                "last",
            ),
        )
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )

    capitalist_liquid_wealth = (
        annual["capitalist_bank_deposits"] + annual["capitalist_vault_cash"] + annual["capitalist_firm_cash"]
    )
    capitalist_augmented_assets = (
        capitalist_liquid_wealth
        + annual["capitalist_productive_capital"]
        + annual["capitalist_inventory_value"]
    )
    worker_augmented_asset_denominator = annual["worker_bank_deposits"] + capitalist_augmented_assets
    employment_count = annual["labor_force"] * annual["avg_employment_rate"]
    annual_gdp_deflator = _safe_ratio(annual["gdp_nominal"], annual["real_gdp_nominal"])
    average_wage = _safe_ratio(annual["total_wages"], employment_count)
    potential_real_gdp = annual["potential_real_gdp"]
    derived_columns = {
        "cpi": annual["end_price_index"],
        "gdp_deflator": annual_gdp_deflator,
        "gdp_per_capita_annual": _safe_ratio(annual["gdp_nominal"], annual["population"]),
        "real_gdp_nominal": annual["real_gdp_nominal"],
        "employment_count": employment_count,
        "average_wage": average_wage,
        "real_average_wage": _safe_ratio(average_wage, annual["end_price_index"]),
        "potential_real_gdp": potential_real_gdp,
        "potential_gdp_nominal": potential_real_gdp * annual_gdp_deflator,
        "output_gap_share": _safe_ratio(potential_real_gdp - annual["real_gdp_nominal"], potential_real_gdp),
        "gdp_growth_yoy": annual["gdp_nominal"].pct_change(),
        "inflation_yoy": annual_gdp_deflator.pct_change(),
        "cpi_inflation_yoy": annual["end_price_index"].pct_change(),
        "population_growth_yoy": annual["end_population"].pct_change(),
        "perceived_utility_growth_yoy": annual["average_perceived_utility"].pct_change(),
        "birth_rate": _safe_ratio(annual["births"], annual["population"]),
        "death_rate": _safe_ratio(annual["deaths"], annual["population"]),
        "child_share": _safe_ratio(annual["children"], annual["population"]),
        "adult_share": _safe_ratio(annual["adults"], annual["population"]),
        "senior_share": _safe_ratio(annual["seniors"], annual["population"]),
        "female_share": _safe_ratio(annual["women"], annual["population"]),
        "male_share": _safe_ratio(annual["men"], annual["population"]),
        "fertile_women_share": _safe_ratio(annual["fertile_women"], annual["women"]),
        "fertile_capable_women_birth_rate": _safe_ratio(
            annual["fertile_capable_women_with_births"],
            annual["fertile_capable_women"],
        ),
        "fertile_capable_women_low_desire_share": _safe_ratio(
            annual["fertile_capable_women_low_desire_no_birth"],
            annual["fertile_capable_women"],
        ),
        "dependency_ratio": _safe_ratio(annual["children"] + annual["seniors"], annual["adults"]),
        "capital_growth_yoy": annual["total_capital_stock"].pct_change(),
        "inventory_growth_yoy": annual["total_inventory_units"].pct_change(),
        "worker_savings_rate": _safe_ratio(annual["worker_voluntary_saved"], annual["worker_cash_available"]),
        "worker_involuntary_retention_rate": _safe_ratio(
            annual["worker_involuntary_retained"],
            annual["worker_cash_available"],
        ),
        "worker_consumption_spending": (
            annual["worker_cash_available"] - annual["worker_cash_saved"]
        ).clip(lower=0.0),
        "education_poverty_gap": (
            annual["poverty_rate_without_university"] - annual["poverty_rate_with_university"]
        ),
        "poor_origin_university_mobility_lift": (
            annual["low_resource_origin_university_upward_mobility_share"]
            - annual["low_resource_origin_nonuniversity_upward_mobility_share"]
        ),
        "worker_consumption_share_gdp": _safe_ratio(
            (
                annual["worker_cash_available"] - annual["worker_cash_saved"]
            ).clip(lower=0.0),
            annual["gdp_nominal"],
        ),
        "household_final_consumption_share_gdp": _safe_ratio(
            annual["household_final_consumption"],
            annual["gdp_nominal"],
        ),
        "government_final_consumption_share_gdp": _safe_ratio(
            annual["government_final_consumption"],
            annual["gdp_nominal"],
        ),
        "gross_fixed_capital_formation_share_gdp": _safe_ratio(
            annual["gross_fixed_capital_formation"],
            annual["gdp_nominal"],
        ),
        "change_in_inventories_share_gdp": _safe_ratio(
            annual["change_in_inventories"],
            annual["gdp_nominal"],
        ),
        "gross_capital_formation_share_gdp": _safe_ratio(
            annual["gross_capital_formation"],
            annual["gdp_nominal"],
        ),
        "government_education_spending_share_gdp": _safe_ratio(
            annual["government_education_spending"],
            annual["gdp_nominal"],
        ),
        "government_public_administration_spending_share_gdp": _safe_ratio(
            annual["government_public_administration_spending"],
            annual["gdp_nominal"],
        ),
        "government_infrastructure_spending_share_gdp": _safe_ratio(
            annual["government_infrastructure_spending"],
            annual["gdp_nominal"],
        ),
        "government_school_spending_share_gdp": _safe_ratio(
            annual["government_school_spending"],
            annual["gdp_nominal"],
        ),
        "government_university_spending_share_gdp": _safe_ratio(
            annual["government_university_spending"],
            annual["gdp_nominal"],
        ),
        "government_school_unit_cost": _safe_ratio(
            annual["government_school_spending"],
            annual["government_school_units"],
        ),
        "government_university_unit_cost": _safe_ratio(
            annual["government_university_spending"],
            annual["government_university_units"],
        ),
        "government_school_unit_cost_ratio_private_price": _safe_ratio(
            _safe_ratio(annual["government_school_spending"], annual["government_school_units"]),
            annual["school_average_price"],
        ),
        "government_university_unit_cost_ratio_private_price": _safe_ratio(
            _safe_ratio(annual["government_university_spending"], annual["government_university_units"]),
            annual["university_average_price"],
        ),
        "children_studying_ratio": annual["school_enrollment_share"],
        "adults_with_school_credential_ratio": annual["school_completion_share"],
        "adults_with_university_credential_ratio": annual["university_completion_share"],
        "net_exports_share_gdp": _safe_ratio(
            annual["net_exports"],
            annual["gdp_nominal"],
        ),
        "gdp_expenditure_gap_share_gdp": _safe_ratio(
            annual["gdp_expenditure_gap"],
            annual["gdp_nominal"],
        ),
        "capitalist_liquid_wealth": capitalist_liquid_wealth,
        "capitalist_augmented_assets": capitalist_augmented_assets,
        "worker_net_financial_position": annual["worker_bank_deposits"] - annual["worker_credit_outstanding"],
        "capitalist_net_financial_position": capitalist_augmented_assets - annual["capitalist_credit_outstanding"],
        "worker_augmented_asset_share": _safe_ratio(
            annual["worker_bank_deposits"],
            worker_augmented_asset_denominator,
        ),
        "total_bank_loans": annual["total_bank_loans_households"] + annual["total_bank_loans_firms"],
        "government_total_spending": (
            annual["government_transfers"]
            + annual["government_procurement_spending"]
            + annual["government_education_spending"]
            + annual["government_public_administration_spending"]
            + annual["government_infrastructure_spending"]
        ),
        "government_tax_burden_gdp": _safe_ratio(annual["government_tax_revenue"], annual["gdp_nominal"]),
        "government_labor_tax_burden_gdp": _safe_ratio(
            annual["government_labor_tax_revenue"],
            annual["gdp_nominal"],
        ),
        "government_payroll_tax_burden_gdp": _safe_ratio(
            annual["government_payroll_tax_revenue"],
            annual["gdp_nominal"],
        ),
        "government_corporate_tax_burden_gdp": _safe_ratio(
            annual["government_corporate_tax_revenue"],
            annual["gdp_nominal"],
        ),
        "government_dividend_tax_burden_gdp": _safe_ratio(
            annual["government_dividend_tax_revenue"],
            annual["gdp_nominal"],
        ),
        "government_wealth_tax_burden_gdp": _safe_ratio(
            annual["government_wealth_tax_revenue"],
            annual["gdp_nominal"],
        ),
        "government_deficit_share_gdp": _safe_ratio(annual["government_deficit"], annual["gdp_nominal"]),
        "commercial_bank_credit_creation_share_money": _safe_ratio(
            annual["commercial_bank_credit_creation"],
            annual["central_bank_money_supply"],
        ),
    }
    annual = _append_derived_columns(annual, derived_columns)
    annual["capitalist_augmented_asset_share"] = 1.0 - annual["worker_augmented_asset_share"]
    annual["real_gdp_growth_yoy"] = annual["real_gdp_nominal"].pct_change()
    return annual.copy()


def simulation_frames(result: SimulationResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history_frame(
        result.history,
        periods_per_year=result.config.periods_per_year,
        target_unemployment=result.config.target_unemployment,
    )
    annual = annual_frame(history, target_unemployment=result.config.target_unemployment)
    return history, annual


def firm_history_frame(result: SimulationResult) -> pd.DataFrame:
    if not result.firm_history:
        return pd.DataFrame()

    return pd.DataFrame.from_records(asdict(snapshot) for snapshot in result.firm_history)


def bank_history_frame(result: SimulationResult) -> pd.DataFrame:
    if not result.bank_history:
        return pd.DataFrame()

    return (
        pd.DataFrame.from_records(asdict(snapshot) for snapshot in result.bank_history)
        .sort_values(["period", "bank_id"])
        .reset_index(drop=True)
    )


def monetary_audit_frame(
    history: list[PeriodSnapshot],
    periods_per_year: int = 12,
    target_unemployment: float = 0.08,
) -> pd.DataFrame:
    frame = history_frame(
        history,
        periods_per_year=periods_per_year,
        target_unemployment=target_unemployment,
    )
    if frame.empty:
        return frame

    audit = frame.copy()
    deposits = audit.get("total_bank_deposits", pd.Series([0.0] * len(audit), index=audit.index)).clip(lower=0.0)
    reserves = audit.get("total_bank_reserves", pd.Series([0.0] * len(audit), index=audit.index)).clip(lower=0.0)
    household_credit = audit.get(
        "total_bank_loans_households",
        pd.Series([0.0] * len(audit), index=audit.index),
    ).clip(lower=0.0)
    firm_credit = audit.get(
        "total_bank_loans_firms",
        pd.Series([0.0] * len(audit), index=audit.index),
    ).clip(lower=0.0)
    bond_holdings = audit.get(
        "total_bank_bond_holdings",
        pd.Series([0.0] * len(audit), index=audit.index),
    ).clip(lower=0.0)
    total_credit = household_credit + firm_credit
    gdp = audit.get("gdp_nominal", pd.Series([0.0] * len(audit), index=audit.index)).clip(lower=0.0)
    broad_money = audit.get("total_liquid_money", pd.Series([0.0] * len(audit), index=audit.index)).clip(lower=0.0)
    monetary_base = audit.get(
        "central_bank_money_supply",
        pd.Series([0.0] * len(audit), index=audit.index),
    ).clip(lower=0.0)
    legal_reserve_ratio = audit.get(
        "average_bank_reserve_ratio",
        pd.Series([0.0] * len(audit), index=audit.index),
    ).clip(lower=0.0)
    required_reserves = deposits * legal_reserve_ratio

    derived_columns = {
        "total_credit": total_credit,
        "household_credit_share": _safe_ratio(household_credit, total_credit),
        "firm_credit_share": _safe_ratio(firm_credit, total_credit),
        "credit_to_gdp": _safe_ratio(total_credit, gdp),
        "household_credit_to_gdp": _safe_ratio(household_credit, gdp),
        "firm_credit_to_gdp": _safe_ratio(firm_credit, gdp),
        "deposits_to_gdp": _safe_ratio(deposits, gdp),
        "broad_money_to_gdp": _safe_ratio(broad_money, gdp),
        "monetary_base_to_gdp": _safe_ratio(monetary_base, gdp),
        "money_velocity_calculated": _safe_ratio(gdp, broad_money),
        "credit_velocity": _safe_ratio(gdp, total_credit),
        "deposit_multiplier": _safe_ratio(deposits, monetary_base),
        "credit_multiplier": _safe_ratio(total_credit, monetary_base),
        "effective_reserve_ratio": _safe_ratio(reserves, deposits),
        "required_reserves": required_reserves,
        "excess_reserves": reserves - required_reserves,
        "excess_reserve_ratio": _safe_ratio(reserves - required_reserves, deposits),
        "funds_deployed_ratio": _safe_ratio(total_credit + bond_holdings, deposits),
        "credit_creation_to_gdp": _safe_ratio(
            audit.get("commercial_bank_credit_creation", pd.Series([0.0] * len(audit), index=audit.index)),
            gdp,
        ),
        "central_bank_issuance_to_money_supply": _safe_ratio(
            audit.get("central_bank_issuance", pd.Series([0.0] * len(audit), index=audit.index)),
            monetary_base,
        ),
        "nonperforming_loan_share_calculated": _safe_ratio(
            audit.get("bank_nonperforming_loans", pd.Series([0.0] * len(audit), index=audit.index)),
            total_credit,
        ),
    }
    audit = _append_derived_columns(audit, derived_columns)

    ordered_columns = [
        "period",
        "year",
        "period_in_year",
        "gdp_nominal",
        "real_gdp_nominal",
        "central_bank_money_supply",
        "central_bank_target_money_supply",
        "central_bank_policy_rate",
        "central_bank_issuance",
        "cumulative_central_bank_issuance",
        "central_bank_monetary_gap_share",
        "total_liquid_money",
        "money_velocity",
        "money_velocity_calculated",
        "broad_money_to_gdp",
        "monetary_base_to_gdp",
        "total_bank_deposits",
        "total_bank_reserves",
        "average_bank_reserve_ratio",
        "effective_reserve_ratio",
        "required_reserves",
        "excess_reserves",
        "excess_reserve_ratio",
        "bank_reserve_coverage_ratio",
        "bank_liquidity_ratio",
        "total_credit",
        "total_bank_loans_households",
        "total_bank_loans_firms",
        "household_credit_share",
        "firm_credit_share",
        "credit_to_gdp",
        "household_credit_to_gdp",
        "firm_credit_to_gdp",
        "credit_velocity",
        "commercial_bank_credit_creation",
        "household_credit_creation",
        "firm_credit_creation",
        "firm_expansion_credit_creation",
        "credit_creation_to_gdp",
        "deposit_multiplier",
        "credit_multiplier",
        "bank_loan_to_deposit_ratio",
        "funds_deployed_ratio",
        "total_bank_bond_holdings",
        "total_bank_assets",
        "total_bank_liabilities",
        "bank_equity",
        "bank_capital_ratio",
        "bank_asset_liability_ratio",
        "bank_insolvent_share",
        "bank_undercapitalized_share",
        "bank_nonperforming_loans",
        "bank_nonperforming_loan_share",
        "nonperforming_loan_share_calculated",
        "household_delinquent_loans",
        "firm_delinquent_loans",
        "bank_writeoffs",
        "bank_loan_restructures",
        "bank_recapitalization",
        "bank_resolution_events",
        "average_bank_deposit_rate",
        "average_bank_loan_rate",
    ]
    available_columns = [column for column in ordered_columns if column in audit.columns]
    return audit[available_columns].copy()


def firm_audit_frame(firm_history_frame: pd.DataFrame, history_frame: pd.DataFrame) -> pd.DataFrame:
    if firm_history_frame.empty:
        return firm_history_frame.copy()

    audit = firm_history_frame.copy()
    defaults = {
        "starting_workers": audit.get("workers", pd.Series([0] * len(audit), index=audit.index)),
        "worker_exits": pd.Series([0] * len(audit), index=audit.index),
        "worker_quits": pd.Series([0] * len(audit), index=audit.index),
        "worker_dismissals": pd.Series([0] * len(audit), index=audit.index),
        "exited_workers_reemployed": pd.Series([0] * len(audit), index=audit.index),
        "payroll_total": pd.Series([0.0] * len(audit), index=audit.index),
        "severance_total": pd.Series([0.0] * len(audit), index=audit.index),
        "effective_marginal_unit_cost": audit.get(
            "unit_cost",
            pd.Series([0.0] * len(audit), index=audit.index),
        ),
        "cash": pd.Series([0.0] * len(audit), index=audit.index),
        "capital": pd.Series([0.0] * len(audit), index=audit.index),
        "technology": pd.Series([0.0] * len(audit), index=audit.index),
        "capital_efficiency_percent": pd.Series([0.0] * len(audit), index=audit.index),
        "technology_level_percent": pd.Series([0.0] * len(audit), index=audit.index),
        "effective_worker_productivity_capacity": audit.get(
            "productivity",
            pd.Series([0.0] * len(audit), index=audit.index),
        ),
        "production": pd.Series([0.0] * len(audit), index=audit.index),
        "installed_production_capacity_units": pd.Series([0.0] * len(audit), index=audit.index),
        "capacity_utilization_rate": pd.Series([0.0] * len(audit), index=audit.index),
        "stockout_rejected_units": pd.Series([0.0] * len(audit), index=audit.index),
        "observed_demand_units": audit.get(
            "sales",
            pd.Series([0.0] * len(audit), index=audit.index),
        ),
        "stockout_pressure": pd.Series([0.0] * len(audit), index=audit.index),
        "capital_investment": pd.Series([0.0] * len(audit), index=audit.index),
        "technology_investment": pd.Series([0.0] * len(audit), index=audit.index),
        "rd_investment_spending": pd.Series([0.0] * len(audit), index=audit.index),
        "industrial_investment_spending": pd.Series([0.0] * len(audit), index=audit.index),
        "investment_goods_units": pd.Series([0.0] * len(audit), index=audit.index),
        "productivity_goods_spending": pd.Series([0.0] * len(audit), index=audit.index),
        "capacity_goods_spending": pd.Series([0.0] * len(audit), index=audit.index),
        "capacity_gain_workers": pd.Series([0.0] * len(audit), index=audit.index),
        "unfilled_investment_budget": pd.Series([0.0] * len(audit), index=audit.index),
        "investment_decision_reason": pd.Series(["sin_decision_registrada"] * len(audit), index=audit.index),
        "investment_animal_spirits": pd.Series([1.0] * len(audit), index=audit.index),
    }
    for column, default_values in defaults.items():
        if column not in audit.columns:
            audit[column] = default_values

    context_columns = ["period"]
    for column in ("average_wage", "unemployment_rate"):
        if column in history_frame.columns:
            context_columns.append(column)
    if len(context_columns) > 1:
        audit = audit.merge(history_frame[context_columns], on="period", how="left")
    else:
        audit["average_wage"] = pd.NA
        audit["unemployment_rate"] = pd.NA

    audit["firm_income_total"] = audit["revenue"]
    audit["firm_profit_total"] = audit["profit"]
    audit["desired_workers_next_period"] = audit["desired_workers"]
    audit["price_minus_marginal_unit_cost"] = audit["price"] - audit["effective_marginal_unit_cost"]
    audit["price_to_marginal_unit_cost_ratio"] = audit["price"] / audit["effective_marginal_unit_cost"].clip(lower=0.1)
    audit["sales_realization_ratio"] = audit["sales"] / audit["expected_sales"].clip(lower=1.0)
    revenue_base = audit["revenue"].clip(lower=1.0)
    tangible_and_rd_investment = (
        audit["industrial_investment_spending"].clip(lower=0.0)
        + audit["rd_investment_spending"].clip(lower=0.0)
    )
    liquidity_base = (audit["cash"].clip(lower=0.0) + tangible_and_rd_investment).clip(lower=1.0)
    audit["productivity_investment_propensity"] = audit["technology_investment"] / revenue_base
    audit["total_investment_propensity"] = tangible_and_rd_investment / revenue_base
    audit["liquidity_reinvestment_rate"] = tangible_and_rd_investment / liquidity_base
    audit["period_total_spending_outflow"] = (
        audit["total_cost"].clip(lower=0.0)
        + tangible_and_rd_investment
    )
    period_money_base = (
        audit["period_total_spending_outflow"]
        + audit["cash"].clip(lower=0.0)
    ).clip(lower=1e-9)
    audit["firm_period_marginal_propensity_to_spend"] = (
        audit["period_total_spending_outflow"] / period_money_base
    )
    audit["firm_period_propensity_to_bank_deposit"] = (
        audit["cash"].clip(lower=0.0) / period_money_base
    )

    def diagnose_no_productivity_investment(row) -> str:
        if row["productivity_goods_spending"] > 1e-9:
            return "si_invirtio_en_producto_A_productividad"
        if row["rd_investment_spending"] > 1e-9:
            return "si_invirtio_en_ID_intangible_productividad"
        if row["technology_level_percent"] >= 92.0:
            return "cerca_del_techo_tecnologico_rendimiento_marginal_bajo"
        decision_reason = str(row["investment_decision_reason"])
        if "mix_B_capacidad" in decision_reason or row["capacity_goods_spending"] > 1e-9:
            return "priorizo_producto_B_por_saturacion_de_capacidad_instalada"
        if decision_reason in ("sin_liquidez_para_invertir", "liquidez_reservada_insuficiente_para_invertir"):
            return "liquidez_insuficiente_o_reservada_para_operar"
        if decision_reason == "restriccion_financiera_impidio_invertir":
            return "restriccion_financiera_impidio_invertir"
        if decision_reason == "liquidez_desaparecio_antes_de_comprar_maquinaria":
            return "liquidez_desaparecio_antes_de_ejecutar_compra"
        if decision_reason in (
            "no_habia_maquinaria_disponible",
            "presupuesto_no_alcanzo_para_unidad_minima_maquinaria",
        ):
            return decision_reason
        if decision_reason == "no_vio_senal_suficiente_para_invertir":
            return "sin_senal_suficiente_de_demanda_o_cuello_de_botella_laboral"
        if decision_reason in ("empresa_nueva_sin_decision_previa", "firma_inactiva", "sin_decision_registrada"):
            return decision_reason
        if row["industrial_investment_spending"] <= 1e-9:
            if row["firm_profit_total"] < 0.0:
                return "perdidas_o_margen_insuficiente_para_financiar_productividad"
            if row["stockout_pressure"] <= 0.05 and row["vacancies"] <= 0:
                return "sin_stockout_ni_vacantes_que_justifiquen_producto_A"
            return "no_ejecuto_inversion_productiva_revisar_liquidez_y_senal"
        return "invirtio_en_maquinaria_pero_no_en_producto_A"

    audit["no_productivity_capital_investment_reason"] = audit.apply(
        diagnose_no_productivity_investment,
        axis=1,
    )

    def diagnose_loss(row) -> str:
        if row["firm_profit_total"] >= 0.0:
            return "sin_perdida"
        marginal_cost = max(0.1, row["effective_marginal_unit_cost"])
        price = max(0.0, row["price"])
        sales_realization = max(0.0, row["sales_realization_ratio"])
        if row["sales"] <= 0.0 and row["firm_income_total"] <= 0.0:
            return "perdida_por_sin_ventas"
        if price < marginal_cost * 0.98:
            return "perdida_probable_por_precio_bajo_vs_costo_marginal"
        if sales_realization < 0.70 and price >= marginal_cost * 0.98:
            return "perdida_probable_por_precio_alto_o_demanda_debil"
        if row["sales"] >= row["expected_sales"] * 0.85 and row["total_cost"] > row["firm_income_total"]:
            return "perdida_probable_por_costos_fijos_nomina_o_escala"
        if row["inventory"] > max(1.0, row["expected_sales"]) and row["sales"] < row["expected_sales"] * 0.85:
            return "perdida_probable_por_exceso_inventario_demanda_insuficiente"
        return "perdida_indeterminada_revisar_costos_demanda"

    def recommend_loss_action(cause: str) -> str:
        if cause == "sin_perdida":
            return "sin_accion_correctiva"
        if cause == "perdida_probable_por_precio_bajo_vs_costo_marginal":
            return "subir_precio_reducir_costo_o_liquidar_solo_si_inventario_conviene"
        if cause in ("perdida_probable_por_precio_alto_o_demanda_debil", "perdida_por_sin_ventas"):
            return "probar_precio_menor_si_aumenta_utilidad_esperada"
        if cause == "perdida_probable_por_costos_fijos_nomina_o_escala":
            return "ajustar_capacidad_nomina_capital_o_productividad"
        if cause == "perdida_probable_por_exceso_inventario_demanda_insuficiente":
            return "bajar_produccion_y_liquidar_inventario_solo_si_conviene"
        return "revisar_costos_demanda_inventario_y_precio"

    audit["probable_loss_cause"] = audit.apply(diagnose_loss, axis=1)
    audit["recommended_loss_response"] = audit["probable_loss_cause"].map(recommend_loss_action)
    ordered_columns = [
        "period",
        "year",
        "period_in_year",
        "firm_id",
        "sector",
        "active",
        "age",
        "starting_workers",
        "workers",
        "desired_workers",
        "vacancies",
        "vacancy_duration",
        "expected_sales",
        "sales",
        "production",
        "inventory",
        "price",
        "capital",
        "capital_efficiency_percent",
        "technology",
        "technology_level_percent",
        "investment_animal_spirits",
        "productivity_investment_propensity",
        "total_investment_propensity",
        "liquidity_reinvestment_rate",
        "capital_investment",
        "technology_investment",
        "technology_gain",
        "rd_investment_spending",
        "industrial_investment_spending",
        "investment_goods_units",
        "productivity_goods_spending",
        "no_productivity_capital_investment_reason",
        "capacity_goods_spending",
        "capacity_gain_workers",
        "unfilled_investment_budget",
        "investment_decision_reason",
        "productivity",
        "effective_worker_productivity_capacity",
        "installed_production_capacity_units",
        "capacity_utilization_rate",
        "stockout_rejected_units",
        "observed_demand_units",
        "stockout_pressure",
        "effective_marginal_unit_cost",
        "price_minus_marginal_unit_cost",
        "price_to_marginal_unit_cost_ratio",
        "sales_realization_ratio",
        "probable_loss_cause",
        "recommended_loss_response",
        "cash",
        "period_total_spending_outflow",
        "firm_period_marginal_propensity_to_spend",
        "firm_period_propensity_to_bank_deposit",
        "payroll_total",
        "severance_total",
        "total_cost",
        "firm_income_total",
        "firm_profit_total",
        "desired_workers_next_period",
        "worker_exits",
        "worker_quits",
        "worker_dismissals",
        "exited_workers_reemployed",
        "average_wage",
        "unemployment_rate",
    ]
    available_columns = [column for column in ordered_columns if column in audit.columns]
    return audit[available_columns].sort_values(["period", "sector", "firm_id"]).reset_index(drop=True)


def family_audit_frame(simulation) -> pd.DataFrame:
    family_history = getattr(simulation, "family_history", None)
    if family_history:
        return pd.DataFrame.from_records(
            asdict(snapshot) for snapshot in family_history
        ).sort_values(["period", "family_id"]).reset_index(drop=True)

    simulation._refresh_period_household_caches()
    simulation._refresh_period_family_cache()

    groups = simulation._family_groups()
    if not groups:
        return pd.DataFrame()

    sector_prices = {
        spec.key: simulation._average_sector_price(spec.key)
        for spec in SECTOR_SPECS
    }
    necessary_essential_demand_units = max(0.0, simulation._period_essential_demand_units)
    essential_offer_units = max(0.0, simulation._period_essential_production_units)
    necessary_demand_to_offer_ratio = (
        necessary_essential_demand_units / max(1.0, essential_offer_units)
        if essential_offer_units > 0.0
        else (1.0 if necessary_essential_demand_units > 0.0 else 0.0)
    )
    def family_basic_goods_coverage(members) -> tuple[float, float]:
        needed_value = 0.0
        covered_value = 0.0
        for member in members:
            for sector_key in ("food", "housing", "clothing"):
                desired_units = max(0.0, simulation._household_sector_desired_units(member, sector_key))
                bought_units = max(0.0, member.last_consumption.get(sector_key, 0.0))
                price = max(0.0, sector_prices[sector_key])
                needed_value += desired_units * price
                covered_value += min(desired_units, bought_units) * price
        coverage_ratio = covered_value / max(1e-9, needed_value) if needed_value > 0.0 else 1.0
        return min(1.0, max(0.0, coverage_ratio)), needed_value

    def shortfall_reason(
        *,
        coverage_ratio: float,
        needed_value: float,
        family_cash_available: float,
        family_voluntary_saved_cash: float,
        family_involuntary_retained_cash: float,
    ) -> str:
        if needed_value <= 0.0:
            return "sin_necesidad_basica_registrada"
        if coverage_ratio >= 0.995:
            return "cobertura_completa"
        affordability_gap = family_cash_available < needed_value * 0.98
        supply_gap = (
            necessary_essential_demand_units > 0.0
            and essential_offer_units < necessary_essential_demand_units * 0.98
        )
        if affordability_gap and supply_gap:
            return "no_alcanzo_y_no_habia_oferta_suficiente"
        if supply_gap:
            return "no_habia_inventario_u_oferta_suficiente"
        if affordability_gap:
            return "no_alcanzo_el_efectivo_disponible"
        shortfall_value = needed_value * max(0.0, 1.0 - coverage_ratio)
        if shortfall_value <= 1e-9:
            return "friccion_o_preferencia_no_comprar"
        if family_voluntary_saved_cash >= shortfall_value * 0.98:
            return "decidieron_ahorrar_o_no_comprar"
        if family_involuntary_retained_cash >= shortfall_value * 0.98:
            return "retencion_involuntaria_no_gastada"
        return "presupuesto_asignado_insuficiente_a_basicos"

    rows: list[dict[str, float | int | str]] = []
    for family_id, members in groups.items():
        alive_members = [member for member in members if member.alive]
        if not alive_members:
            continue
        adults = [
            member
            for member in alive_members
            if simulation._household_age_years(member) >= simulation.config.entry_age_years
        ]
        labor_capable_members = [
            member for member in alive_members if simulation._household_labor_capacity(member) > 0.0
        ]
        employed_members = sum(1 for member in labor_capable_members if member.employed_by is not None)

        essential_basket_cost = sum(simulation._essential_budget(member) for member in alive_members)
        school_target_units = sum(
            simulation._household_sector_desired_units(member, "school")
            for member in alive_members
        )
        public_school_units = max(
            0.0,
            getattr(simulation, "_period_family_public_education_units", {}).get((family_id, "school"), 0.0),
        )
        private_school_units = max(0.0, school_target_units - public_school_units)
        private_school_basket_cost = private_school_units * sector_prices["school"]

        family_income_total = sum(max(0.0, member.last_income) for member in alive_members)
        family_cash_available = max(0.0, simulation._period_family_cash_available.get(family_id, 0.0))
        family_period_income_cash = max(
            0.0,
            simulation._period_family_period_income_cash.get(family_id, 0.0),
        )
        total_basic_basket_cost = essential_basket_cost + private_school_basket_cost
        period_income_covers_basic_basket = family_period_income_cash + 1e-9 >= total_basic_basket_cost
        family_start_savings_cash = max(
            0.0,
            simulation._period_family_start_savings_cash.get(family_id, 0.0),
        )
        family_saved_cash = max(0.0, simulation._period_family_cash_saved.get(family_id, 0.0))
        family_spent_cash = max(0.0, simulation._period_family_cash_spent.get(family_id, 0.0))
        family_income_spent_cash = max(
            0.0,
            simulation._period_family_income_spent_cash.get(family_id, 0.0),
        )
        family_savings_spent_cash = max(
            0.0,
            simulation._period_family_savings_spent_cash.get(family_id, 0.0),
        )
        family_period_net_saving_cash = simulation._period_family_net_saving_cash.get(
            family_id,
            family_saved_cash - family_start_savings_cash,
        )
        family_voluntary_saved_cash = max(0.0, simulation._period_family_voluntary_saved_cash.get(family_id, 0.0))
        family_involuntary_retained_cash = max(
            0.0,
            simulation._period_family_involuntary_retained_cash.get(family_id, 0.0),
        )
        family_expected_salary = (
            sum(max(0.0, member.reservation_wage) for member in labor_capable_members)
            / max(1, len(labor_capable_members))
            if labor_capable_members
            else 0.0
        )
        accepted_members = [member for member in labor_capable_members if member.employed_by is not None]
        family_accepted_salary = (
            sum(max(0.0, member.contract_wage) for member in accepted_members)
            / max(1, len(accepted_members))
            if accepted_members
            else 0.0
        )
        basic_goods_coverage_ratio, basic_goods_needed_value = family_basic_goods_coverage(alive_members)
        basic_goods_shortfall_reason = shortfall_reason(
            coverage_ratio=basic_goods_coverage_ratio,
            needed_value=basic_goods_needed_value,
            family_cash_available=family_cash_available,
            family_voluntary_saved_cash=family_voluntary_saved_cash,
            family_involuntary_retained_cash=family_involuntary_retained_cash,
        )
        if family_cash_available > 0.0:
            propensity_to_save = min(1.0, max(0.0, family_saved_cash / family_cash_available))
            propensity_to_spend = min(1.0, max(0.0, family_spent_cash / family_cash_available))
        else:
            propensity_to_save = 0.0
            propensity_to_spend = 0.0

        rows.append(
            {
                "period": simulation.period,
                "year": ((simulation.period - 1) // max(1, simulation.config.periods_per_year)) + 1,
                "period_in_year": ((simulation.period - 1) % max(1, simulation.config.periods_per_year)) + 1,
                "family_id": family_id,
                "family_members": len(alive_members),
                "adult_members": len(adults),
                "labor_capable_members": len(labor_capable_members),
                "employed_members": employed_members,
                "total_basic_basket_cost_including_school": total_basic_basket_cost,
                "private_school_basket_cost": private_school_basket_cost,
                "total_family_income": family_income_total,
                "family_employment_rate": employed_members / max(1, len(labor_capable_members)),
                "family_cash_available": family_cash_available,
                "family_period_income_cash": family_period_income_cash,
                "period_income_covers_basic_basket": period_income_covers_basic_basket,
                "family_start_savings_cash": family_start_savings_cash,
                "family_cash_spent": family_spent_cash,
                "family_income_spent_cash": family_income_spent_cash,
                "family_savings_spent_cash": family_savings_spent_cash,
                "family_period_net_saving_cash": family_period_net_saving_cash,
                "family_voluntary_saved_cash": family_voluntary_saved_cash,
                "family_involuntary_retained_cash": family_involuntary_retained_cash,
                "family_expected_salary": family_expected_salary,
                "family_accepted_salary": family_accepted_salary,
                "basic_goods_coverage_percent": 100.0 * basic_goods_coverage_ratio,
                "basic_goods_shortfall_reason": basic_goods_shortfall_reason,
                "marginal_propensity_to_spend": propensity_to_spend,
                "marginal_propensity_to_save": propensity_to_save,
                "necessary_essential_demand_units": necessary_essential_demand_units,
                "essential_offer_units": essential_offer_units,
                "necessary_demand_to_offer_ratio": necessary_demand_to_offer_ratio,
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(["period", "family_id"]).reset_index(drop=True)


def firm_period_summary(firm_history_frame: pd.DataFrame) -> pd.DataFrame:
    if firm_history_frame.empty:
        return firm_history_frame.copy()

    summary = (
        firm_history_frame.groupby("period", as_index=False)
        .agg(
            year=("year", "last"),
            period_in_year=("period_in_year", "last"),
            active_firms=("active", "sum"),
            average_workers=("workers", "mean"),
            average_desired_workers=("desired_workers", "mean"),
            average_vacancies=("vacancies", "mean"),
            total_workers=("workers", "sum"),
            total_desired_workers=("desired_workers", "sum"),
            total_vacancies=("vacancies", "sum"),
            average_price=("price", "mean"),
            average_wage_offer=("wage_offer", "mean"),
            average_cash=("cash", "mean"),
            average_capital=("capital", "mean"),
            average_inventory=("inventory", "mean"),
            average_productivity=("productivity", "mean"),
            average_input_cost_per_unit=("input_cost_per_unit", "mean"),
            average_transport_cost_per_unit=("transport_cost_per_unit", "mean"),
            average_fixed_overhead=("fixed_overhead", "mean"),
            average_capital_charge=("capital_charge", "mean"),
            average_unit_cost=("unit_cost", "mean"),
            average_technology=("technology", "mean"),
            average_technology_investment=("technology_investment", "mean"),
            average_technology_gain=("technology_gain", "mean"),
            average_market_share=("market_share", "mean"),
            average_sales=("sales", "mean"),
            average_revenue=("revenue", "mean"),
            average_total_cost=("total_cost", "mean"),
            average_production=("production", "mean"),
            average_profit=("profit", "mean"),
            average_loss_streak=("loss_streak", "mean"),
        )
        .sort_values("period")
        .reset_index(drop=True)
    )

    summary["labor_gap"] = summary["total_desired_workers"] - summary["total_workers"]
    summary["vacancy_rate"] = summary["total_vacancies"] / summary["total_desired_workers"].replace(0, pd.NA)
    summary["worker_fill_rate"] = summary["total_workers"] / summary["total_desired_workers"].replace(0, pd.NA)
    return summary


def firm_year_summary(firm_history_frame: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    if firm_history_frame.empty:
        return firm_history_frame.copy()

    summary = (
        firm_history_frame.groupby("year", as_index=False)
        .agg(
            active_share=("active", "mean"),
            active_firm_months=("active", "sum"),
            unique_firms=("firm_id", "nunique"),
            average_workers=("workers", "mean"),
            average_desired_workers=("desired_workers", "mean"),
            average_vacancies=("vacancies", "mean"),
            average_price=("price", "mean"),
            average_wage_offer=("wage_offer", "mean"),
            average_cash=("cash", "mean"),
            average_capital=("capital", "mean"),
            average_inventory=("inventory", "mean"),
            average_productivity=("productivity", "mean"),
            average_input_cost_per_unit=("input_cost_per_unit", "mean"),
            average_transport_cost_per_unit=("transport_cost_per_unit", "mean"),
            average_fixed_overhead=("fixed_overhead", "mean"),
            average_capital_charge=("capital_charge", "mean"),
            average_unit_cost=("unit_cost", "mean"),
            average_technology=("technology", "mean"),
            average_technology_investment=("technology_investment", "mean"),
            average_technology_gain=("technology_gain", "mean"),
            average_market_share=("market_share", "mean"),
            average_sales=("sales", "mean"),
            average_revenue=("revenue", "mean"),
            average_total_cost=("total_cost", "mean"),
            average_production=("production", "mean"),
            average_profit=("profit", "mean"),
            average_loss_streak=("loss_streak", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    summary["average_active_firms"] = summary["active_share"] * summary["unique_firms"]
    summary["yearly_labor_gap"] = summary["average_desired_workers"] - summary["average_workers"]
    summary["yearly_vacancy_rate"] = summary["average_vacancies"] / summary["average_desired_workers"].replace(0, pd.NA)
    summary["yearly_worker_fill_rate"] = summary["average_workers"] / summary["average_desired_workers"].replace(0, pd.NA)
    summary["active_firm_months_per_year"] = summary["active_firm_months"] / max(1, periods_per_year)
    return summary
