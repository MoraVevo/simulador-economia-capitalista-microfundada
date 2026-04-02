from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .domain import FirmPeriodSnapshot, PeriodSnapshot, SimulationResult


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, pd.NA)


def _append_derived_columns(frame: pd.DataFrame, derived_columns: dict[str, pd.Series]) -> pd.DataFrame:
    if not derived_columns:
        return frame.copy()
    derived_frame = pd.DataFrame(derived_columns, index=frame.index)
    return pd.concat([frame, derived_frame], axis=1).copy()


def history_frame(
    history: list[PeriodSnapshot],
    periods_per_year: int = 12,
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records([asdict(snapshot) for snapshot in history])
    if frame.empty:
        return frame

    periods_per_year = max(1, periods_per_year)
    capitalist_liquid_wealth = (
        frame["capitalist_bank_deposits"] + frame["capitalist_vault_cash"] + frame["capitalist_firm_cash"]
    )
    capitalist_augmented_assets = (
        capitalist_liquid_wealth
        + frame["capitalist_productive_capital"]
        + frame["capitalist_inventory_value"]
    )
    worker_augmented_asset_denominator = frame["worker_bank_deposits"] + capitalist_augmented_assets
    derived_columns = {
        "inflation_rate": frame["price_index"].pct_change(),
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
        "real_gdp_nominal": _safe_ratio(frame["gdp_nominal"], frame["price_index"]),
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
        "capitalist_liquid_wealth": capitalist_liquid_wealth,
        "capitalist_augmented_assets": capitalist_augmented_assets,
        "worker_net_financial_position": frame["worker_bank_deposits"] - frame["worker_credit_outstanding"],
        "capitalist_net_financial_position": capitalist_augmented_assets - frame["capitalist_credit_outstanding"],
        "worker_augmented_asset_share": _safe_ratio(frame["worker_bank_deposits"], worker_augmented_asset_denominator),
        "total_bank_loans": frame["total_bank_loans_households"] + frame["total_bank_loans_firms"],
        "government_total_spending": frame["government_transfers"] + frame["government_procurement_spending"],
        "government_tax_burden_gdp": _safe_ratio(frame["government_tax_revenue"], frame["gdp_nominal"]),
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


def annual_frame(history_frame: pd.DataFrame) -> pd.DataFrame:
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
            gdp_per_capita_monthly=("gdp_per_capita", "mean"),
            total_wages=("total_wages", "sum"),
            total_sales_units=("total_sales_units", "sum"),
            potential_demand_units=("potential_demand_units", "sum"),
            demand_fulfillment_rate=("demand_fulfillment_rate", "mean"),
            essential_demand_units=("essential_demand_units", "sum"),
            essential_sales_units=("essential_sales_units", "sum"),
            essential_fulfillment_rate=("essential_fulfillment_rate", "mean"),
            average_food_meals_per_person=("average_food_meals_per_person", "mean"),
            food_sufficient_share=("food_sufficient_share", "mean"),
            food_subsistence_share=("food_subsistence_share", "mean"),
            food_acute_hunger_share=("food_acute_hunger_share", "mean"),
            food_severe_hunger_share=("food_severe_hunger_share", "mean"),
            average_health_fragility=("average_health_fragility", "mean"),
            average_perceived_utility=("average_perceived_utility", "mean"),
            total_sales_revenue=("total_sales_revenue", "sum"),
            total_production_units=("total_production_units", "sum"),
            period_investment_spending=("period_investment_spending", "sum"),
            business_cost_recycled=("business_cost_recycled", "sum"),
            business_cost_to_firms=("business_cost_to_firms", "sum"),
            business_cost_to_households=("business_cost_to_households", "sum"),
            business_cost_to_owners=("business_cost_to_owners", "sum"),
            inheritance_transfers=("inheritance_transfers", "sum"),
            bankruptcy_cash_recoveries=("bankruptcy_cash_recoveries", "sum"),
            total_profit=("total_profit", "sum"),
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
            government_corporate_tax_revenue=("government_corporate_tax_revenue", "sum"),
            government_dividend_tax_revenue=("government_dividend_tax_revenue", "sum"),
            government_wealth_tax_revenue=("government_wealth_tax_revenue", "sum"),
            government_transfers=("government_transfers", "sum"),
            government_unemployment_support=("government_unemployment_support", "sum"),
            government_child_allowance=("government_child_allowance", "sum"),
            government_basic_support=("government_basic_support", "sum"),
            government_procurement_spending=("government_procurement_spending", "sum"),
            government_bond_issuance=("government_bond_issuance", "sum"),
            government_deficit=("government_deficit", "sum"),
            government_surplus=("government_surplus", "sum"),
            labor_share_gdp=("labor_share_gdp", "mean"),
            profit_share_gdp=("profit_share_gdp", "mean"),
            investment_share_gdp=("investment_share_gdp", "mean"),
            capitalist_consumption_share_gdp=("capitalist_consumption_share_gdp", "mean"),
            government_spending_share_gdp=("government_spending_share_gdp", "mean"),
            dividend_share_gdp=("dividend_share_gdp", "mean"),
            retained_profit_share_gdp=("retained_profit_share_gdp", "mean"),
            central_bank_money_supply=("central_bank_money_supply", "last"),
            central_bank_target_money_supply=("central_bank_target_money_supply", "last"),
            central_bank_policy_rate=("central_bank_policy_rate", "last"),
            central_bank_issuance=("central_bank_issuance", "sum"),
            cumulative_central_bank_issuance=("cumulative_central_bank_issuance", "last"),
            household_credit_creation=("household_credit_creation", "sum"),
            firm_credit_creation=("firm_credit_creation", "sum"),
            commercial_bank_credit_creation=("commercial_bank_credit_creation", "sum"),
            average_bank_deposit_rate=("average_bank_deposit_rate", "mean"),
            average_bank_loan_rate=("average_bank_loan_rate", "mean"),
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
    derived_columns = {
        "gdp_per_capita_annual": _safe_ratio(annual["gdp_nominal"], annual["population"]),
        "real_gdp_nominal": _safe_ratio(annual["gdp_nominal"], annual["end_price_index"]),
        "gdp_growth_yoy": annual["gdp_nominal"].pct_change(),
        "inflation_yoy": annual["end_price_index"].pct_change(),
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
        "worker_consumption_share_gdp": _safe_ratio(
            (
                annual["worker_cash_available"] - annual["worker_cash_saved"]
            ).clip(lower=0.0),
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
        "government_total_spending": annual["government_transfers"] + annual["government_procurement_spending"],
        "government_tax_burden_gdp": _safe_ratio(annual["government_tax_revenue"], annual["gdp_nominal"]),
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
    history = history_frame(result.history, periods_per_year=result.config.periods_per_year)
    annual = annual_frame(history)
    return history, annual


def firm_history_frame(result: SimulationResult) -> pd.DataFrame:
    if not result.firm_history:
        return pd.DataFrame()

    return pd.DataFrame.from_records(
        {
            "period": snapshot.period,
            "year": snapshot.year,
            "period_in_year": snapshot.period_in_year,
            "firm_id": snapshot.firm_id,
            "sector": snapshot.sector,
            "active": snapshot.active,
            "workers": snapshot.workers,
            "desired_workers": snapshot.desired_workers,
            "vacancies": snapshot.vacancies,
            "price": snapshot.price,
            "wage_offer": snapshot.wage_offer,
            "cash": snapshot.cash,
            "capital": snapshot.capital,
            "inventory": snapshot.inventory,
            "productivity": snapshot.productivity,
            "input_cost_per_unit": snapshot.input_cost_per_unit,
            "transport_cost_per_unit": snapshot.transport_cost_per_unit,
            "fixed_overhead": snapshot.fixed_overhead,
            "capital_charge": snapshot.capital_charge,
            "unit_cost": snapshot.unit_cost,
            "markup_tolerance": snapshot.markup_tolerance,
            "volume_preference": snapshot.volume_preference,
            "inventory_aversion": snapshot.inventory_aversion,
            "employment_inertia": snapshot.employment_inertia,
            "price_aggressiveness": snapshot.price_aggressiveness,
            "cash_conservatism": snapshot.cash_conservatism,
            "market_share_ambition": snapshot.market_share_ambition,
            "forecast_caution": snapshot.forecast_caution,
            "technology": snapshot.technology,
            "technology_investment": snapshot.technology_investment,
            "technology_gain": snapshot.technology_gain,
            "sales": snapshot.sales,
            "expected_sales": snapshot.expected_sales,
            "revenue": snapshot.revenue,
            "production": snapshot.production,
            "profit": snapshot.profit,
            "total_cost": snapshot.total_cost,
            "loss_streak": snapshot.loss_streak,
            "market_share": snapshot.market_share,
            "market_fragility_belief": snapshot.market_fragility_belief,
            "forecast_error_belief": snapshot.forecast_error_belief,
            "target_inventory": snapshot.target_inventory,
            "age": snapshot.age,
        }
        for snapshot in result.firm_history
    )


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
