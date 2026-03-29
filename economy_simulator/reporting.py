from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .domain import FirmPeriodSnapshot, PeriodSnapshot, SimulationResult


def history_frame(
    history: list[PeriodSnapshot],
    periods_per_year: int = 12,
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records([asdict(snapshot) for snapshot in history])
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["year"] = ((frame["period"] - 1) // max(1, periods_per_year)) + 1
    frame["period_in_year"] = ((frame["period"] - 1) % max(1, periods_per_year)) + 1
    frame["inflation_rate"] = frame["price_index"].pct_change()
    frame["gdp_growth"] = frame["gdp_nominal"].pct_change()
    frame["gdp_per_capita_growth"] = frame["gdp_per_capita"].pct_change()
    frame["population_growth"] = frame["population"].pct_change()
    frame["birth_death_balance"] = frame["births"] - frame["deaths"]
    frame["child_share"] = frame["children"] / frame["population"].replace(0, pd.NA)
    frame["adult_share"] = frame["adults"] / frame["population"].replace(0, pd.NA)
    frame["senior_share"] = frame["seniors"] / frame["population"].replace(0, pd.NA)
    frame["dependency_ratio"] = (frame["children"] + frame["seniors"]) / frame["adults"].replace(0, pd.NA)
    frame["real_gdp_nominal"] = frame["gdp_nominal"] / frame["price_index"].replace(0, pd.NA)
    frame["real_gdp_growth"] = frame["real_gdp_nominal"].pct_change()
    frame["capital_growth"] = frame["total_capital_stock"].pct_change()
    frame["inventory_growth"] = frame["total_inventory_units"].pct_change()
    return frame


def annual_frame(history_frame: pd.DataFrame) -> pd.DataFrame:
    if history_frame.empty:
        return history_frame.copy()

    annual = (
        history_frame.groupby("year", as_index=False)
        .agg(
            population=("population", "mean"),
            women=("women", "mean"),
            men=("men", "mean"),
            fertile_women=("fertile_women", "mean"),
            children=("children", "mean"),
            adults=("adults", "mean"),
            seniors=("seniors", "mean"),
            labor_force=("labor_force", "mean"),
            family_units=("family_units", "mean"),
            end_population=("population", "last"),
            end_women=("women", "last"),
            end_men=("men", "last"),
            end_fertile_women=("fertile_women", "last"),
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
            total_sales_revenue=("total_sales_revenue", "sum"),
            total_production_units=("total_production_units", "sum"),
            period_investment_spending=("period_investment_spending", "sum"),
            total_profit=("total_profit", "sum"),
            avg_employment_rate=("employment_rate", "mean"),
            avg_unemployment_rate=("unemployment_rate", "mean"),
            average_worker_savings=("average_worker_savings", "mean"),
            end_price_index=("price_index", "last"),
            gini_household_savings=("gini_household_savings", "last"),
            gini_owner_wealth=("gini_owner_wealth", "last"),
            capitalist_controlled_assets=("capitalist_controlled_assets", "last"),
            capitalist_asset_share=("capitalist_asset_share", "last"),
            total_bankruptcies=("bankruptcies", "sum"),
            total_capital_stock=("total_capital_stock", "last"),
            total_inventory_units=("total_inventory_units", "last"),
            total_household_savings=("total_household_savings", "last"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    annual["gdp_per_capita_annual"] = annual["gdp_nominal"] / annual["population"].replace(0, pd.NA)
    annual["real_gdp_nominal"] = annual["gdp_nominal"] / annual["end_price_index"].replace(0, pd.NA)
    annual["gdp_growth_yoy"] = annual["gdp_nominal"].pct_change()
    annual["real_gdp_growth_yoy"] = annual["real_gdp_nominal"].pct_change()
    annual["inflation_yoy"] = annual["end_price_index"].pct_change()
    annual["population_growth_yoy"] = annual["end_population"].pct_change()
    annual["birth_rate"] = annual["births"] / annual["population"].replace(0, pd.NA)
    annual["death_rate"] = annual["deaths"] / annual["population"].replace(0, pd.NA)
    annual["child_share"] = annual["children"] / annual["population"].replace(0, pd.NA)
    annual["adult_share"] = annual["adults"] / annual["population"].replace(0, pd.NA)
    annual["senior_share"] = annual["seniors"] / annual["population"].replace(0, pd.NA)
    annual["female_share"] = annual["women"] / annual["population"].replace(0, pd.NA)
    annual["male_share"] = annual["men"] / annual["population"].replace(0, pd.NA)
    annual["fertile_women_share"] = annual["fertile_women"] / annual["women"].replace(0, pd.NA)
    annual["dependency_ratio"] = (annual["children"] + annual["seniors"]) / annual["adults"].replace(0, pd.NA)
    annual["capital_growth_yoy"] = annual["total_capital_stock"].pct_change()
    annual["inventory_growth_yoy"] = annual["total_inventory_units"].pct_change()
    return annual


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
            "technology": snapshot.technology,
            "technology_investment": snapshot.technology_investment,
            "technology_gain": snapshot.technology_gain,
            "sales": snapshot.sales,
            "revenue": snapshot.revenue,
            "production": snapshot.production,
            "profit": snapshot.profit,
            "total_cost": snapshot.total_cost,
            "loss_streak": snapshot.loss_streak,
            "market_share": snapshot.market_share,
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
            average_production=("production", "mean"),
            average_profit=("profit", "mean"),
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
