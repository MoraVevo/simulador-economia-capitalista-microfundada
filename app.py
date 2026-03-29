from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from economy_simulator import EconomySimulation, ESSENTIAL_SECTOR_KEYS, SECTOR_BY_KEY, SimulationConfig
from economy_simulator.reporting import annual_frame, firm_history_frame, firm_period_summary, firm_year_summary, simulation_frames

INITIAL_HOUSEHOLDS = 10000


COLUMN_LABELS = {
    "gdp_nominal": "PIB nominal",
    "real_gdp_nominal": "PIB real aproximado",
    "price_index": "Indice de precios",
    "inflation_rate": "Inflacion",
    "population": "Poblacion",
    "women": "Mujeres",
    "men": "Hombres",
    "fertile_women": "Mujeres fertiles",
    "births": "Nacimientos",
    "deaths": "Muertes",
    "average_age": "Edad promedio",
    "children": "Ninos",
    "adults": "Adultos",
    "seniors": "Mayores",
    "labor_force": "Fuerza laboral",
    "children_with_guardian": "Ninos con apoyo familiar",
    "orphans": "Huerfanos",
    "family_units": "Unidades familiares",
    "average_family_income": "Ingreso familiar promedio",
    "average_family_resources": "Recursos familiares promedio",
    "average_family_basic_basket_cost": "Canasta basica familiar promedio",
    "family_income_to_basket_ratio": "Cobertura ingreso/canasta",
    "family_resources_to_basket_ratio": "Cobertura recursos/canasta",
    "families_income_below_basket_share": "Familias con ingreso bajo canasta",
    "families_resources_below_basket_share": "Familias con recursos bajos canasta",
    "average_worker_savings": "Ahorro promedio de trabajadores",
    "employment_rate": "Empleo",
    "unemployment_rate": "Desempleo",
    "average_workers": "Trabajadores promedio",
    "average_desired_workers": "Trabajadores deseados promedio",
    "average_vacancies": "Vacantes promedio",
    "total_workers": "Trabajadores totales",
    "total_desired_workers": "Trabajadores deseados totales",
    "total_vacancies": "Vacantes totales",
    "labor_gap": "Brecha laboral",
    "worker_fill_rate": "Cobertura de contratacion",
    "vacancy_rate": "Tasa de vacancia",
    "average_price": "Precio promedio",
    "average_wage_offer": "Salario ofrecido promedio",
    "average_cash": "Caja promedio",
    "average_capital": "Capital promedio",
    "average_inventory": "Inventario promedio",
    "average_productivity": "Productividad promedio",
    "average_technology": "Tecnologia promedio",
    "average_technology_investment": "Inversion tecnologica promedio",
    "average_technology_gain": "Ganancia tecnologica promedio",
    "average_sales": "Ventas promedio",
    "average_revenue": "Ingresos promedio",
    "average_total_cost": "Costo total promedio",
    "average_production": "Produccion promedio",
    "average_profit": "Ganancia promedio",
    "labor_supply_gap": "Exceso de oferta laboral",
    "total_capital_stock": "Stock de capital",
    "total_inventory_units": "Inventarios",
    "total_household_savings": "Ahorro de hogares",
    "potential_demand_units": "Demanda potencial",
    "demand_fulfillment_rate": "Cobertura de demanda",
    "essential_demand_units": "Demanda basica necesaria",
    "essential_sales_units": "Compras basicas realizadas",
    "essential_fulfillment_rate": "Cobertura de bienes basicos",
    "gini_household_savings": "Gini del ahorro",
    "gini_owner_wealth": "Gini de riqueza de duenos",
    "capitalist_controlled_assets": "Activos controlados por capitalistas",
    "capitalist_asset_share": "Participacion de dinero liquido capitalista",
    "gdp_per_capita": "PIB per capita",
    "gdp_per_capita_annual": "PIB per capita anual",
    "gdp_per_capita_monthly": "PIB per capita promedio",
    "gdp_growth": "Crecimiento del PIB",
    "gdp_growth_yoy": "Crecimiento anual del PIB",
    "real_gdp_growth": "Crecimiento del PIB real",
    "real_gdp_growth_yoy": "Crecimiento anual del PIB real",
    "inflation_yoy": "Inflacion anual",
    "population_growth": "Crecimiento poblacional",
    "population_growth_yoy": "Crecimiento anual poblacional",
    "birth_death_balance": "Balance demografico",
    "end_population": "Poblacion final",
    "end_women": "Mujeres finales",
    "end_men": "Hombres finales",
    "end_fertile_women": "Mujeres fertiles finales",
    "end_children": "Ninos finales",
    "end_adults": "Adultos finales",
    "end_seniors": "Mayores finales",
    "end_labor_force": "Fuerza laboral final",
    "birth_rate": "Tasa de natalidad",
    "death_rate": "Tasa de mortalidad",
    "child_share": "Participacion ninos",
    "adult_share": "Participacion adultos",
    "senior_share": "Participacion mayores",
    "female_share": "Participacion mujeres",
    "male_share": "Participacion hombres",
    "fertile_women_share": "Participacion mujeres fertiles",
    "dependency_ratio": "Relacion de dependencia",
    "avg_unemployment_rate": "Desempleo promedio",
    "total_bankruptcies": "Quiebras totales",
    "year": "Anio",
    "period": "Periodo",
    "period_in_year": "Periodo del anio",
    "end_price_index": "Indice de precios final",
    "capital_growth_yoy": "Crecimiento anual del capital",
    "inventory_growth_yoy": "Crecimiento anual de inventarios",
}

FIRM_LABELS = {
    "firm_id": "Firma",
    "sector": "Sector",
    "active": "Activa",
    "workers": "Trabajadores",
    "price": "Precio",
    "wage_offer": "Salario ofrecido",
    "cash": "Efectivo",
    "capital": "Capital",
    "inventory": "Inventario",
    "revenue": "Ingresos",
    "total_cost": "Costo total",
    "profit": "Ganancia",
    "sales": "Ventas",
    "input_cost_per_unit": "Costo insumo por unidad",
    "transport_cost_per_unit": "Costo transporte por unidad",
    "fixed_overhead": "Gasto fijo",
    "capital_charge": "Cargo de capital",
    "unit_cost": "Costo unitario",
    "markup_tolerance": "Tolerancia al margen",
    "volume_preference": "Preferencia por volumen",
    "inventory_aversion": "Aversion al inventario",
    "employment_inertia": "Inercia laboral",
    "price_aggressiveness": "Agresividad de precio",
    "cash_conservatism": "Conservadurismo de caja",
    "market_share_ambition": "Ambicion de cuota",
    "technology": "Tecnologia",
    "technology_investment": "Inversion en tecnologia",
    "technology_gain": "Ganancia tecnologica",
    "loss_streak": "Racha de perdidas",
    "market_share": "Participacion de mercado",
    "desired_workers": "Trabajadores deseados",
    "vacancies": "Vacantes",
    "target_inventory": "Inventario objetivo",
    "age": "Antiguedad",
}

SECTOR_LABELS = {
    "Basic food": "Alimentos basicos",
    "Housing and essential services": "Vivienda y servicios esenciales",
    "Clothing and hygiene": "Ropa e higiene",
    "Non-essential manufactured goods": "Bienes manufacturados no esenciales",
    "Leisure / entertainment / simple technology": "Ocio / entretenimiento / tecnologia simple",
}


st.set_page_config(
    page_title="Simulador economico",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at top left, #f8fbff 0%, #eef2f7 48%, #e5ebf3 100%);
    color: #0f172a;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
}
p, li, label, span {
    color: #1f2937;
}
.stMetric {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    padding: 0.85rem 0.9rem;
    border-radius: 14px;
}
.panel {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
    border-radius: 18px;
    padding: 1rem 1rem 0.75rem 1rem;
}
.small-note {
    color: #475569;
    font-size: 0.93rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def run_model(periods: int, seed: int, periods_per_year: int, firms_per_sector: int):
    config = SimulationConfig(
        periods=periods,
        households=INITIAL_HOUSEHOLDS,
        seed=seed,
        periods_per_year=periods_per_year,
        firms_per_sector=firms_per_sector,
    )
    result = EconomySimulation(config).run()
    history_df, annual_df = simulation_frames(result)
    return result, history_df, annual_df


def money(value: float) -> str:
    return f"{value:,.2f}"


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def fmt_delta(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return f"{value:.1%}"


def make_line_chart(frame: pd.DataFrame, x: str, y_cols: list[str], title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    palette = px.colors.qualitative.Set2

    for index, column in enumerate(y_cols):
        if column not in frame.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame[x],
                y=frame[column],
                mode="lines",
                name=COLUMN_LABELS.get(column, column.replace("_", " ").title()),
                line=dict(width=2.4, color=palette[index % len(palette)]),
            )
        )

    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0f172a"),
        legend_title_text="",
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        yaxis_title=y_title,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    return fig


def build_firm_table(result) -> pd.DataFrame:
    records = []
    for firm in result.firms:
        spec = SECTOR_BY_KEY[firm.sector]
        records.append(
            {
                "firm_id": firm.id,
                "sector": SECTOR_LABELS.get(spec.name, spec.name),
                "active": "Si" if firm.active else "No",
                "workers": len(firm.workers),
                "price": firm.price,
                "wage_offer": firm.wage_offer,
                "cash": firm.cash,
                "capital": firm.capital,
                "inventory": firm.inventory,
                "revenue": firm.last_revenue,
                "total_cost": firm.last_total_cost,
                "profit": firm.last_profit,
                "sales": firm.last_sales,
                "markup_tolerance": firm.markup_tolerance,
                "volume_preference": firm.volume_preference,
                "inventory_aversion": firm.inventory_aversion,
                "employment_inertia": firm.employment_inertia,
                "price_aggressiveness": firm.price_aggressiveness,
                "cash_conservatism": firm.cash_conservatism,
                "market_share_ambition": firm.market_share_ambition,
                "technology": firm.technology,
                "technology_investment": firm.last_technology_investment,
                "technology_gain": firm.last_technology_gain,
                "loss_streak": firm.loss_streak,
                "age": firm.age,
            }
        )
    table = pd.DataFrame.from_records(records)
    if not table.empty:
        table = table.sort_values("sector")
    return table


def build_distribution_data(result):
    active_households = [household for household in result.households if household.alive]
    household_savings = pd.Series([household.savings for household in active_households], name="savings")
    owner_wealth = pd.Series([owner.wealth for owner in result.entrepreneurs], name="wealth")
    return household_savings, owner_wealth


def build_sector_productivity_data(
    firm_history_frame: pd.DataFrame,
    sector_key: str,
    time_col: str = "period",
) -> pd.DataFrame:
    sector_history = firm_history_frame[firm_history_frame["sector"] == sector_key].copy()
    if sector_history.empty:
        return sector_history

    summary = (
        sector_history.groupby(time_col, as_index=False)
        .agg(
            year=("year", "last"),
            period_in_year=("period_in_year", "last"),
            total_workers=("workers", "sum"),
            total_production=("production", "sum"),
            average_firm_productivity=("productivity", "mean"),
        )
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    summary["production_per_worker"] = summary["total_production"] / summary["total_workers"].replace(0, pd.NA)
    return summary


def build_sector_group_productivity_data(
    firm_history_frame: pd.DataFrame,
    population_frame: pd.DataFrame,
    sector_keys: tuple[str, ...],
    time_col: str = "period",
    population_column: str = "population",
) -> pd.DataFrame:
    sector_history = firm_history_frame[firm_history_frame["sector"].isin(sector_keys)].copy()
    if sector_history.empty:
        return sector_history

    summary = (
        sector_history.groupby(time_col, as_index=False)
        .agg(
            year=("year", "last"),
            period_in_year=("period_in_year", "last"),
            total_workers=("workers", "sum"),
            total_production=("production", "sum"),
            average_firm_productivity=("productivity", "mean"),
        )
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    summary = summary.merge(population_frame[[time_col, population_column]], on=time_col, how="left")
    summary["production_per_worker"] = summary["total_production"] / summary["total_workers"].replace(0, pd.NA)
    summary["production_per_person"] = summary["total_production"] / summary[population_column].replace(0, pd.NA)
    return summary


def build_sector_labor_market_data(firm_period_frame: pd.DataFrame) -> pd.DataFrame:
    if firm_period_frame.empty:
        return firm_period_frame.copy()

    summary = (
        firm_period_frame.groupby("sector", as_index=False)
        .agg(
            total_firms=("firm_id", "count"),
            active_firms=("active", "sum"),
            total_workers=("workers", "sum"),
            total_desired_workers=("desired_workers", "sum"),
            total_vacancies=("vacancies", "sum"),
            average_price=("price", "mean"),
            average_wage_offer=("wage_offer", "mean"),
        )
        .sort_values("sector")
        .reset_index(drop=True)
    )
    summary["labor_gap"] = summary["total_desired_workers"] - summary["total_workers"]
    summary["vacancy_rate"] = summary["total_vacancies"] / summary["total_desired_workers"].replace(0, pd.NA)
    summary["worker_fill_rate"] = summary["total_workers"] / summary["total_desired_workers"].replace(0, pd.NA)
    summary["sector_name"] = summary["sector"].map(lambda sector_key: SECTOR_BY_KEY[sector_key].name)
    summary["sector_label"] = summary["sector_name"].map(lambda name: SECTOR_LABELS.get(name, name))
    return summary


def build_basic_goods_price_composition(
    firm_period_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, float | None, float | None]:
    essential_firms = firm_period_frame[firm_period_frame["sector"].isin(ESSENTIAL_SECTOR_KEYS)].copy()
    if essential_firms.empty:
        return pd.DataFrame(), None, None

    component_totals = {
        "Trabajo": 0.0,
        "Insumos": 0.0,
        "Transporte": 0.0,
        "Gastos fijos": 0.0,
        "Cargo de capital": 0.0,
    }
    total_production = 0.0
    total_revenue = 0.0

    for _, row in essential_firms.iterrows():
        production = float(row["production"])
        if production <= 0.0:
            continue

        total_production += production
        labor_total = max(0.0, float(row["wage_offer"]) * float(row["workers"]))
        input_total = max(0.0, float(row["input_cost_per_unit"]) * production)
        transport_total = max(0.0, float(row["transport_cost_per_unit"]) * production)
        fixed_total = max(0.0, float(row["fixed_overhead"]))
        capital_total = max(0.0, float(row["capital_charge"]))
        revenue_total = max(0.0, float(row["price"]) * production)

        component_totals["Trabajo"] += labor_total
        component_totals["Insumos"] += input_total
        component_totals["Transporte"] += transport_total
        component_totals["Gastos fijos"] += fixed_total
        component_totals["Cargo de capital"] += capital_total
        total_revenue += revenue_total

    if total_production <= 0.0:
        return pd.DataFrame(), None, None

    average_price = total_revenue / total_production
    component_per_unit = {
        name: value / total_production for name, value in component_totals.items()
    }
    average_unit_cost = sum(component_per_unit.values())
    margin_per_unit = max(0.0, average_price - average_unit_cost)

    composition = pd.DataFrame(
        {
            "componente": [
                "Trabajo",
                "Insumos",
                "Transporte",
                "Gastos fijos",
                "Cargo de capital",
                "Margen",
            ],
            "valor": [
                component_per_unit["Trabajo"],
                component_per_unit["Insumos"],
                component_per_unit["Transporte"],
                component_per_unit["Gastos fijos"],
                component_per_unit["Cargo de capital"],
                margin_per_unit,
            ],
        }
    )
    composition = composition[composition["valor"] > 0].reset_index(drop=True)
    return composition, average_price, average_unit_cost


st.title("Simulador economico")
st.caption(
    "Las cifras nominales se muestran en las unidades monetarias internas del simulador. "
    "La simulacion usa un paso interno discreto, pero la interfaz se agrupa por anios."
)

with st.sidebar:
    st.header("Escenario")
    years = st.slider("Anios", min_value=1, max_value=200, value=20, step=1)
    seed = st.number_input("Semilla", min_value=1, max_value=999999, value=7, step=1)
    periods_per_year = st.selectbox("Periodos por anio", options=[4, 6, 12], index=2)
    firms_per_sector = st.slider("Firmas por sector", min_value=1, max_value=30, value=20, step=1)
    total_periods = years * periods_per_year
    st.markdown(f"La poblacion inicial queda fija en {INITIAL_HOUSEHOLDS} hogares/agentes para la linea base.")
    st.markdown("La simulacion es determinista con los mismos parametros y la misma semilla.")
    if "view_year" not in st.session_state:
        st.session_state.view_year = years
    st.session_state.view_year = min(st.session_state.view_year, years)
    nav_prev, nav_center, nav_next = st.columns(3)
    if nav_prev.button("Anio anterior", use_container_width=True, disabled=st.session_state.view_year <= 1):
        st.session_state.view_year -= 1
        st.rerun()
    if nav_center.button("Volver al final", use_container_width=True):
        st.session_state.view_year = years
        st.rerun()
    if nav_next.button("Anio siguiente", use_container_width=True, disabled=st.session_state.view_year >= years):
        st.session_state.view_year += 1
        st.rerun()


result, history_df, annual_df = run_model(total_periods, seed, periods_per_year, firms_per_sector)
firm_history_df = firm_history_frame(result)
household_savings, owner_wealth = build_distribution_data(result)
selected_year = min(st.session_state.view_year, years)
history_view_df = history_df[history_df["year"] <= selected_year].copy()
annual_view_df = annual_frame(history_view_df)
history_view_df = annual_view_df.copy()
history_view_df["period"] = history_view_df["year"]
firm_history_view_df = firm_history_df[firm_history_df["year"] <= selected_year].copy()
firm_year_view_df = firm_history_view_df[firm_history_view_df["year"] == selected_year].copy()
firm_year_view_df = firm_year_view_df.sort_values(["sector", "price"]).copy()
firm_summary_view_df = firm_year_summary(firm_history_view_df, periods_per_year=periods_per_year)
food_productivity_view_df = build_sector_productivity_data(firm_history_view_df, "food", time_col="year")
food_productivity_view_df["period"] = food_productivity_view_df["year"]
basic_goods_view_df = build_sector_group_productivity_data(
    firm_history_view_df,
    annual_view_df,
    ESSENTIAL_SECTOR_KEYS,
    time_col="year",
    population_column="end_population",
)
basic_goods_view_df["period"] = basic_goods_view_df["year"]
basic_price_composition_df, basic_price_average, basic_price_cost = build_basic_goods_price_composition(
    firm_year_view_df
)
sector_labor_market_df = build_sector_labor_market_data(firm_year_view_df)
firm_summary_view_df = firm_summary_view_df.merge(
    annual_view_df[["year", "labor_force", "avg_unemployment_rate"]],
    on="year",
    how="left",
)
firm_summary_view_df["period"] = firm_summary_view_df["year"]
firm_summary_view_df["labor_supply_gap"] = firm_summary_view_df["labor_force"] - firm_summary_view_df["average_desired_workers"]
history_view = history_view_df.rename(columns=COLUMN_LABELS)
annual_view = annual_view_df.rename(columns=COLUMN_LABELS)
annual_full_view = annual_df.rename(columns=COLUMN_LABELS)
firm_view = firm_year_view_df.rename(columns=FIRM_LABELS)
firm_summary_view = firm_summary_view_df.rename(columns=COLUMN_LABELS)

latest_year = annual_view_df.iloc[-1] if not annual_view_df.empty else None
previous_year = annual_view_df.iloc[-2] if len(annual_view_df) > 1 else None

annual_gdp_delta = fmt_delta(latest_year["gdp_growth_yoy"]) if latest_year is not None else None
annual_population_delta = fmt_delta(latest_year["population_growth_yoy"]) if latest_year is not None else None

annual_gdp_pc_delta = fmt_delta(
    latest_year["gdp_per_capita_annual"] / previous_year["gdp_per_capita_annual"] - 1
    if latest_year is not None and previous_year is not None
    else None
)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Resumen macro y demografico")
st.caption(
    f"Mostrando el anio {selected_year} de {years}. Usa los botones de la barra lateral para avanzar o retroceder anio por anio."
)
st.markdown(
    "El tablero usa la contabilidad interna de la simulacion. El PIB es un proxy nominal "
    "construido a partir de ventas mas reinversion de las firmas, y la poblacion cambia "
    "con nacimientos y muertes."
)
st.caption(
    "El sustento infantil sale primero del ingreso y ahorro de madre y padre. "
    "El tutor solo entra como respaldo si los padres ya no estan vivos."
)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("PIB anual", money(latest_year["gdp_nominal"]), delta=annual_gdp_delta)
with metric_cols[1]:
    annual_gdp_value = latest_year["gdp_nominal"]
    st.metric("PIB anual acumulado", money(annual_gdp_value), delta=annual_gdp_delta)
with metric_cols[2]:
    annual_gdp_pc_value = latest_year["gdp_per_capita_annual"]
    st.metric("PIB per capita", money(annual_gdp_pc_value), delta=annual_gdp_pc_delta)
with metric_cols[3]:
    st.metric(
        "Poblacion total",
        f"{int(latest_year['end_population']):,}",
        delta=annual_population_delta,
    )

metric_cols_2 = st.columns(4)
with metric_cols_2[0]:
    st.metric("Desempleo", pct(latest_year["avg_unemployment_rate"]))
with metric_cols_2[1]:
    st.metric("Mujeres", f"{int(latest_year['end_women'])}")
with metric_cols_2[2]:
    st.metric("Hombres", f"{int(latest_year['end_men'])}")
with metric_cols_2[3]:
    st.metric("Mujeres fertiles", f"{int(latest_year['end_fertile_women'])}")

metric_cols_3 = st.columns(4)
with metric_cols_3[0]:
    st.metric("Ninos", f"{int(latest_year['end_children'])}")
with metric_cols_3[1]:
    st.metric("Adultos", f"{int(latest_year['end_adults'])}")
with metric_cols_3[2]:
    st.metric("Mayores", f"{int(latest_year['end_seniors'])}")
with metric_cols_3[3]:
    st.metric("Fuerza laboral", f"{int(latest_year['end_labor_force'])}")

metric_cols_4b = st.columns(3)
with metric_cols_4b[0]:
    st.metric("Nacimientos", f"{int(latest_year['births'])}")
with metric_cols_4b[1]:
    st.metric("Muertes", f"{int(latest_year['deaths'])}")
with metric_cols_4b[2]:
    st.metric("Edad promedio", f"{latest_year['average_age']:.1f} anos")

metric_cols_4c = st.columns(2)
with metric_cols_4c[0]:
    st.metric("Ninos con familia", f"{int(latest_year['children_with_guardian'])}")
with metric_cols_4c[1]:
    st.metric("Huerfanos", f"{int(latest_year['orphans'])}")

metric_cols_4d = st.columns(4)
with metric_cols_4d[0]:
    st.metric("Canasta basica familiar", money(latest_year["average_family_basic_basket_cost"]))
with metric_cols_4d[1]:
    st.metric("Ingreso familiar", money(latest_year["average_family_income"]))
with metric_cols_4d[2]:
    st.metric("Recursos familiares", money(latest_year["average_family_resources"]))
with metric_cols_4d[3]:
    st.metric("Cobertura ingreso/canasta", pct(latest_year["family_income_to_basket_ratio"]))

metric_cols_4e = st.columns(2)
with metric_cols_4e[0]:
    st.metric("Familias con ingreso bajo canasta", pct(latest_year["families_income_below_basket_share"]))
with metric_cols_4e[1]:
    st.metric("Familias con recursos bajos canasta", pct(latest_year["families_resources_below_basket_share"]))

metric_cols_4f = st.columns(2)
with metric_cols_4f[0]:
    st.metric("Ahorro promedio de trabajadores", money(latest_year["average_worker_savings"]))

latest_food_productivity = None
if not food_productivity_view_df.empty:
    latest_food_productivity = food_productivity_view_df.iloc[-1]["production_per_worker"]
with metric_cols_4f[1]:
    if latest_food_productivity is None or pd.isna(latest_food_productivity):
        st.metric("Produccion basica por trabajador", "n/a")
    else:
        st.metric("Produccion basica por trabajador", f"{latest_food_productivity:.2f} unidades")

metric_cols_4g = st.columns(2)
latest_basic_goods_per_person = None
if not basic_goods_view_df.empty:
    latest_basic_goods_per_person = basic_goods_view_df.iloc[-1]["production_per_person"]
with metric_cols_4g[0]:
    if latest_year is None or pd.isna(latest_year["essential_fulfillment_rate"]):
        st.metric("Cobertura de bienes basicos", "n/a")
    else:
        st.metric("Cobertura de bienes basicos", pct(latest_year["essential_fulfillment_rate"]))
with metric_cols_4g[1]:
    if latest_basic_goods_per_person is None or pd.isna(latest_basic_goods_per_person):
        st.metric("Produccion basica por persona", "n/a")
    else:
        st.metric("Produccion basica por persona", f"{latest_basic_goods_per_person:.2f} unidades")

metric_cols_4 = st.columns(4)
with metric_cols_4[0]:
    st.metric("Inflacion interanual", pct(latest_year["inflation_yoy"]))
with metric_cols_4[1]:
    st.metric("Cobertura de demanda", pct(latest_year["demand_fulfillment_rate"]))
with metric_cols_4[2]:
    st.metric("Stock de capital", money(latest_year["total_capital_stock"]))
with metric_cols_4[3]:
    st.metric("Gini del ahorro de hogares", f"{latest_year['gini_household_savings']:.2f}")

metric_cols_5 = st.columns(2)
with metric_cols_5[0]:
    st.metric("Gini de riqueza de duenos", f"{latest_year['gini_owner_wealth']:.2f}")
with metric_cols_5[1]:
    st.metric("Participacion de dinero liquido capitalista", pct(latest_year["capitalist_asset_share"]))

st.markdown("</div>", unsafe_allow_html=True)

tab_monthly, tab_annual, tab_distribution, tab_firms, tab_data = st.tabs(
    ["Evolucion anual", "Resumen anual", "Distribucion", "Empresas", "Datos"]
)

with tab_monthly:
    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["population"],
                title="Poblacion total",
                y_title="Personas",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with top_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["births", "deaths"],
                title="Nacimientos y muertes",
                y_title="Personas",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c0, c1 = st.columns(2)
    with c0:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["children", "adults", "seniors"],
                title="Estructura por edades",
                y_title="Personas",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["population", "labor_force"],
                title="Poblacion y fuerza laboral",
                y_title="Personas",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c2a, c2b = st.columns(2)
    with c2a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["women", "men"],
                title="Mujeres y hombres",
                y_title="Personas",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["fertile_women"],
                title="Mujeres en edad fertil",
                y_title="Personas",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            history_view_df,
            x="period",
            y_cols=["average_worker_savings"],
            title="Ahorro promedio de los trabajadores",
            y_title="Unidades monetarias",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            history_view_df,
            x="period",
            y_cols=["dependency_ratio"],
            title="Relacion de dependencia",
            y_title="Relacion",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["gdp_nominal", "real_gdp_nominal"],
                title="PIB y produccion real aproximada",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["price_index", "inflation_rate"],
                title="Precios e inflacion",
                y_title="Indice / tasa",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["employment_rate", "unemployment_rate"],
                title="Mercado laboral",
                y_title="Tasa",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["total_sales_units", "potential_demand_units"],
                title="Oferta y demanda",
                y_title="Unidades",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["average_family_basic_basket_cost", "average_family_income", "average_family_resources"],
                title="Canasta basica e ingreso familiar",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["family_income_to_basket_ratio", "family_resources_to_basket_ratio"],
                title="Cobertura familiar frente a la canasta",
                y_title="Proporcion",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c_food_1, c_food_2 = st.columns(2)
    with c_food_1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                food_productivity_view_df,
                x="period",
                y_cols=["total_production"],
                title="Produccion anual de alimentos basicos",
                y_title="Unidades",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_food_2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                food_productivity_view_df,
                x="period",
                y_cols=["production_per_worker"],
                title="Produccion de alimentos basicos por trabajador",
                y_title="Unidades por trabajador",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c_basic_1, c_basic_2 = st.columns(2)
    with c_basic_1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["essential_demand_units", "essential_sales_units"],
                title="Bienes basicos requeridos y comprados",
                y_title="Unidades",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_basic_2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                basic_goods_view_df,
                x="period",
                y_cols=["production_per_person", "production_per_worker"],
                title="Produccion basica por persona y por trabajador",
                y_title="Unidades",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            history_view_df,
            x="period",
            y_cols=["gini_household_savings", "gini_owner_wealth"],
            title="Desigualdad",
            y_title="Gini",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_annual:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            annual_view_df,
            x="year",
            y_cols=["gdp_nominal", "gdp_per_capita_annual"],
            title="PIB anual y PIB per capita",
            y_title="Unidades monetarias",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            annual_view_df,
            x="year",
            y_cols=["end_population"],
            title="Poblacion anual",
            y_title="Personas",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            annual_view_df,
            x="year",
            y_cols=["child_share", "adult_share", "senior_share"],
            title="Composicion poblacional",
            y_title="Participacion",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            annual_view_df,
            x="year",
            y_cols=["female_share", "male_share"],
            title="Composicion por sexo",
            y_title="Participacion",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            annual_view_df,
            x="year",
            y_cols=["fertile_women_share"],
            title="Participacion de mujeres fertiles",
            y_title="Participacion",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            annual_view_df,
            x="year",
            y_cols=["dependency_ratio"],
            title="Relacion de dependencia anual",
            y_title="Relacion",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            annual_view_df,
            x="year",
            y_cols=["average_worker_savings"],
            title="Ahorro promedio de trabajadores anual",
            y_title="Unidades monetarias",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                annual_view_df,
                x="year",
                y_cols=["gdp_growth_yoy", "real_gdp_growth_yoy"],
                title="Crecimiento interanual",
                y_title="Tasa de crecimiento",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                annual_view_df,
                x="year",
                y_cols=["inflation_yoy", "avg_unemployment_rate"],
                title="Inflacion y desempleo",
                y_title="Tasa",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            annual_view_df,
            x="year",
            y_cols=["inflation_yoy"],
            title="Inflacion por anio",
            y_title="Tasa",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                annual_view_df,
                x="year",
                y_cols=["population_growth_yoy"],
                title="Crecimiento poblacional anual",
                y_title="Tasa",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                annual_view_df,
                x="year",
                y_cols=["birth_rate", "death_rate"],
                title="Natalidad y mortalidad",
                y_title="Tasa",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(
        annual_view[
            [
                "Anio",
                "Poblacion",
                "Poblacion final",
                "Mujeres",
                "Hombres",
                "Mujeres fertiles",
                "Mujeres finales",
                "Hombres finales",
                "Mujeres fertiles finales",
                "Ninos",
                "Adultos",
                "Mayores",
                "Fuerza laboral",
                "Nacimientos",
                "Muertes",
                "Edad promedio",
                "Ahorro promedio de trabajadores",
                "Unidades familiares",
                "Ingreso familiar promedio",
                "Recursos familiares promedio",
                "Canasta basica familiar promedio",
                "Cobertura ingreso/canasta",
                "Cobertura recursos/canasta",
                "Familias con ingreso bajo canasta",
                "Familias con recursos bajos canasta",
                "Participacion ninos",
                "Participacion adultos",
                "Participacion mayores",
                "Participacion mujeres",
                "Participacion hombres",
                "Participacion mujeres fertiles",
                "Relacion de dependencia",
                "PIB nominal",
                "PIB per capita anual",
                "Crecimiento anual del PIB",
                "Crecimiento anual del PIB real",
                "Crecimiento anual poblacional",
                "Tasa de natalidad",
                "Tasa de mortalidad",
                "Inflacion anual",
                "Desempleo promedio",
                "Quiebras totales",
                "Stock de capital",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_distribution:
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig = px.histogram(
            household_savings.to_frame(),
            x="savings",
            nbins=20,
            title="Distribucion del ahorro de hogares",
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
            margin=dict(l=10, r=10, t=50, b=10),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig = px.histogram(
            owner_wealth.to_frame(),
            x="wealth",
            nbins=10,
            title="Distribucion de la riqueza de duenos",
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
            margin=dict(l=10, r=10, t=50, b=10),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            history_view_df,
            x="period",
            y_cols=["gini_household_savings", "gini_owner_wealth"],
            title="Evolucion del Gini",
            y_title="Gini",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_firms:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Empresas del anio seleccionado")
    st.caption(
        f"Anio {selected_year}. Aqui puedes ver por que una firma si o no contrato mas: "
        "trabajadores deseados, vacantes, caja, salario ofrecido, productividad y costos."
    )
    sector_options = ["Todos los sectores"] + list(SECTOR_LABELS.values())
    selected_sector = st.selectbox("Sector para analizar", sector_options, index=0)
    firm_detail_view = firm_view.copy()
    if selected_sector != "Todos los sectores":
        firm_detail_view = firm_detail_view[firm_detail_view["Sector"] == selected_sector].copy()
    if selected_sector == "Todos los sectores":
        firm_detail_view = firm_detail_view.sort_values(["Sector", "Participacion de mercado"], ascending=[True, False])
    else:
        firm_detail_view = firm_detail_view.sort_values(["Participacion de mercado", "Costo unitario"], ascending=[False, True])
    chart_view = firm_detail_view.sort_values("Participacion de mercado", ascending=False).head(15).copy()

    st.dataframe(
        firm_detail_view[
            [
                "Firma",
                "Sector",
                "Activa",
                "Trabajadores",
                "Trabajadores deseados",
                "Vacantes",
                "Precio",
                "Salario ofrecido",
                "Efectivo",
                "Capital",
                "Inventario",
                "Inventario objetivo",
                "Costo insumo por unidad",
                "Costo transporte por unidad",
                "Gasto fijo",
                "Cargo de capital",
                "Costo unitario",
                "Ingresos",
                "Costo total",
                "Ganancia",
                "Racha de perdidas",
                "Tolerancia al margen",
                "Preferencia por volumen",
                "Aversion al inventario",
                "Inercia laboral",
                "Agresividad de precio",
                "Conservadurismo de caja",
                "Ambicion de cuota",
                "Tecnologia",
                "Inversion en tecnologia",
                "Ganancia tecnologica",
                "Participacion de mercado",
                "Ventas",
                "Antiguedad",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not firm_detail_view.empty:
        first_row_left, first_row_right = st.columns(2)
        with first_row_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            market_share_fig = px.bar(
                chart_view,
                x="Firma",
                y="Participacion de mercado",
                color="Sector" if selected_sector == "Todos los sectores" else None,
                title="Cuota de mercado por firma",
                hover_data=["Precio", "Costo unitario", "Trabajadores", "Ganancia"],
            )
            market_share_fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                font=dict(color="#0f172a"),
                margin=dict(l=10, r=10, t=50, b=10),
                height=360,
                legend_title_text="",
                yaxis_title="Participacion",
            )
            st.plotly_chart(market_share_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with first_row_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        cost_fig = px.bar(
            chart_view,
            x="Firma",
            y=["Ingresos", "Costo total"],
            barmode="group",
            title="Ingresos y costo total por firma",
        )
        cost_fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
            margin=dict(l=10, r=10, t=50, b=10),
            height=360,
            legend_title_text="",
            yaxis_title="Unidades monetarias",
        )
        st.plotly_chart(cost_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    first_row_left, first_row_right = st.columns(2)
    with first_row_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                firm_summary_view_df,
                x="period",
                y_cols=["labor_force", "average_desired_workers", "labor_supply_gap"],
                title="Oferta laboral y demanda de trabajo",
                y_title="Trabajadores",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with first_row_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                firm_summary_view_df,
                x="period",
                y_cols=["average_revenue", "average_total_cost", "average_profit"],
                title="Ingresos, costo total y ganancia promedio",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    second_row_left, second_row_right = st.columns(2)
    with second_row_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                firm_summary_view_df,
                x="period",
                y_cols=["average_price", "average_wage_offer"],
                title="Precio y salario ofrecido promedio",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with second_row_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                firm_summary_view_df,
                x="period",
                y_cols=[
                    "average_input_cost_per_unit",
                    "average_transport_cost_per_unit",
                    "average_fixed_overhead",
                    "average_capital_charge",
                ],
                title="Desglose de costos promedio",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Composicion del precio de bienes basicos")
    st.caption(
        "Promedio ponderado por produccion de firmas activas de alimentos, vivienda y ropa/higiene "
        "en el anio seleccionado."
    )
    if basic_price_average is not None and basic_price_cost is not None:
        st.caption(
            f"Precio promedio: {money(basic_price_average)} | "
            f"Costo unitario promedio: {money(basic_price_cost)}"
        )
        if basic_price_average < basic_price_cost:
            st.warning(
                "El precio promedio de los bienes basicos esta por debajo del costo unitario promedio. "
                "La porcion de margen se muestra en cero para evitar valores negativos en el pastel."
            )
    if basic_price_composition_df.empty:
        st.info("No hay firmas basicas activas en este anio para descomponer el precio.")
    else:
        price_breakdown_fig = px.pie(
            basic_price_composition_df,
            names="componente",
            values="valor",
            hole=0.38,
            title="Desglose del precio promedio",
            color="componente",
            color_discrete_sequence=["#1D4ED8", "#0891B2", "#0F766E", "#A16207", "#7C3AED", "#B45309"],
        )
        price_breakdown_fig.update_traces(
            textinfo="percent+label",
            sort=False,
            hovertemplate="%{label}: %{value:.2f} unidades<extra></extra>",
        )
        price_breakdown_fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
            margin=dict(l=10, r=10, t=50, b=10),
            height=400,
            legend_title_text="",
        )
        st.plotly_chart(price_breakdown_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            firm_summary_view_df,
            x="period",
            y_cols=["average_technology", "average_technology_gain"],
            title="Tecnologia promedio e incremento tecnologico",
            y_title="Indice",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    sector_chart = firm_view.copy()
    sector_chart["Sector"] = sector_chart["Sector"].str.replace(" / ", "\n", regex=False)
    sector_chart["Trabajadores deseados"] = sector_chart["Trabajadores deseados"].astype(float)
    sector_chart["Vacantes"] = sector_chart["Vacantes"].astype(float)
    bar = px.bar(
        sector_chart,
        x="Sector",
        y=["Trabajadores", "Trabajadores deseados"],
        barmode="group",
        title="Contratacion por sector en el anio seleccionado",
    )
    bar.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0f172a"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        legend_title_text="",
        yaxis_title="Trabajadores",
    )
    st.plotly_chart(bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Desempleo por sector")
    st.caption(
        "Se mide como la brecha entre trabajadores deseados y contratados por sector. "
        "Es una lectura de demanda laboral insatisfecha, no desempleo de personas."
    )
    if sector_labor_market_df.empty:
        st.info("No hay datos de firmas para este anio.")
    else:
        sector_unemployment_fig = px.bar(
            sector_labor_market_df.sort_values("vacancy_rate", ascending=False),
            x="sector_label",
            y="vacancy_rate",
            title="Desempleo sectorial aproximado anual",
            hover_data={
                "sector_name": False,
                "total_firms": True,
                "active_firms": True,
                "total_workers": True,
                "total_desired_workers": True,
                "total_vacancies": True,
                "labor_gap": True,
                "worker_fill_rate": ":.1%",
                "vacancy_rate": ":.1%",
            },
        )
        sector_unemployment_fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#0f172a"),
            margin=dict(l=10, r=10, t=50, b=10),
            height=360,
            legend_title_text="",
            yaxis_title="Tasa",
            xaxis_title="Sector",
        )
        sector_unemployment_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(sector_unemployment_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            firm_summary_view_df,
            x="period",
            y_cols=["average_market_share"],
            title="Participacion promedio de mercado",
            y_title="Participacion",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Historial anual seleccionado")
    st.dataframe(history_view.drop(columns=["Periodo"], errors="ignore"), width="stretch", hide_index=True)
    st.download_button(
        "Descargar CSV anual seleccionado",
        history_view.to_csv(index=False).encode("utf-8"),
        file_name="economy_annual_selected.csv",
        mime="text/csv",
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Historial anual completo")
    st.dataframe(annual_full_view.drop(columns=["Periodo"], errors="ignore"), width="stretch", hide_index=True)
    st.download_button(
        "Descargar CSV anual",
        annual_full_view.to_csv(index=False).encode("utf-8"),
        file_name="economy_annual_history.csv",
        mime="text/csv",
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)
