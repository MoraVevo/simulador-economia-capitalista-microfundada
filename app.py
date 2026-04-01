from __future__ import annotations

from dataclasses import replace

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from economy_simulator import EconomySimulation, ESSENTIAL_SECTOR_KEYS, SECTOR_BY_KEY, SECTOR_SPECS, SimulationConfig
from economy_simulator.domain import SimulationResult
from economy_simulator.reporting import firm_history_frame, firm_period_summary, simulation_frames

INITIAL_HOUSEHOLDS = 10000
MONTH_PRESETS = [12, 24, 60, 120, 240]
PERIODS_PER_YEAR = 12
RUN_MODEL_CACHE_VERSION = 25
CENTRAL_BANK_RULE_OPTIONS = {
    "Emision por crecimiento de bienes": "goods_growth",
    "Fisher": "fisher",
    "Dividendo por productividad": "productivity_dividend",
}
CENTRAL_BANK_RULE_LABELS = {value: label for label, value in CENTRAL_BANK_RULE_OPTIONS.items()}
DEFAULT_POLICY_CONFIG = SimulationConfig()


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
    "fertile_families": "Familias con madre fértil",
    "fertile_families_with_births": "Familias fértiles que dieron a luz",
    "average_family_income": "Ingreso familiar promedio",
    "average_family_basic_basket_cost": "Canasta basica familiar promedio",
    "family_income_to_basket_ratio": "Cobertura ingreso/canasta",
    "families_income_below_basket_share": "Familias con ingreso bajo canasta",
    "fertile_capable_women": "Mujeres fertiles capaces (intervalo cumplido)",
    "fertile_capable_women_low_desire_no_birth": "Mujeres fertiles capaces con deseo insuficiente",
    "fertile_capable_women_with_births": "Mujeres fertiles capaces que dieron a luz",
    "fertile_capable_women_birth_rate": "Tasa de nacimientos entre mujeres fertiles capaces (intervalo cumplido)",
    "fertile_capable_women_low_desire_share": "Participacion sin deseo suficiente",
    "average_worker_savings": "Ahorro promedio de trabajadores",
    "worker_cash_available": "Liquidez disponible de trabajadores",
    "worker_cash_saved": "Liquidez retenida total de trabajadores",
    "worker_voluntary_saved": "Ahorro voluntario de trabajadores",
    "worker_involuntary_retained": "Liquidez retenida por racionamiento",
    "worker_bank_deposits": "Depositos bancarios de trabajadores",
    "worker_credit_outstanding": "Deuda bancaria de trabajadores",
    "worker_consumption_spending": "Gasto de trabajadores",
    "worker_net_financial_position": "Posicion financiera neta trabajadora",
    "worker_savings_rate": "Tasa agregada de ahorro voluntario",
    "worker_involuntary_retention_rate": "Tasa de retencion por racionamiento",
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
    "expected_sales": "Ventas esperadas",
    "expected_sales_change": "Cambio de ventas esperadas",
    "sales_realization": "Ventas / esperado",
    "forecast_error_abs": "Error absoluto de pronostico",
    "forecast_caution": "Cautela de pronostico",
    "forecast_error_belief": "Error de pronostico percibido",
    "market_fragility_belief": "Fragilidad de mercado percibida",
    "demand_elasticity": "Elasticidad de demanda",
    "learning_maturity": "Madurez de aprendizaje",
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
    "average_food_meals_per_person": "Comidas promedio por persona",
    "food_sufficient_share": "Poblacion con alimentacion suficiente",
    "food_subsistence_share": "Poblacion en subsistencia alimentaria",
    "food_acute_hunger_share": "Poblacion con hambre aguda",
    "food_severe_hunger_share": "Poblacion con hambre severa",
    "average_health_fragility": "Fragilidad de salud promedio",
    "labor_cost_per_product": "Costo laboral por unidad producida",
    "essential_basket_equivalents_produced": "Canastas esenciales completas producidas",
    "basic_goods_labor_cost_per_unit": "Costo laboral unitario basico",
    "basic_goods_input_cost_per_unit": "Costo de insumos unitario basico",
    "basic_goods_transport_cost_per_unit": "Costo de transporte unitario basico",
    "basic_goods_fixed_cost_per_unit": "Costo fijo unitario basico",
    "basic_goods_capital_cost_per_unit": "Costo de capital unitario basico",
    "basic_goods_total_unit_cost": "Costo total unitario basico",
    "basic_goods_average_sale_price": "Precio promedio de venta basico",
    "basic_goods_margin_per_unit": "Margen unitario basico",
    "gini_household_savings": "Gini del ahorro",
    "gini_owner_wealth": "Gini de riqueza de duenos",
    "capitalist_bank_deposits": "Depositos bancarios capitalistas",
    "capitalist_vault_cash": "Caja ociosa capitalista",
    "capitalist_firm_cash": "Caja de firmas",
    "capitalist_credit_outstanding": "Deuda bancaria empresarial",
    "capitalist_productive_capital": "Capital productivo capitalista",
    "capitalist_inventory_value": "Inventarios valorizados capitalistas",
    "capitalist_liquid_wealth": "Riqueza liquida capitalista",
    "capitalist_augmented_assets": "Activos ampliados capitalistas",
    "capitalist_net_financial_position": "Posicion financiera neta capitalista",
    "worker_augmented_asset_share": "Participacion ampliada trabajadora",
    "capitalist_augmented_asset_share": "Participacion ampliada capitalista",
    "capitalist_controlled_assets": "Activos controlados por capitalistas",
    "capitalist_asset_share": "Participacion de activos capitalistas",
    "capitalist_liquid_share": "Participacion de dinero liquido capitalista",
    "worker_liquid_share": "Participacion de dinero liquido trabajador",
    "price_to_unit_cost": "Precio / costo unitario",
    "goods_monetary_mass": "Masa monetaria de bienes",
    "government_treasury_cash": "Caja del Estado",
    "government_debt_outstanding": "Deuda publica",
    "government_tax_revenue": "Recaudacion fiscal",
    "government_corporate_tax_revenue": "Impuesto corporativo",
    "government_dividend_tax_revenue": "Impuesto a dividendos",
    "government_wealth_tax_revenue": "Impuesto a riqueza",
    "government_total_spending": "Gasto total del Estado",
    "government_transfers": "Transferencias del Estado",
    "government_unemployment_support": "Seguro de desempleo",
    "government_child_allowance": "Subsidio infantil",
    "government_basic_support": "Subsidio de canasta",
    "government_procurement_spending": "Compras publicas esenciales",
    "government_bond_issuance": "Emision de bonos publicos",
    "government_deficit": "Deficit fiscal",
    "government_deficit_share_gdp": "Deficit fiscal / PIB",
    "government_surplus": "Superavit fiscal",
    "government_tax_burden_gdp": "Carga tributaria / PIB",
    "government_corporate_tax_burden_gdp": "Impuesto corporativo / PIB",
    "government_dividend_tax_burden_gdp": "Impuesto a dividendos / PIB",
    "government_wealth_tax_burden_gdp": "Impuesto a riqueza / PIB",
    "labor_share_gdp": "Participacion salarial del PIB",
    "profit_share_gdp": "Participacion de ganancias en el PIB",
    "investment_share_gdp": "Participacion de inversion en el PIB",
    "capitalist_consumption_share_gdp": "Consumo capitalista / PIB",
    "government_spending_share_gdp": "Gasto estatal / PIB",
    "dividend_share_gdp": "Dividendos / PIB",
    "retained_profit_share_gdp": "Ganancia retenida / PIB",
    "central_bank_money_supply": "Oferta monetaria",
    "central_bank_target_money_supply": "Meta de oferta monetaria",
    "central_bank_policy_rate": "Tasa lider del banco central",
    "central_bank_issuance": "Emision monetaria del periodo",
    "cumulative_central_bank_issuance": "Emision monetaria acumulada",
    "household_credit_creation": "Credito bancario nuevo a trabajadores",
    "firm_credit_creation": "Credito bancario nuevo a empresas",
    "commercial_bank_credit_creation": "Creacion bancaria de dinero",
    "commercial_bank_credit_creation_share_money": "Creacion bancaria / oferta monetaria",
    "average_bank_deposit_rate": "Tasa pasiva bancaria promedio",
    "average_bank_loan_rate": "Tasa activa bancaria promedio",
    "total_bank_deposits": "Depositos totales bancarios",
    "total_bank_reserves": "Reservas bancarias",
    "total_bank_loans_households": "Prestamos bancarios a trabajadores",
    "total_bank_loans_firms": "Prestamos bancarios a empresas",
    "total_bank_loans": "Prestamos bancarios totales",
    "total_bank_bond_holdings": "Bonos publicos en bancos",
    "total_bank_assets": "Activos bancarios totales",
    "total_bank_liabilities": "Pasivos bancarios totales",
    "bank_equity": "Patrimonio neto bancario",
    "bank_capital_ratio": "Capital bancario / activos",
    "bank_asset_liability_ratio": "Activos / pasivos bancarios",
    "bank_reserve_coverage_ratio": "Cobertura de reservas bancarias",
    "bank_liquidity_ratio": "Liquidez bancaria / depositos",
    "bank_loan_to_deposit_ratio": "Prestamos / depositos bancarios",
    "bank_insolvent_share": "Participacion de bancos insolventes",
    "money_velocity": "Velocidad del dinero",
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
    "fertile_women_birth_rate": "Tasa de madres fértiles que dieron a luz",
    "fertile_family_birth_rate": "Tasa de familias fértiles que dieron a luz",
    "fertile_capable_family_birth_rate": "Tasa de nacimientos entre familias fertiles capaces",
    "fertile_capable_family_low_desire_share": "Participacion sin deseo suficiente",
    "child_share": "Participacion ninos",
    "adult_share": "Participacion adultos",
    "senior_share": "Participacion mayores",
    "female_share": "Participacion mujeres",
    "male_share": "Participacion hombres",
    "fertile_women_share": "Participacion mujeres fertiles",
    "dependency_ratio": "Relacion de dependencia",
    "avg_unemployment_rate": "Desempleo promedio",
    "total_bankruptcies": "Quiebras totales",
    "bankruptcies": "Quiebras",
    "year": "Anio",
    "period": "Mes",
    "period_in_year": "Periodo del anio",
    "end_price_index": "Indice de precios final",
    "capital_growth_yoy": "Crecimiento anual del capital",
    "inventory_growth_yoy": "Crecimiento anual de inventarios",
    "total_wages": "Salarios pagados",
    "total_sales_units": "Unidades vendidas",
    "total_sales_revenue": "Ingresos por ventas",
    "total_production_units": "Produccion total",
    "period_investment_spending": "Gasto de inversion",
    "total_profit": "Ganancia total",
    "total_liquid_money": "Dinero liquido total",
    "business_cost_recycled": "Costos reciclados",
    "business_cost_to_firms": "Costos enviados a firmas",
    "business_cost_to_households": "Costos enviados a hogares",
    "business_cost_to_owners": "Costos enviados a duenos",
    "inheritance_transfers": "Transferencias por herencia",
    "bankruptcy_cash_recoveries": "Recuperaciones de caja por quiebra",
}

HIDDEN_HISTORY_COLUMNS = [
    "average_family_resources",
    "family_resources_to_basket_ratio",
    "families_resources_below_basket_share",
]

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
    "forecast_caution": "Cautela de pronostico",
    "forecast_error_belief": "Error de pronostico percibido",
    "market_fragility_belief": "Fragilidad de mercado percibida",
    "technology": "Tecnologia",
    "technology_investment": "Inversion en tecnologia",
    "technology_gain": "Ganancia tecnologica",
    "expected_sales": "Ventas esperadas",
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
    "Manufacturing / industrial goods": "Manufactura / bienes industriales",
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

def _with_policy_values(config: SimulationConfig, policy_values: dict[str, float | str]) -> SimulationConfig:
    return replace(config, **policy_values)


def _build_simulation_result(simulation: EconomySimulation) -> SimulationResult:
    return SimulationResult(
        config=simulation.config,
        history=simulation.history,
        firm_history=simulation.firm_history,
        households=simulation.households,
        entrepreneurs=simulation.entrepreneurs,
        firms=simulation.firms,
        central_bank=simulation.central_bank,
        banks=simulation.banks,
        government=simulation.government,
    )


def default_policy_values() -> dict[str, float | str]:
    return {
        "central_bank_rule": DEFAULT_POLICY_CONFIG.central_bank_rule,
        "central_bank_goods_growth_pass_through": DEFAULT_POLICY_CONFIG.central_bank_goods_growth_pass_through,
        "central_bank_target_velocity": DEFAULT_POLICY_CONFIG.central_bank_target_velocity,
        "central_bank_target_annual_inflation": DEFAULT_POLICY_CONFIG.central_bank_target_annual_inflation,
        "central_bank_policy_rate_base": DEFAULT_POLICY_CONFIG.central_bank_policy_rate_base,
        "central_bank_policy_rate_floor": DEFAULT_POLICY_CONFIG.central_bank_policy_rate_floor,
        "central_bank_policy_rate_ceiling": DEFAULT_POLICY_CONFIG.central_bank_policy_rate_ceiling,
        "central_bank_productivity_dividend_share": DEFAULT_POLICY_CONFIG.central_bank_productivity_dividend_share,
        "reserve_ratio": DEFAULT_POLICY_CONFIG.reserve_ratio,
        "bank_bond_allocation_share": DEFAULT_POLICY_CONFIG.bank_bond_allocation_share,
        "bank_deposit_rate_share": DEFAULT_POLICY_CONFIG.bank_deposit_rate_share,
        "government_corporate_tax_rate_low": DEFAULT_POLICY_CONFIG.government_corporate_tax_rate_low,
        "government_corporate_tax_rate_mid": DEFAULT_POLICY_CONFIG.government_corporate_tax_rate_mid,
        "government_corporate_tax_rate_high": DEFAULT_POLICY_CONFIG.government_corporate_tax_rate_high,
        "government_dividend_tax_rate_low": DEFAULT_POLICY_CONFIG.government_dividend_tax_rate_low,
        "government_dividend_tax_rate_mid": DEFAULT_POLICY_CONFIG.government_dividend_tax_rate_mid,
        "government_dividend_tax_rate_high": DEFAULT_POLICY_CONFIG.government_dividend_tax_rate_high,
        "government_wealth_tax_rate": DEFAULT_POLICY_CONFIG.government_wealth_tax_rate,
        "government_wealth_tax_threshold_multiple": DEFAULT_POLICY_CONFIG.government_wealth_tax_threshold_multiple,
        "government_unemployment_benefit_share": DEFAULT_POLICY_CONFIG.government_unemployment_benefit_share,
        "government_child_allowance_share": DEFAULT_POLICY_CONFIG.government_child_allowance_share,
        "government_basic_support_gap_share": DEFAULT_POLICY_CONFIG.government_basic_support_gap_share,
        "government_procurement_gap_share": DEFAULT_POLICY_CONFIG.government_procurement_gap_share,
        "government_procurement_price_sensitivity": DEFAULT_POLICY_CONFIG.government_procurement_price_sensitivity,
        "government_spending_scale": DEFAULT_POLICY_CONFIG.government_spending_scale,
        "government_spending_efficiency": DEFAULT_POLICY_CONFIG.government_spending_efficiency,
    }


def render_policy_controls(
    prefix: str,
    defaults: dict[str, float | str],
    *,
    title: str,
) -> dict[str, float | str]:
    st.markdown(f"**{title}**")
    central_bank_rule_label = st.selectbox(
        "Regla del banco central",
        options=list(CENTRAL_BANK_RULE_OPTIONS),
        index=list(CENTRAL_BANK_RULE_OPTIONS.values()).index(
            str(defaults.get("central_bank_rule", DEFAULT_POLICY_CONFIG.central_bank_rule))
        ),
        key=f"{prefix}_central_bank_rule",
    )
    values: dict[str, float | str] = {
        "central_bank_rule": CENTRAL_BANK_RULE_OPTIONS[central_bank_rule_label],
    }
    with st.expander("Banco central y bancos", expanded=False):
        values["central_bank_goods_growth_pass_through"] = st.slider(
            "Intensidad de emision monetaria",
            min_value=0.0,
            max_value=2.0,
            value=float(defaults["central_bank_goods_growth_pass_through"]),
            step=0.05,
            key=f"{prefix}_goods_pass",
        )
        values["central_bank_target_velocity"] = st.slider(
            "Velocidad objetivo del dinero",
            min_value=0.05,
            max_value=1.00,
            value=float(defaults["central_bank_target_velocity"]),
            step=0.01,
            key=f"{prefix}_velocity",
        )
        values["central_bank_target_annual_inflation"] = st.slider(
            "Meta anual de inflacion",
            min_value=0.00,
            max_value=0.20,
            value=float(defaults["central_bank_target_annual_inflation"]),
            step=0.005,
            key=f"{prefix}_inflation_target",
        )
        values["central_bank_policy_rate_base"] = st.slider(
            "Tasa lider base",
            min_value=0.00,
            max_value=0.20,
            value=float(defaults["central_bank_policy_rate_base"]),
            step=0.005,
            key=f"{prefix}_policy_base",
        )
        values["central_bank_policy_rate_floor"] = st.slider(
            "Piso de tasa lider",
            min_value=0.00,
            max_value=0.12,
            value=float(defaults["central_bank_policy_rate_floor"]),
            step=0.005,
            key=f"{prefix}_policy_floor",
        )
        values["central_bank_policy_rate_ceiling"] = st.slider(
            "Techo de tasa lider",
            min_value=0.02,
            max_value=0.30,
            value=float(defaults["central_bank_policy_rate_ceiling"]),
            step=0.005,
            key=f"{prefix}_policy_ceiling",
        )
        values["central_bank_productivity_dividend_share"] = st.slider(
            "Pass-through del dividendo por productividad",
            min_value=0.0,
            max_value=2.0,
            value=float(defaults["central_bank_productivity_dividend_share"]),
            step=0.05,
            key=f"{prefix}_prod_dividend",
        )
        values["reserve_ratio"] = st.slider(
            "Encaje bancario",
            min_value=0.00,
            max_value=0.50,
            value=float(defaults["reserve_ratio"]),
            step=0.01,
            key=f"{prefix}_reserve_ratio",
        )
        values["bank_bond_allocation_share"] = st.slider(
            "Preferencia bancaria por bonos publicos",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["bank_bond_allocation_share"]),
            step=0.05,
            key=f"{prefix}_bond_alloc",
        )
        values["bank_deposit_rate_share"] = st.slider(
            "Traslado de tasa lider a tasa pasiva",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["bank_deposit_rate_share"]),
            step=0.05,
            key=f"{prefix}_deposit_rate_share",
        )
    with st.expander("Estado e impuestos", expanded=False):
        values["government_corporate_tax_rate_low"] = st.slider(
            "ISR corporativo bajo",
            min_value=0.00,
            max_value=0.40,
            value=float(defaults["government_corporate_tax_rate_low"]),
            step=0.01,
            key=f"{prefix}_corp_low",
        )
        values["government_corporate_tax_rate_mid"] = st.slider(
            "ISR corporativo medio",
            min_value=0.00,
            max_value=0.50,
            value=float(defaults["government_corporate_tax_rate_mid"]),
            step=0.01,
            key=f"{prefix}_corp_mid",
        )
        values["government_corporate_tax_rate_high"] = st.slider(
            "ISR corporativo alto",
            min_value=0.00,
            max_value=0.60,
            value=float(defaults["government_corporate_tax_rate_high"]),
            step=0.01,
            key=f"{prefix}_corp_high",
        )
        values["government_dividend_tax_rate_low"] = st.slider(
            "ISR a dividendos bajo",
            min_value=0.00,
            max_value=0.30,
            value=float(defaults["government_dividend_tax_rate_low"]),
            step=0.01,
            key=f"{prefix}_div_low",
        )
        values["government_dividend_tax_rate_mid"] = st.slider(
            "ISR a dividendos medio",
            min_value=0.00,
            max_value=0.40,
            value=float(defaults["government_dividend_tax_rate_mid"]),
            step=0.01,
            key=f"{prefix}_div_mid",
        )
        values["government_dividend_tax_rate_high"] = st.slider(
            "ISR a dividendos alto",
            min_value=0.00,
            max_value=0.50,
            value=float(defaults["government_dividend_tax_rate_high"]),
            step=0.01,
            key=f"{prefix}_div_high",
        )
        values["government_wealth_tax_rate"] = st.slider(
            "IUSI / impuesto patrimonial",
            min_value=0.000,
            max_value=0.050,
            value=float(defaults["government_wealth_tax_rate"]),
            step=0.001,
            key=f"{prefix}_wealth_tax",
        )
        values["government_wealth_tax_threshold_multiple"] = st.slider(
            "Umbral patrimonial (multiplo de salario de vida)",
            min_value=1.0,
            max_value=80.0,
            value=float(defaults["government_wealth_tax_threshold_multiple"]),
            step=1.0,
            key=f"{prefix}_wealth_threshold",
        )
    with st.expander("Gasto del Estado", expanded=False):
        values["government_spending_scale"] = st.slider(
            "Escala total del gasto del Estado",
            min_value=0.0,
            max_value=3.0,
            value=float(defaults["government_spending_scale"]),
            step=0.05,
            key=f"{prefix}_spend_scale",
        )
        values["government_spending_efficiency"] = st.slider(
            "Calidad del gasto del Estado",
            min_value=0.10,
            max_value=1.00,
            value=float(defaults["government_spending_efficiency"]),
            step=0.05,
            key=f"{prefix}_spend_eff",
        )
        values["government_unemployment_benefit_share"] = st.slider(
            "Seguro de desempleo",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["government_unemployment_benefit_share"]),
            step=0.05,
            key=f"{prefix}_unemp_support",
        )
        values["government_child_allowance_share"] = st.slider(
            "Subsidio infantil",
            min_value=0.0,
            max_value=0.8,
            value=float(defaults["government_child_allowance_share"]),
            step=0.02,
            key=f"{prefix}_child_allowance",
        )
        values["government_basic_support_gap_share"] = st.slider(
            "Cobertura estatal de la brecha de canasta",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["government_basic_support_gap_share"]),
            step=0.05,
            key=f"{prefix}_basic_support",
        )
        values["government_procurement_gap_share"] = st.slider(
            "Compras publicas sobre brecha esencial",
            min_value=0.0,
            max_value=1.5,
            value=float(defaults["government_procurement_gap_share"]),
            step=0.05,
            key=f"{prefix}_proc_gap",
        )
        values["government_procurement_price_sensitivity"] = st.slider(
            "Sensibilidad del Estado al precio en compras",
            min_value=0.1,
            max_value=2.0,
            value=float(defaults["government_procurement_price_sensitivity"]),
            step=0.05,
            key=f"{prefix}_proc_price",
        )
    values["central_bank_policy_rate_ceiling"] = max(
        float(values["central_bank_policy_rate_ceiling"]),
        float(values["central_bank_policy_rate_floor"]),
    )
    values["central_bank_policy_rate_base"] = min(
        max(
            float(values["central_bank_policy_rate_base"]),
            float(values["central_bank_policy_rate_floor"]),
        ),
        float(values["central_bank_policy_rate_ceiling"]),
    )
    return values


@st.cache_data(show_spinner=False)
def run_model(
    months: int,
    seed: int,
    firms_per_sector: int,
    base_policy_values: dict[str, float | str],
    policy_change_period: int | None,
    shock_policy_values: dict[str, float | str] | None,
    cache_version: int = RUN_MODEL_CACHE_VERSION,
):
    config = SimulationConfig(
        periods=months,
        households=INITIAL_HOUSEHOLDS,
        seed=seed,
        periods_per_year=PERIODS_PER_YEAR,
        firms_per_sector=firms_per_sector,
    )
    config = _with_policy_values(config, base_policy_values)
    simulation = EconomySimulation(config)
    if policy_change_period is None or not shock_policy_values:
        result = simulation.run()
    else:
        for month in range(1, months + 1):
            if month == policy_change_period:
                simulation.config = _with_policy_values(simulation.config, shock_policy_values)
            simulation.step()
        result = _build_simulation_result(simulation)
    history_df, _ = simulation_frames(result)
    return result, history_df


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


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


def make_unique_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    counts: dict[str, int] = {}
    renamed_columns: list[str] = []
    for column in frame.columns:
        seen = counts.get(column, 0)
        if seen == 0:
            renamed_columns.append(column)
        else:
            renamed_columns.append(f"{column} ({seen + 1})")
        counts[column] = seen + 1
    result = frame.copy()
    result.columns = renamed_columns
    return result


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


def _money_flow_link_color(value: float, max_value: float) -> str:
    if max_value <= 0.0:
        return "rgba(59,130,246,0.35)"
    intensity = max(0.0, min(1.0, value / max_value))
    red = int(59 + (239 - 59) * intensity)
    green = int(130 + (68 - 130) * intensity)
    blue = int(246 + (68 - 246) * intensity)
    alpha = 0.25 + 0.60 * intensity
    return f"rgba({red},{green},{blue},{alpha:.3f})"


def make_institution_flow_chart(period_row: pd.Series) -> go.Figure:
    nodes = [
        "Banco central",
        "Bancos comerciales",
        "Estado",
        "Casa A",
        "Casa B",
        "Empresa A",
        "Empresa B",
    ]
    node_index = {label: index for index, label in enumerate(nodes)}
    node_x = [0.03, 0.22, 0.42, 0.60, 0.60, 0.84, 0.84]
    node_y = [0.44, 0.44, 0.12, 0.34, 0.70, 0.24, 0.60]

    worker_spending = max(
        0.0,
        float(period_row.get("worker_consumption_spending", 0.0)),
    )
    wage_flow = max(0.0, float(period_row.get("total_wages", 0.0)))
    transfers = max(0.0, float(period_row.get("government_transfers", 0.0)))
    procurement = max(0.0, float(period_row.get("government_procurement_spending", 0.0)))
    tax_flow = max(0.0, float(period_row.get("government_tax_revenue", 0.0)))
    bonds = max(0.0, float(period_row.get("government_bond_issuance", 0.0)))
    central_bank_issue = max(0.0, float(period_row.get("central_bank_issuance", 0.0)))
    household_credit = max(0.0, float(period_row.get("household_credit_creation", 0.0)))
    firm_credit = max(0.0, float(period_row.get("firm_credit_creation", 0.0)))
    worker_savings = max(0.0, float(period_row.get("worker_voluntary_saved", 0.0)))

    links: list[tuple[str, str, float]] = [
        ("Banco central", "Bancos comerciales", central_bank_issue),
        ("Bancos comerciales", "Estado", bonds),
        ("Bancos comerciales", "Casa A", household_credit * 0.5),
        ("Bancos comerciales", "Casa B", household_credit * 0.5),
        ("Bancos comerciales", "Empresa A", firm_credit * 0.5),
        ("Bancos comerciales", "Empresa B", firm_credit * 0.5),
        ("Estado", "Casa A", transfers * 0.5),
        ("Estado", "Casa B", transfers * 0.5),
        ("Estado", "Empresa A", procurement * 0.5),
        ("Estado", "Empresa B", procurement * 0.5),
        ("Casa A", "Empresa A", worker_spending * 0.25),
        ("Casa A", "Empresa B", worker_spending * 0.25),
        ("Casa B", "Empresa A", worker_spending * 0.25),
        ("Casa B", "Empresa B", worker_spending * 0.25),
        ("Empresa A", "Casa A", wage_flow * 0.25),
        ("Empresa A", "Casa B", wage_flow * 0.25),
        ("Empresa B", "Casa A", wage_flow * 0.25),
        ("Empresa B", "Casa B", wage_flow * 0.25),
        ("Casa A", "Bancos comerciales", worker_savings * 0.5),
        ("Casa B", "Bancos comerciales", worker_savings * 0.5),
        ("Empresa A", "Estado", tax_flow * 0.5),
        ("Empresa B", "Estado", tax_flow * 0.5),
    ]
    links = [link for link in links if link[2] > 1e-9]
    max_flow = max((value for _, _, value in links), default=0.0)
    visible_threshold = max_flow * 0.035
    prominent_links = [link for link in links if link[2] >= visible_threshold]
    if len(prominent_links) >= 8:
        links = prominent_links

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    label=nodes,
                    x=node_x,
                    y=node_y,
                    pad=26,
                    thickness=26,
                    color=[
                        "rgba(15,23,42,0.90)",
                        "rgba(30,64,175,0.90)",
                        "rgba(100,116,139,0.90)",
                        "rgba(71,85,105,0.90)",
                        "rgba(100,116,139,0.90)",
                        "rgba(22,163,74,0.90)",
                        "rgba(21,128,61,0.90)",
                    ],
                    line=dict(color="rgba(15,23,42,0.28)", width=1.0),
                ),
                link=dict(
                    source=[node_index[source] for source, _, _ in links],
                    target=[node_index[target] for _, target, _ in links],
                    value=[value for _, _, value in links],
                    color=[_money_flow_link_color(value, max_flow) for _, _, value in links],
                    hovertemplate="%{source.label} -> %{target.label}<br>%{value:,.2f}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        title="Flujo de dinero entre instituciones del mes seleccionado",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0f172a", size=13),
        margin=dict(l=24, r=24, t=70, b=20),
        height=620,
        annotations=[
            dict(
                x=0.01,
                y=1.07,
                xref="paper",
                yref="paper",
                text="Rojo = flujo alto, azul = flujo bajo. Se ocultan flujos muy pequenos para que se lea mejor.",
                showarrow=False,
                font=dict(size=12, color="#475569"),
                align="left",
            )
        ],
    )
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
    labor_bill_summary = (
        sector_history.assign(labor_bill=sector_history["wage_offer"] * sector_history["workers"])
        .groupby(time_col, as_index=False)
        .agg(total_labor_bill=("labor_bill", "sum"))
    )
    summary = summary.merge(labor_bill_summary, on=time_col, how="left")
    summary["total_labor_bill"] = summary["total_labor_bill"].fillna(0.0)
    summary = summary.merge(population_frame[[time_col, population_column]], on=time_col, how="left")
    summary["production_per_worker"] = summary["total_production"] / summary["total_workers"].replace(0, pd.NA)
    summary["production_per_person"] = summary["total_production"] / summary[population_column].replace(0, pd.NA)
    summary["labor_cost_per_product"] = summary["total_labor_bill"] / summary["total_production"].replace(0, pd.NA)
    return summary


def build_essential_basket_survival_data(
    firm_history_frame: pd.DataFrame,
    population_frame: pd.DataFrame,
    time_col: str = "period",
    population_column: str = "population",
) -> pd.DataFrame:
    result = population_frame[[time_col, population_column]].copy()
    essential_history = firm_history_frame[firm_history_frame["sector"].isin(ESSENTIAL_SECTOR_KEYS)].copy()
    if essential_history.empty:
        result["essential_basket_equivalents_produced"] = 0.0
        return result

    sector_production = (
        essential_history.groupby([time_col, "sector"], as_index=False)
        .agg(total_production=("production", "sum"))
        .sort_values([time_col, "sector"])
    )
    sector_pivot = sector_production.pivot(index=time_col, columns="sector", values="total_production").reset_index()
    sector_pivot.columns.name = None
    result = result.merge(sector_pivot, on=time_col, how="left")

    total_essential_need = sum(
        spec.essential_need for spec in SECTOR_SPECS if spec.key in ESSENTIAL_SECTOR_KEYS
    )
    basket_equivalent_cols: list[str] = []
    for sector_key in ESSENTIAL_SECTOR_KEYS:
        if sector_key not in result.columns:
            result[sector_key] = 0.0
        result[sector_key] = result[sector_key].fillna(0.0)
        sector_need_share = SECTOR_BY_KEY[sector_key].essential_need / max(1e-9, total_essential_need)
        basket_equivalent_col = f"{sector_key}_basket_equivalents"
        result[basket_equivalent_col] = result[sector_key] / max(1e-9, sector_need_share)
        basket_equivalent_cols.append(basket_equivalent_col)

    result["essential_basket_equivalents_produced"] = result[basket_equivalent_cols].min(axis=1)
    return result[[time_col, population_column, "essential_basket_equivalents_produced"]]


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


def build_basic_goods_price_time_series(
    firm_history_frame: pd.DataFrame,
    time_col: str = "period",
) -> pd.DataFrame:
    essential_history = firm_history_frame[firm_history_frame["sector"].isin(ESSENTIAL_SECTOR_KEYS)].copy()
    if essential_history.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int]] = []
    for period_value, period_rows in essential_history.groupby(time_col):
        producing_rows = period_rows[period_rows["production"] > 0.0].copy()
        if producing_rows.empty:
            continue

        total_production = float(producing_rows["production"].sum())
        if total_production <= 0.0:
            continue

        labor_total = float((producing_rows["wage_offer"] * producing_rows["workers"]).clip(lower=0.0).sum())
        input_total = float((producing_rows["input_cost_per_unit"] * producing_rows["production"]).clip(lower=0.0).sum())
        transport_total = float(
            (producing_rows["transport_cost_per_unit"] * producing_rows["production"]).clip(lower=0.0).sum()
        )
        fixed_total = float(producing_rows["fixed_overhead"].clip(lower=0.0).sum())
        capital_total = float(producing_rows["capital_charge"].clip(lower=0.0).sum())
        revenue_total = float((producing_rows["price"] * producing_rows["production"]).clip(lower=0.0).sum())

        labor_cost_per_unit = labor_total / total_production
        input_cost_per_unit = input_total / total_production
        transport_cost_per_unit = transport_total / total_production
        fixed_cost_per_unit = fixed_total / total_production
        capital_cost_per_unit = capital_total / total_production
        total_unit_cost = (
            labor_cost_per_unit
            + input_cost_per_unit
            + transport_cost_per_unit
            + fixed_cost_per_unit
            + capital_cost_per_unit
        )
        average_sale_price = revenue_total / total_production

        row: dict[str, float | int] = {
            time_col: int(period_value),
            "basic_goods_labor_cost_per_unit": labor_cost_per_unit,
            "basic_goods_input_cost_per_unit": input_cost_per_unit,
            "basic_goods_transport_cost_per_unit": transport_cost_per_unit,
            "basic_goods_fixed_cost_per_unit": fixed_cost_per_unit,
            "basic_goods_capital_cost_per_unit": capital_cost_per_unit,
            "basic_goods_total_unit_cost": total_unit_cost,
            "basic_goods_average_sale_price": average_sale_price,
            "basic_goods_margin_per_unit": average_sale_price - total_unit_cost,
        }
        if "year" in period_rows.columns:
            row["year"] = int(period_rows["year"].iloc[-1])
        if "period_in_year" in period_rows.columns:
            row["period_in_year"] = int(period_rows["period_in_year"].iloc[-1])
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(time_col).reset_index(drop=True)


def describe_firm_month(row: pd.Series) -> str:
    notes = []
    training_phase_value = row.get("training_phase", False)
    training_phase = bool(training_phase_value) if pd.notna(training_phase_value) else False
    if training_phase:
        notes.append("seguia en fase de aprendizaje del mercado")

    price_change = row.get("price_change", 0.0)
    if pd.notna(price_change) and price_change > 0.01:
        notes.append(f"Subio precio {price_change:.1%}")
    elif pd.notna(price_change) and price_change < -0.01:
        notes.append(f"Bajo precio {abs(price_change):.1%}")
    else:
        notes.append("Mantuvo precio casi estable")

    sales_gap = row.get("sales_gap", 0.0)
    if pd.notna(sales_gap) and sales_gap < -0.20:
        notes.append("vendio muy por debajo de lo que esperaba")
    elif pd.notna(sales_gap) and sales_gap > 0.10:
        notes.append("vendio por encima de lo esperado")
    expected_sales_change = row.get("expected_sales_change", 0.0)
    if pd.notna(expected_sales_change) and expected_sales_change > 0.20:
        notes.append("elevo fuerte su expectativa de ventas")

    family_ratio = row.get("family_income_to_basket_ratio", pd.NA)
    low_resources = row.get("families_income_below_basket_share", pd.NA)
    essential_cover = row.get("essential_fulfillment_rate", pd.NA)
    if pd.notna(family_ratio) and family_ratio < 1.0:
        notes.append("los hogares promedio no cubrian la canasta")
    if pd.notna(low_resources) and low_resources > 0.35:
        notes.append("muchas familias estaban bajo la canasta")
    if pd.notna(essential_cover) and essential_cover < 0.95:
        notes.append("la cobertura de bienes basicos estaba fragil")
    worker_rationing = row.get("worker_involuntary_retention_rate", pd.NA)
    if pd.notna(worker_rationing) and worker_rationing > 0.15:
        notes.append("parte del efectivo obrero quedo retenido por racionamiento")
    severe_hunger = row.get("food_severe_hunger_share", pd.NA)
    if pd.notna(severe_hunger) and severe_hunger > 0.05:
        notes.append("ya habia hambre severa en parte de la poblacion")

    inventory_ratio = row.get("inventory_ratio", 1.0)
    if pd.notna(inventory_ratio) and inventory_ratio > 1.20:
        notes.append("tenia inventario alto")
    cash_cover_months = row.get("cash_cover_months", 0.0)
    if pd.notna(cash_cover_months) and cash_cover_months < 1.0:
        notes.append("la caja cubria menos de un mes de compromisos")

    profit = row.get("profit", 0.0)
    if pd.notna(profit) and profit < 0.0:
        notes.append("cerro el mes con perdida")
    if (
        pd.notna(price_change)
        and price_change > 0.02
        and pd.notna(family_ratio)
        and family_ratio < 1.0
        and pd.notna(sales_gap)
        and sales_gap < -0.10
    ):
        notes.append("encarecio pese a demanda fragil y luego sobreestimo ventas")
    active = row.get("active", True)
    active_next = row.get("active_next", True)
    if pd.notna(active) and bool(active) and pd.notna(active_next) and not bool(active_next):
        notes.append("salio del mercado al mes siguiente")
    elif pd.notna(active) and not bool(active):
        notes.append("ya estaba inactiva")

    return ". ".join(notes) + "."


def build_firm_diagnostic_data(
    firm_history_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
    firm_id: int,
    learning_warmup_periods: int,
) -> pd.DataFrame:
    firm_data = firm_history_frame[firm_history_frame["firm_id"] == firm_id].copy()
    if firm_data.empty:
        return firm_data

    context_columns = [
        "period",
        "population",
        "unemployment_rate",
        "average_family_basic_basket_cost",
        "average_family_income",
        "family_income_to_basket_ratio",
        "families_income_below_basket_share",
        "essential_fulfillment_rate",
        "demand_fulfillment_rate",
        "average_food_meals_per_person",
        "food_sufficient_share",
        "food_severe_hunger_share",
        "average_health_fragility",
        "worker_savings_rate",
        "worker_involuntary_retention_rate",
        "worker_liquid_share",
        "capitalist_liquid_share",
        "central_bank_issuance",
        "price_index",
        "total_liquid_money",
    ]
    firm_data = firm_data.sort_values("period").reset_index(drop=True)
    available_context_columns = [column for column in context_columns if column in history_frame.columns]
    firm_data = firm_data.merge(history_frame[available_context_columns], on="period", how="left")
    if "expected_sales" not in firm_data.columns:
        firm_data["expected_sales"] = firm_data.get("sales", pd.Series([0.0] * len(firm_data))).copy()
    if "forecast_caution" not in firm_data.columns:
        firm_data["forecast_caution"] = 1.0
    if "forecast_error_belief" not in firm_data.columns:
        firm_data["forecast_error_belief"] = 0.15
    if "market_fragility_belief" not in firm_data.columns:
        firm_data["market_fragility_belief"] = 0.0
    if "demand_elasticity" not in firm_data.columns:
        firm_data["demand_elasticity"] = pd.NA
    if "learning_maturity" not in firm_data.columns:
        warmup_scale = max(1, learning_warmup_periods)
        firm_data["learning_maturity"] = ((firm_data["period"] - 1).clip(lower=0) / warmup_scale).clip(0.0, 1.0)
    firm_data["price_change"] = firm_data["price"].pct_change()
    firm_data["wage_change"] = firm_data["wage_offer"].pct_change()
    firm_data["expected_sales_change"] = firm_data["expected_sales"].pct_change()
    firm_data["sales_realization"] = firm_data["sales"] / firm_data["expected_sales"].replace(0, pd.NA)
    firm_data["sales_gap"] = firm_data["sales_realization"] - 1.0
    firm_data["forecast_error_abs"] = (firm_data["sales_realization"] - 1.0).abs()
    firm_data["inventory_ratio"] = firm_data["inventory"] / firm_data["target_inventory"].replace(0, pd.NA)
    firm_data["price_to_unit_cost"] = firm_data["price"] / firm_data["unit_cost"].replace(0, pd.NA)
    firm_data["training_phase"] = firm_data["period"] <= max(0, learning_warmup_periods)
    firm_data["monthly_commitments"] = (
        firm_data["wage_offer"] * firm_data["workers"]
        + firm_data["fixed_overhead"]
        + firm_data["capital_charge"]
    )
    firm_data["cash_cover_months"] = firm_data["cash"] / firm_data["monthly_commitments"].replace(0, pd.NA)
    active_next = firm_data["active"].shift(-1)
    firm_data["active_next"] = active_next.where(active_next.notna(), firm_data["active"]).astype(bool)
    firm_data["decision_note"] = firm_data.apply(describe_firm_month, axis=1)
    return firm_data


st.title("Simulador economico")
st.caption(
    "Las cifras nominales se muestran en las unidades monetarias internas del simulador. "
    "Cada periodo equivale a un mes y toda la interfaz se muestra en escala mensual."
)

with st.sidebar:
    st.header("Escenario")
    preset_months = st.selectbox("Meses predefinidos", options=MONTH_PRESETS, index=3)
    use_custom_months = st.checkbox("Escribir meses manualmente", value=False)
    if use_custom_months:
        duration_months = st.number_input("Meses totales", min_value=1, max_value=2400, value=240, step=1)
    else:
        duration_months = preset_months
    seed = st.number_input("Semilla", min_value=1, max_value=999999, value=7, step=1)
    firms_per_sector = st.slider("Firmas por sector", min_value=1, max_value=80, value=40, step=1)
    base_policy_values = render_policy_controls(
        "base",
        default_policy_values(),
        title="Politica inicial",
    )
    policy_change_enabled = st.checkbox("Activar cambio de politica por periodo", value=False)
    policy_change_period: int | None = None
    shock_policy_values: dict[str, float | str] | None = None
    if policy_change_enabled:
        policy_change_period = int(
            st.slider(
                "Aplicar nuevo paquete desde el mes",
                min_value=1,
                max_value=int(duration_months),
                value=max(2, min(int(duration_months), int(duration_months) // 2 or 1)),
                step=1,
            )
        )
        shock_policy_values = render_policy_controls(
            "shock",
            base_policy_values,
            title="Politica desde el cambio",
        )
    total_months = int(duration_months)
    st.caption(
        f"Duracion objetivo: {total_months} meses. "
        "Cada periodo del simulador equivale a 1 mes."
    )
    st.caption(
        "Puedes correr una politica inicial y, si activas el cambio de politica, aplicar un segundo "
        "paquete de variables desde el mes que elijas."
    )
    st.markdown(f"La poblacion inicial queda fija en {INITIAL_HOUSEHOLDS} hogares/agentes para la linea base.")
    st.markdown("La simulacion es determinista con los mismos parametros y la misma semilla.")
    if "view_period" not in st.session_state:
        st.session_state.view_period = total_months
    st.session_state.view_period = min(st.session_state.view_period, total_months)
    nav_prev, nav_center, nav_next = st.columns(3)
    if nav_prev.button("Mes anterior", use_container_width=True, disabled=st.session_state.view_period <= 1):
        st.session_state.view_period -= 1
        st.rerun()
    if nav_center.button("Volver al final", use_container_width=True):
        st.session_state.view_period = total_months
        st.rerun()
    if nav_next.button("Mes siguiente", use_container_width=True, disabled=st.session_state.view_period >= total_months):
        st.session_state.view_period += 1
        st.rerun()


result, history_df = run_model(
    total_months,
    seed,
    firms_per_sector,
    base_policy_values,
    policy_change_period,
    shock_policy_values,
    cache_version=RUN_MODEL_CACHE_VERSION,
)
firm_history_df = firm_history_frame(result)
household_savings, owner_wealth = build_distribution_data(result)
selected_period = min(st.session_state.view_period, total_months)
history_view_df = history_df[history_df["period"] <= selected_period].copy()
firm_history_view_df = firm_history_df[firm_history_df["period"] <= selected_period].copy()
firm_period_view_df = firm_history_view_df[firm_history_view_df["period"] == selected_period].copy()
firm_period_view_df = firm_period_view_df.sort_values(["sector", "price"]).copy()
firm_summary_view_df = firm_period_summary(firm_history_view_df)
food_productivity_view_df = build_sector_productivity_data(firm_history_view_df, "food", time_col="period")
basic_goods_view_df = build_sector_group_productivity_data(
    firm_history_view_df,
    history_view_df,
    ESSENTIAL_SECTOR_KEYS,
    time_col="period",
    population_column="population",
)
essential_survival_view_df = build_essential_basket_survival_data(
    firm_history_view_df,
    history_view_df,
    time_col="period",
    population_column="population",
)
basic_goods_price_view_df = build_basic_goods_price_time_series(
    firm_history_view_df,
    time_col="period",
)
sector_labor_market_df = build_sector_labor_market_data(firm_period_view_df)
firm_summary_view_df = firm_summary_view_df.merge(
    history_view_df[["period", "labor_force", "unemployment_rate"]],
    on="period",
    how="left",
)
firm_summary_view_df["labor_supply_gap"] = firm_summary_view_df["labor_force"] - firm_summary_view_df["total_desired_workers"]
history_view = make_unique_columns(
    history_view_df.drop(columns=HIDDEN_HISTORY_COLUMNS, errors="ignore").rename(columns=COLUMN_LABELS)
)
history_full_view = make_unique_columns(
    history_df.drop(columns=HIDDEN_HISTORY_COLUMNS, errors="ignore").rename(columns=COLUMN_LABELS)
)
firm_view = firm_period_view_df.rename(columns=FIRM_LABELS)

latest_period = history_view_df.iloc[-1] if not history_view_df.empty else None
monthly_gdp_delta = fmt_delta(latest_period["gdp_growth"]) if latest_period is not None else None
monthly_population_delta = fmt_delta(latest_period["population_growth"]) if latest_period is not None else None
monthly_gdp_pc_delta = fmt_delta(latest_period["gdp_per_capita_growth"]) if latest_period is not None else None

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Resumen macro y demografico")
st.caption(
    f"Mostrando el mes {selected_period} de {total_months}. Usa los botones de la barra lateral para avanzar o retroceder mes por mes."
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
    st.metric("PIB del mes", money(latest_period["gdp_nominal"]), delta=monthly_gdp_delta)
with metric_cols[1]:
    st.metric("PIB per capita del mes", money(latest_period["gdp_per_capita"]), delta=monthly_gdp_pc_delta)
with metric_cols[2]:
    st.metric(
        "Poblacion total",
        f"{int(latest_period['population']):,}",
        delta=monthly_population_delta,
    )
with metric_cols[3]:
    st.metric("Inflacion mensual", pct(latest_period["inflation_rate"]))

metric_cols_2 = st.columns(4)
with metric_cols_2[0]:
    st.metric("Desempleo", pct(latest_period["unemployment_rate"]))
with metric_cols_2[1]:
    st.metric("Mujeres", f"{int(latest_period['women'])}")
with metric_cols_2[2]:
    st.metric("Hombres", f"{int(latest_period['men'])}")
with metric_cols_2[3]:
    st.metric("Mujeres fertiles", f"{int(latest_period['fertile_women'])}")

metric_cols_3 = st.columns(4)
with metric_cols_3[0]:
    st.metric("Ninos", f"{int(latest_period['children'])}")
with metric_cols_3[1]:
    st.metric("Adultos", f"{int(latest_period['adults'])}")
with metric_cols_3[2]:
    st.metric("Mayores", f"{int(latest_period['seniors'])}")
with metric_cols_3[3]:
    st.metric("Fuerza laboral", f"{int(latest_period['labor_force'])}")

metric_cols_4b = st.columns(3)
with metric_cols_4b[0]:
    st.metric("Nacimientos", f"{int(latest_period['births'])}")
with metric_cols_4b[1]:
    st.metric("Muertes", f"{int(latest_period['deaths'])}")
with metric_cols_4b[2]:
    st.metric("Edad promedio", f"{latest_period['average_age']:.1f} anos")

metric_cols_4c = st.columns(2)
with metric_cols_4c[0]:
    st.metric("Ninos con familia", f"{int(latest_period['children_with_guardian'])}")
with metric_cols_4c[1]:
    st.metric("Huerfanos", f"{int(latest_period['orphans'])}")

metric_cols_4d = st.columns(3)
with metric_cols_4d[0]:
    st.metric("Canasta basica familiar", money(latest_period["average_family_basic_basket_cost"]))
with metric_cols_4d[1]:
    st.metric("Ingreso familiar", money(latest_period["average_family_income"]))
with metric_cols_4d[2]:
    st.metric("Cobertura ingreso/canasta", pct(latest_period["family_income_to_basket_ratio"]))

metric_cols_4e = st.columns(2)
with metric_cols_4e[0]:
    st.metric("Familias con ingreso bajo canasta", pct(latest_period["families_income_below_basket_share"]))
with metric_cols_4e[1]:
    st.metric(
        "Retencion por racionamiento",
        pct(latest_period["worker_involuntary_retention_rate"]),
    )

metric_cols_4f = st.columns(2)
with metric_cols_4f[0]:
    st.metric("Ahorro promedio de trabajadores", money(latest_period["average_worker_savings"]))

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
    if latest_period is None or pd.isna(latest_period["essential_fulfillment_rate"]):
        st.metric("Cobertura de bienes basicos", "n/a")
    else:
        st.metric("Cobertura de bienes basicos", pct(latest_period["essential_fulfillment_rate"]))
with metric_cols_4g[1]:
    if latest_basic_goods_per_person is None or pd.isna(latest_basic_goods_per_person):
        st.metric("Produccion basica por persona", "n/a")
    else:
        st.metric("Produccion basica por persona", f"{latest_basic_goods_per_person:.2f} unidades")

metric_cols_food = st.columns(3)
with metric_cols_food[0]:
    st.metric("Comidas promedio por persona", f"{latest_period['average_food_meals_per_person']:.1f}")
with metric_cols_food[1]:
    st.metric("Alimentacion suficiente", pct(latest_period["food_sufficient_share"]))
with metric_cols_food[2]:
    st.metric("Hambre severa", pct(latest_period["food_severe_hunger_share"]))

metric_cols_4 = st.columns(4)
with metric_cols_4[0]:
    st.metric("Indice de precios", f"{latest_period['price_index']:.2f}")
with metric_cols_4[1]:
    st.metric("Cobertura de demanda", pct(latest_period["demand_fulfillment_rate"]))
with metric_cols_4[2]:
    st.metric("Stock de capital", money(latest_period["total_capital_stock"]))
with metric_cols_4[3]:
    st.metric("Gini del ahorro de hogares", f"{latest_period['gini_household_savings']:.2f}")

metric_cols_5 = st.columns(2)
with metric_cols_5[0]:
    st.metric("Gini de riqueza de duenos", f"{latest_period['gini_owner_wealth']:.2f}")
with metric_cols_5[1]:
    st.metric("Participacion de dinero liquido capitalista", pct(latest_period["capitalist_liquid_share"]))

metric_cols_6 = st.columns(3)
with metric_cols_6[0]:
    st.metric("Oferta monetaria", money(latest_period["central_bank_money_supply"]))
with metric_cols_6[1]:
    st.metric("Emision monetaria", money(latest_period["central_bank_issuance"]))
with metric_cols_6[2]:
    st.metric("Tasa lider del banco central", pct(latest_period["central_bank_policy_rate"]))

metric_cols_7 = st.columns(3)
with metric_cols_7[0]:
    st.metric("Recaudacion fiscal", money(latest_period["government_tax_revenue"]))
with metric_cols_7[1]:
    public_spending = latest_period["government_transfers"] + latest_period["government_procurement_spending"]
    st.metric("Gasto publico", money(public_spending))
with metric_cols_7[2]:
    st.metric("Deficit fiscal / PIB", pct(latest_period["government_deficit_share_gdp"]))

metric_cols_7b = st.columns(2)
with metric_cols_7b[0]:
    st.metric("Deuda publica", money(latest_period["government_debt_outstanding"]))
with metric_cols_7b[1]:
    st.metric("Caja del Estado", money(latest_period["government_treasury_cash"]))

metric_cols_8 = st.columns(4)
with metric_cols_8[0]:
    st.metric("Creacion bancaria de dinero", money(latest_period["commercial_bank_credit_creation"]))
with metric_cols_8[1]:
    st.metric("Activos / pasivos bancarios", f"{latest_period['bank_asset_liability_ratio']:.2f}")
with metric_cols_8[2]:
    st.metric("Cobertura de reservas", f"{latest_period['bank_reserve_coverage_ratio']:.2f}")
with metric_cols_8[3]:
    st.metric("Prestamos / depositos", f"{latest_period['bank_loan_to_deposit_ratio']:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

tab_monthly, tab_summary, tab_distribution, tab_firms, tab_data = st.tabs(
    ["Evolucion mensual", "Resumen mensual", "Distribucion", "Empresas", "Datos"]
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
            y_cols=["worker_voluntary_saved", "worker_involuntary_retained"],
            title="Destino monetario de los trabajadores",
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
            y_cols=["worker_savings_rate", "worker_involuntary_retention_rate"],
            title="Ahorro voluntario vs retencion por racionamiento",
            y_title="Proporcion",
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
        key="monthly_dependency_ratio_chart",
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
                y_cols=["average_family_basic_basket_cost", "average_family_income"],
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
                y_cols=["family_income_to_basket_ratio", "families_income_below_basket_share"],
                title="Ingreso familiar frente a la canasta",
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
                title="Produccion mensual de alimentos basicos",
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

    c_food_3, c_food_4 = st.columns(2)
    with c_food_3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "food_sufficient_share",
                    "food_subsistence_share",
                    "food_acute_hunger_share",
                    "food_severe_hunger_share",
                ],
                title="Estados alimentarios de la poblacion",
                y_title="Proporcion",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_food_4:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["average_food_meals_per_person", "average_health_fragility"],
                title="Comidas promedio y fragilidad de salud",
                y_title="Nivel",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c_basic_1, c_basic_2 = st.columns(2)
    with c_basic_1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                basic_goods_view_df,
                x="period",
                y_cols=["production_per_worker"],
                title="Productividad por trabajador en bienes necesarios",
                y_title="Unidades por trabajador",
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
                y_cols=["labor_cost_per_product"],
                title="Precio relativo del trabajo por producto producido",
                y_title="Unidades monetarias por unidad",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            essential_survival_view_df,
            x="period",
            y_cols=[
                "essential_basket_equivalents_produced",
                "population",
            ],
            title="Canastas esenciales completas producidas vs poblacion",
            y_title="Canastas completas / personas",
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

    c_money_1, c_money_2 = st.columns(2)
    with c_money_1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "worker_augmented_asset_share",
                    "capitalist_augmented_asset_share",
                    "worker_liquid_share",
                    "capitalist_liquid_share",
                ],
                title="Concentracion ampliada de dinero y activos entre clases",
                y_title="Participacion",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_money_2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "worker_bank_deposits",
                    "worker_credit_outstanding",
                    "capitalist_liquid_wealth",
                    "capitalist_credit_outstanding",
                ],
                title="Dinero bancario y credito entre trabajadores y capitalistas",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c_money_3, c_money_4 = st.columns(2)
    with c_money_3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "capitalist_productive_capital",
                    "capitalist_inventory_value",
                    "total_bank_bond_holdings",
                ],
                title="Capital productivo, inventarios y bonos publicos",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_money_4:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["central_bank_policy_rate", "average_bank_deposit_rate", "average_bank_loan_rate"],
                title="Tasas del banco central y banca comercial",
                y_title="Tasa",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c_state_1, c_state_2 = st.columns(2)
    with c_state_1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "government_tax_revenue",
                    "government_total_spending",
                    "government_deficit",
                    "government_surplus",
                ],
                title="Flujos fiscales del Estado",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_state_2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "government_tax_burden_gdp",
                    "government_corporate_tax_burden_gdp",
                    "government_dividend_tax_burden_gdp",
                    "government_wealth_tax_burden_gdp",
                ],
                title="Carga de impuestos por tipo",
                y_title="Participacion del PIB",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c_state_3, c_state_4 = st.columns(2)
    with c_state_3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "government_deficit_share_gdp",
                    "government_tax_burden_gdp",
                    "government_spending_share_gdp",
                ],
                title="Deficit fiscal y carga del Estado como porcentaje del PIB",
                y_title="Participacion del PIB",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_state_4:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "central_bank_issuance",
                    "household_credit_creation",
                    "firm_credit_creation",
                    "commercial_bank_credit_creation",
                ],
                title="Emision del banco central y creacion bancaria de dinero",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c_state_5, c_state_6 = st.columns(2)
    with c_state_5:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["government_treasury_cash", "government_debt_outstanding"],
                title="Caja y deuda publica",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_state_6:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "central_bank_issuance",
                    "commercial_bank_credit_creation_share_money",
                ],
                title="Emision y credito nuevo como proporcion monetaria",
                y_title="Tasa / participacion",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    c_bank_1, c_bank_2 = st.columns(2)
    with c_bank_1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "total_bank_assets",
                    "total_bank_liabilities",
                    "bank_equity",
                ],
                title="Balance agregado de bancos comerciales",
                y_title="Unidades monetarias",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c_bank_2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=[
                    "bank_asset_liability_ratio",
                    "bank_reserve_coverage_ratio",
                    "bank_loan_to_deposit_ratio",
                    "bank_liquidity_ratio",
                    "bank_insolvent_share",
                ],
                title="Salud y prudencia de bancos comerciales",
                y_title="Ratio / participacion",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            history_view_df,
            x="period",
            y_cols=[
                "labor_share_gdp",
                "profit_share_gdp",
                "investment_share_gdp",
                "capitalist_consumption_share_gdp",
                "government_spending_share_gdp",
                "dividend_share_gdp",
                "retained_profit_share_gdp",
            ],
            title="Shares de flujo del PIB",
            y_title="Participacion",
        ),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_institution_flow_chart(latest_period),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_summary:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            history_view_df,
            x="period",
            y_cols=["gdp_nominal", "gdp_per_capita"],
            title="PIB mensual y PIB per capita",
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
            y_cols=["population"],
            title="Poblacion mensual",
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
            history_view_df,
            x="period",
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
            history_view_df,
            x="period",
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
            history_view_df,
            x="period",
            y_cols=["dependency_ratio"],
            title="Relacion de dependencia",
            y_title="Relacion",
        ),
        key="summary_dependency_ratio_chart",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            history_view_df,
            x="period",
            y_cols=["average_worker_savings"],
            title="Ahorro promedio de trabajadores",
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
                history_view_df,
                x="period",
                y_cols=["gdp_growth", "real_gdp_growth"],
                title="Crecimiento mensual",
                y_title="Tasa de crecimiento",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["inflation_rate", "unemployment_rate"],
                title="Inflacion y desempleo",
                y_title="Tasa",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(
        make_line_chart(
            history_view_df,
            x="period",
            y_cols=["inflation_rate"],
            title="Inflacion mensual",
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
                history_view_df,
                x="period",
                y_cols=["population_growth"],
                title="Crecimiento poblacional mensual",
                y_title="Tasa",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                history_view_df,
                x="period",
                y_cols=["birth_rate", "death_rate"],
                title="Natalidad y mortalidad",
                y_title="Tasa",
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
                y_cols=["fertile_capable_women", "fertile_capable_women_low_desire_no_birth"],
                title="Mujeres fertiles con capacidad economica e intervalo cumplido",
                y_title="Cantidad",
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
                y_cols=["fertile_capable_women_birth_rate"],
                title="Tasa de nacimientos entre mujeres fertiles capaces (intervalo cumplido)",
                y_title="Proporcion",
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(
        history_view.drop(columns=["Anio", "Periodo del anio"], errors="ignore")[
            [
                "Mes",
                "Poblacion",
                "Mujeres",
                "Hombres",
                "Mujeres fertiles",
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
                "Canasta basica familiar promedio",
                "Cobertura ingreso/canasta",
                "Familias con ingreso bajo canasta",
                "Tasa agregada de ahorro voluntario",
                "Tasa de retencion por racionamiento",
                "Participacion ninos",
                "Participacion adultos",
                "Participacion mayores",
                "Participacion mujeres",
                "Participacion hombres",
                "Participacion mujeres fertiles",
                "Relacion de dependencia",
                "PIB nominal",
                "PIB per capita",
                "Crecimiento del PIB",
                "Crecimiento del PIB real",
                "Crecimiento poblacional",
                "Tasa de natalidad",
                "Tasa de mortalidad",
                "Inflacion",
                "Desempleo",
                "Quiebras",
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
    st.subheader("Empresas del mes seleccionado")
    st.caption(
        f"Mes {selected_period}. Aqui puedes ver por que una firma si o no contrato mas: "
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
    else:
        st.info("No hay firmas activas para este mes.")

    first_row_left, first_row_right = st.columns(2)
    with first_row_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(
            make_line_chart(
                firm_summary_view_df,
                x="period",
                y_cols=["labor_force", "total_desired_workers", "labor_supply_gap"],
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
    st.subheader("Costos y precio de venta de bienes basicos")
    st.caption(
        "Serie temporal mensual ponderada por produccion de firmas activas de alimentos, vivienda y ropa/higiene."
    )
    latest_basic_cost = None
    latest_basic_price = None
    if not basic_goods_price_view_df.empty:
        latest_basic_cost = basic_goods_price_view_df.iloc[-1]["basic_goods_total_unit_cost"]
        latest_basic_price = basic_goods_price_view_df.iloc[-1]["basic_goods_average_sale_price"]
    if latest_basic_price is not None and latest_basic_cost is not None:
        st.caption(
            f"Ultimo mes visible - Precio promedio: {money(latest_basic_price)} | "
            f"Costo unitario promedio: {money(latest_basic_cost)}"
        )
        if latest_basic_price < latest_basic_cost:
            st.warning(
                "El precio promedio de los bienes basicos esta por debajo del costo unitario promedio. "
                "Eso implica margen unitario negativo en el ultimo mes visible."
            )
    if basic_goods_price_view_df.empty:
        st.info("No hay firmas basicas activas en los meses visibles para construir la serie.")
    else:
        st.plotly_chart(
            make_line_chart(
                basic_goods_price_view_df,
                x="period",
                y_cols=[
                    "basic_goods_average_sale_price",
                    "basic_goods_total_unit_cost",
                    "basic_goods_labor_cost_per_unit",
                    "basic_goods_input_cost_per_unit",
                    "basic_goods_transport_cost_per_unit",
                    "basic_goods_fixed_cost_per_unit",
                    "basic_goods_capital_cost_per_unit",
                ],
                title="Precio de venta y componentes de costo (promedio ponderado)",
                y_title="Unidades monetarias por unidad",
            ),
            use_container_width=True,
        )
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
        title="Contratacion por sector en el mes seleccionado",
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
        st.info("No hay datos de firmas para este mes.")
    else:
        sector_unemployment_fig = px.bar(
            sector_labor_market_df.sort_values("vacancy_rate", ascending=False),
            x="sector_label",
            y="vacancy_rate",
            title="Vacancia sectorial aproximada del mes",
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

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Diagnostico mensual de una firma")
    st.caption(
        "Selecciona una firma y revisa mes por mes que esperaba vender, que decision tomo, en que fase de aprendizaje estaba "
        "y como se encontraban los demandantes."
    )
    firm_selector_options = []
    for _, row in firm_detail_view.iterrows():
        firm_selector_options.append(
            (
                f"Firma {int(row['Firma'])} | {row['Sector']} | {'activa' if row['Activa'] == 'Si' else 'inactiva'}",
                int(row["Firma"]),
            )
        )
    if not firm_selector_options:
        st.info("No hay firmas disponibles para diagnostico en este filtro.")
    else:
        selected_firm_label = st.selectbox(
            "Firma para diagnostico detallado",
            options=[label for label, _ in firm_selector_options],
            index=0,
        )
        selected_firm_id = dict(firm_selector_options)[selected_firm_label]
        firm_diagnostic_df = build_firm_diagnostic_data(
            firm_history_view_df,
            history_view_df,
            selected_firm_id,
            result.config.firm_learning_warmup_periods,
        )

        if firm_diagnostic_df.empty:
            st.info("No hay historial disponible para esta firma.")
        else:
            latest_firm_month = firm_diagnostic_df.iloc[-1]
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Precio actual", money(latest_firm_month["price"]), delta=fmt_delta(latest_firm_month["price_change"]))
            with metric_cols[1]:
                sales_realization = latest_firm_month["sales_realization"]
                st.metric(
                    "Ventas / esperado",
                    "n/a" if pd.isna(sales_realization) else f"{sales_realization:.2f}x",
                )
            with metric_cols[2]:
                st.metric("Caja", money(latest_firm_month["cash"]))
            with metric_cols[3]:
                basket_ratio = latest_firm_month["family_income_to_basket_ratio"]
                st.metric(
                    "Ingreso/canasta hogares",
                    "n/a" if pd.isna(basket_ratio) else f"{basket_ratio:.2f}x",
                )

            audit_cols = st.columns(4)
            with audit_cols[0]:
                mean_abs_error = firm_diagnostic_df["forecast_error_abs"].dropna().mean()
                st.metric("Error medio pronostico", "n/a" if pd.isna(mean_abs_error) else f"{mean_abs_error:.2f}")
            with audit_cols[1]:
                tense_price_hikes = firm_diagnostic_df[
                    (firm_diagnostic_df["price_change"] > 0.02)
                    & (firm_diagnostic_df["family_income_to_basket_ratio"] < 1.0)
                ].shape[0]
                st.metric("Meses subio precio con hogares tensos", f"{tense_price_hikes}")
            with audit_cols[2]:
                warmup_months = int(firm_diagnostic_df["training_phase"].sum())
                st.metric("Meses en aprendizaje", f"{warmup_months}")
            with audit_cols[3]:
                demand_elasticity = latest_firm_month.get("demand_elasticity", pd.NA)
                st.metric(
                    "Elasticidad usada",
                    "n/a" if pd.isna(demand_elasticity) else f"{demand_elasticity:.2f}",
                )

            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(
                    make_line_chart(
                        firm_diagnostic_df,
                        x="period",
                        y_cols=["price", "unit_cost", "wage_offer"],
                        title="Precio, costo unitario y salario",
                        y_title="Unidades monetarias",
                    ),
                    key=f"firm_diag_price_{selected_firm_id}",
                    use_container_width=True,
                )
            with chart_right:
                st.plotly_chart(
                    make_line_chart(
                        firm_diagnostic_df,
                        x="period",
                        y_cols=["sales", "expected_sales", "inventory", "target_inventory"],
                        title="Ventas, expectativa e inventario",
                        y_title="Unidades",
                    ),
                    key=f"firm_diag_sales_{selected_firm_id}",
                    use_container_width=True,
                )

            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(
                    make_line_chart(
                        firm_diagnostic_df,
                        x="period",
                        y_cols=["cash", "profit"],
                        title="Caja y ganancia de la firma",
                        y_title="Unidades monetarias",
                    ),
                    key=f"firm_diag_cash_{selected_firm_id}",
                    use_container_width=True,
                )
            with chart_right:
                st.plotly_chart(
                    make_line_chart(
                        firm_diagnostic_df,
                        x="period",
                        y_cols=[
                            "family_income_to_basket_ratio",
                            "families_income_below_basket_share",
                            "essential_fulfillment_rate",
                            "food_severe_hunger_share",
                            "unemployment_rate",
                        ],
                        title="Contexto social del mismo mes",
                        y_title="Ratio / tasa",
                    ),
                    key=f"firm_diag_context_{selected_firm_id}",
                    use_container_width=True,
                )

            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(
                    make_line_chart(
                        firm_diagnostic_df,
                        x="period",
                        y_cols=[
                            "demand_elasticity",
                            "forecast_error_belief",
                            "market_fragility_belief",
                            "learning_maturity",
                        ],
                        title="Aprendizaje de demanda y fragilidad percibida",
                        y_title="Indice / ratio",
                    ),
                    key=f"firm_diag_learning_{selected_firm_id}",
                    use_container_width=True,
                )
            with chart_right:
                st.plotly_chart(
                    make_line_chart(
                        firm_diagnostic_df,
                        x="period",
                        y_cols=[
                            "worker_savings_rate",
                            "worker_involuntary_retention_rate",
                            "worker_liquid_share",
                            "capitalist_liquid_share",
                        ],
                        title="Estado financiero de demandantes y concentracion",
                        y_title="Ratio / tasa",
                    ),
                    key=f"firm_diag_demanders_{selected_firm_id}",
                    use_container_width=True,
                )

            with st.expander("Cronologia narrativa mes por mes", expanded=False):
                for _, row in firm_diagnostic_df.iterrows():
                    st.markdown(f"Mes {int(row['period'])}: {row['decision_note']}")

            diagnostic_view = firm_diagnostic_df[
                [
                    "period",
                    "active",
                    "price",
                    "price_change",
                    "wage_offer",
                    "wage_change",
                    "workers",
                    "desired_workers",
                    "expected_sales",
                    "expected_sales_change",
                    "sales",
                    "sales_realization",
                    "forecast_error_abs",
                    "inventory_ratio",
                    "price_to_unit_cost",
                    "cash",
                    "cash_cover_months",
                    "profit",
                    "demand_elasticity",
                    "forecast_caution",
                    "forecast_error_belief",
                    "market_fragility_belief",
                    "learning_maturity",
                    "training_phase",
                    "population",
                    "unemployment_rate",
                    "family_income_to_basket_ratio",
                    "families_income_below_basket_share",
                    "essential_fulfillment_rate",
                    "demand_fulfillment_rate",
                    "average_food_meals_per_person",
                    "food_sufficient_share",
                    "food_severe_hunger_share",
                    "average_health_fragility",
                    "worker_savings_rate",
                    "worker_involuntary_retention_rate",
                    "worker_liquid_share",
                    "capitalist_liquid_share",
                    "central_bank_issuance",
                    "decision_note",
                ]
            ].copy()
            diagnostic_view["active"] = diagnostic_view["active"].map({True: "Si", False: "No"})
            diagnostic_view["training_phase"] = diagnostic_view["training_phase"].map({True: "Si", False: "No"})
            diagnostic_view = diagnostic_view.rename(
                columns={
                    "period": "Mes",
                    "active": "Activa",
                    "price": "Precio",
                    "price_change": "Cambio de precio",
                    "wage_offer": "Salario ofrecido",
                    "wage_change": "Cambio de salario",
                    "workers": "Trabajadores",
                    "desired_workers": "Trabajadores deseados",
                    "expected_sales": "Ventas esperadas",
                    "expected_sales_change": "Cambio de ventas esperadas",
                    "sales": "Ventas reales",
                    "sales_realization": "Ventas / esperado",
                    "forecast_error_abs": "Error absoluto de pronostico",
                    "inventory_ratio": "Inventario / objetivo",
                    "price_to_unit_cost": "Precio / costo unitario",
                    "cash": "Caja",
                    "cash_cover_months": "Meses de caja",
                    "profit": "Ganancia",
                    "demand_elasticity": "Elasticidad de demanda",
                    "forecast_caution": "Cautela de pronostico",
                    "forecast_error_belief": "Error de pronostico percibido",
                    "market_fragility_belief": "Fragilidad de mercado percibida",
                    "learning_maturity": "Madurez de aprendizaje",
                    "training_phase": "Fase de aprendizaje",
                    "population": "Poblacion",
                    "unemployment_rate": "Desempleo",
                    "family_income_to_basket_ratio": "Ingreso/canasta hogares",
                    "families_income_below_basket_share": "Familias bajo canasta",
                    "essential_fulfillment_rate": "Cobertura bienes basicos",
                    "demand_fulfillment_rate": "Cobertura total de demanda",
                    "average_food_meals_per_person": "Comidas promedio por persona",
                    "food_sufficient_share": "Poblacion con comida suficiente",
                    "food_severe_hunger_share": "Poblacion con hambre severa",
                    "average_health_fragility": "Fragilidad de salud",
                    "worker_savings_rate": "Ahorro voluntario trabajador",
                    "worker_involuntary_retention_rate": "Retencion obrera por racionamiento",
                    "worker_liquid_share": "Participacion liquida trabajadora",
                    "capitalist_liquid_share": "Participacion liquida capitalista",
                    "central_bank_issuance": "Emision monetaria",
                    "decision_note": "Lectura del mes",
                }
            )
            st.dataframe(diagnostic_view, width="stretch", hide_index=True)
            st.download_button(
                "Descargar CSV del diagnostico de la firma",
                dataframe_to_csv_bytes(diagnostic_view),
                file_name=f"firm_{selected_firm_id}_diagnostic.csv",
                mime="text/csv",
                width="stretch",
                key=f"download_firm_diag_{selected_firm_id}",
                on_click="ignore",
            )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Historial mensual hasta el mes seleccionado")
    st.dataframe(
        history_view.drop(columns=["Anio", "Periodo del anio"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Descargar CSV mensual seleccionado",
        dataframe_to_csv_bytes(history_view.drop(columns=["Anio", "Periodo del anio"], errors="ignore")),
        file_name="economy_monthly_selected.csv",
        mime="text/csv",
        width="stretch",
        key="download_monthly_selected",
        on_click="ignore",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Historial mensual completo")
    st.dataframe(
        history_full_view.drop(columns=["Anio", "Periodo del anio"], errors="ignore"),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Descargar CSV mensual",
        dataframe_to_csv_bytes(history_full_view.drop(columns=["Anio", "Periodo del anio"], errors="ignore")),
        file_name="economy_monthly_history.csv",
        mime="text/csv",
        width="stretch",
        key="download_monthly_full",
        on_click="ignore",
    )
    st.markdown("</div>", unsafe_allow_html=True)
