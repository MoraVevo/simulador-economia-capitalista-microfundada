from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import tempfile
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .policies import CountryProfile, country_profiles
from .scenario_runner import run_scenario_history


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run country-style scenarios in parallel and export one PDF per profile.")
    parser.add_argument("--periods", type=int, default=240, help="Simulation periods to run per profile.")
    parser.add_argument("--households", type=int, default=10000, help="Worker households per scenario.")
    parser.add_argument("--firms-per-sector", type=int, default=40, help="Initial firms per sector.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for all profiles.")
    parser.add_argument("--periods-per-year", type=int, default=12, help="Simulation periods in one year.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf"),
        help="Directory where the country PDFs and CSV summaries will be written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes. Use 0 to auto-pick up to the number of profiles.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print one progress line per scenario every N periods. Use 0 to silence progress.",
    )
    parser.add_argument(
        "--phase-split-period",
        type=int,
        default=0,
        help="If > 0 and < periods, also export phase PDFs split at this period.",
    )
    return parser


def _safe_float(value: float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)


def _fmt_money(value: float | int | None) -> str:
    return f"{_safe_float(value):,.2f}"


def _fmt_pct(value: float | int | None) -> str:
    return f"{_safe_float(value):.1%}"


def _slugify(name: str) -> str:
    normalized = (
        name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    cleaned = ["-" if not ch.isalnum() else ch for ch in normalized]
    return "".join(cleaned).strip("-").replace("--", "-")


def _rolling_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.mean())


def _phase_label(start_period: int, end_period: int, phase_index: int) -> str:
    return f"Fase {phase_index} ({start_period}-{end_period})"


def _run_profile(payload: dict[str, object]) -> tuple[str, str]:
    name = str(payload["scenario_name"])
    frame_json = run_scenario_history(
        months=int(payload["months"]),
        seed=int(payload["seed"]),
        firms_per_sector=int(payload["firms_per_sector"]),
        households=int(payload["households"]),
        periods_per_year=int(payload["periods_per_year"]),
        base_policy_values=dict(payload["policy_values"]),
        policy_change_period=None,
        shock_policy_values=None,
        scenario_name=name,
        log_every=int(payload["log_every"]),
    )
    return name, frame_json


def _annualize_core_history(monthly: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, str] = {
        "population": "mean",
        "fertile_women": "mean",
        "births": "sum",
        "deaths": "sum",
        "labor_force": "mean",
        "employment_count": "mean",
        "unemployment_rate": "mean",
        "family_income_to_basket_ratio": "mean",
        "gdp_nominal": "sum",
        "real_gdp_nominal": "sum",
        "potential_gdp_nominal": "sum",
        "output_gap_share": "mean",
        "cpi": "last",
        "gdp_deflator": "last",
        "inflation_rate": "mean",
        "gdp_growth": "mean",
        "population_growth": "mean",
        "average_wage": "mean",
        "real_average_wage": "mean",
        "essential_demand_units": "sum",
        "essential_production_units": "sum",
        "essential_sales_units": "sum",
        "people_full_essential_coverage": "mean",
        "full_essential_coverage_share": "mean",
        "average_food_meals_per_person": "mean",
        "bank_equity": "last",
        "bank_capital_ratio": "mean",
        "bank_insolvent_share": "mean",
        "bank_undercapitalized_share": "mean",
        "central_bank_money_supply": "last",
        "central_bank_target_money_supply": "last",
        "central_bank_policy_rate": "mean",
        "central_bank_issuance": "sum",
        "central_bank_monetary_gap_share": "mean",
        "average_bank_reserve_ratio": "mean",
        "government_tax_revenue": "sum",
        "government_labor_tax_revenue": "sum",
        "government_payroll_tax_revenue": "sum",
        "government_total_spending": "sum",
        "government_deficit": "sum",
        "government_debt_outstanding": "last",
        "government_school_spending": "sum",
        "government_university_spending": "sum",
        "government_school_units": "sum",
        "government_university_units": "sum",
        "school_average_price": "mean",
        "university_average_price": "mean",
        "recession_flag": "mean",
        "recession_intensity": "mean",
        "government_countercyclical_spending": "sum",
        "government_countercyclical_support_multiplier": "mean",
        "government_countercyclical_procurement_multiplier": "mean",
        "household_final_consumption_share_gdp": "mean",
        "government_final_consumption_share_gdp": "mean",
        "government_infrastructure_spending_share_gdp": "mean",
        "government_spending_share_gdp": "mean",
        "government_tax_burden_gdp": "mean",
        "gross_capital_formation_share_gdp": "mean",
        "investment_knowledge_multiplier": "mean",
        "public_capital_stock": "last",
        "net_exports_share_gdp": "mean",
        "gdp_expenditure_gap_share_gdp": "mean",
        "government_deficit_share_gdp": "mean",
        "school_enrollment_share": "mean",
        "university_enrollment_share": "mean",
        "school_completion_share": "mean",
        "university_completion_share": "mean",
        "low_resource_school_enrollment_share": "mean",
        "low_resource_university_enrollment_share": "mean",
        "low_resource_university_student_share": "mean",
        "low_resource_origin_upward_mobility_share": "mean",
        "low_resource_origin_university_completion_share": "mean",
        "poor_origin_university_mobility_lift": "mean",
        "school_income_premium": "mean",
        "university_income_premium": "mean",
        "poverty_rate_without_university": "mean",
        "poverty_rate_with_university": "mean",
        "skilled_job_fill_rate": "mean",
    }
    available = {column: rule for column, rule in aggregations.items() if column in monthly.columns}
    annual = monthly.groupby("year", as_index=False).agg(available)
    if annual.empty:
        return annual
    if {"gdp_nominal", "real_gdp_nominal"}.issubset(annual.columns):
        annual["gdp_deflator"] = annual["gdp_nominal"] / annual["real_gdp_nominal"].replace(0, pd.NA)
    annual["inflation_yoy"] = annual.get("gdp_deflator", annual["cpi"]).pct_change()
    annual["cpi_inflation_yoy"] = annual["cpi"].pct_change()
    annual["gdp_growth_yoy"] = annual["gdp_nominal"].pct_change()
    annual["population_growth_yoy"] = annual["population"].pct_change()
    annual["avg_unemployment_rate"] = annual.get("unemployment_rate", pd.Series(dtype=float))
    if {"government_labor_tax_revenue", "gdp_nominal"}.issubset(annual.columns):
        annual["government_labor_tax_burden_gdp"] = (
            annual["government_labor_tax_revenue"] / annual["gdp_nominal"].replace(0, pd.NA)
        )
    if {"government_payroll_tax_revenue", "gdp_nominal"}.issubset(annual.columns):
        annual["government_payroll_tax_burden_gdp"] = (
            annual["government_payroll_tax_revenue"] / annual["gdp_nominal"].replace(0, pd.NA)
        )
    if {"government_school_spending", "government_school_units"}.issubset(annual.columns):
        annual["government_school_unit_cost"] = (
            annual["government_school_spending"] / annual["government_school_units"].replace(0, pd.NA)
        )
    if {"government_university_spending", "government_university_units"}.issubset(annual.columns):
        annual["government_university_unit_cost"] = (
            annual["government_university_spending"] / annual["government_university_units"].replace(0, pd.NA)
        )
    if {"government_school_unit_cost", "school_average_price"}.issubset(annual.columns):
        annual["government_school_unit_cost_ratio_private_price"] = (
            annual["government_school_unit_cost"] / annual["school_average_price"].replace(0, pd.NA)
        )
    if {"government_university_unit_cost", "university_average_price"}.issubset(annual.columns):
        annual["government_university_unit_cost_ratio_private_price"] = (
            annual["government_university_unit_cost"] / annual["university_average_price"].replace(0, pd.NA)
        )
    if "school_enrollment_share" in annual.columns:
        annual["children_studying_ratio"] = annual["school_enrollment_share"]
    if "school_completion_share" in annual.columns:
        annual["adults_with_school_credential_ratio"] = annual["school_completion_share"]
    if "university_completion_share" in annual.columns:
        annual["adults_with_university_credential_ratio"] = annual["university_completion_share"]
    return annual


def _style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8)


def _plot_line_set(ax, frame: pd.DataFrame, x: str, columns: list[tuple[str, str]], ylabel: str) -> None:
    for column, label in columns:
        if column in frame.columns:
            ax.plot(frame[x], frame[column], label=label, linewidth=1.8)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Ano", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8)


def _make_macro_figure(annual: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    _plot_line_set(
        axes[0, 0],
        annual,
        "year",
        [
            ("gdp_nominal", "PIB nominal"),
            ("real_gdp_nominal", "PIB real"),
            ("potential_gdp_nominal", "PIB potencial"),
        ],
        "Unidades monetarias",
    )
    _style_axis(axes[0, 0], "PIB nominal, real y potencial", "Unidades monetarias")

    _plot_line_set(
        axes[0, 1],
        annual,
        "year",
        [
            ("inflation_yoy", "Inflacion PIB"),
            ("avg_unemployment_rate", "Desempleo"),
            ("output_gap_share", "Brecha de producto"),
        ],
        "Tasa",
    )
    _style_axis(axes[0, 1], "Inflacion, desempleo y brecha", "Tasa")

    _plot_line_set(
        axes[1, 0],
        annual,
        "year",
        [
            ("cpi", "IPC"),
            ("average_wage", "Salario promedio"),
            ("real_average_wage", "Salario real"),
        ],
        "Indice o dinero",
    )
    _style_axis(axes[1, 0], "IPC y salarios", "Indice o dinero")

    ax = axes[1, 1]
    for column, label in [
        ("essential_production_units", "Producidos"),
        ("essential_sales_units", "Comprados"),
        ("essential_demand_units", "Necesarios"),
    ]:
        if column in annual.columns:
            ax.plot(annual["year"], annual[column], label=label, linewidth=1.8)
    if "full_essential_coverage_share" in annual.columns:
        ax2 = ax.twinx()
        ax2.plot(
            annual["year"],
            annual["full_essential_coverage_share"],
            color="#2F855A",
            linewidth=1.8,
            linestyle="--",
            label="Cobertura completa",
        )
        ax2.set_ylabel("Cobertura", fontsize=9)
        ax2.tick_params(axis="y", labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="best")
    else:
        ax.legend(fontsize=8, loc="best")
    _style_axis(ax, "Canasta necesaria producida vs comprada", "Unidades")
    ax.set_xlabel("Ano", fontsize=9)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_public_sector_figure(annual: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    plotted = False
    if "population" in annual.columns:
        ax.plot(
            annual["year"],
            annual["population"],
            label="Poblacion total",
            linewidth=2.2,
            color="#1d4ed8",
        )
        plotted = True
    for column, label, color in [
        ("fertile_women", "Mujeres fertiles", "#b45309"),
        ("births", "Nacimientos", "#16a34a"),
        ("deaths", "Muertes", "#dc2626"),
    ]:
        if column in annual.columns:
            ax.plot(annual["year"], annual[column], label=label, linewidth=1.6, color=color)
            plotted = True
    if plotted:
        ax.legend(fontsize=8, loc="best")
    _style_axis(ax, "Poblacion total, natalidad y mortalidad", "Personas")

    ax = axes[0, 1]
    for column, label in [
        ("government_tax_revenue", "Ingresos"),
        ("government_labor_tax_revenue", "Impuestos al trabajo"),
        ("government_payroll_tax_revenue", "Contribuciones nomina"),
        ("government_total_spending", "Gasto total"),
        ("government_deficit", "Deficit"),
        ("government_countercyclical_spending", "Gasto anticiclico"),
    ]:
        if column in annual.columns:
            ax.plot(annual["year"], annual[column], label=label, linewidth=1.8)
    if "government_debt_outstanding" in annual.columns or "recession_intensity" in annual.columns:
        ax2 = ax.twinx()
        if "government_debt_outstanding" in annual.columns:
            ax2.plot(
                annual["year"],
                annual["government_debt_outstanding"],
                color="#805AD5",
                linewidth=1.8,
                linestyle="--",
                label="Deuda",
            )
        if "recession_intensity" in annual.columns:
            ax2.plot(
                annual["year"],
                annual["recession_intensity"],
                color="#C53030",
                linewidth=1.6,
                linestyle=":",
                label="Intensidad recesion",
            )
        ax2.set_ylabel("Deuda o intensidad", fontsize=9)
        ax2.tick_params(axis="y", labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="best")
    else:
        ax.legend(fontsize=8, loc="best")
    _style_axis(ax, "Gobierno: recaudo, gasto y deuda", "Unidades monetarias")

    ax = axes[1, 0]
    for column, label in [
        ("bank_equity", "Patrimonio bancario"),
        ("bank_capital_ratio", "Capital/activos"),
        ("bank_insolvent_share", "Bancos insolventes"),
        ("bank_undercapitalized_share", "Bancos subcapitalizados"),
    ]:
        if column in annual.columns:
            ax.plot(annual["year"], annual[column], label=label, linewidth=1.8)
    ax.legend(fontsize=8, loc="best")
    _style_axis(ax, "Bancos comerciales", "Unidades o tasa")

    ax = axes[1, 1]
    for column, label in [
        ("central_bank_money_supply", "Oferta monetaria"),
        ("central_bank_target_money_supply", "Oferta objetivo"),
        ("central_bank_issuance", "Emision"),
        ("central_bank_policy_rate", "Tasa lider"),
        ("central_bank_monetary_gap_share", "Brecha monetaria"),
        ("average_bank_reserve_ratio", "Encaje efectivo"),
    ]:
        if column in annual.columns:
            ax.plot(annual["year"], annual[column], label=label, linewidth=1.8)
    ax.legend(fontsize=8, loc="best")
    _style_axis(ax, "Banco central", "Unidades o tasa")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_expenditure_figure(annual: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    ax = axes[0]
    expenditure_columns = [
        ("household_final_consumption_share_gdp", "Consumo hogares"),
        ("government_final_consumption_share_gdp", "Consumo gobierno"),
        ("gross_capital_formation_share_gdp", "Formacion de capital"),
        ("net_exports_share_gdp", "Exportaciones netas"),
    ]
    plotted = False
    for column, label in expenditure_columns:
        if column in annual.columns:
            ax.plot(annual["year"], annual[column], label=label, linewidth=1.8)
            plotted = True
    if "gdp_expenditure_gap_share_gdp" in annual.columns:
        ax.plot(
            annual["year"],
            annual["gdp_expenditure_gap_share_gdp"],
            label="Brecha identidad gasto",
            linewidth=1.4,
            linestyle="--",
            color="#805AD5",
        )
        plotted = True
    if plotted:
        ax.legend(fontsize=8, loc="best")
    _style_axis(ax, "Distribucion del PIB por gasto", "Participacion del PIB")
    ax.set_xlabel("Ano", fontsize=9)

    ax = axes[1]
    for column, label in [
        ("government_tax_burden_gdp", "T / PIB"),
        ("government_labor_tax_burden_gdp", "Trabajo / PIB"),
        ("government_payroll_tax_burden_gdp", "Nomina / PIB"),
        ("government_deficit_share_gdp", "Deficit / PIB"),
        ("government_spending_share_gdp", "Gasto Estado / PIB"),
    ]:
        if column in annual.columns:
            ax.plot(annual["year"], annual[column], label=label, linewidth=1.8)
    ax.legend(fontsize=8, loc="best")
    _style_axis(ax, "Carga tributaria y tamano del Estado", "Participacion del PIB")
    ax.set_xlabel("Ano", fontsize=9)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_education_figure(annual: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    _plot_line_set(
        axes[0, 0],
        annual,
        "year",
        [
            ("children_studying_ratio", "Ninos estudiando / ninos totales"),
            ("university_enrollment_share", "Matricula universitaria"),
            ("adults_with_school_credential_ratio", "Adultos con escolaridad completa / adultos totales"),
            ("adults_with_university_credential_ratio", "Adultos con titulo universitario / adultos totales"),
        ],
        "Participacion",
    )
    _style_axis(axes[0, 0], "Escolaridad y universidad", "Participacion")

    _plot_line_set(
        axes[0, 1],
        annual,
        "year",
        [
            ("low_resource_school_enrollment_share", "Escuela desde hogares pobres"),
            ("low_resource_university_enrollment_share", "Universidad desde hogares pobres"),
            ("low_resource_university_student_share", "Universitarios desde hogares pobres"),
        ],
        "Participacion",
    )
    _style_axis(axes[0, 1], "Acceso educativo desde hogares pobres", "Participacion")

    _plot_line_set(
        axes[1, 0],
        annual,
        "year",
        [
            ("low_resource_origin_upward_mobility_share", "Movilidad ascendente"),
            ("low_resource_origin_university_completion_share", "Titulo universitario"),
            ("poor_origin_university_mobility_lift", "Ventaja por universidad"),
        ],
        "Participacion o brecha",
    )
    _style_axis(axes[1, 0], "Movilidad social desde origen pobre", "Participacion o brecha")

    _plot_line_set(
        axes[1, 1],
        annual,
        "year",
        [
            ("school_income_premium", "Prima escolar"),
            ("university_income_premium", "Prima universitaria"),
            ("poverty_rate_without_university", "Pobreza sin universidad"),
            ("poverty_rate_with_university", "Pobreza con universidad"),
            ("skilled_job_fill_rate", "Cobertura vacantes cualificadas"),
        ],
        "Ratio o tasa",
    )
    _style_axis(axes[1, 1], "Retornos educativos y trabajo cualificado", "Ratio o tasa")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary_tables(annual: pd.DataFrame) -> tuple[list[list[str]], list[list[str]]]:
    if annual.empty:
        return [], []

    summary_window = annual.tail(min(3, len(annual))).copy()
    window_start = int(summary_window["year"].iloc[0])
    window_end = int(summary_window["year"].iloc[-1])
    summary_rows = [
        ["Ventana del resumen", f"{window_start}-{window_end}"],
        ["PIB nominal promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "gdp_nominal"))],
        ["PIB real promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "real_gdp_nominal"))],
        ["PIB potencial promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "potential_gdp_nominal"))],
        ["Inflacion promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "inflation_yoy"))],
        ["Desempleo promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "avg_unemployment_rate"))],
        ["IPC promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "cpi"))],
        ["Salario promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "average_wage"))],
        ["Salario real promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "real_average_wage"))],
        ["Cobertura completa promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "full_essential_coverage_share"))],
        ["Patrimonio bancario promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "bank_equity"))],
        ["Deficit fiscal promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "government_deficit"))],
        ["Deficit fiscal / PIB promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "government_deficit_share_gdp"))],
        ["Gasto total del Estado / PIB promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "government_spending_share_gdp"))],
        ["Consumo final del gobierno / PIB promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "government_final_consumption_share_gdp"))],
        ["Infraestructura publica / PIB promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "government_infrastructure_spending_share_gdp"))],
        ["Carga tributaria / PIB promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "government_tax_burden_gdp"))],
        ["Impuestos al trabajo / PIB promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "government_labor_tax_burden_gdp"))],
        ["Contribuciones sobre nomina / PIB promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "government_payroll_tax_burden_gdp"))],
        ["Costo unitario escuela publica / precio privado", _fmt_pct(_rolling_mean(summary_window, "government_school_unit_cost_ratio_private_price"))],
        ["Costo unitario universidad publica / precio privado", _fmt_pct(_rolling_mean(summary_window, "government_university_unit_cost_ratio_private_price"))],
        ["Multiplicador conocimiento-inversion promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "investment_knowledge_multiplier"))],
        ["Stock de capital publico final", _fmt_money(annual.iloc[-1].get("public_capital_stock"))],
        ["Intensidad de recesion promedio ultimos 3 anos", _fmt_pct(_rolling_mean(summary_window, "recession_intensity"))],
        ["Gasto anticiclico promedio ultimos 3 anos", _fmt_money(_rolling_mean(summary_window, "government_countercyclical_spending"))],
    ]
    def _series_with_fallback(*columns: str) -> pd.Series:
        for column in columns:
            if column in annual.columns:
                return annual[column]
        return pd.Series([float("nan")] * len(annual), dtype=float)

    children_studying = _series_with_fallback("children_studying_ratio", "school_enrollment_share")
    adults_school = _series_with_fallback("adults_with_school_credential_ratio", "school_completion_share")
    adults_university = _series_with_fallback(
        "adults_with_university_credential_ratio",
        "university_completion_share",
    )
    mobility_rows = [
        ["Ninos estudiando / ninos totales", _fmt_pct(children_studying.mean())],
        ["Matricula universitaria promedio", _fmt_pct(annual["university_enrollment_share"].mean())],
        ["Adultos con escolaridad completa / adultos totales", _fmt_pct(adults_school.mean())],
        ["Adultos con titulo universitario / adultos totales", _fmt_pct(adults_university.mean())],
        ["Escuela desde hogares pobres", _fmt_pct(annual["low_resource_school_enrollment_share"].mean())],
        ["Universidad desde hogares pobres", _fmt_pct(annual["low_resource_university_enrollment_share"].mean())],
        ["Movilidad ascendente origen pobre", _fmt_pct(annual["low_resource_origin_upward_mobility_share"].mean())],
        ["Prima universitaria promedio", _fmt_pct(annual["university_income_premium"].mean())],
        ["Pobreza con universidad", _fmt_pct(annual["poverty_rate_with_university"].mean())],
        ["Pobreza sin universidad", _fmt_pct(annual["poverty_rate_without_university"].mean())],
        ["Cobertura vacantes cualificadas", _fmt_pct(annual["skilled_job_fill_rate"].mean())],
    ]
    return summary_rows, mobility_rows


def _policy_table_rows(policy_values: dict[str, float | str]) -> list[list[str]]:
    selected_keys = [
        ("target_unemployment", "Desempleo objetivo"),
        ("central_bank_target_annual_inflation", "Meta anual de inflacion"),
        ("central_bank_policy_rate_base", "Tasa lider base"),
        ("reserve_ratio", "Encaje bancario"),
        ("government_corporate_tax_rate_low", "Impuesto corporativo bajo"),
        ("government_corporate_tax_rate_mid", "Impuesto corporativo medio"),
        ("government_corporate_tax_rate_high", "Impuesto corporativo alto"),
        ("government_dividend_tax_rate_low", "Impuesto dividendos bajo"),
        ("government_dividend_tax_rate_mid", "Impuesto dividendos medio"),
        ("government_dividend_tax_rate_high", "Impuesto dividendos alto"),
        ("government_labor_tax_rate_low", "Impuesto laboral bajo"),
        ("government_labor_tax_rate_mid", "Impuesto laboral medio"),
        ("government_labor_tax_rate_high", "Impuesto laboral alto"),
        ("government_payroll_tax_rate", "Contribucion sobre nomina"),
        ("government_wealth_tax_rate", "Impuesto al patrimonio"),
        ("government_unemployment_benefit_share", "Seguro de desempleo"),
            ("government_child_allowance_share", "Transferencia por hijos"),
            ("public_school_budget_share", "Presupuesto escolar publico"),
            ("public_school_support_package_share", "Paquete efectivo escuela publica"),
            ("public_university_budget_share", "Presupuesto universitario publico"),
        ("public_university_support_package_share", "Paquete efectivo universidad publica"),
        ("public_education_low_resource_priority_bonus", "Prioridad educativa hogares pobres"),
        ("public_school_support_continuity_bonus", "Continuidad escuela publica"),
        ("public_university_support_continuity_bonus", "Continuidad universidad publica"),
        ("public_administration_budget_share", "Presupuesto administracion publica"),
        ("government_infrastructure_budget_share", "Presupuesto infraestructura publica"),
        ("public_administration_payroll_share", "Parte salarial administracion publica"),
        ("public_administration_employment_floor_share", "Piso empleo administracion publica"),
        (
            "public_administration_employment_state_size_sensitivity",
            "Sensibilidad empleo admin al tamano del Estado",
        ),
        ("public_administration_employment_cap_share", "Tope empleo administracion publica"),
        ("government_final_consumption_floor_share_gdp", "Piso consumo final gobierno / PIB"),
            ("government_spending_scale", "Escala de gasto publico"),
            ("government_spending_efficiency", "Eficiencia del gasto"),
        ("government_recession_unemployment_buffer", "Buffer de desempleo para recesion"),
        ("government_recession_output_gap_threshold", "Umbral de brecha para recesion"),
        ("government_countercyclical_support_multiplier_max", "Tope apoyo anticiclico"),
        ("government_countercyclical_procurement_multiplier_max", "Tope compras anticiclicas"),
    ]
    rows = []
    for key, label in selected_keys:
        value = policy_values.get(key)
        if isinstance(value, str):
            rows.append([label, value])
        else:
            rows.append([label, _fmt_pct(value)])
    return rows


def _build_pdf_report(
    *,
    profile: CountryProfile,
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    output_pdf: Path,
    phase_label: str | None = None,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.3 * cm,
        rightMargin=1.3 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    summary_rows, mobility_rows = _summary_tables(annual)
    policy_rows = _policy_table_rows(profile.values)

    with tempfile.TemporaryDirectory(prefix="economy-report-") as tmp_dir:
        tmp = Path(tmp_dir)
        macro_png = tmp / "macro.png"
        public_png = tmp / "public.png"
        expenditure_png = tmp / "expenditure.png"
        education_png = tmp / "education.png"
        _make_macro_figure(annual, macro_png)
        _make_public_sector_figure(annual, public_png)
        _make_expenditure_figure(annual, expenditure_png)
        _make_education_figure(annual, education_png)

        story = [
            Paragraph(
                (
                    f"Reporte macroeconomico simulado - {profile.name}"
                    if phase_label is None
                    else f"Reporte macroeconomico simulado - {profile.name} - {phase_label}"
                ),
                styles["Title"],
            ),
            Spacer(1, 0.3 * cm),
            Paragraph(
                (
                    f"{profile.description} "
                    "Escenario corrido en paralelo con el mismo motor del simulador. "
                    "Las graficas estan agregadas por ano para facilitar lectura en horizontes largos."
                    + (
                        ""
                        if phase_label is None
                        else f" Este PDF resume un tramo parcial de la simulacion: {phase_label}."
                    )
                ),
                styles["BodyText"],
            ),
            Spacer(1, 0.35 * cm),
            Paragraph("Resumen promedio de los ultimos 3 anos", styles["Heading2"]),
            Table(
                [["Metrica", "Valor"]] + summary_rows,
                colWidths=[8.4 * cm, 7.0 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F7FAFC")]),
                    ]
                ),
            ),
            Spacer(1, 0.35 * cm),
            Paragraph("Movilidad y educacion", styles["Heading2"]),
            Table(
                [["Metrica", "Promedio / valor"]] + mobility_rows,
                colWidths=[8.4 * cm, 7.0 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6FFFA")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F7FAFC")]),
                    ]
                ),
            ),
            Spacer(1, 0.35 * cm),
            Paragraph("Supuestos de politica", styles["Heading2"]),
            Table(
                [["Parametro", "Valor"]] + policy_rows,
                colWidths=[10.5 * cm, 4.9 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEFCBF")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F7FAFC")]),
                    ]
                ),
            ),
            PageBreak(),
            Paragraph("Macro y canasta basica", styles["Heading2"]),
            Spacer(1, 0.15 * cm),
            Image(str(macro_png), width=18.0 * cm, height=13.4 * cm),
            Spacer(1, 0.15 * cm),
            Paragraph(
                "Incluye PIB nominal, real y potencial; inflacion, desempleo, salarios e indicadores de canasta esencial producida, comprada y cubierta.",
                styles["Small"],
            ),
            PageBreak(),
            Paragraph("PIB por gasto y deficit fiscal", styles["Heading2"]),
            Spacer(1, 0.15 * cm),
            Image(str(expenditure_png), width=18.0 * cm, height=7.8 * cm),
            Spacer(1, 0.15 * cm),
            Paragraph(
                "Resume la composicion del PIB por gasto y el deficit fiscal en relacion con el producto.",
                styles["Small"],
            ),
            PageBreak(),
            Paragraph("Demografia, gobierno, bancos y banco central", styles["Heading2"]),
            Spacer(1, 0.15 * cm),
            Image(str(public_png), width=18.0 * cm, height=13.4 * cm),
            Spacer(1, 0.15 * cm),
            Paragraph(
                "Se muestran poblacion total, natalidad, mortalidad, mujeres fertiles, balance fiscal, salud bancaria y trayectoria monetaria.",
                styles["Small"],
            ),
            PageBreak(),
            Paragraph("Educacion y movilidad social", styles["Heading2"]),
            Spacer(1, 0.15 * cm),
            Image(str(education_png), width=18.0 * cm, height=13.4 * cm),
            Spacer(1, 0.15 * cm),
            Paragraph(
                "Se resumen matricula, acceso desde hogares pobres, movilidad desde origen bajo canasta y retornos educativos.",
                styles["Small"],
            ),
        ]
        doc.build(story)


def _phase_slice(monthly: pd.DataFrame, start_period: int, end_period: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_monthly = monthly[(monthly["period"] >= start_period) & (monthly["period"] <= end_period)].copy()
    phase_annual = _annualize_core_history(phase_monthly)
    return phase_monthly, phase_annual


def run_country_reports(
    *,
    periods: int,
    households: int,
    firms_per_sector: int,
    seed: int,
    periods_per_year: int,
    output_dir: Path,
    workers: int = 0,
    log_every: int = 25,
    phase_split_period: int = 0,
) -> list[Path]:
    profiles = country_profiles()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_payloads = [
        {
            "scenario_name": scenario_name,
            "months": periods,
            "seed": seed,
            "firms_per_sector": firms_per_sector,
            "households": households,
            "periods_per_year": periods_per_year,
            "policy_values": profile.values,
            "log_every": log_every,
        }
        for scenario_name, profile in profiles.items()
    ]
    max_workers = workers if workers > 0 else min(len(scenario_payloads), max(1, (os.cpu_count() or 1) - 1))
    results: dict[str, pd.DataFrame] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_profile, payload): payload["scenario_name"] for payload in scenario_payloads}
        for future in concurrent.futures.as_completed(future_map):
            scenario_name, frame_json = future.result()
            results[scenario_name] = pd.read_json(StringIO(frame_json), orient="split")

    pdf_paths: list[Path] = []
    for scenario_name, profile in profiles.items():
        monthly = results[scenario_name]
        annual = _annualize_core_history(monthly)
        slug = _slugify(scenario_name)
        monthly.to_csv(output_dir / f"{slug}_mensual.csv", index=False)
        annual.to_csv(output_dir / f"{slug}_anual.csv", index=False)
        pdf_path = output_dir / f"{slug}.pdf"
        _build_pdf_report(
            profile=profile,
            monthly=monthly,
            annual=annual,
            output_pdf=pdf_path,
        )
        pdf_paths.append(pdf_path)
        if 0 < phase_split_period < periods:
            phase_ranges = [
                (1, phase_split_period, 1),
                (phase_split_period + 1, periods, 2),
            ]
            for start_period, end_period, phase_index in phase_ranges:
                phase_monthly, phase_annual = _phase_slice(monthly, start_period, end_period)
                if phase_monthly.empty or phase_annual.empty:
                    continue
                phase_suffix = f"fase_{phase_index}_{start_period}_{end_period}"
                phase_monthly.to_csv(output_dir / f"{slug}_{phase_suffix}_mensual.csv", index=False)
                phase_annual.to_csv(output_dir / f"{slug}_{phase_suffix}_anual.csv", index=False)
                phase_pdf_path = output_dir / f"{slug}_{phase_suffix}.pdf"
                _build_pdf_report(
                    profile=profile,
                    monthly=phase_monthly,
                    annual=phase_annual,
                    output_pdf=phase_pdf_path,
                    phase_label=_phase_label(start_period, end_period, phase_index),
                )
                pdf_paths.append(phase_pdf_path)
    return pdf_paths


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    pdf_paths = run_country_reports(
        periods=args.periods,
        households=args.households,
        firms_per_sector=args.firms_per_sector,
        seed=args.seed,
        periods_per_year=args.periods_per_year,
        output_dir=args.output_dir,
        workers=args.workers,
        log_every=args.log_every,
        phase_split_period=args.phase_split_period,
    )
    print("PDF reports written:")
    for path in pdf_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
