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
        "government_tax_revenue": "sum",
        "government_total_spending": "sum",
        "government_deficit": "sum",
        "government_debt_outstanding": "last",
        "recession_flag": "mean",
        "recession_intensity": "mean",
        "government_countercyclical_spending": "sum",
        "government_countercyclical_support_multiplier": "mean",
        "government_countercyclical_procurement_multiplier": "mean",
        "household_final_consumption_share_gdp": "mean",
        "government_final_consumption_share_gdp": "mean",
        "gross_capital_formation_share_gdp": "mean",
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
    annual["inflation_yoy"] = annual["cpi"].pct_change()
    annual["gdp_growth_yoy"] = annual["gdp_nominal"].pct_change()
    annual["population_growth_yoy"] = annual["population"].pct_change()
    annual["avg_unemployment_rate"] = annual.get("unemployment_rate", pd.Series(dtype=float))
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
            ("inflation_yoy", "Inflacion"),
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
    _plot_line_set(
        axes[0, 0],
        annual,
        "year",
        [
            ("births", "Nacimientos"),
            ("deaths", "Muertes"),
            ("fertile_women", "Mujeres fertiles"),
        ],
        "Personas",
    )
    _style_axis(axes[0, 0], "Natalidad, mortalidad y mujeres fertiles", "Personas")

    ax = axes[0, 1]
    for column, label in [
        ("government_tax_revenue", "Ingresos"),
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
        ("government_deficit_share_gdp", "Deficit / PIB"),
        ("government_tax_revenue", "Ingresos fiscales"),
        ("government_total_spending", "Gasto publico"),
    ]:
        if column in annual.columns:
            ax.plot(annual["year"], annual[column], label=label, linewidth=1.8)
    ax.legend(fontsize=8, loc="best")
    _style_axis(ax, "Deficit fiscal y tamano del Estado", "Tasa o unidades")
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
            ("school_enrollment_share", "Matricula escolar"),
            ("university_enrollment_share", "Matricula universitaria"),
            ("school_completion_share", "Escolaridad completa"),
            ("university_completion_share", "Titulo universitario"),
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
    latest = annual.iloc[-1]
    summary_rows = [
        ["Ano final", str(int(latest["year"]))],
        ["PIB nominal final", _fmt_money(latest.get("gdp_nominal"))],
        ["PIB real final", _fmt_money(latest.get("real_gdp_nominal"))],
        ["PIB potencial final", _fmt_money(latest.get("potential_gdp_nominal"))],
        ["Inflacion final", _fmt_pct(latest.get("inflation_yoy"))],
        ["Desempleo final", _fmt_pct(latest.get("avg_unemployment_rate"))],
        ["IPC final", _fmt_money(latest.get("cpi"))],
        ["Salario promedio final", _fmt_money(latest.get("average_wage"))],
        ["Salario real final", _fmt_money(latest.get("real_average_wage"))],
        ["Cobertura completa final", _fmt_pct(latest.get("full_essential_coverage_share"))],
        ["Patrimonio bancario final", _fmt_money(latest.get("bank_equity"))],
        ["Deficit fiscal final", _fmt_money(latest.get("government_deficit"))],
        ["Deficit fiscal / PIB final", _fmt_pct(latest.get("government_deficit_share_gdp"))],
        ["Intensidad de recesion final", _fmt_pct(latest.get("recession_intensity"))],
        ["Gasto anticiclico final", _fmt_money(latest.get("government_countercyclical_spending"))],
    ]
    mobility_rows = [
        ["Matricula escolar promedio", _fmt_pct(annual["school_enrollment_share"].mean())],
        ["Matricula universitaria promedio", _fmt_pct(annual["university_enrollment_share"].mean())],
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
        ("government_wealth_tax_rate", "Impuesto al patrimonio"),
        ("government_unemployment_benefit_share", "Seguro de desempleo"),
        ("government_child_allowance_share", "Transferencia por hijos"),
        ("public_school_budget_share", "Presupuesto escolar publico"),
        ("public_university_budget_share", "Presupuesto universitario publico"),
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
            Paragraph(f"Reporte macroeconomico simulado - {profile.name}", styles["Title"]),
            Spacer(1, 0.3 * cm),
            Paragraph(
                (
                    f"{profile.description} "
                    "Escenario corrido en paralelo con el mismo motor del simulador. "
                    "Las graficas estan agregadas por ano para facilitar lectura en horizontes largos."
                ),
                styles["BodyText"],
            ),
            Spacer(1, 0.35 * cm),
            Paragraph("Resumen final", styles["Heading2"]),
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
                "Se muestran natalidad, mortalidad, mujeres fertiles, balance fiscal, salud bancaria y trayectoria monetaria.",
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
    )
    print("PDF reports written:")
    for path in pdf_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
