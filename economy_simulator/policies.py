from __future__ import annotations

from dataclasses import dataclass

from .domain import SimulationConfig


DEFAULT_POLICY_CONFIG = SimulationConfig()


@dataclass(frozen=True)
class CountryProfile:
    name: str
    short_label: str
    description: str
    values: dict[str, float | str]


def default_policy_values() -> dict[str, float | str]:
    return {
        "target_unemployment": DEFAULT_POLICY_CONFIG.target_unemployment,
        "birth_interval_periods": DEFAULT_POLICY_CONFIG.birth_interval_periods,
        "annual_birth_rate": DEFAULT_POLICY_CONFIG.annual_birth_rate,
        "annual_birth_rate_capable_single": DEFAULT_POLICY_CONFIG.annual_birth_rate_capable_single,
        "annual_birth_rate_capable_partnered": DEFAULT_POLICY_CONFIG.annual_birth_rate_capable_partnered,
        "annual_birth_rate_noncapable": DEFAULT_POLICY_CONFIG.annual_birth_rate_noncapable,
        "initial_school_completion_share": DEFAULT_POLICY_CONFIG.initial_school_completion_share,
        "initial_university_completion_share": DEFAULT_POLICY_CONFIG.initial_university_completion_share,
        "public_school_budget_share": DEFAULT_POLICY_CONFIG.public_school_budget_share,
        "public_university_budget_share": DEFAULT_POLICY_CONFIG.public_university_budget_share,
        "initial_private_school_firms": DEFAULT_POLICY_CONFIG.initial_private_school_firms,
        "initial_private_university_firms": DEFAULT_POLICY_CONFIG.initial_private_university_firms,
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
        "bank_household_max_debt_to_income": DEFAULT_POLICY_CONFIG.bank_household_max_debt_to_income,
        "bank_household_max_interest_share": DEFAULT_POLICY_CONFIG.bank_household_max_interest_share,
        "bank_firm_min_interest_coverage": DEFAULT_POLICY_CONFIG.bank_firm_min_interest_coverage,
        "bank_firm_max_debt_to_revenue": DEFAULT_POLICY_CONFIG.bank_firm_max_debt_to_revenue,
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
        "government_countercyclical_enabled": DEFAULT_POLICY_CONFIG.government_countercyclical_enabled,
        "government_recession_unemployment_buffer": DEFAULT_POLICY_CONFIG.government_recession_unemployment_buffer,
        "government_recession_output_gap_threshold": DEFAULT_POLICY_CONFIG.government_recession_output_gap_threshold,
        "government_recession_lookback_periods": DEFAULT_POLICY_CONFIG.government_recession_lookback_periods,
        "government_countercyclical_transfer_weight": DEFAULT_POLICY_CONFIG.government_countercyclical_transfer_weight,
        "government_countercyclical_procurement_weight": DEFAULT_POLICY_CONFIG.government_countercyclical_procurement_weight,
        "government_countercyclical_support_multiplier_max": (
            DEFAULT_POLICY_CONFIG.government_countercyclical_support_multiplier_max
        ),
        "government_countercyclical_procurement_multiplier_max": (
            DEFAULT_POLICY_CONFIG.government_countercyclical_procurement_multiplier_max
        ),
    }


def guatemala_profile() -> CountryProfile:
    values = default_policy_values()
    values.update(
        {
            "target_unemployment": 0.05,
            "birth_interval_periods": 8,
            "annual_birth_rate": 0.18,
            "annual_birth_rate_capable_single": 0.11,
            "annual_birth_rate_capable_partnered": 0.56,
            "annual_birth_rate_noncapable": 0.035,
            "initial_school_completion_share": 0.62,
            "initial_university_completion_share": 0.12,
            "public_school_budget_share": 0.015,
            "public_university_budget_share": 0.010,
            "initial_private_school_firms": 4,
            "initial_private_university_firms": 1,
            "central_bank_goods_growth_pass_through": 0.75,
            "central_bank_target_velocity": 0.45,
            "central_bank_target_annual_inflation": 0.045,
            "central_bank_policy_rate_base": 0.07,
            "central_bank_policy_rate_floor": 0.03,
            "central_bank_policy_rate_ceiling": 0.18,
            "central_bank_productivity_dividend_share": 0.25,
            "reserve_ratio": 0.05,
            "bank_bond_allocation_share": 0.20,
            "bank_deposit_rate_share": 0.15,
            "bank_household_max_debt_to_income": 5.0,
            "bank_household_max_interest_share": 0.28,
            "bank_firm_min_interest_coverage": 1.25,
            "bank_firm_max_debt_to_revenue": 4.5,
            "government_corporate_tax_rate_low": 0.05,
            "government_corporate_tax_rate_mid": 0.10,
            "government_corporate_tax_rate_high": 0.15,
            "government_dividend_tax_rate_low": 0.02,
            "government_dividend_tax_rate_mid": 0.05,
            "government_dividend_tax_rate_high": 0.08,
            "government_wealth_tax_rate": 0.000,
            "government_wealth_tax_threshold_multiple": 40.0,
            "government_unemployment_benefit_share": 0.05,
            "government_child_allowance_share": 0.05,
            "government_basic_support_gap_share": 0.08,
            "government_procurement_gap_share": 0.08,
            "government_procurement_price_sensitivity": 1.60,
            "government_spending_scale": 0.35,
            "government_spending_efficiency": 0.55,
            "government_recession_unemployment_buffer": 0.05,
            "government_recession_output_gap_threshold": 0.07,
            "government_recession_lookback_periods": 4,
            "government_countercyclical_transfer_weight": 0.45,
            "government_countercyclical_procurement_weight": 0.55,
            "government_countercyclical_support_multiplier_max": 1.40,
            "government_countercyclical_procurement_multiplier_max": 1.55,
        }
    )
    return CountryProfile(
        name="Guatemala (mas liberal)",
        short_label="Guatemala",
        description=(
            "Perfil con menor presion tributaria, red de proteccion mas delgada, "
            "intermediacion financiera mas corta y base educativa inicial mas baja."
        ),
        values=values,
    )


def united_states_profile() -> CountryProfile:
    values = default_policy_values()
    values.update(
        {
            "target_unemployment": 0.045,
            "birth_interval_periods": 10,
            "annual_birth_rate": 0.12,
            "annual_birth_rate_capable_single": 0.08,
            "annual_birth_rate_capable_partnered": 0.38,
            "annual_birth_rate_noncapable": 0.025,
            "initial_school_completion_share": 0.90,
            "initial_university_completion_share": 0.38,
            "public_school_budget_share": 0.018,
            "public_university_budget_share": 0.012,
            "initial_private_school_firms": 10,
            "initial_private_university_firms": 3,
            "central_bank_goods_growth_pass_through": 0.85,
            "central_bank_target_velocity": 0.28,
            "central_bank_target_annual_inflation": 0.020,
            "central_bank_policy_rate_base": 0.045,
            "central_bank_policy_rate_floor": 0.01,
            "central_bank_policy_rate_ceiling": 0.12,
            "central_bank_productivity_dividend_share": 0.38,
            "reserve_ratio": 0.10,
            "bank_bond_allocation_share": 0.35,
            "bank_deposit_rate_share": 0.35,
            "bank_household_max_debt_to_income": 9.5,
            "bank_household_max_interest_share": 0.40,
            "bank_firm_min_interest_coverage": 1.10,
            "bank_firm_max_debt_to_revenue": 7.0,
            "government_corporate_tax_rate_low": 0.12,
            "government_corporate_tax_rate_mid": 0.20,
            "government_corporate_tax_rate_high": 0.26,
            "government_dividend_tax_rate_low": 0.06,
            "government_dividend_tax_rate_mid": 0.12,
            "government_dividend_tax_rate_high": 0.18,
            "government_wealth_tax_rate": 0.000,
            "government_wealth_tax_threshold_multiple": 30.0,
            "government_unemployment_benefit_share": 0.22,
            "government_child_allowance_share": 0.10,
            "government_basic_support_gap_share": 0.18,
            "government_procurement_gap_share": 0.22,
            "government_procurement_price_sensitivity": 1.05,
            "government_spending_scale": 0.75,
            "government_spending_efficiency": 0.72,
            "government_recession_unemployment_buffer": 0.03,
            "government_recession_output_gap_threshold": 0.05,
            "government_recession_lookback_periods": 3,
            "government_countercyclical_transfer_weight": 0.65,
            "government_countercyclical_procurement_weight": 0.80,
            "government_countercyclical_support_multiplier_max": 1.85,
            "government_countercyclical_procurement_multiplier_max": 2.10,
        }
    )
    return CountryProfile(
        name="Estados Unidos (mixto)",
        short_label="Estados Unidos",
        description=(
            "Perfil con mercado financiero mas profundo, mayor educacion inicial, "
            "estado de bienestar intermedio y presencia privada mas fuerte en educacion superior."
        ),
        values=values,
    )


def norway_profile() -> CountryProfile:
    values = default_policy_values()
    values.update(
        {
            "target_unemployment": 0.035,
            "birth_interval_periods": 11,
            "annual_birth_rate": 0.10,
            "annual_birth_rate_capable_single": 0.06,
            "annual_birth_rate_capable_partnered": 0.30,
            "annual_birth_rate_noncapable": 0.02,
            "initial_school_completion_share": 0.96,
            "initial_university_completion_share": 0.42,
            "public_school_budget_share": 0.022,
            "public_university_budget_share": 0.020,
            "initial_private_school_firms": 2,
            "initial_private_university_firms": 1,
            "central_bank_goods_growth_pass_through": 0.95,
            "central_bank_target_velocity": 0.30,
            "central_bank_target_annual_inflation": 0.020,
            "central_bank_policy_rate_base": 0.03,
            "central_bank_policy_rate_floor": 0.01,
            "central_bank_policy_rate_ceiling": 0.10,
            "central_bank_productivity_dividend_share": 0.60,
            "reserve_ratio": 0.14,
            "bank_bond_allocation_share": 0.45,
            "bank_deposit_rate_share": 0.55,
            "bank_household_max_debt_to_income": 7.5,
            "bank_household_max_interest_share": 0.30,
            "bank_firm_min_interest_coverage": 1.20,
            "bank_firm_max_debt_to_revenue": 5.5,
            "government_corporate_tax_rate_low": 0.20,
            "government_corporate_tax_rate_mid": 0.28,
            "government_corporate_tax_rate_high": 0.35,
            "government_dividend_tax_rate_low": 0.15,
            "government_dividend_tax_rate_mid": 0.22,
            "government_dividend_tax_rate_high": 0.30,
            "government_wealth_tax_rate": 0.012,
            "government_wealth_tax_threshold_multiple": 12.0,
            "government_unemployment_benefit_share": 0.60,
            "government_child_allowance_share": 0.30,
            "government_basic_support_gap_share": 0.65,
            "government_procurement_gap_share": 0.75,
            "government_procurement_price_sensitivity": 0.70,
            "government_spending_scale": 1.35,
            "government_spending_efficiency": 0.90,
            "government_recession_unemployment_buffer": 0.02,
            "government_recession_output_gap_threshold": 0.04,
            "government_recession_lookback_periods": 3,
            "government_countercyclical_transfer_weight": 0.80,
            "government_countercyclical_procurement_weight": 0.95,
            "government_countercyclical_support_multiplier_max": 2.20,
            "government_countercyclical_procurement_multiplier_max": 2.60,
        }
    )
    return CountryProfile(
        name="Noruega (economia del bienestar)",
        short_label="Noruega",
        description=(
            "Perfil con mayor carga tributaria, red publica mas robusta, "
            "educacion inicial alta y banca mas prudente con mayor estabilizacion."
        ),
        values=values,
    )


def social_state_intensive_profile() -> CountryProfile:
    values = default_policy_values()
    values.update(
        {
            "target_unemployment": 0.030,
            "birth_interval_periods": 11,
            "annual_birth_rate": 0.095,
            "annual_birth_rate_capable_single": 0.055,
            "annual_birth_rate_capable_partnered": 0.28,
            "annual_birth_rate_noncapable": 0.018,
            "initial_school_completion_share": 0.975,
            "initial_university_completion_share": 0.50,
            "public_school_budget_share": 0.028,
            "public_university_budget_share": 0.025,
            "initial_private_school_firms": 1,
            "initial_private_university_firms": 1,
            "central_bank_goods_growth_pass_through": 0.92,
            "central_bank_target_velocity": 0.24,
            "central_bank_target_annual_inflation": 0.018,
            "central_bank_policy_rate_base": 0.025,
            "central_bank_policy_rate_floor": 0.005,
            "central_bank_policy_rate_ceiling": 0.08,
            "central_bank_productivity_dividend_share": 0.70,
            "reserve_ratio": 0.18,
            "bank_bond_allocation_share": 0.55,
            "bank_deposit_rate_share": 0.62,
            "bank_household_max_debt_to_income": 6.5,
            "bank_household_max_interest_share": 0.26,
            "bank_firm_min_interest_coverage": 1.30,
            "bank_firm_max_debt_to_revenue": 4.8,
            "government_corporate_tax_rate_low": 0.24,
            "government_corporate_tax_rate_mid": 0.33,
            "government_corporate_tax_rate_high": 0.42,
            "government_dividend_tax_rate_low": 0.18,
            "government_dividend_tax_rate_mid": 0.27,
            "government_dividend_tax_rate_high": 0.36,
            "government_wealth_tax_rate": 0.018,
            "government_wealth_tax_threshold_multiple": 10.0,
            "government_unemployment_benefit_share": 0.75,
            "government_child_allowance_share": 0.40,
            "government_basic_support_gap_share": 0.80,
            "government_procurement_gap_share": 0.90,
            "government_procurement_price_sensitivity": 0.60,
            "government_spending_scale": 1.65,
            "government_spending_efficiency": 0.95,
            "government_recession_unemployment_buffer": 0.015,
            "government_recession_output_gap_threshold": 0.035,
            "government_recession_lookback_periods": 2,
            "government_countercyclical_transfer_weight": 0.95,
            "government_countercyclical_procurement_weight": 1.05,
            "government_countercyclical_support_multiplier_max": 2.60,
            "government_countercyclical_procurement_multiplier_max": 3.00,
        }
    )
    return CountryProfile(
        name="Estado social intensivo (benchmark)",
        short_label="Benchmark social",
        description=(
            "Perfil sintetico de intervencion social alta: mayor gasto y tributacion, "
            "educacion publica mas amplia, transferencias mas profundas y banca mas prudente."
        ),
        values=values,
    )


def country_profiles() -> dict[str, CountryProfile]:
    profiles = [
        guatemala_profile(),
        united_states_profile(),
        norway_profile(),
        social_state_intensive_profile(),
    ]
    return {profile.name: profile for profile in profiles}


def scenario_policy_presets() -> dict[str, dict[str, float | str]]:
    return {name: profile.values for name, profile in country_profiles().items()}
