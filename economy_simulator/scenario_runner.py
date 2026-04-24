from __future__ import annotations

import json
import sys
from dataclasses import replace
from time import perf_counter

import pandas as pd

from .domain import SimulationConfig, SimulationResult
from .engine import EconomySimulation
from .reporting import core_history_frame, family_audit_frame, firm_audit_frame, firm_history_frame


def _with_policy_values(config: SimulationConfig, policy_values: dict[str, float | str]) -> SimulationConfig:
    return replace(config, **policy_values)


def _scenario_sample_seed(base_seed: int, scenario_name: str, salt: str) -> int:
    scenario_offset = sum((index + 1) * ord(char) for index, char in enumerate(f"{scenario_name}:{salt}"))
    return int(base_seed) * 10_003 + scenario_offset


def _sample_audit_entities(
    frame: pd.DataFrame,
    *,
    id_column: str,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    if frame.empty or sample_size <= 0 or id_column not in frame.columns:
        return frame.iloc[0:0].copy()

    unique_ids = pd.Series(frame[id_column].dropna().unique())
    if unique_ids.empty:
        return frame.iloc[0:0].copy()

    target_size = min(int(sample_size), len(unique_ids))
    if target_size >= len(unique_ids):
        sampled_ids = set(unique_ids.tolist())
    else:
        sampled_ids = set(unique_ids.sample(n=target_size, random_state=random_state).tolist())
    sampled = frame[frame[id_column].isin(sampled_ids)].copy()
    sort_columns = [column for column in ("period", "year", "period_in_year", id_column) if column in sampled.columns]
    if sort_columns:
        sampled = sampled.sort_values(sort_columns)
    return sampled.reset_index(drop=True)


def _run_scenario(
    *,
    months: int,
    seed: int,
    firms_per_sector: int,
    households: int,
    periods_per_year: int,
    base_policy_values: dict[str, float | str],
    policy_change_period: int | None,
    shock_policy_values: dict[str, float | str] | None,
    scenario_name: str | None = None,
    log_every: int = 0,
    track_firm_history: bool = False,
    track_family_history: bool = False,
) -> tuple[SimulationResult, EconomySimulation]:
    config = SimulationConfig(
        periods=months,
        households=households,
        seed=seed,
        periods_per_year=periods_per_year,
        firms_per_sector=firms_per_sector,
        track_firm_history=track_firm_history,
        track_family_history=track_family_history,
    )
    config = _with_policy_values(config, base_policy_values)
    simulation = EconomySimulation(config)
    label = scenario_name or "Escenario"
    started = perf_counter()
    if log_every > 0:
        print(
            f"[{label}] inicio: periodos={months} hogares={households} firmas/sector={firms_per_sector}",
            file=sys.stderr,
            flush=True,
        )
    if policy_change_period is None or not shock_policy_values:
        for month in range(1, months + 1):
            simulation.step()
            if log_every > 0 and (month == 1 or month % max(1, log_every) == 0 or month == months):
                latest = simulation.history[-1]
                previous = simulation.history[-2] if len(simulation.history) > 1 else None
                inflation_rate = (
                    (latest.gdp_deflator / previous.gdp_deflator) - 1.0
                    if previous is not None and previous.gdp_deflator > 0.0
                    else 0.0
                )
                elapsed = perf_counter() - started
                print(
                    (
                        f"[{label}] periodo {month}/{months} "
                        f"poblacion={latest.population} desempleo={latest.unemployment_rate:.1%} "
                        f"inflacion={inflation_rate:.1%} pib={latest.gdp_nominal:,.0f} t={elapsed:.1f}s"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            if simulation.history and simulation.history[-1].population <= 0:
                break
        result = SimulationResult(
            config=simulation.config,
            history=simulation.history,
            firm_history=simulation.firm_history,
            households=simulation.households,
            entrepreneurs=simulation.entrepreneurs,
            firms=simulation.firms,
            central_bank=simulation.central_bank,
            banks=simulation.banks,
            government=simulation.government,
            family_history=simulation.family_history,
        )
    else:
        for month in range(1, months + 1):
            if month == policy_change_period:
                simulation.config = _with_policy_values(simulation.config, shock_policy_values)
            simulation.step()
            if log_every > 0 and (month == 1 or month % max(1, log_every) == 0 or month == months):
                latest = simulation.history[-1]
                previous = simulation.history[-2] if len(simulation.history) > 1 else None
                inflation_rate = (
                    (latest.gdp_deflator / previous.gdp_deflator) - 1.0
                    if previous is not None and previous.gdp_deflator > 0.0
                    else 0.0
                )
                elapsed = perf_counter() - started
                print(
                    (
                        f"[{label}] periodo {month}/{months} "
                        f"poblacion={latest.population} desempleo={latest.unemployment_rate:.1%} "
                        f"inflacion={inflation_rate:.1%} pib={latest.gdp_nominal:,.0f} t={elapsed:.1f}s"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        result = SimulationResult(
            config=simulation.config,
            history=simulation.history,
            firm_history=simulation.firm_history,
            households=simulation.households,
            entrepreneurs=simulation.entrepreneurs,
            firms=simulation.firms,
            central_bank=simulation.central_bank,
            banks=simulation.banks,
            government=simulation.government,
            family_history=simulation.family_history,
        )
    if log_every > 0 and result.history:
        latest = result.history[-1]
        print(
            (
                f"[{label}] completado: periodos={latest.period} poblacion={latest.population} "
                f"desempleo={latest.unemployment_rate:.1%} pib={latest.gdp_nominal:,.0f}"
            ),
            file=sys.stderr,
            flush=True,
        )
    return result, simulation


def run_scenario_history(
    *,
    months: int,
    seed: int,
    firms_per_sector: int,
    households: int,
    periods_per_year: int,
    base_policy_values: dict[str, float | str],
    policy_change_period: int | None,
    shock_policy_values: dict[str, float | str] | None,
    scenario_name: str | None = None,
    log_every: int = 0,
) -> str:
    result, _simulation = _run_scenario(
        months=months,
        seed=seed,
        firms_per_sector=firms_per_sector,
        households=households,
        periods_per_year=periods_per_year,
        base_policy_values=base_policy_values,
        policy_change_period=policy_change_period,
        shock_policy_values=shock_policy_values,
        scenario_name=scenario_name,
        log_every=log_every,
        track_firm_history=False,
        track_family_history=False,
    )
    history_df = core_history_frame(
        result.history,
        periods_per_year=result.config.periods_per_year,
        target_unemployment=result.config.target_unemployment,
    )
    return history_df.to_json(orient="split")


def run_scenario_export_bundle(
    *,
    months: int,
    seed: int,
    firms_per_sector: int,
    households: int,
    periods_per_year: int,
    base_policy_values: dict[str, float | str],
    policy_change_period: int | None,
    shock_policy_values: dict[str, float | str] | None,
    scenario_name: str | None = None,
    log_every: int = 0,
    audit_firm_sample_size: int = 0,
    audit_family_sample_size: int = 0,
) -> str:
    result, simulation = _run_scenario(
        months=months,
        seed=seed,
        firms_per_sector=firms_per_sector,
        households=households,
        periods_per_year=periods_per_year,
        base_policy_values=base_policy_values,
        policy_change_period=policy_change_period,
        shock_policy_values=shock_policy_values,
        scenario_name=scenario_name,
        log_every=log_every,
        track_firm_history=True,
        track_family_history=True,
    )
    monthly_df = core_history_frame(
        result.history,
        periods_per_year=result.config.periods_per_year,
        target_unemployment=result.config.target_unemployment,
    )
    firms_df = firm_history_frame(result)
    firm_audit_df = firm_audit_frame(firms_df, monthly_df)
    family_audit_df = family_audit_frame(simulation)
    sample_label = scenario_name or "Escenario"
    if audit_firm_sample_size > 0:
        firm_audit_df = _sample_audit_entities(
            firm_audit_df,
            id_column="firm_id",
            sample_size=audit_firm_sample_size,
            random_state=_scenario_sample_seed(seed, sample_label, "firm_audit"),
        )
    if audit_family_sample_size > 0:
        family_audit_df = _sample_audit_entities(
            family_audit_df,
            id_column="family_id",
            sample_size=audit_family_sample_size,
            random_state=_scenario_sample_seed(seed, sample_label, "family_audit"),
        )
    payload = {
        "monthly": monthly_df.to_json(orient="split"),
        "firm_audit": firm_audit_df.to_json(orient="split"),
        "family_audit": family_audit_df.to_json(orient="split"),
    }
    return json.dumps(payload)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    output = run_scenario_history(**payload)
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
