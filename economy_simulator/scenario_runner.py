from __future__ import annotations

import json
import sys
from dataclasses import replace
from time import perf_counter

from .domain import SimulationConfig, SimulationResult
from .engine import EconomySimulation
from .reporting import core_history_frame


def _with_policy_values(config: SimulationConfig, policy_values: dict[str, float | str]) -> SimulationConfig:
    return replace(config, **policy_values)


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
    config = SimulationConfig(
        periods=months,
        households=households,
        seed=seed,
        periods_per_year=periods_per_year,
        firms_per_sector=firms_per_sector,
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
                elapsed = perf_counter() - started
                print(
                    (
                        f"[{label}] periodo {month}/{months} "
                        f"poblacion={latest.population} desempleo={latest.unemployment_rate:.1%} "
                        f"pib={latest.gdp_nominal:,.0f} t={elapsed:.1f}s"
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
        )
    else:
        for month in range(1, months + 1):
            if month == policy_change_period:
                simulation.config = _with_policy_values(simulation.config, shock_policy_values)
            simulation.step()
            if log_every > 0 and (month == 1 or month % max(1, log_every) == 0 or month == months):
                latest = simulation.history[-1]
                elapsed = perf_counter() - started
                print(
                    (
                        f"[{label}] periodo {month}/{months} "
                        f"poblacion={latest.population} desempleo={latest.unemployment_rate:.1%} "
                        f"pib={latest.gdp_nominal:,.0f} t={elapsed:.1f}s"
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
    history_df = core_history_frame(
        result.history,
        periods_per_year=result.config.periods_per_year,
        target_unemployment=result.config.target_unemployment,
    )
    return history_df.to_json(orient="split")


def main() -> int:
    payload = json.loads(sys.stdin.read())
    output = run_scenario_history(**payload)
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
