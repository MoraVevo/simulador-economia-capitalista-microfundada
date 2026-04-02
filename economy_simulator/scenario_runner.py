from __future__ import annotations

import json
import sys
from dataclasses import replace

from .domain import SimulationConfig, SimulationResult
from .engine import EconomySimulation
from .reporting import simulation_frames


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
    if policy_change_period is None or not shock_policy_values:
        result = simulation.run()
    else:
        for month in range(1, months + 1):
            if month == policy_change_period:
                simulation.config = _with_policy_values(simulation.config, shock_policy_values)
            simulation.step()
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
    history_df, _ = simulation_frames(result)
    return history_df.to_json(orient="split")


def main() -> int:
    payload = json.loads(sys.stdin.read())
    output = run_scenario_history(**payload)
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
