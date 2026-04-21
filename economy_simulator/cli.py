from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .domain import SimulationConfig
from .engine import EconomySimulation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the economy simulator.")
    parser.add_argument("--periods", type=int, default=120, help="Number of periods to simulate.")
    parser.add_argument("--households", type=int, default=5000, help="Number of worker households.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=12,
        help="Number of simulation periods that make one macroeconomic year.",
    )
    parser.add_argument(
        "--firms-per-sector",
        type=int,
        default=20,
        help="Number of firms to create in each sector at startup.",
    )
    parser.add_argument("--output", type=Path, help="Write the period history to JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--log-every", type=int, default=10, help="Print one line every N periods.")
    return parser


def _print_snapshot(snapshot) -> None:
    print(
        f"Period {snapshot.period:>4}: "
        f"population={snapshot.population:4d} "
        f"children={snapshot.children:4d} "
        f"adults={snapshot.adults:4d} "
        f"seniors={snapshot.seniors:4d} "
        f"labor_force={snapshot.labor_force:4d} "
        f"guarded_children={snapshot.children_with_guardian:4d} "
        f"orphans={snapshot.orphans:4d} "
        f"employment={snapshot.employment_rate:5.1%} "
        f"unemployment={snapshot.unemployment_rate:5.1%} "
        f"births={snapshot.births:2d} "
        f"deaths={snapshot.deaths:2d} "
        f"output={snapshot.total_production_units:8.2f} "
        f"sales={snapshot.total_sales_revenue:8.2f} "
        f"profit={snapshot.total_profit:8.2f} "
        f"gini={snapshot.gini_household_savings:4.2f}"
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = SimulationConfig(
        periods=args.periods,
        households=args.households,
        seed=args.seed,
        periods_per_year=args.periods_per_year,
        firms_per_sector=args.firms_per_sector,
    )
    sim = EconomySimulation(config)

    for _ in range(config.periods):
        snapshot = sim.step()
        if not args.quiet and (snapshot.period == 1 or snapshot.period % max(1, args.log_every) == 0):
            _print_snapshot(snapshot)

    final = sim.history[-1]
    print(
        "\nFinal summary\n"
        f"  periods: {config.periods}\n"
        f"  periods per year: {config.periods_per_year}\n"
        f"  firms per sector: {config.firms_per_sector}\n"
        f"  households: {config.households}\n"
        f"  year: {final.year}\n"
        f"  period in year: {final.period_in_year}\n"
        f"  population: {final.population}\n"
        f"  children: {final.children}\n"
        f"  adults: {final.adults}\n"
        f"  seniors: {final.seniors}\n"
        f"  labor force: {final.labor_force}\n"
        f"  children with guardian: {final.children_with_guardian}\n"
        f"  orphans: {final.orphans}\n"
        f"  unemployment: {final.unemployment_rate:5.1%}\n"
        f"  births in last period: {final.births}\n"
        f"  deaths in last period: {final.deaths}\n"
        f"  monthly GDP: {final.gdp_nominal:,.2f}\n"
        f"  GDP per capita: {final.gdp_per_capita:,.2f}\n"
        f"  average age: {final.average_age:,.1f}\n"
        f"  total wages: {final.total_wages:,.2f}\n"
        f"  total sales revenue: {final.total_sales_revenue:,.2f}\n"
        f"  total investment spending: {final.period_investment_spending:,.2f}\n"
        f"  recycled business costs: {final.business_cost_recycled:,.2f}\n"
        f"  total liquid money: {final.total_liquid_money:,.2f}\n"
        f"  capital stock: {final.total_capital_stock:,.2f}\n"
        f"  inventory stock: {final.total_inventory_units:,.2f}\n"
        f"  total profit: {final.total_profit:,.2f}\n"
        f"  household savings: {final.total_household_savings:,.2f}\n"
        f"  household savings gini: {final.gini_household_savings:4.2f}\n"
        f"  owner wealth gini: {final.gini_owner_wealth:4.2f}\n"
    )

    if args.output:
        payload = {
            "config": asdict(config),
            "history": [asdict(snapshot) for snapshot in sim.history],
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"History written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
