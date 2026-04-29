from __future__ import annotations

from pathlib import Path

import economy_simulator.batch_reports as batch_reports
from economy_simulator.policies import united_states_profile


def main() -> int:
    us_profile = united_states_profile()
    batch_reports.country_profiles = lambda: {us_profile.name: us_profile}

    batch_reports.run_country_reports(
        periods=120,
        households=10000,
        firms_per_sector=40,
        seed=7,
        periods_per_year=12,
        output_dir=Path("output/run_eeuu_10000_h40_p120_f120_hh300"),
        workers=1,
        audit_firms_sample=120,
        audit_families_sample=300,
        log_every=1,
        phase_split_period=0,
        include_csv=False,
        include_pdf=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
