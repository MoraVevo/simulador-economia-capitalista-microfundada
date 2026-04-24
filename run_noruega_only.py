from __future__ import annotations

from pathlib import Path

import economy_simulator.batch_reports as batch_reports
from economy_simulator.policies import norway_profile


def main() -> int:
    no_profile = norway_profile()
    batch_reports.country_profiles = lambda: {no_profile.name: no_profile}

    batch_reports.run_country_reports(
        periods=30,
        households=10000,
        firms_per_sector=40,
        seed=7,
        periods_per_year=12,
        output_dir=Path("output/run_noruega_10000_h40_p30"),
        workers=1,
        audit_firms_sample=40,
        audit_families_sample=40,
        log_every=1,
        phase_split_period=0,
        include_csv=False,
        include_pdf=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
