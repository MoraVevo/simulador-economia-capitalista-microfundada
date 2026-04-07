from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _clamp_numba(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@njit(cache=True)
def compute_household_baseline_demand_arrays(
    age_years: np.ndarray,
    need_scale: np.ndarray,
    food_share: np.ndarray,
    housing_share: np.ndarray,
    clothing_share: np.ndarray,
    manufactured_share: np.ndarray,
    leisure_share: np.ndarray,
    *,
    entry_age_years: float,
    senior_age_years: float,
    max_age_years: float,
    child_consumption_multiplier: float,
    senior_consumption_multiplier: float,
    discretionary_scale: float,
    food_price: float,
    housing_price: float,
    clothing_price: float,
) -> tuple[np.ndarray, np.ndarray]:
    household_count = len(age_years)
    desired_units = np.empty((household_count, 5), dtype=np.float64)
    essential_budgets = np.empty(household_count, dtype=np.float64)

    senior_span = max(1.0, max_age_years - senior_age_years)
    child_span = max(1.0, entry_age_years)

    for index in range(household_count):
        age = age_years[index]
        if age < entry_age_years:
            progress = _clamp_numba(age / child_span, 0.0, 1.0)
            consumption_multiplier = _clamp_numba(
                child_consumption_multiplier
                + (1.0 - child_consumption_multiplier) * progress,
                child_consumption_multiplier,
                1.0,
            )
        elif age < senior_age_years:
            consumption_multiplier = 1.0
        elif age >= max_age_years:
            consumption_multiplier = senior_consumption_multiplier
        else:
            progress = _clamp_numba((age - senior_age_years) / senior_span, 0.0, 1.0)
            consumption_multiplier = _clamp_numba(
                senior_consumption_multiplier - 0.10 * progress,
                0.65,
                1.0,
            )

        base_units = need_scale[index] * consumption_multiplier
        food_units = base_units * food_share[index]
        housing_units = base_units * housing_share[index]
        clothing_units = base_units * clothing_share[index]
        manufactured_units = base_units * discretionary_scale * manufactured_share[index]
        leisure_units = base_units * discretionary_scale * leisure_share[index]

        desired_units[index, 0] = food_units
        desired_units[index, 1] = housing_units
        desired_units[index, 2] = clothing_units
        desired_units[index, 3] = manufactured_units
        desired_units[index, 4] = leisure_units
        essential_budgets[index] = (
            food_units * food_price
            + housing_units * housing_price
            + clothing_units * clothing_price
        )

    return desired_units, essential_budgets
