from __future__ import annotations

from statistics import mean


def clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def safe_mean(values: list[float] | tuple[float, ...], default: float = 0.0) -> float:
    if not values:
        return default
    return mean(values)


def gini(values: list[float] | tuple[float, ...]) -> float:
    cleaned = [max(0.0, value) for value in values if value is not None]
    if not cleaned:
        return 0.0

    sorted_values = sorted(cleaned)
    total = sum(sorted_values)
    if total <= 0:
        return 0.0

    weighted_sum = 0.0
    for index, value in enumerate(sorted_values, start=1):
        weighted_sum += index * value

    n = len(sorted_values)
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n

