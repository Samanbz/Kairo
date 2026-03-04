"""
Traffic modeling helpers and logic.
"""

import numpy as np

WEATHER_PROFILES = {
    "Clear": {"speedFactor": 1.0, "accel": 2.6, "decel": 4.5},
    "Rain": {"speedFactor": 0.85, "accel": 2.0, "decel": 3.5},
    "Snow": {"speedFactor": 0.60, "accel": 1.2, "decel": 2.0},
}


def get_traffic_intensity(hour_of_day: float, day_of_week: str) -> float:
    """Returns a traffic intensity factor (0.1 to 1.0) based on time of day and day of week."""
    is_weekend = day_of_week in ["Saturday", "Sunday"]

    if is_weekend:
        # Single peak at 1pm (13:00) for weekends
        peak = np.exp(-((hour_of_day - 13) ** 2) / (2 * 2.5**2))
        base_load = 0.2
        # Scale to max ~0.9
        intensity = base_load + 0.7 * peak
    else:
        # Weekday: Morning Peak (8am), Evening Peak (5pm)
        morning_peak = np.exp(-((hour_of_day - 8) ** 2) / (2 * 2**2))
        evening_peak = np.exp(-((hour_of_day - 17) ** 2) / (2 * 2**2))

        base_load = 0.15
        intensity = base_load + 0.85 * max(morning_peak, evening_peak)

    return float(np.clip(intensity, 0.1, 1.0))


def get_sumo_period(intensity: float, min_p=0.02, max_p=0.8) -> float:
    """Converts intensity to SUMO period (period is seconds per vehicle)."""
    return min_p + (1.0 - intensity) * (max_p - min_p)
