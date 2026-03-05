import csv
import random
from pathlib import Path

# Darmstadt coordinates
CENTER_LAT = 49.8728
CENTER_LNG = 8.6512
DEPOT_LAT = 49.8900
DEPOT_LNG = 8.6300


def generate_location(is_depot=False):
    if is_depot:
        return {"lat": DEPOT_LAT, "lng": DEPOT_LNG}

    # Gaussian distribution around Darmstadt center
    # 0.02 degrees is roughly ~2km
    lat = random.gauss(CENTER_LAT, 0.02)
    lng = random.gauss(CENTER_LNG, 0.02)

    # Clip coordinates to bounds roughly representing Darmstadt and surroundings
    lat = max(49.75, min(lat, 49.95))
    lng = max(8.55, min(lng, 8.75))

    return {"lat": round(lat, 5), "lng": round(lng, 5)}


def generate_orders(num_orders=150):
    orders = []
    for i in range(num_orders):
        # Weight: lognormal simulates many small packages, fewer large ones
        weight = round(random.lognormvariate(1, 0.8), 1)
        weight = max(1.0, min(weight, 50.0))

        # Distribution of time windows
        window_type = random.choices(["morning", "afternoon", "full_day"], weights=[0.3, 0.3, 0.4])[
            0
        ]

        if window_type == "morning":
            tw_start = 8 * 60
            tw_end = 12 * 60
        elif window_type == "afternoon":
            tw_start = 12 * 60
            tw_end = 16 * 60
        else:
            tw_start = 8 * 60
            tw_end = 18 * 60

        # Add some random variance to the windows (e.g. +/- 30 or 60 mins)
        tw_start += random.randint(-1, 2) * 30
        tw_end += random.randint(-1, 2) * 30

        # Bound to reasonable day hours (06:00 to 20:00 max)
        tw_start = max(6 * 60, min(tw_start, 18 * 60))
        tw_end = max(8 * 60, min(tw_end, 20 * 60))

        if tw_end <= tw_start + 60:
            tw_end = tw_start + 120  # Minimum 2 hour window

        orders.append(
            {
                "id": f"ORD_{i + 1:04d}",
                "location": generate_location(),
                "weight": weight,
                "volume": round(weight * random.uniform(0.5, 1.5), 2),
                "time_window_start": int(tw_start),
                "time_window_end": int(tw_end),
                "service_time": int(
                    random.choices([3, 5, 10, 15], weights=[0.4, 0.4, 0.15, 0.05])[0]
                ),
            }
        )
    return orders


def generate_vehicles(num_vehicles=15):
    vehicles = []
    for i in range(num_vehicles):
        # Most vehicles are standard 1000kg vans, some 500kg cars, few 1500kg trucks
        capacity = random.choices([500.0, 1000.0, 1500.0], weights=[0.2, 0.6, 0.2])[0]

        vehicles.append(
            {
                "id": f"VEH_{i + 1:03d}",
                "start_location": generate_location(is_depot=True),
                "end_location": generate_location(is_depot=True),
                "capacity": float(capacity),
                "shift_start": 8 * 60,  # 08:00 AM
                "shift_end": 18 * 60,  # 18:00 PM
                "speed_factor": round(random.uniform(0.9, 1.1), 2),
            }
        )
    return vehicles


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "data" / "test_sets"
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)  # Set seed for reproducible datasets

    orders = generate_orders(150)
    with open(output_dir / "realistic_orders.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "lat",
                "lng",
                "weight",
                "volume",
                "time_window_start",
                "time_window_end",
                "service_time",
            ]
        )
        for order in orders:
            writer.writerow(
                [
                    order["id"],
                    order["location"]["lat"],
                    order["location"]["lng"],
                    order["weight"],
                    order["volume"],
                    order["time_window_start"],
                    order["time_window_end"],
                    order["service_time"],
                ]
            )

    vehicles = generate_vehicles(15)
    with open(output_dir / "realistic_vehicles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "start_lat",
                "start_lng",
                "end_lat",
                "end_lng",
                "capacity",
                "shift_start",
                "shift_end",
                "speed_factor",
            ]
        )
        for vehicle in vehicles:
            writer.writerow(
                [
                    vehicle["id"],
                    vehicle["start_location"]["lat"],
                    vehicle["start_location"]["lng"],
                    vehicle["end_location"]["lat"],
                    vehicle["end_location"]["lng"],
                    vehicle["capacity"],
                    vehicle["shift_start"],
                    vehicle["shift_end"],
                    vehicle["speed_factor"],
                ]
            )

    print(f"Generated {len(orders)} orders and {len(vehicles)} vehicles.")
    print(f"Output saved to: {output_dir.resolve()}")
