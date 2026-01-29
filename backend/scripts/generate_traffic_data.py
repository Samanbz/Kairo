import argparse
import concurrent.futures
import json
import math
import os
import random
import subprocess
import sys
from functools import partial

import libsumo as traci
import numpy as np
import pandas as pd
import requests
import sumolib
from tqdm import tqdm


def setup_args():
    parser = argparse.ArgumentParser(description="Generate traffic training data pipeline")
    parser.add_argument("--lat", type=float, default=49.869421, help="Center Latitude")
    parser.add_argument("--lon", type=float, default=8.668005, help="Center Longitude")
    parser.add_argument("--dist", type=int, default=5000, help="Radius/Distance in meters to fetch")
    parser.add_argument(
        "--output-dir", type=str, default="sumo_data", help="Directory to store SUMO files"
    )
    parser.add_argument(
        "--output-csv", type=str, default="traffic_training_data.csv", help="Output CSV filename"
    )
    parser.add_argument(
        "--sumo-cmd", type=str, default="sumo", help="SUMO binary (ignored if using libsumo)"
    )
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration in seconds")
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of simulation episodes to run"
    )
    parser.add_argument("--name", type=str, default="network", help="Base name for generated files")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of parallel workers"
    )
    parser.add_argument(
        "--step-length", type=float, default=1.0, help="Simulation step length in seconds"
    )
    return parser.parse_args()


# Traffic Modeling Helpers


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

    return np.clip(intensity, 0.1, 1.0)


def get_sumo_period(intensity: float, min_p=0.4, max_p=5.0) -> float:
    """Converts intensity (1.0=high) to SUMO period (0.4s=high freq)."""
    return min_p + (1.0 - intensity) * (max_p - min_p)


def fetch_map(lat, lon, dist, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    osm_path = os.path.join(output_dir, "map.osm")
    meta_path = os.path.join(output_dir, "map_meta.json")

    # Check for existing map matching parameters
    if os.path.exists(osm_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if (
                math.isclose(meta.get("lat"), lat, rel_tol=1e-5)
                and math.isclose(meta.get("lon"), lon, rel_tol=1e-5)
                and meta.get("dist") == dist
            ):
                print(
                    f"✅ Found existing map matching parameters at '{osm_path}'. Skipping download."
                )
                return osm_path
        except Exception:
            pass  # On error, proceed to fetch

    print(f"🌍 Fetching map data for {lat}, {lon} with size {dist}m...")

    # Earth radius approx 6378137 meters
    # 1 degree lat ~= 111111 meters
    delta_lat = dist / 111111
    delta_lon = dist / (111111 * math.cos(math.radians(lat)))

    north = lat + delta_lat
    south = lat - delta_lat
    east = lon + delta_lon
    west = lon - delta_lon

    if north < south:
        north, south = south, north

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:xml][timeout:180];
    (
      node({south},{west},{north},{east});
      way["highway"]({south},{west},{north},{east});
      relation["highway"]({south},{west},{north},{east});
    );
    out meta;
    >;
    out meta;
    """

    try:
        response = requests.post(overpass_url, data=overpass_query)
        if response.status_code == 200:
            with open(osm_path, "wb") as f:
                f.write(response.content)

            # Save metadata
            with open(meta_path, "w") as f:
                json.dump({"lat": lat, "lon": lon, "dist": dist}, f)

            return osm_path
        else:
            print(f"❌ API Error {response.status_code}:\n{response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        sys.exit(1)


def build_sumo_network(osm_path, output_dir, name):
    print("🛠️  Converting OSM to SUMO Network...")
    net_file = os.path.join(output_dir, f"{name}.net.xml")

    subprocess.run(
        [
            "netconvert",
            "--osm-files",
            osm_path,
            "-o",
            net_file,
            # 1. Geometry Fixes
            "--geometry.remove",  # Remove unneeded shape nodes
            "--roundabouts.guess",  # Fix roundabouts
            "--ramps.guess",  # Fix highway on-ramps
            # 2. Junction Merging (The Magic Fix)
            "--junctions.join",  # Merge close junctions
            "--junctions.join-dist",
            "20",  # Merge nodes within 20m (Aggressive!)
            "--junctions.corner-detail",
            "5",
            # 3. Traffic Light Smarts
            "--tls.guess-signals",
            "--tls.discard-simple",  # Remove lights at tiny intersections
            "--tls.join",  # Merge traffic lights at complex junctions
            "--tls.default-type",
            "actuated",  # Smart lights (green when car detected)
            # 4. Pruning (Remove tiny garbage edges)
            "--remove-edges.isolated",  # Remove islands
            "--keep-edges.min-speed",
            "5",  # Remove walking paths/service roads (<18km/h)
            "--no-warnings",
        ],
        check=True,
    )


def generate_route_file(net_file, output_dir, name, duration, period, unique_id):
    """Generates demand for a specific episode."""
    trips_file = os.path.join(output_dir, f"{name}_{unique_id}.trips.xml")
    routes_file = os.path.join(output_dir, f"{name}_{unique_id}.rou.xml")

    # 2. randomTrips
    random_trips_script = os.path.join(
        os.environ.get("SUMO_HOME", "/usr/share/sumo"), "tools/randomTrips.py"
    )
    if not os.path.exists(random_trips_script):
        # Fallback common location or check path
        random_trips_script = "/usr/share/sumo/tools/randomTrips.py"

    if not os.path.exists(random_trips_script):
        raise FileNotFoundError("Could not find randomTrips.py. Please set SUMO_HOME.")

    subprocess.run(
        [
            sys.executable,
            random_trips_script,
            "-n",
            net_file,
            "-o",
            trips_file,
            "-e",
            str(duration),
            "--period",
            str(period),
            "--validate",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 3. duarouter
    subprocess.run(
        [
            "duarouter",
            "-n",
            net_file,
            "--route-files",
            trips_file,
            "-o",
            routes_file,
            "--ignore-errors",
            "--no-warnings",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return routes_file, trips_file


# Simulation Logic

WEATHER_PROFILES = {
    "Clear": {"speedFactor": 1.0, "accel": 2.6, "decel": 4.5},
    "Rain": {"speedFactor": 0.85, "accel": 2.0, "decel": 3.5},
    "Snow": {"speedFactor": 0.60, "accel": 1.2, "decel": 2.0},
}


def get_static_features(net_file):
    print("   Parsing road network topology...")
    net = sumolib.net.readNet(net_file)
    features = {}
    for edge in net.getEdges():
        edge_id = edge.getID()
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        features[edge_id] = {
            "street_type": edge.getType(),
            "speed_limit": edge.getSpeed(),
            "in_degree": len(from_node.getIncoming()),
            "out_degree": len(to_node.getOutgoing()),
            "lane_count": edge.getLaneNumber(),
        }
    return features


def collect_edge_data(traci_module, edge_id, static_data, weather, day_of_week, hour_of_day):
    if edge_id.startswith(":") or edge_id not in static_data:
        return None

    current_speed = traci_module.edge.getLastStepMeanSpeed(edge_id)
    vehicle_count = traci_module.edge.getLastStepVehicleNumber(edge_id)

    if vehicle_count > 0:
        static = static_data[edge_id]
        max_speed = static["speed_limit"]
        tau = min(current_speed / max_speed, 1.0) if max_speed > 0 else 0
        return {
            "weather": weather,
            "day_of_week": day_of_week,
            "hour_of_day": hour_of_day,
            "edge_id": edge_id,
            "street_type": static["street_type"],
            "in_degree": static["in_degree"],
            "out_degree": static["out_degree"],
            "lane_count": static["lane_count"],
            "max_speed": max_speed,
            "current_speed": current_speed,
            "vehicle_count": vehicle_count,
            "congestion": tau,
        }
    return None


def worker_simulation_task(
    episode_idx,
    net_file,
    output_dir,
    base_name,
    duration,
    static_features,
    step_length,
):
    """
    Worker function executed in a separate process.
    """

    # Randomize parameters for this episodes
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weathers = ["Clear", "Rain", "Snow"]
    weather_weights = [0.7, 0.2, 0.1]

    day = random.choice(days)
    hour = random.randint(0, 23)
    weather = np.random.choice(weathers, p=weather_weights)

    # Unique ID for this process's files
    unique_id = f"{os.getpid()}_{episode_idx}"

    # Logic for traffic intensity
    intensity = get_traffic_intensity(hour, day)
    period = get_sumo_period(intensity)

    # Generate Routes
    try:
        route_file, trips_file = generate_route_file(
            net_file, output_dir, base_name, duration, period, unique_id
        )
    except Exception as e:
        return {"error": str(e)}

    # Output file for this episode (partial)
    partial_csv = os.path.join(output_dir, f"partial_{unique_id}.csv")

    # Start Simulation
    # Note: libsumo doesn't use sumo-cmd, it links directly.
    # args are list of strings
    sumo_args = [
        "sumo",  # Dummy arg 0
        "-n",
        net_file,
        "-r",
        route_file,
        "--no-step-log",
        "true",
        "--no-warnings",
        "true",
        "--time-to-teleport",
        "120",  # Teleport car if stuck for >120s (default is 300s)
        "--collision.action",
        "remove",  # Remove cars if they crash (prevents permablocks)
        "--step-length",
        str(step_length),
        # Avoid creating log files in parallel to prevent write conflicts if same name
        "--log",
        os.devnull,
    ]

    traci.start(sumo_args)

    # Set weather physics
    params = WEATHER_PROFILES[weather]
    try:
        traci.vehicletype.setAccel("DEFAULT_VEHTYPE", params["accel"])
        traci.vehicletype.setDecel("DEFAULT_VEHTYPE", params["decel"])
        traci.vehicletype.setSpeedFactor("DEFAULT_VEHTYPE", params["speedFactor"])
    except Exception:
        print("⚠️ Warning: Could not set vehicle type parameters in SUMO.")

    batch_data = []
    header_written = os.path.exists(partial_csv)

    step = 0
    # Simulate
    # step_length is handled by SUMO. simulationStep() advances by one step.
    # If step-length is 1.0, step+=1. If 0.5, step+=1 means 1 step (0.5s).
    # We want to sample every 10 minutes (600 seconds).
    # steps_per_sample = 600 / step_length
    steps_per_sample = int(600 / step_length)
    if steps_per_sample < 1:
        steps_per_sample = 1

    total_steps = int(duration / step_length)

    try:
        for _ in range(total_steps):
            traci.simulationStep()
            step += 1

            if step % steps_per_sample == 0:
                # Calculate effective hour of day based on elapsed time
                elapsed_seconds = step * step_length
                current_hour = (hour + elapsed_seconds / 3600.0) % 24.0

                edge_list = traci.edge.getIDList()
                for edge_id in edge_list:
                    data = collect_edge_data(
                        traci, edge_id, static_features, weather, day, current_hour
                    )
                    if data:
                        batch_data.append(data)

                # Flush to disk periodically to save memory
                if len(batch_data) > 5000:
                    df = pd.DataFrame(batch_data)
                    df.to_csv(partial_csv, mode="a", header=not header_written, index=False)
                    header_written = True
                    batch_data = []

        # Flush remaining
        if batch_data:
            df = pd.DataFrame(batch_data)
            df.to_csv(partial_csv, mode="a", header=not header_written, index=False)

    finally:
        traci.close()
        # Cleanup route files
        if os.path.exists(route_file):
            os.remove(route_file)
        if os.path.exists(trips_file):
            os.remove(trips_file)

    return partial_csv


def main():
    args = setup_args()

    # 1. Fetch Map
    osm_path = fetch_map(args.lat, args.lon, args.dist, args.output_dir)

    # 2. Build Static Network
    net_file = build_sumo_network(osm_path, args.output_dir, args.name)
    static_features = get_static_features(net_file)

    print(f"🚀 Starting parallel generation: {args.episodes} episodes, {args.workers} workers")
    print(f"   Step Size: {args.step_length}s, Duration: {args.duration}s")

    # 3. Parallel Execution
    tasks = []
    for i in range(args.episodes):
        tasks.append(i)

    # Helper to clean up arguments
    worker_func = partial(
        worker_simulation_task,
        net_file=net_file,
        output_dir=args.output_dir,
        base_name=args.name,
        duration=args.duration,
        static_features=static_features,
        step_length=args.step_length,
    )

    generated_files = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Wrap with tqdm
        future_to_episode = {executor.submit(worker_func, i): i for i in range(args.episodes)}

        for future in tqdm(
            concurrent.futures.as_completed(future_to_episode),
            total=args.episodes,
            desc="Total Progress",
        ):
            try:
                result = future.result()
                if isinstance(result, str) and os.path.exists(result):
                    generated_files.append(result)
                elif isinstance(result, dict) and "error" in result:
                    print(f"⚠️ Episode failed: {result['error']}")
            except Exception as exc:
                print(f"❌ Worker exception: {exc}")

    # 4. Merge Data
    print("💾 Merging csv files...")
    if generated_files:
        target_csv = args.output_csv
        first_file = True

        with open(target_csv, "w") as outfile:
            for fname in tqdm(generated_files, desc="Merging"):
                with open(fname, "r") as infile:
                    header = infile.readline()
                    if first_file:
                        outfile.write(header)
                        first_file = False

                    # Write rest
                    for line in infile:
                        outfile.write(line)

                # Cleanup partial
                os.remove(fname)

    print(f"\n✅ Data Generation Complete. Output: {args.output_csv}")


if __name__ == "__main__":
    main()
