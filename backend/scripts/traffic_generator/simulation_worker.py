import concurrent.futures
import json
import os
import random
import shutil
import tempfile
import traceback

import libsumo as traci
import numpy as np
import pandas as pd

from .modeling import WEATHER_PROFILES, get_sumo_period, get_traffic_intensity
from .network_builder import generate_trips_xml


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
            "betweenness": static["betweenness"],
            "closeness": static["closeness"],
            "max_speed": max_speed,
            "current_speed": current_speed,
            "vehicle_count": vehicle_count,
            "congestion": tau,
        }
    return None


def worker_simulation_task(
    episode_idx, net_file, output_dir, base_name, duration, static_features_path, job_id
):
    """
    Worker function executed in a separate process.
    Run in a unique temporary directory to avoid file collisions.
    """
    try:
        return _worker_simulation_inner(
            episode_idx, net_file, output_dir, base_name, duration, static_features_path, job_id
        )
    except Exception:
        return {"error": traceback.format_exc()}


def _worker_simulation_inner(
    episode_idx, net_file, output_dir, base_name, duration, static_features_path, job_id
):
    # Create a unique temporary directory for this worker
    worker_tmp_dir = tempfile.mkdtemp(prefix=f"sumo_worker_{job_id}_{episode_idx}_{os.getpid()}_")
    original_cwd = os.getcwd()

    try:
        # Move to the temp directory
        os.chdir(worker_tmp_dir)

        # Load static features locally to avoid pickling overhead
        with open(static_features_path, "r") as f:
            static_features = json.load(f)

        # Randomize parameters for this episodes
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weathers = ["Clear", "Rain", "Snow"]
        weather_weights = [0.7, 0.2, 0.1]

        day = random.choice(days)
        hour = random.randint(0, 23)
        weather = np.random.choice(weathers, p=weather_weights)

        # Unique ID for this process's files, including job_id for cluster uniqueness
        unique_id = f"{job_id}_{episode_idx}_{os.getpid()}"

        # Logic for traffic intensity
        intensity = get_traffic_intensity(hour, day)
        period = get_sumo_period(intensity)

        # Generate Routes
        try:
            temp_trips = os.path.abspath("trips.trips.xml")

            trip_count = generate_trips_xml(
                net_file=net_file,
                output_path=temp_trips,
                duration=duration,
                period=period,
                min_distance=1000,
            )

            if trip_count == 0:
                return {"error": "No valid trips generated (check network connectivity)"}

            route_file = temp_trips

        except Exception as e:
            return {"error": str(e)}

        # Output file for this episode (partial)
        partial_csv = os.path.join(output_dir, f"partial_{unique_id}.csv")

        # Start Simulation
        STEP_LENGTH = 1.0

        sumo_args = [
            "sumo",
            "-n",
            net_file,
            "-r",
            route_file,
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--ignore-route-errors",
            "--time-to-teleport",
            "30",
            "--collision.action",
            "remove",
            "--step-length",
            str(STEP_LENGTH),
            "--default.action-step-length",
            str(STEP_LENGTH),
            "--routing-algorithm",
            "dijkstra",
            "--end",
            str(duration),
            "--log",
            os.devnull,
        ]

        traci.start(sumo_args)

        # Set vehicle types and weather physics
        try:
            w_factor = WEATHER_PROFILES[weather]["accel"] / 2.6

            traci.vehicletype.copy("DEFAULT_VEHTYPE", "passenger")
            traci.vehicletype.setAccel("passenger", 2.6 * w_factor)
            traci.vehicletype.setImperfection("passenger", 0.5)
            traci.vehicletype.setParameter("passenger", "device.rerouting.probability", "1.0")
            traci.vehicletype.setParameter("passenger", "device.rerouting.period", "60")

            traci.vehicletype.copy("DEFAULT_VEHTYPE", "truck")
            traci.vehicletype.setLength("truck", 12.0)
            traci.vehicletype.setAccel("truck", 1.2 * w_factor)
            traci.vehicletype.setParameter("truck", "device.rerouting.probability", "1.0")
            traci.vehicletype.setParameter("truck", "device.rerouting.period", "60")
        except Exception as e:
            print(f"⚠️ Warning: Could not set vehicle types in SUMO: {e}")

        all_edges = traci.edge.getIDList()

        valid_edges = [e for e in all_edges if not e.startswith(":") and e in static_features]
        num_edges = len(valid_edges)

        batch_data = []
        header_written = os.path.exists(partial_csv)

        io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        io_futures = []

        def flush_data(data, filepath, write_header):
            df = pd.DataFrame(data)
            df.to_csv(filepath, mode="a", header=write_header, index=False)

        AGGREGATION_PERIOD = 600.0
        current_time = 0.0
        last_flush_time = 0.0

        stats_s_sum = np.zeros(num_edges, dtype=np.float32)
        stats_cnt = np.zeros(num_edges, dtype=np.int32)
        stats_max_occ = np.zeros(num_edges, dtype=np.int16)

        while current_time < duration:
            traci.simulationStep()
            current_time += STEP_LENGTH

            for veh_id in traci.simulation.getDepartedIDList():
                if random.random() < 0.15:
                    traci.vehicle.setType(veh_id, "truck")
                else:
                    traci.vehicle.setType(veh_id, "passenger")

            for e_idx, e in enumerate(valid_edges):
                n_veh = traci.edge.getLastStepVehicleNumber(e)
                if n_veh > 0:
                    v_speed = traci.edge.getLastStepMeanSpeed(e)
                    stats_s_sum[e_idx] += v_speed
                    stats_cnt[e_idx] += 1
                    if n_veh > stats_max_occ[e_idx]:
                        stats_max_occ[e_idx] = n_veh

            if current_time - last_flush_time >= AGGREGATION_PERIOD:
                current_hour = (hour + current_time / 3600.0) % 24.0

                active_indices = np.where(stats_cnt > 0)[0]

                for idx in active_indices:
                    e = valid_edges[idx]
                    avg_speed = stats_s_sum[idx] / stats_cnt[idx]
                    max_occ = stats_max_occ[idx]

                    static = static_features[e]
                    max_spd = static["speed_limit"]
                    tau = min(avg_speed / max_spd, 1.0) if max_spd > 0 else 0

                    rec = {
                        "weather": weather,
                        "day_of_week": day,
                        "hour_of_day": current_hour,
                        "edge_id": e,
                        "street_type": static["street_type"],
                        "in_degree": static["in_degree"],
                        "out_degree": static["out_degree"],
                        "lane_count": static["lane_count"],
                        "betweenness": static["betweenness"],
                        "closeness": static["closeness"],
                        "max_speed": max_spd,
                        "current_speed": float(avg_speed),
                        "vehicle_count": int(max_occ),
                        "congestion": float(tau),
                    }
                    batch_data.append(rec)

                stats_s_sum.fill(0)
                stats_cnt.fill(0)
                stats_max_occ.fill(0)

                if len(batch_data) > 0:
                    io_futures.append(
                        io_executor.submit(flush_data, batch_data, partial_csv, not header_written)
                    )
                    header_written = True
                    batch_data = []

                last_flush_time = current_time

        if batch_data:
            io_futures.append(
                io_executor.submit(flush_data, batch_data, partial_csv, not header_written)
            )

        for f in io_futures:
            f.result()

        io_executor.shutdown()

    finally:
        try:
            traci.close()
        except Exception:
            pass

        if "route_file" in locals() and os.path.exists(route_file):
            try:
                os.remove(route_file)
            except OSError:
                pass

        os.chdir(original_cwd)
        try:
            shutil.rmtree(worker_tmp_dir)
        except OSError:
            pass

    return partial_csv
