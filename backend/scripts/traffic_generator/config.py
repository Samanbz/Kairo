import argparse
import os


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

    # Cluster & Robustness Args
    default_workers = os.cpu_count() or 1

    parser.add_argument(
        "--workers", type=int, default=default_workers, help="Number of parallel workers"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default="local",
        help="Unique Identifier for this job/node",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip map download and network build (prevents race conditions on shared FS)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging of partial files (useful for massive datasets)",
    )

    return parser.parse_args()
