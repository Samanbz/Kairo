import multiprocessing
import os
import sys
from functools import partial

from tqdm import tqdm

from .config import setup_args
from .data_aggregator import merge_csv_files
from .graph_features import get_static_features
from .map_provider import fetch_map
from .network_builder import build_sumo_network
from .simulation_worker import worker_simulation_task


def main():
    args = setup_args()

    # 1. Fetch Map (Only if root/setup is allowed)
    if not args.skip_setup:
        osm_path = fetch_map(args.lat, args.lon, args.dist, args.output_dir)
        # 2. Build Static Network
        net_file = build_sumo_network(osm_path, args.output_dir, args.name)
    else:
        # If skipping setup, assume files exist
        net_file = os.path.join(args.output_dir, f"{args.name}.net.xml")
        if not os.path.exists(net_file):
            print(f"❌ Error: {net_file} not found. Run without --skip-setup first.")
            sys.exit(1)
        print("⏭️  Skipping map download and build (--skip-setup active).")

    # Load from cache if available
    # Only ensure cache exists, do not load large dict into main process memory if not needed
    get_static_features(net_file, args.output_dir)
    static_features_path = os.path.join(args.output_dir, "static_features_cache.json")

    print(
        f"🚀 Starting parallel generation: {args.episodes} episodes, "
        f"{args.workers} workers, Job: {args.job_id}"
    )

    # 3. Parallel Execution

    # Use 'spawn' context for libsumo safety in multi-threaded/node environments
    ctx = multiprocessing.get_context("spawn")

    # Important: Ensure main process uses absolute path for net_file as workers will chdir
    net_file = os.path.abspath(net_file)
    args.output_dir = os.path.abspath(args.output_dir)
    static_features_path = os.path.abspath(static_features_path)

    # Helper to clean up arguments
    worker_func = partial(
        worker_simulation_task,
        net_file=net_file,
        output_dir=args.output_dir,
        base_name=args.name,
        duration=args.duration,
        static_features_path=static_features_path,
        job_id=args.job_id,
    )

    generated_files = []
    failed_episodes = 0

    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        try:
            for result in tqdm(
                pool.imap_unordered(worker_func, range(args.episodes)),
                total=args.episodes,
                desc="Simulation Progress",
            ):
                if isinstance(result, str) and os.path.exists(result):
                    generated_files.append(result)
                elif isinstance(result, dict) and "error" in result:
                    failed_episodes += 1
                    tqdm.write(f"⚠️ Episode failed: {result['error'][:200]}")

        except KeyboardInterrupt:
            print("\n🛑 Interrupted! Terminating pool...")
            pool.terminate()
            pool.join()
            sys.exit(1)

    if failed_episodes:
        print(f"\n⚠️ {failed_episodes}/{args.episodes} episodes failed.")

    # 4. Merge Data (Optional)
    if args.no_merge:
        print(f"✅ Data Generation Complete. Partial files kept in {args.output_dir}")
    elif generated_files:
        merge_csv_files(generated_files, args.output_csv)
    else:
        print("⚠️ No files generated.")


if __name__ == "__main__":
    main()
