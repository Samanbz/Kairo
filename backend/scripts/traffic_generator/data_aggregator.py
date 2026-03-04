import os

import pandas as pd
from tqdm import tqdm


def merge_csv_files(generated_files, target_csv):
    print("💾 Merging csv files...")
    first_file = True

    with open(target_csv, "w") as outfile:
        for fname in tqdm(generated_files, desc="Merging"):
            with open(fname, "r") as infile:
                header = infile.readline()
                if first_file:
                    outfile.write(header)
                    first_file = False

                # Stream write rest
                for line in infile:
                    outfile.write(line)

            # Cleanup partial
            try:
                os.remove(fname)
            except OSError:
                pass

    print(f"\n✅ Data Generation Complete. Output: {target_csv}")

    # Only analyze if file is small enough (arbitrary check) or user requests it
    # For huge files, pandas read_csv will crash
    if os.path.getsize(target_csv) < 500 * 1024 * 1024:  # 500MB Limit for check
        df = pd.read_csv(target_csv)
        print(f"   Total Records: {len(df)}")
        anomalies = df[(df["current_speed"] == 0.0) & (df["vehicle_count"] == 1)]
        portion_anomalies = len(anomalies) / len(df) * 100.0
        print(f"   Anomalies (0 speed & 1 vehicle): {len(anomalies)} ({portion_anomalies:.2f}%)")
    else:
        print("   (Skipping anomaly check for large file > 500MB)")
