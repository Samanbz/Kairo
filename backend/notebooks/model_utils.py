import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_traffic_data(csv_path="../data/traffic_training_data.csv"):
    """
    Loads and preprocesses traffic data for generalized regression.
    """
    print(f"Loading data from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback for when running from root
        print("File not found in CWD, trying full path...")
        data = pd.read_csv("backend/data/traffic_training_data.csv")

    # 1. Filter Anomalies (Parking artifacts)
    initial_len = len(data)
    # Remove rows where cars are stuck (0 speed) but only 1 car is there (likely parked)
    data = data[~((data["vehicle_count"] == 1) & (data["current_speed"] == 0.0))]
    print(f"Filtered {initial_len - len(data)} anomaly rows.")

    # 2. Feature Engineering
    # Cyclical Time
    data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)

    # Topological / Road Structure Features
    # Avoid division by zero
    data["lane_count"] = data["lane_count"].replace(0, 1)

    # "Bottleneck Intensity": High betweenness squeezed into few lanes
    data["bottleneck_intensity"] = data["betweenness"] / data["lane_count"]

    # "Road Capacity": Theoretical max flow
    data["road_capacity"] = data["lane_count"] * data["max_speed"]

    # Rush Hour Flag
    data["is_rush_hour"] = data["hour_of_day"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    data["is_weekend"] = data["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    # 3. Define Feature Sets
    features = [
        "hour_of_day",
        "day_of_week",
        "weather",  # Environment
        "street_type",
        "lane_count",
        "max_speed",  # Static Road Specs
        "in_degree",
        "out_degree",  # Local Connectivity
        "betweenness",
        "closeness",  # Global Topology
        "bottleneck_intensity",
        "road_capacity",  # Engineered Topology
        "hour_sin",
        "hour_cos",  # Cyclical
        "is_rush_hour",
        "is_weekend",  # Temporal Flags
    ]

    cat_cols = ["weather", "day_of_week", "street_type", "is_rush_hour", "is_weekend"]

    # Ensure categorical types
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")

    target = "congestion"

    # Drop NAs
    data = data.dropna(subset=[target] + [f for f in features if f in data.columns])

    print(f"Final Dataset Shape: {data.shape}")
    print(f"Features used: {features}")

    return data, features, cat_cols, target


def get_train_test_split(data, features, target, test_size=0.2):
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Standard evaluation for traffic models
    """
    preds = model.predict(X_test)

    # Clip to valid range [0, 1]
    preds = np.clip(preds, 0, 1)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    print(f"\n--- {model_name} Performance ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    return preds


def plot_benchmark_profile(model, features, model_name="Model"):
    """
    Plots the standard 'Rainy Monday Main Artery' test case
    """
    hours = range(0, 24)
    # Create synthetic query DataFrame
    # Needs to match training schema exactly
    demo_df = pd.DataFrame(
        {
            "hour_of_day": hours,
            "day_of_week": ["Friday"] * 24,
            "weather": ["Rain"] * 24,
            "street_type": ["highway.motorway"] * 24,
            "lane_count": [2] * 24,
            "max_speed": [13.89] * 24,
            "in_degree": [2] * 24,
            "out_degree": [2] * 24,
            "betweenness": [0.05] * 24,  # High centrality
            "closeness": [0.02] * 24,
        }
    )

    # Apply same engineering
    demo_df["hour_sin"] = np.sin(2 * np.pi * demo_df["hour_of_day"] / 24)
    demo_df["hour_cos"] = np.cos(2 * np.pi * demo_df["hour_of_day"] / 24)
    demo_df["bottleneck_intensity"] = demo_df["betweenness"] / demo_df["lane_count"]
    demo_df["road_capacity"] = demo_df["lane_count"] * demo_df["max_speed"]
    demo_df["is_rush_hour"] = demo_df["hour_of_day"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    demo_df["is_weekend"] = 0

    # Cast categories
    cat_cols = ["weather", "day_of_week", "street_type", "is_rush_hour", "is_weekend"]

    # Note: We loop and check if cols exist just to be safe, though we defined them
    for col in cat_cols:
        demo_df[col] = demo_df[col].astype("category")

    preds = model.predict(demo_df[features])
    preds = np.clip(preds, 0, 1)

    plt.figure(figsize=(10, 5))
    plt.plot(hours, preds, marker="o", label=f"{model_name}", linewidth=2)
    plt.title(f"Traffic Profile: Rainy Monday (Main Artery)\n{model_name}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Speed Efficiency (1.0 = Free Flow)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
