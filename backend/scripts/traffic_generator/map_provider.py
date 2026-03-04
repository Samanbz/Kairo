import json
import math
import os
import sys

import requests


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
                math.isclose(meta.get("lat", 0), lat, rel_tol=1e-5)
                and math.isclose(meta.get("lon", 0), lon, rel_tol=1e-5)
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
