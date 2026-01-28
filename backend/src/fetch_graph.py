import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox


def fetch_darmstadt_graph():
    """
    Downloads and cleans the driveable street network for a given place.
    """
    print("Downloading graph for Darmstadt...")

    # Download the graph
    # simplify=True is default, but good to be explicit
    G = ox.graph_from_point(
        (49.8728, 8.6512),  # Darmstadt center coordinates
        dist=10000,  # 10 km radius
        network_type="all",  # The only thing that has worked so far
        simplify=True,
        truncate_by_edge=True,
    )

    print(f"Original graph stats - Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

    return G


def clean_graph(G):
    """
    Removes isolated nodes and keeps the largest strongly connected component.
    """
    print("Cleaning graph (keeping largest strongly connected component)...")

    kill_list = [
        "footway",
        "cycleway",
        "path",
        "pedestrian",
        "steps",
        "track",
        "corridor",
        "elevator",
        "platform",
        "bridleway",
    ]

    edges_to_remove = []

    # 3. Iterate and Mark for Death
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get("highway")

        # Helper to check if the tag is in our kill list
        # (OSM tags can be lists like ['footway', 'cycleway'])
        is_bad = False
        if isinstance(highway, list):
            if any(h in kill_list for h in highway):
                is_bad = True
        elif highway in kill_list:
            is_bad = True

        if is_bad:
            edges_to_remove.append((u, v, k))

    # 4. Execute Removal
    G.remove_edges_from(edges_to_remove)

    # 5. Clean up the isolated dots left behind
    G.remove_nodes_from(list(nx.isolates(G)))
    # In a directed graph (drive network), strongly connected ensures
    # you can get from A to B and back to A within the component.
    # For logistics, this is crucial.

    G_clean = ox.truncate.largest_component(G, strongly=False)

    print(f"Cleaned graph stats - Nodes: {len(G_clean.nodes)}, Edges: {len(G_clean.edges)}")

    return G_clean


def save_graph(G, output_path):
    """
    Saves the graph to a .graphml file.
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving graph to {output_path}...")
    ox.save_graphml(G, filepath=output_path)
    print("Done.")


if __name__ == "__main__":
    # Define output path
    # Determine the project root relative to this script
    current_dir = Path(__file__).parent.resolve()
    # Assuming script is in backend/src, backend is parent
    backend_dir = current_dir.parent
    data_dir = backend_dir / "data"
    output_path = data_dir / "darmstadt_drive.graphml"

    try:
        G = fetch_darmstadt_graph()
        G_clean = clean_graph(G)
        save_graph(G_clean, output_path)
        ox.plot_graph(G_clean, figsize=(20, 20))
    except Exception as e:
        print(f"An error occurred: {e}")
