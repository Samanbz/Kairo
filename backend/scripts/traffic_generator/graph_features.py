import json
import os

import networkx as nx
import sumolib


def get_static_features(net_file, output_dir):
    cache_file = os.path.join(output_dir, "static_features_cache.json")

    # Check if cache exists and is fresher than net_file
    if os.path.exists(cache_file):
        net_mtime = os.path.getmtime(net_file)
        cache_mtime = os.path.getmtime(cache_file)

        if cache_mtime > net_mtime:
            print(f"   Loading static features from cache: {cache_file}")
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"   Cache load failed ({e}), recomputing...")
        else:
            print("   Cache is stale (network file changed), recomputing...")

    print("   Parsing road network topology...")
    net = sumolib.net.readNet(net_file)

    # Build NetworkX Graph for Topology Analysis
    print("   Building graph for centrality analysis...")
    G = nx.DiGraph()
    for edge in net.getEdges():
        # SUMO Edges are directed. Weight by length to find "shortest paths" (arteries)
        G.add_edge(edge.getFromNode().getID(), edge.getToNode().getID(), weight=edge.getLength())

    # Compute Centrality
    # Betweenness: High score = "Bridge" road (Critical for network flow)
    # Closeness: High score = "Central" road (Close to everything else, e.g. City Center)
    # Note: This can be slow for large maps, but is computed only once per static map.
    print("   Computing Betweenness Centrality (Topology Importance)...")
    betweenness = nx.betweenness_centrality(G, weight="weight", k=1000)  # k=None implies exact calc

    print("   Computing Closeness Centrality (Geometric Centrality)...")
    closeness = nx.closeness_centrality(G, distance="weight")

    features = {}
    for edge in net.getEdges():
        edge_id = edge.getID()
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        from_id = from_node.getID()

        features[edge_id] = {
            "street_type": edge.getType(),
            "speed_limit": edge.getSpeed(),
            "in_degree": len(from_node.getIncoming()),
            "out_degree": len(to_node.getOutgoing()),
            "lane_count": edge.getLaneNumber(),
            "betweenness": betweenness.get(from_id, 0.0),
            "closeness": closeness.get(from_id, 0.0),
        }

    # Save cache
    try:
        print(f"   Saving static features (including centrality) to {cache_file}...")
        with open(cache_file, "w") as f:
            json.dump(features, f)
    except Exception as e:
        print(f"   Warning: Could not save static feature cache: {e}")

    return features
