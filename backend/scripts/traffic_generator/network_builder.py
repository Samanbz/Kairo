import math
import os
import subprocess

import numpy as np
import sumolib


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

    return net_file


def generate_trips_xml(net_file, output_path, duration, period, min_distance=1000):
    """
    Generate a SUMO trips XML file using sumolib directly.
    SUMO will route these trips on-the-fly during simulation.
    """
    net = sumolib.net.readNet(net_file)

    # Get all edges that allow passenger vehicles
    all_edges = [
        e
        for e in net.getEdges()
        if e.allows("passenger") and not e.isSpecial() and e.getLength() > 50
    ]

    if len(all_edges) < 2:
        raise RuntimeError(f"Network has too few usable edges ({len(all_edges)})")

    # Identify fringe edges (dead-ends / network boundary) for realistic trip origins/destinations
    fringe_edges = [
        e
        for e in all_edges
        if len([s for s in e.getOutgoing().keys() if s.allows("passenger")]) == 0
        or len([s for s in e.getIncoming().keys() if s.allows("passenger")]) == 0
    ]

    # Weight edges by speed^4 (same as randomTrips --speed-exponent 4.0)
    edge_weights = np.array([e.getSpeed() ** 4 for e in all_edges], dtype=np.float64)
    edge_weights /= edge_weights.sum()

    # Fringe factor: fringe edges are 10x more likely to be origins/destinations
    fringe_ids = set(e.getID() for e in fringe_edges)
    fringe_weights = np.array(
        [10.0 if e.getID() in fringe_ids else 1.0 for e in all_edges], dtype=np.float64
    )
    origin_weights = edge_weights * fringe_weights
    origin_weights /= origin_weights.sum()

    # Pre-compute edge positions for distance filtering (use 'from' node coords)
    edge_positions = np.array([e.getFromNode().getCoord() for e in all_edges])

    num_trips = int(duration / period)

    with open(output_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            '<trips xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/trips_file.xsd">\n'
        )

        trip_count = 0
        max_attempts_per_trip = 20

        for i in range(num_trips):
            depart = i * period

            for _ in range(max_attempts_per_trip):
                src_idx = np.random.choice(len(all_edges), p=origin_weights)
                dst_idx = np.random.choice(len(all_edges), p=origin_weights)

                if src_idx == dst_idx:
                    continue

                # Euclidean distance check (fast proxy for min_distance)
                dx = edge_positions[src_idx][0] - edge_positions[dst_idx][0]
                dy = edge_positions[src_idx][1] - edge_positions[dst_idx][1]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist >= min_distance:
                    src_edge = all_edges[src_idx]
                    dst_edge = all_edges[dst_idx]
                    f.write(
                        f'    <trip id="t_{trip_count}" depart="{depart:.2f}" '
                        f'from="{src_edge.getID()}" to="{dst_edge.getID()}" '
                        f'departLane="best" departSpeed="max" departPos="random"/>\n'
                    )
                    trip_count += 1
                    break

        f.write("</trips>\n")

    return trip_count
