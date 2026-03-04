from typing import Any, Dict, List

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from models.domain import Order, RouteRequest, Vehicle


class CVRPTWSolver:
    def __init__(self, request: RouteRequest, traffic_model=None):
        self.request = request
        self.orders = request.orders
        self.vehicles = request.vehicles
        self.traffic_model = traffic_model

    def _create_data_model(self) -> Dict[str, Any]:
        """
        Structures the data for OR-Tools to consume.
        - Nodes: [Depots (Vehicles' Origins)] + [Orders]
        - A matrix of distances and a matrix of durations among all nodes.
        Note: We temporarily assume each vehicle has unique origin (Depot) just mapped sequentially.
        """
        data = {}

        # Consolidate all nodes
        locations = [v.origin for v in self.vehicles] + [o.location for o in self.orders]
        data["locations"] = locations
        num_locations = len(locations)
        data["num_vehicles"] = len(self.vehicles)
        data["starts"] = list(range(len(self.vehicles)))
        # Suppose vehicles can end anywhere, or must return? For periodic, they might not return.
        # We can set 'ends' equal to starts, or allow arbitrary ends
        data["ends"] = list(range(len(self.vehicles)))

        # Distances and Durations: Dummy calculation (Euclidean distance mapped to minutes)
        # MUST REPLACE with actual OSRM/RoutingPy matrix modulated by the traffic_model!
        import math

        def node_dist(l1, l2):
            # Very rough distance simulation
            return int(math.hypot(l1.lat - l2.lat, l1.lng - l2.lng) * 111000)

        distance_matrix = []
        duration_matrix = []
        for i, src in enumerate(locations):
            dist_row, dur_row = [], []
            for j, dest in enumerate(locations):
                d = node_dist(src, dest)
                dist_row.append(d)
                # Traffic Model modifier simulation
                speed = 10.0  # meters per second
                dur_row.append(int(d / speed / 60))  # duration in minutes
            distance_matrix.append(dist_row)
            duration_matrix.append(dur_row)

        data["distance_matrix"] = distance_matrix
        data["duration_matrix"] = duration_matrix
        data["time_windows"] = [(0, 1440) for _ in self.vehicles] + [
            (o.time_window_start, o.time_window_end) for o in self.orders
        ]
        data["service_times"] = [0 for _ in self.vehicles] + [o.service_time for o in self.orders]
        data["demands"] = [0 for _ in self.vehicles] + [int(o.weight) for o in self.orders]
        data["vehicle_capacities"] = [int(v.capacity_weight) for v in self.vehicles]

        return data

    def solve(self):
        data = self._create_data_model()
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["starts"], data["ends"]
        )
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Demand constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_capacities"],
            True,  # start cumul to zero
            "Capacity",
        )

        # Time constraint
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["duration_matrix"][from_node][to_node] + data["service_times"][from_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        routing.AddDimension(
            time_callback_index,
            1440,  # max slack (let's say maximum wait time)
            1440,  # max time (let's say 24 hours bounding = 1440 minutes)
            False,  # start cumul to zero
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        for location_idx, time_window in enumerate(data["time_windows"]):
            if location_idx < data["num_vehicles"]:
                continue  # skip depot ends setup for now
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # Optimize search
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.time_limit.seconds = 5

        # Solve
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            return self._format_solution(data, manager, routing, solution)
        return {"error": "No solution found."}

    def _format_solution(self, data, manager, routing, solution):
        # Parses the OR-Tools solution into a dict for presentation to the frontend
        routes = []
        time_dimension = routing.GetDimensionOrDie("Time")

        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route_dict = {"vehicle_id": self.vehicles[vehicle_id].vehicle_id, "stops": []}
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                load = 0 if node_index < data["num_vehicles"] else data["demands"][node_index]

                route_dict["stops"].append(
                    {
                        "location_idx": node_index,
                        "order_id": None
                        if node_index < data["num_vehicles"]
                        else self.orders[node_index - data["num_vehicles"]].order_id,
                        "eta_min": solution.Min(time_var),
                        "eta_max": solution.Max(time_var),
                        "load_change": load,
                    }
                )
                index = solution.Value(routing.NextVar(index))
            routes.append(route_dict)
        return routes
