import csv
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from models.domain import Order, RouteRequest, Vehicle
from services.solver import CVRPTWSolver
from services.traffic import TrafficModel

router = APIRouter(prefix="/routing", tags=["Routing"])


@router.post("/solve")
async def solve_routing_problem(request: RouteRequest):
    """
    Accepts JSON representation of orders, vehicles, and scenario configs,
    and returns optimized routes mapped out.
    """
    # Initialize mock traffic model for now
    traffic_model = TrafficModel()

    # Process
    solver = CVRPTWSolver(request, traffic_model)
    solution = solver.solve()

    if "error" in solution:
        raise HTTPException(status_code=400, detail=solution["error"])

    return {"status": "success", "routes": solution}


@router.post("/upload_orders")
async def upload_orders_csv(file: UploadFile = File(...)):
    """
    Helper endpoint to accept a CSV upload, parse it into Order Pydantic models.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    contents = await file.read()
    decoded = contents.decode("utf-8").splitlines()
    reader = csv.DictReader(decoded)

    parsed_orders = []
    for row in reader:
        try:
            # Simple assumption mapping row mappings:
            # required CSV headers: order_id, lat, lng, weight, time_start, time_end
            parsed_orders.append(
                Order(
                    order_id=row["order_id"],
                    location={"lat": float(row["lat"]), "lng": float(row["lng"])},
                    weight=float(row["weight"]),
                    time_window_start=int(row["time_start"]),
                    time_window_end=int(row["time_end"]),
                    service_time=int(row.get("service_time", 5)),
                )
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing row {row}: {str(e)}")

    return {"parsed_count": len(parsed_orders), "orders": parsed_orders}
