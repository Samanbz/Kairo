from datetime import time, timedelta
from typing import List, Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    lat: float = Field(..., description="Latitude of the location")
    lng: float = Field(..., description="Longitude of the location")
    osm_node_id: Optional[int] = Field(None, description="Nearest OSM Node ID")


class Order(BaseModel):
    order_id: str = Field(..., description="Unique identifier for the order")
    location: Location
    weight: float = Field(..., description="Weight of the parcel (e.g., kg)")
    volume: Optional[float] = Field(0.0, description="Volume of the parcel (e.g., m^3)")
    time_window_start: int = Field(
        ..., description="Earliest delivery time allowed (minutes from day start)"
    )
    time_window_end: int = Field(
        ..., description="Latest delivery time allowed (minutes from day start)"
    )
    service_time: int = Field(5, description="Unloading time required at dropoff in minutes")


class Vehicle(BaseModel):
    vehicle_id: str = Field(..., description="Unique identifier for the vehicle")
    origin: Location = Field(..., description="Starting location of the vehicle (e.g., Warehouse)")
    capacity_weight: float = Field(..., description="Maximum payload weight constraint")
    capacity_volume: Optional[float] = Field(None, description="Maximum payload volume constraint")
    shift_start: int = Field(
        ..., description="Earliest time vehicle can leave origin (minutes from day start)"
    )
    shift_end: int = Field(
        ..., description="Latest time vehicle must return to origin (minutes from day start)"
    )
    speed_factor: float = Field(
        1.0, description="Speed multiplier (e.g., 0.8 for heavily loaded truck)"
    )


class ScenarioConfig(BaseModel):
    time_of_day: int = Field(8, description="Hour of day (0-23) used for traffic prediction")
    weather_condition: str = Field(
        "Clear", description="Weather setting used for traffic prediction"
    )
    day_of_week: str = Field("Monday", description="Day of week used for traffic prediction")


class RouteRequest(BaseModel):
    orders: List[Order]
    vehicles: List[Vehicle]
    config: ScenarioConfig
