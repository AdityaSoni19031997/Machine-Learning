"""
Design The Following System ->

Reference (for problem statement) -> https://workat.tech/machine-coding/practice/design-parking-lot-qm6hwq4wkhp8

# The Problem Statement
    - A parking lot is an area where cars can be parked for a certain amount of time. 
    - A parking lot can have multiple floors with each floor having a different number of slots.
    - Each parking slot being suitable for different types of vehicles.

# Minimal Requirements...
    Details about the Parking Lot -:
        - Create the parking lot.
        - Add floors to the parking lot.
        - Add a parking lot slot to any of the floors.
        - Given a vehicle, it finds the first available slot, books it, creates a ticket, parks the vehicle, and finally returns the ticket.
        - Unparks a vehicle given the ticket id.
        - Displays the number of free slots per floor for a specific vehicle type.
        - Displays all the free slots per floor for a specific vehicle type.
        - Displays all the occupied slots per floor for a specific vehicle type.

    Details about the Vehicles:
        - Every vehicle will have a type, registration number, and color.
        - Different Types of Vehicles:
            - Car
            - Bike
            - Truck

    Details about the Parking Slots:
        - Each type of slot can park a specific type of vehicle.
        - No other vehicle should be allowed by the system.
        - Finding the first available slot should be based on:
            - The slot should be of the same type as the vehicle.
            - The slot should be on the lowest possible floor in the parking lot.
            - The slot should have the lowest possible slot number on the floor.
        - Numbered serially from 1 to n for each floor where n is the number of parking slots on that floor.

    Details about the Parking Lot Floors:
        - Numbered serially from 1 to n where n is the number of floors.
        - Might contain one or more parking lot slots of different types.
        - We will assume that the first slot on each floor will be for a truck, the next 2 for bikes, and all the other slots for cars.

    Details about the Tickets:
        - The ticket id would be of the following format:
            <parking_lot_id>_<floor_no>_<slot_no>
            Example: PR1234_2_5 (denotes 5th slot of 2nd floor of parking lot PR1234)

    We can assume that there will only be 1 parking lot.
"""

# TODO Machine Level Coding


from itertools import cycle, islice
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List


class Address:
    country: str
    state: str
    city: str
    street: str
    zip_code: str


class ParkingVehicleType(Enum):
    BIKE: str = "bike"
    CAR: str = "car"
    TRUCK: str = "truck"


class ParkingTicketStatus(Enum):
    PAID: str = "paid"
    ACTIVE: str = "active"


@dataclass
class Vehicle:
    vehicle_color: str
    vehicle_license_no: str
    vehicle_type: ParkingVehicleType


class ParkingSlot:
    slot_id: int
    is_available: bool = field(default=True)
    slot_cost_per_hr: float
    slot_created_on: datetime = field(default=datetime.now())
    slot_type: ParkingVehicleType
    parked_vehicle: Vehicle

    def verify_slot_and_parked_vehicle_type(self):
        ...


class ParkingDisplayBoard:
    no_slots_free: int


@dataclass
class ParkingFloor:
    level_id: int
    slots: List[ParkingSlot]
    display_board: ParkingDisplayBoard
    slots_type: Dict[int, str]


@dataclass
class ParkingLot:
    lot_address: Address = field(default_factory=Address)
    parking_floor: List[ParkingFloor] = field(default_factory=list)


class ParkingTicket:
    ticket_id: int
    level_id: int
    slot_id: int
    parking_charge: float
    parking_end_time: datetime
    parking_start_time: datetime
    parking_ticket_status: ParkingTicketStatus
    parking_type: ParkingVehicleType

    def total_cost(self) -> float:
        ...

    def update_end_time(self) -> None:
        ...


class BIKE(Vehicle):
    vehicle_type = ParkingVehicleType.BIKE


class CAR(Vehicle):
    vehicle_type = ParkingVehicleType.CAR


class TRUCK(Vehicle):
    vehicle_type = ParkingVehicleType.TRUCK


class Account:
    name: str
    email: str
    password: str
    emp_id: str
    emp_addr: Address


class Admin(Account):
    FLOOR_LEVEL_ID = 0
    SLOT_ID = 0
    TICKET_ID = 0

    def __init__(self, name, email, password, emp_id, emp_addr) -> None:
        super().__init__(name, email, password, emp_id, emp_addr)
        ...

    def create_parking_lot(self, lot_address: Address):
        self.parking_lot = ParkingLot(lot_address=lot_address)

    def add_a_parking_floor(self, no_of_slots_in_the_floor: int):
        ...

    def add_parking_slots(self):
        ...
