from src.db import Owner
from src.plate import LicensePlate


class LicensePlatesDB:
    def __init__(self):
        self.plates = dict()

        self.last_searched_plate = None

    def __contains__(self, item):
        return str(item) in self.plates.keys()

    def add_relation(self, owner: Owner, license_plate: LicensePlate):
        self.plates[str(license_plate)] = owner

    def get_owner(self, license_plate: LicensePlate):
        self.last_searched_plate = license_plate

        return self.plates[str(license_plate)]
