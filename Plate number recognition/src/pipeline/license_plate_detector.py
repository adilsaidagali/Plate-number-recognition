from src import plate


class LicensePlateDetector:
    @staticmethod
    def detect_plate(image):
        return plate.detect_carplate(image)
