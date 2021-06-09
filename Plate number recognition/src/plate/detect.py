import cv2

from src.local_utils.constants import LICENSE_PLATE_CASCADES_FILE
from src.subjects import Plate


def detect_carplate(image):
    carplate_overlay = image.copy()

    carplate_haar_cascade = cv2.CascadeClassifier(LICENSE_PLATE_CASCADES_FILE)
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=5)

    plate = None

    for x, y, w, h in carplate_rects:
        tmp_plate = Plate(x, y, x + w, y + h, 0)

        if not plate or tmp_plate.square > plate.square:
            plate = tmp_plate

    return plate
