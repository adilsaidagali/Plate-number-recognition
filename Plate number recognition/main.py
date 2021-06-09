import cv2
import numpy as np
from PIL import ImageGrab

from src.db import LicensePlatesDB, Owner
from src.pipeline import (
    CarDetector,
    LicensePlateDetector,
    RuLicensePlateRecognizer,
    KzLicensePlateRecognizer,
)
from src.plate import RuLicensePlate, KzLicensePlate

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

car_detector = CarDetector()
license_plate_recognizer = KzLicensePlateRecognizer()

db = LicensePlatesDB()
db.add_relation(
    Owner('Иван', 'Иванович', 'Иванов', 'M', '04.04.1984'),
    RuLicensePlate(series='MMM', number='700', region='48')
)
db.add_relation(
    Owner('Евгения', 'Евгеньевна', 'Евгеньева', 'Ж', '08.08.1998'),
    RuLicensePlate(series='OOO', number='888', region='19')
)
db.add_relation(
    Owner('Уйсумбаева', 'Шнар', 'Кабудлхамитовна', 'Ж', '01.07.1974'),
    KzLicensePlate(series='YX', number='171', region='14')
)
db.add_relation(
    Owner('Анар', 'Онербайкызы', 'Рахимжанова',  'Ж', '06.06.1996'),
    KzLicensePlate(series='PHL', number='029', region='10')
)
db.add_relation(
    Owner('Азамат', 'Муратович', 'Арманов',  'М', '09.12.1989'),
    KzLicensePlate(series='AZA', number='777', region='13')
)
db.add_relation(
    Owner('Кабдула', 'Исмаилович', 'Болатов', 'М', '09.12.1989'),
    KzLicensePlate(series='VP', number='010', region='01')
)


def process_frame_camera():
    global webcam, car_detector

    check, image = webcam.read()

    if check:
        image = np.uint8(np.array(image, dtype=np.float))
        process_frame(image)


def process_frame_grab():
    global webcam, car_detector

    image = ImageGrab.grab(bbox=(800, 100, 1600, 1400))
    image = np.uint8(np.array(image, dtype=np.float))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    process_frame(image)


def process_frame(image):
    display_image = image.copy()

    car = car_detector.detect_car(image)

    if car:
        carplate = LicensePlateDetector.detect_plate(image)

        color = car.Color.REGULAR

        if carplate:
            if car.x1 <= carplate.x1 < car.x2 and \
                    car.x1 < carplate.x2 <= car.x2 and \
                    car.y1 <= carplate.y1 < car.y2 and \
                    car.y1 < carplate.y2 <= car.y2:
                license_plate = license_plate_recognizer.recognize_license_plate(image, carplate)
                if license_plate:
                    cv2.putText(
                        display_image,
                        str(license_plate),
                        (min(carplate.x1, carplate.x2), min(carplate.y1, carplate.y2)),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        car.Color.HIGHLIGHTED,
                        2,
                    )

                    if license_plate in db:
                        color = car.Color.HIGHLIGHTED

                        if str(db.last_searched_plate) != str(license_plate):
                            print(str(license_plate), '--', db.get_owner(license_plate))

                cv2.rectangle(
                    img=display_image,
                    pt1=carplate.rect[0],
                    pt2=carplate.rect[1],
                    color=car.Color.HIGHLIGHTED,
                    thickness=2
                )

        cv2.rectangle(
            img=display_image,
            pt1=car.rect[0],
            pt2=car.rect[1],
            color=color,
            thickness=2
        )

    cv2.imshow('License plate recognition', display_image)


if __name__ == '__main__':
    while True:
        process_frame_grab()

        if cv2.waitKey(1) == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
