import os

import cv2
import numpy as np

from src import plate
from src.local_utils.constants import (
    RU_CHARACTERS_IMAGES_PATH,
    RU_CHARACTERS_BEST_MODEL_FILE,
    KZ_CHARACTERS_IMAGES_PATH,
    KZ_CHARACTERS_BEST_MODEL_FILE,
    CHARACTERS_SIZE,
    )
from src.networks.character_classifier import CharacterClassifier
from src.plate import (
    LicensePlate,
    LicensePlateError,
    RuLicensePlate,
    KzLicensePlate,
    )


class LicensePlateRecognizer:
    MIN_RATIO = None
    MAX_RATIO = None
    MIN_HEIGHT = None
    MAX_HEIGHT = None
    MIN_CHARACTERS_COUNT = None
    MAX_CHARACTERS_COUNT = None
    
    def __init__(self, characters_images_path, weights_file_path, license_plate_format: LicensePlate):
        self.classnames = os.listdir(characters_images_path)
        self.model = CharacterClassifier(len(self.classnames))
        self.model.load_weights(weights_file_path)
        self.license_plate_impl = license_plate_format
    
    def recognize_license_plate(self, image, plate_rect):
        plate_image = plate.get_clean(image[
                                      plate_rect.y1:plate_rect.y2,
                                      plate_rect.x1:plate_rect.x2,
                                      ].copy()
                                      )

        _, thresholded = cv2.threshold(plate_image, 10, 255, cv2.THRESH_BINARY)
        plate_image = cv2.bitwise_not(thresholded)
        contours, _ = cv2.findContours(plate_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crop_characters = []
        
        for contour in self.sort_contours(contours):
            x, y, w, h = cv2.boundingRect(contour)
            ratio = h / w
            
            if self.MIN_RATIO <= ratio <= self.MAX_RATIO:
                if self.MIN_HEIGHT <= h / plate_image.shape[0] <= self.MAX_HEIGHT:
                    current_character = cv2.bitwise_not(plate_image[y:y + h, x:x + w].copy())
                    crop_characters.append(cv2.resize(current_character, dsize=CHARACTERS_SIZE))
        
        if self.MIN_CHARACTERS_COUNT <= len(crop_characters) <= self.MAX_CHARACTERS_COUNT:
            characters = [
                self.classnames[np.argmax(self.model.predict(self.image2data(img))[0])] for img in crop_characters
                ]
            try:
                return self.license_plate_impl.from_characters(characters)
            except LicensePlateError:
                return None
        else:
            return None
    
    @staticmethod
    def sort_contours(contours, reverse=False):
        if len(contours) == 0:
            return contours
        
        i = 0
        boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        
        return contours
    
    @staticmethod
    def image2data(image):
        res = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return np.array([res.astype('float32') / 255])


class RuLicensePlateRecognizer(LicensePlateRecognizer):
    MIN_RATIO = 0.8
    MAX_RATIO = 2.5
    MIN_HEIGHT = 0.3
    MAX_HEIGHT = 0.9
    MIN_CHARACTERS_COUNT = 8
    MAX_CHARACTERS_COUNT = 9
    
    def __init__(self):
        super().__init__(RU_CHARACTERS_IMAGES_PATH, RU_CHARACTERS_BEST_MODEL_FILE, RuLicensePlate)


class KzLicensePlateRecognizer(LicensePlateRecognizer):
    MIN_RATIO = 0.9
    MAX_RATIO = 2.5
    MIN_HEIGHT = 0.35
    MAX_HEIGHT = 0.8
    MIN_CHARACTERS_COUNT = 7
    MAX_CHARACTERS_COUNT = 8
    
    def __init__(self):
        super().__init__(KZ_CHARACTERS_IMAGES_PATH, KZ_CHARACTERS_BEST_MODEL_FILE, KzLicensePlate)
