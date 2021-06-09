import json
from argparse import Namespace

import cv2
import numpy as np

from src.local_utils import inference_utils
from src.local_utils.constants import (
    MODEL_CONFIG_FILE,
    CARS_BEST_MODEL_FILE,
    CARS_LABELS_FILE,
    CONFIDENCE_THRESHOLD,
    PREDICTIONS,
    COORD_DELTA,
)
from src.subjects import Car


class CarDetector:
    def __init__(self):
        self.args = Namespace()
        self.args.config = MODEL_CONFIG_FILE
        self.args.weights = CARS_BEST_MODEL_FILE
        self.args.label_maps = CARS_LABELS_FILE
        self.args.confidence_threshold = CONFIDENCE_THRESHOLD
        self.args.num_predictions = PREDICTIONS

        with open(self.args.config, mode='r') as config_file:
            config = json.load(config_file)

        self.input_size = config['model']['input_size']

        self.model, \
        self.label_maps, \
        self.process_input_fn = inference_utils.inference_ssd_mobilenetv2(config, self.args)

        self.model.load_weights(self.args.weights)

    def detect_car(self, image):
        img = image.copy()

        image_height, image_width, _ = img.shape
        height_scale, width_scale = self.input_size / image_height, self.input_size / image_width

        img = cv2.cvtColor(cv2.resize(img, (self.input_size, self.input_size)), cv2.COLOR_BGR2RGB)
        img = self.process_input_fn(img)

        img = np.expand_dims(img, axis=0)
        y_pred = self.model.predict(img)

        car = None

        for i, pred in enumerate(y_pred[0]):
            confidence_score = pred[1]
            score = confidence_score * 100

            if 1 >= confidence_score > self.args.confidence_threshold:
                classname = self.label_maps[int(pred[0]) - 1].upper()

                if classname == 'CAR':
                    tmp_car = Car(
                        x1=max(min(int(pred[2] / width_scale), image_width - 1), 0),
                        y1=max(min(int(pred[3] / height_scale), image_height - 1), 0),
                        x2=max(min(int(pred[4] / width_scale), image_width - 1), 0),
                        y2=max(min(int(pred[5] / height_scale), image_height - 1), 0),
                        score=score,
                    )

                    if not car or tmp_car.square > car.square:
                        if COORD_DELTA <= tmp_car.x1 <= (image_width - COORD_DELTA - 1) and \
                                COORD_DELTA <= tmp_car.x2 <= (image_width - COORD_DELTA - 1) and \
                                COORD_DELTA <= tmp_car.y1 <= (image_height - COORD_DELTA - 1) and \
                                COORD_DELTA <= tmp_car.y2 <= (image_height - COORD_DELTA - 1):
                            car = tmp_car

        return car
