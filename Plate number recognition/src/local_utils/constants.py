import os

# region Dataset

SRC_PATH = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'src'))
WORKSPACE_PATH = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'workspace'))
DATA_PATH = os.path.abspath(os.path.join(WORKSPACE_PATH, 'dataset'))
RU_DATA_PATH = os.path.abspath(os.path.join(DATA_PATH, 'ru'))
KZ_DATA_PATH = os.path.abspath(os.path.join(DATA_PATH, 'kz'))

# region RU

RU_CARS_PATH = os.path.abspath(os.path.join(RU_DATA_PATH, 'cars'))
RU_CARS_IMAGES_PATH = os.path.abspath(os.path.join(RU_CARS_PATH, 'images'))
RU_CARS_ANNOTATIONS_PATH = os.path.abspath(os.path.join(RU_CARS_PATH, 'annotations'))

RU_PLATES_PATH = os.path.abspath(os.path.join(RU_DATA_PATH, 'plates'))
RU_PLATES_IMAGES_PATH = os.path.abspath(os.path.join(RU_PLATES_PATH, 'images'))
RU_PLATES_ANNOTATIONS_PATH = os.path.abspath(os.path.join(RU_PLATES_PATH, 'annotations'))

RU_CHARACTERS_PATH = os.path.abspath(os.path.join(RU_DATA_PATH, 'characters'))
RU_CHARACTERS_IMAGES_PATH = os.path.abspath(os.path.join(RU_CHARACTERS_PATH, 'images'))

# endregion RU

# region KZ

KZ_CARS_PATH = os.path.abspath(os.path.join(KZ_DATA_PATH, 'cars'))
KZ_CARS_IMAGES_PATH = os.path.abspath(os.path.join(KZ_CARS_PATH, 'images'))
KZ_CARS_ANNOTATIONS_PATH = os.path.abspath(os.path.join(KZ_CARS_PATH, 'annotations'))

KZ_PLATES_PATH = os.path.abspath(os.path.join(KZ_DATA_PATH, 'plates'))
KZ_PLATES_IMAGES_PATH = os.path.abspath(os.path.join(KZ_PLATES_PATH, 'images'))
KZ_PLATES_ANNOTATIONS_PATH = os.path.abspath(os.path.join(KZ_PLATES_PATH, 'annotations'))

KZ_CHARACTERS_PATH = os.path.abspath(os.path.join(KZ_DATA_PATH, 'characters'))
KZ_CHARACTERS_IMAGES_PATH = os.path.abspath(os.path.join(KZ_CHARACTERS_PATH, 'images'))

# endregion KZ

# endregion Dataset

# region Output

OUTPUT_PATH = os.path.abspath(os.path.join(WORKSPACE_PATH, 'output'))
RU_OUTPUT_PATH = os.path.abspath(os.path.join(OUTPUT_PATH, 'ru'))
KZ_OUTPUT_PATH = os.path.abspath(os.path.join(OUTPUT_PATH, 'kz'))

CARS_HISTORY_FILE = os.path.abspath(os.path.join(OUTPUT_PATH, 'cars_history.csv'))
CARS_BEST_MODEL_FILE = os.path.abspath(os.path.join(OUTPUT_PATH, 'cars_model_best.h5'))

PLATES_HISTORY_FILE = os.path.abspath(os.path.join(OUTPUT_PATH, 'plates_history.csv'))
PLATES_BEST_MODEL_FILE = os.path.abspath(os.path.join(OUTPUT_PATH, 'plates_model_best.h5'))

# region RU

RU_CHARACTERS_HISTORY_FILE = os.path.abspath(os.path.join(RU_OUTPUT_PATH, 'characters_history.csv'))
RU_CHARACTERS_BEST_MODEL_FILE = os.path.abspath(os.path.join(RU_OUTPUT_PATH, 'characters_model_best.h5'))

# endregion RU

# region KZ

KZ_CHARACTERS_HISTORY_FILE = os.path.abspath(os.path.join(KZ_OUTPUT_PATH, 'characters_history.csv'))
KZ_CHARACTERS_BEST_MODEL_FILE = os.path.abspath(os.path.join(KZ_OUTPUT_PATH, 'characters_model_best.h5'))

# endregion KZ

# endregion Output

# region Validation

RANDOM_SEED = 69228148822869
VALIDATION_SPLIT = 0.2

# endregion Validation

# region Training

CARS_TRAINING_SET_FILE = os.path.abspath(os.path.join(RU_CARS_PATH, 'training_set.txt'))
CARS_VALIDATION_SET_FILE = os.path.abspath(os.path.join(RU_CARS_PATH, 'validation_set.txt'))
CARS_LABELS_FILE = os.path.abspath(os.path.join(RU_CARS_PATH, 'labels.txt'))

PLATES_TRAINING_SET_FILE = os.path.abspath(os.path.join(RU_PLATES_PATH, 'training_set.txt'))
PLATES_VALIDATION_SET_FILE = os.path.abspath(os.path.join(RU_PLATES_PATH, 'validation_set.txt'))

# region RU

RU_PLATES_LABELS_FILE = os.path.abspath(os.path.join(RU_PLATES_PATH, 'labels.txt'))

# region RU

# region KZ

KZ_PLATES_LABELS_FILE = os.path.abspath(os.path.join(KZ_PLATES_PATH, 'labels.txt'))

# region KZ

# region RU

RU_CHARACTERS_LABELS_FILE = os.path.abspath(os.path.join(RU_CHARACTERS_PATH, 'labels.txt'))

# endregion RU

# region KZ

KZ_CHARACTERS_LABELS_FILE = os.path.abspath(os.path.join(KZ_CHARACTERS_PATH, 'labels.txt'))

# endregion KZ

# endregion Training

# region Model

MODEL_CONFIG_FILE = os.path.abspath(os.path.join(SRC_PATH, 'configs', 'ssd300_mobilenetv2.json'))

# endregion Model

# region Inference

CONFIDENCE_THRESHOLD = 0.9
PREDICTIONS = 10
COORD_DELTA = 5

HAAR_CASCADES_PATH = os.path.abspath(os.path.join(SRC_PATH, 'haar_cascades'))
LICENSE_PLATE_CASCADES_FILE = os.path.abspath(os.path.join(HAAR_CASCADES_PATH, 'haarcascade_russian_plate_number.xml'))

# endregion Inference

# region Plates

PLATES_SIZE = (520, 112)

# endregion Plates

# region Characters

CHARACTERS_SIZE = (32, 32)
WHITENING_WIDTH = 10

# endregion Characters
