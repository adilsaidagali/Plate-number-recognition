from tensorflow.keras.applications import mobilenet_v2

from src.data_generators import SSD_DATA_GENERATOR
from src.local_utils import data_utils


def get_data_generator(config, args):
    model_config = config['model']

    training_samples = data_utils.get_samples_from_split(
        split_file=args.training_split,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )
    num_training_samples = len(training_samples)

    validation_samples = data_utils.get_samples_from_split(
        split_file=args.validation_split,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir
    )
    num_validation_samples = len(validation_samples)

    print(f'Creating data generator for {model_config["name"]}')

    with open(args.label_maps, "r") as label_map_file:
        label_maps = [i.strip("\n") for i in label_map_file.readlines()]

    training_data_generator = SSD_DATA_GENERATOR(
        samples=training_samples,
        config=config,
        label_maps=label_maps,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=mobilenet_v2.preprocess_input
    )
    print('\tTraining data generator created')
    print(f'\tTraining set size: {num_training_samples}')

    validation_data_generator = SSD_DATA_GENERATOR(
        samples=validation_samples,
        config=config,
        label_maps=label_maps,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        augment=args.augment,
        process_input_fn=mobilenet_v2.preprocess_input
    )
    print('\tValidation data generator created')
    print(f'\tValidation set size: {num_validation_samples}')

    print('\n\tDone')

    return training_data_generator, num_training_samples, validation_data_generator, num_validation_samples


def get_characters_data_generator():
    pass