from tensorflow.keras.applications import mobilenet_v2

from src.networks import SSD_MOBILENETV2


def inference_ssd_mobilenetv2(config, args):
    with open(args.label_maps, mode='r') as file:
        label_maps = [line.strip('\n') for line in file.readlines()]

    model = SSD_MOBILENETV2(
        config,
        label_maps,
        is_training=False,
        num_predictions=args.num_predictions)

    process_input_fn = mobilenet_v2.preprocess_input

    return model, label_maps, process_input_fn
