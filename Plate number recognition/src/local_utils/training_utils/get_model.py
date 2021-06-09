from src.networks import SSD_MOBILENETV2


def get_model(config, args):
    with open(args.label_maps, mode='r') as label_map_file:
        label_maps = [i.strip('\n') for i in label_map_file.readlines()]

    return SSD_MOBILENETV2(
        config=config,
        label_maps=label_maps,
        is_training=True
    )
