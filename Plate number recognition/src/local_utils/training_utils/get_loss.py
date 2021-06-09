from src.losses import SSD_LOSS


def get_loss(config):
    training_config = config['training']

    return SSD_LOSS(
        alpha=training_config['alpha'],
        min_negative_boxes=training_config['min_negative_boxes'],
        negative_boxes_ratio=training_config['negative_boxes_ratio']
    )
