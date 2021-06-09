from tensorflow.keras.optimizers import SGD


def get_optimizer(args):
    return SGD(
        lr=args.learning_rate,
        momentum=0.9,
        decay=0.0005,
        nesterov=False
    )
