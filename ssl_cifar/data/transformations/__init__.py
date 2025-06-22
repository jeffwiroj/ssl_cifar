from .simsiam import simsiam_augmentation


def get_ssl_augmentations(ssl_type="simsiam"):
    if ssl_type == "simsiam":
        return simsiam_augmentation
