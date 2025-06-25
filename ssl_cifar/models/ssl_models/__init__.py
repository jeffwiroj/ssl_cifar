from .barlow_twins import BarlowTwins
from .simsiam import SimSiam


def get_ssl_model(model_name="simsiam", **kwargs):
    if model_name == "simsiam":
        return SimSiam(**kwargs)
    elif model_name == "barlow_twins":
        return BarlowTwins(**kwargs)
