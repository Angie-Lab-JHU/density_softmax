from algorithms.VAE.src.models.wide_resnet_28_10 import Wide_Resnet_28_10
from algorithms.VAE.src.models.lenet_5 import Lenet_5


nets_map = {"lenet_5": Lenet_5, "wide_resnet_28_10": Wide_Resnet_28_10}


def get_model(name):
    if name not in nets_map:
        raise ValueError("Name of model unknown %s" % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn
