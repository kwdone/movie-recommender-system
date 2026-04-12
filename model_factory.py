from models.svd_model import SVDModel
from models.svd_decomposition import SVDRecommender as SVDCF
from models.svd_surprise import SurpriseSVDWrapper
from models.item_cf import ItemBasedCF

def get_model(model_name):
    if model_name == "svd_model":
        return SVDModel()
    elif model_name == "naive_svd":
        return SVDCF()
    elif model_name == "item_cf":
        return ItemBasedCF()
    elif model_name == "surprise_svd":
        return SurpriseSVDWrapper()
    else: 
        raise ValueError("Unknown model")
