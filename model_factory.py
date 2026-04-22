from models.svd_model import SVDModel
from models.svd_decomposition import SVDRecommender as SVDCF
from models.svd_surprise import SurpriseSVDWrapper
from models.item_cf import ItemBasedCF
from content.feature_extractor import ContentAnalyzer

MODEL_REGISTRY = {
    "svd_model": {"class": SVDModel, "type": "cf"},
    "naive_svd": {"class": SVDCF, "type": "cf"},
    "item_cf": {"class": ItemBasedCF, "type": "cf"},
    "surprise_svd": {"class": SurpriseSVDWrapper, "type": "cf"},
    "content_based": {"class": ContentAnalyzer, "type": "cb"}
}

def get_model(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    entry = MODEL_REGISTRY[model_name]
    return entry["class"](), entry["type"]
