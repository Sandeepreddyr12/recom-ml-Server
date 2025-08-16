import pickle
import pandas as pd
from config import MODEL_FILES
import logging
import os

logger = logging.getLogger(__name__)

def load_models():
    """Load all required ML models and data"""
    try:
        models = {}
        required_models = [
            'svd_model',
            'nmf_model',
            'knn_model',
            'product_similarity_df',
            'product_popularity',
            'products_df'
        ]
        
        for name in required_models:
            path = MODEL_FILES[name]
            logger.info(f"Loading {name} from {path}")
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            logger.info(f"Successfully loaded {name}")

        # Load interactions_df from e-commerce-data directory
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'e-commerce-data', 'interactions.csv')
        logger.info(f"Loading interactions_df from {data_path}")
        models['interactions_df'] = pd.read_csv(data_path)
        logger.info("Successfully loaded interactions_df")
        
        return models
    except FileNotFoundError as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError(f"Failed to load models: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading models: {e}")
        raise RuntimeError(f"Unexpected error loading models: {e}")
