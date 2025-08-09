import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = os.path.join(BASE_DIR, "e-commerce-models")

# Model file paths
MODEL_FILES = {
    'svd_model': os.path.join(MODELS_PATH, 'svd_model.pkl'),
    'nmf_model': os.path.join(MODELS_PATH, 'nmf_model.pkl'),
    'knn_model': os.path.join(MODELS_PATH, 'knn_model.pkl'),
    'product_similarity_df': os.path.join(MODELS_PATH, 'product_similarity_df.pkl'),
    'product_popularity': os.path.join(MODELS_PATH, 'product_popularity.pkl'),
    'products_df': os.path.join(MODELS_PATH, 'products_df.pkl')
}

# API Settings
API_V1_STR = "/api/v1"
PROJECT_NAME = "E-commerce Recommendation System"
