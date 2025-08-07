from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import os
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from surprise import SVD, NMF, KNNBasic

app = FastAPI()

# Load the models and data
MODELS_PATH = os.path.join(os.path.dirname(__file__), "e-commerce-models")

class HybridRecommendationSystem:
    def __init__(self, svd_model, nmf_model, knn_model, product_similarity_df,
                product_popularity, products_df, interactions_df):
        self.svd_model = svd_model
        self.nmf_model = nmf_model
        self.knn_model = knn_model
        self.product_similarity_df = product_similarity_df
        self.product_popularity = product_popularity
        self.products_df = products_df
        self.interactions_df = interactions_df

    def get_user_interactions(self, user_id):
        """Get products a user has interacted with"""
        if self.interactions_df is not None and user_id in self.interactions_df['userId'].unique():
            return self.interactions_df[self.interactions_df['userId'] == user_id]['productId'].unique()
        return []

    def collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get collaborative filtering recommendations"""
        all_product_ids = self.products_df['product_id'].unique()
        user_product_ids = self.get_user_interactions(user_id)
        products_to_predict_ids = [p_id for p_id in all_product_ids if p_id not in user_product_ids]

        predictions = []
        for product_id in products_to_predict_ids:
            try:
                svd_pred = self.svd_model.predict(user_id, product_id).est
                nmf_pred = self.nmf_model.predict(user_id, product_id).est
                knn_pred = self.knn_model.predict(user_id, product_id).est
                avg_pred = 0.4 * svd_pred + 0.2 * nmf_pred + 0.4 * knn_pred
                predictions.append((product_id, avg_pred))
            except Exception:
                pass

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def content_based_recommendations(self, user_id, n_recommendations=10):
        """Get content-based recommendations"""
        user_product_ids = self.get_user_interactions(user_id)

        if len(user_product_ids) == 0:
            return []

        recommendations = {}
        for product_id in user_product_ids:
            if product_id in self.product_similarity_df.columns:
                similarities = self.product_similarity_df[product_id]
                for idx, similarity in similarities.items():
                    if idx not in user_product_ids and idx != product_id:
                        if idx not in recommendations:
                            recommendations[idx] = 0
                        recommendations[idx] += similarity

        for product_id in recommendations:
            recommendations[product_id] /= len(user_product_ids)

        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]

    def popularity_recommendations(self, user_id, n_recommendations=10):
        """Get popularity-based recommendations"""
        user_product_ids = self.get_user_interactions(user_id)
        popular_products = self.product_popularity[
            ~self.product_popularity['product_id'].isin(user_product_ids)
        ]

        recommendations = []
        for _, row in popular_products.head(n_recommendations).iterrows():
            recommendations.append((row['product_id'], row['popularity_score']))

        return recommendations

    def get_hybrid_recommendations(self, user_id, n_recommendations=10,
                               cf_weight=0.5, cb_weight=0.3, pop_weight=0.2):
        """Get hybrid recommendations combining all approaches"""
        cf_recs = self.collaborative_recommendations(user_id, n_recommendations * 2)
        cb_recs = self.content_based_recommendations(user_id, n_recommendations * 2)
        pop_recs = self.popularity_recommendations(user_id, n_recommendations * 2)

        final_recommendations = {}

        for product_id, score in cf_recs:
            final_recommendations[product_id] = cf_weight * score

        for product_id, score in cb_recs:
            if product_id in final_recommendations:
                final_recommendations[product_id] += cb_weight * score
            else:
                final_recommendations[product_id] = cb_weight * score

        for product_id, score in pop_recs:
            if product_id in final_recommendations:
                final_recommendations[product_id] += pop_weight * score
            else:
                final_recommendations[product_id] = pop_weight * score

        sorted_recs = sorted(final_recommendations.items(),
                         key=lambda x: x[1], reverse=True)

        recommendations_with_details = []
        for product_id, score in sorted_recs[:n_recommendations]:
            product_details_row = self.products_df[self.products_df['product_id'] == product_id]
            if not product_details_row.empty:
                product_details = product_details_row.iloc[0]
                recommendations_with_details.append({
                    'product_id': product_id,
                    'product_name': product_details['name'],
                    'score': float(score),
                    'category': product_details['category'],
                    'brand': product_details['brand'],
                    'avgRating': float(product_details['avgRating']),
                    'numReviews': int(product_details['numReviews'])
                })

        return recommendations_with_details

def handle_cold_start_user(product_popularity_df, products_df, n_recommendations=10):
    """Handle recommendations for new users with no interaction history"""
    popular_products = product_popularity_df.head(n_recommendations)
    recommendations = []
    for _, row in popular_products.iterrows():
        product_details_row = products_df[products_df['product_id'] == row['product_id']]
        if not product_details_row.empty:
            product_details = product_details_row.iloc[0]
            recommendations.append({
                'product_id': row['product_id'],
                'product_name': row['name'],
                'score': float(row['popularity_score']),
                'category': product_details['category'],
                'brand': product_details['brand'],
                'avgRating': float(row['avgRating']),
                'numReviews': int(row['numReviews'])
            })
    return recommendations

# Load all required models and data
try:
    with open(os.path.join(MODELS_PATH, 'svd_model.pkl'), 'rb') as f:
        svd_model = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'nmf_model.pkl'), 'rb') as f:
        nmf_model = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'knn_model.pkl'), 'rb') as f:
        knn_model = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'product_similarity_df.pkl'), 'rb') as f:
        product_similarity_df = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'product_popularity.pkl'), 'rb') as f:
        product_popularity = pickle.load(f)
    with open(os.path.join(MODELS_PATH, 'products_df.pkl'), 'rb') as f:
        products_df = pickle.load(f)
    interactions_df = pd.read_csv(os.path.join(MODELS_PATH, 'interactions.csv'))
    interactions_df['createdAt'] = pd.to_datetime(interactions_df['createdAt'])

    # Initialize the recommendation system
    hybrid_system = HybridRecommendationSystem(
        svd_model, nmf_model, knn_model,
        product_similarity_df, product_popularity,
        products_df, interactions_df
    )

except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    raise RuntimeError("Failed to load recommendation models") from e

@app.get("/")
async def read_root():
    return {"message": "E-commerce Recommendation System API"}

@app.get("/recommendations/{user_id}", response_model=List[Dict[str, Any]])
async def get_user_recommendations(user_id: str, n_recommendations: int = 10):
    """Get recommendations for a specific user"""
    try:
        if user_id not in interactions_df['userId'].unique():
            print(f"User {user_id} not found. Returning popular products.")
            return handle_cold_start_user(product_popularity, products_df, n_recommendations)
        else:
            recommendations = hybrid_system.get_hybrid_recommendations(user_id, n_recommendations)
            return recommendations
    except Exception as e:
        print(f"Error generating recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
