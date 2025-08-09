from typing import List, Dict, Any
import pandas as pd
from surprise import SVD, NMF, KNNBasic
import logging

logger = logging.getLogger(__name__)

class HybridRecommendationSystem:
    def __init__(self, 
                 svd_model: SVD,
                 nmf_model: NMF,
                 knn_model: KNNBasic,
                 product_similarity_df: pd.DataFrame,
                 product_popularity: pd.DataFrame,
                 products_df: pd.DataFrame):  # Removed interactions_df
        """Initialize the hybrid recommendation system."""
        self.svd_model = svd_model
        self.nmf_model = nmf_model
        self.knn_model = knn_model
        self.product_similarity_df = product_similarity_df
        self.product_popularity = product_popularity
        self.products_df = products_df
        logger.info("HybridRecommendationSystem initialized successfully")

    def get_user_interactions(self, user_id: str) -> List[str]:
        """
        Get products a user has interacted with.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of product IDs the user has interacted with
        """
        try:
            if self.interactions_df is not None and user_id in self.interactions_df['userId'].unique():
                return self.interactions_df[self.interactions_df['userId'] == user_id]['productId'].unique()
            return []
        except Exception as e:
            logger.error(f"Error getting user interactions: {e}")
            return []

    def collaborative_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[tuple]:
        """
        Get collaborative filtering recommendations using ensemble of models.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (product_id, score)
        """
        try:
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
                except Exception as e:
                    logger.warning(f"Error predicting for product {product_id}: {e}")
                    continue

            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []

    def content_based_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[tuple]:
        """
        Get content-based recommendations using product similarities.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (product_id, score)
        """
        try:
            user_product_ids = self.get_user_interactions(user_id)
            if not user_product_ids:
                return []

            recommendations = {}
            for product_id in user_product_ids:
                if product_id in self.product_similarity_df.columns:
                    similarities = self.product_similarity_df[product_id]
                    for idx, similarity in similarities.items():
                        if idx not in user_product_ids and idx != product_id:
                            recommendations[idx] = recommendations.get(idx, 0) + similarity

            for product_id in recommendations:
                recommendations[product_id] /= len(user_product_ids)

            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return sorted_recs[:n_recommendations]
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []

    def popularity_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[tuple]:
        """
        Get popularity-based recommendations.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (product_id, score)
        """
        try:
            user_product_ids = self.get_user_interactions(user_id)
            popular_products = self.product_popularity[
                ~self.product_popularity['product_id'].isin(user_product_ids)
            ]

            recommendations = []
            for _, row in popular_products.head(n_recommendations).iterrows():
                recommendations.append((row['product_id'], row['popularity_score']))
            return recommendations
        except Exception as e:
            logger.error(f"Error in popularity recommendations: {e}")
            return []

    def get_hybrid_recommendations(self, user_id: str, n_recommendations: int = 10,
                               cf_weight: float = 0.5, cb_weight: float = 0.3,
                               pop_weight: float = 0.2) -> List[Dict[str, Any]]:
        """
        Get hybrid recommendations combining all approaches.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return
            cf_weight: Weight for collaborative filtering recommendations
            cb_weight: Weight for content-based recommendations
            pop_weight: Weight for popularity-based recommendations
            
        Returns:
            List of dictionaries containing product recommendations with details
        """
        try:
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

            sorted_recs = sorted(final_recommendations.items(), key=lambda x: x[1], reverse=True)

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
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return []
