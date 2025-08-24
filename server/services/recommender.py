from typing import List, Dict, Any
import pandas as pd
from surprise import SVD, NMF, KNNBasic
import logging
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)




class HybridRecommendationSystem:
    def __init__(self, svd_model, nmf_model, knn_model, product_similarity_df,
                 product_popularity, products_df, interactions_df, db_products):
        self.svd_model = svd_model
        self.nmf_model = nmf_model
        self.knn_model = knn_model
        self.product_similarity_df = product_similarity_df
        self.product_popularity = product_popularity # This should now be the time-decayed version
        self.products_df = products_df # This should now include one-hot encoded and popularity features
        self.interactions_df = interactions_df # This should now include user and temporal features
        self.db_products = db_products


    


    def get_user_interactions(self, user_id):
        """Get products a user has interacted with"""
        # Ensure interactions_df is available and user_id is valid
        if self.interactions_df is not None and user_id in self.interactions_df['userId'].unique():
            # Return product_id instead of name for consistency
            return self.interactions_df[self.interactions_df['userId'] == user_id]['productId'].unique()
        return []


    def collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get collaborative filtering recommendations"""
        # Get all product IDs
        all_product_ids = self.products_df['product_id'].unique()

        # Get product IDs user hasn't interacted with
        user_product_ids = self.get_user_interactions(user_id)
        products_to_predict_ids = [p_id for p_id in all_product_ids if p_id not in user_product_ids]

        # Get predictions from all models
        predictions = []
        for product_id in products_to_predict_ids:
            # Handle potential errors during prediction, e.g., unknown user or item
            try:
                # Surprise models predict based on user_id and item_id (productId in our case)
                svd_pred = self.svd_model.predict(user_id, product_id).est
                nmf_pred = self.nmf_model.predict(user_id, product_id).est
                knn_pred = self.knn_model.predict(user_id, product_id).est

                # Weighted average of predictions
                avg_pred = 0.4 * svd_pred + 0.2 * nmf_pred + 0.4 * knn_pred
                predictions.append((product_id, avg_pred))
            except Exception as e:
                # Optionally log the error or handle specific exceptions
                # print(f"Error predicting for user {user_id} and product {product_id}: {e}")
                pass # Skip this product if prediction fails

        # Sort by prediction score
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def content_based_recommendations(self, user_id, n_recommendations=10):
        """Get content-based recommendations"""
        # Get user's interaction history (product IDs)
        user_product_ids = self.get_user_interactions(user_id)

        if len(user_product_ids) == 0:
            return []

        # Get weighted average similarity for all products
        recommendations = {}
        for product_id in user_product_ids:
            if product_id in self.product_similarity_df.columns:
                similarities = self.product_similarity_df[product_id]
                for idx, similarity in similarities.items():
                    # idx is product_id
                    if idx not in user_product_ids and idx != product_id:
                        if idx not in recommendations:
                            recommendations[idx] = 0
                        recommendations[idx] += similarity

        # Average the similarities
        for product_id in recommendations:
            recommendations[product_id] /= len(user_product_ids)

        # Sort by similarity score
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]

    def popularity_recommendations(self, user_id, n_recommendations=10):
        """Get popularity-based recommendations"""
        # Get product IDs user hasn't interacted with
        user_product_ids = self.get_user_interactions(user_id)

        # Filter out products user has already interacted with based on product_id
        popular_products = self.product_popularity[
            ~self.product_popularity['product_id'].isin(user_product_ids)
        ]

        # Return top popular products (return product_id and score)
        recommendations = []
        for _, row in popular_products.head(n_recommendations).iterrows():
             recommendations.append((row['product_id'], row['popularity_score']))


        return recommendations


    def get_hybrid_recommendations(self, user_id, n_recommendations=10,
                                 cf_weight=0.5, cb_weight=0.3, pop_weight=0.2):
        """Get hybrid recommendations combining all approaches"""
        # Get recommendations from each component (returns product_id and score)
        cf_recs = self.collaborative_recommendations(user_id, n_recommendations * 2)
        cb_recs = self.content_based_recommendations(user_id, n_recommendations * 2)
        pop_recs = self.popularity_recommendations(user_id, n_recommendations * 2)

        # Combine recommendations with weights
        final_recommendations = {}

        # Add collaborative filtering recommendations
        for product_id, score in cf_recs:
            final_recommendations[product_id] = cf_weight * score

        # Add content-based recommendations
        for product_id, score in cb_recs:
            if product_id in final_recommendations:
                final_recommendations[product_id] += cb_weight * score
            else:
                final_recommendations[product_id] = cb_weight * score

        # Add popularity recommendations
        for product_id, score in pop_recs:
            if product_id in final_recommendations:
                final_recommendations[product_id] += pop_weight * score
            else:
                final_recommendations[product_id] = pop_weight * score

        # Sort by final score
        sorted_recs = sorted(final_recommendations.items(),
                           key=lambda x: x[1], reverse=True)


        # Get product details and convert numpy types
        recommendations_with_details = []
        for product_id, score in sorted_recs[:n_recommendations]:
            # Find product details using product_id from db_products
            product_details = next((item for item in self.db_products if item.get('_id') == product_id), None)
            if product_details:
                recommendations_with_details.append(format_product_details(product_details, score))
        return recommendations_with_details

    def get_recommendations_for_product(self, user_id, product_id, n_recommendations=10,
                                        content_weight=0.6, cf_weight=0.2, pop_weight=0.2):
        """
        Get recommendations for a user based on a specific product they are viewing.

        Parameters:
        - user_id: ID of the user.
        - product_id: ID of the product being viewed.
        - n_recommendations: Number of recommendations to return.
        - content_weight: Weight for content-based similarity.
        - cf_weight: Weight for collaborative filtering prediction.
        - pop_weight: Weight for popularity score.
        """
        if product_id not in self.product_similarity_df.index:
            print(f"Product {product_id} not found in similarity matrix. Returning popular products.")
            return handle_cold_start_user(self.product_popularity, self.db_products, n_recommendations)

        # Get content-based similar products
        # Exclude the product itself and products the user has already interacted with
        user_product_ids = self.get_user_interactions(user_id)
        similarities = self.product_similarity_df[product_id].drop(product_id, errors='ignore')
        similarities = similarities[~similarities.index.isin(user_product_ids)]

        # Combine recommendations with weights
        final_recommendations = {}

        for similar_product_id, content_score in similarities.items():
            final_recommendations[similar_product_id] = content_weight * content_score

            # Add collaborative filtering prediction (if user is not cold-start)
            if user_id in self.interactions_df['userId'].unique():
                try:
                    cf_pred = self.svd_model.predict(user_id, similar_product_id).est # Using SVD as the representative CF model
                    final_recommendations[similar_product_id] += cf_weight * cf_pred
                except Exception:
                    pass # Ignore if prediction fails (e.g., unknown item in CF)

            # Add popularity score
            popularity_score = self.product_popularity.loc[
                self.product_popularity['product_id'] == similar_product_id,
                'popularity_score'
            ].iloc[0] if similar_product_id in self.product_popularity['product_id'].values else 0.0
            final_recommendations[similar_product_id] += pop_weight * popularity_score


        # Sort by final score
        sorted_recs = sorted(final_recommendations.items(),
                           key=lambda x: x[1], reverse=True)

        # Get product details and convert numpy types
        recommendations_with_details = []
        for product_id, score in sorted_recs[:n_recommendations]:
            # Find product details using product_id from db_products
            product_details = next((item for item in self.db_products if item.get('_id') == product_id), None)
            if product_details:
                recommendations_with_details.append(format_product_details(product_details, score))

        return recommendations_with_details



def format_product_details(product_details, score):
    rating_dist = product_details.get('ratingDistribution', [])
    if isinstance(rating_dist, list):
        rating_dist_dict = {str(item['rating']): item['count'] for item in rating_dist}
    else:
        rating_dist_dict = rating_dist if isinstance(rating_dist, dict) else {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    return {
        '_id': str(product_details['_id']),
        'name': product_details['name'],
        'score': float(score),
        'category': product_details['category'],
        'brand': product_details['brand'],
        'description': product_details.get('description', ''),
        'price': float(product_details.get('price', 0.0)),
        'listPrice': float(product_details.get('listPrice', 0.0)),
        'images': product_details.get('images', []),
        'colors': product_details.get('colors', []),
        'sizes': product_details.get('sizes', []),
        'tags': product_details.get('tags', []),
        'countInStock': int(product_details.get('countInStock', 0)),
        'slug': product_details.get('slug', ''),
        'avgRating': float(product_details.get('avgRating', 0.0)),
        'numReviews': int(product_details.get('numReviews', 0)),
        'numSales': int(product_details.get('numSales', 0)),
        'isPublished': product_details.get('isPublished', True),
        'createdAt': product_details.get('createdAt', ''),
        'updatedAt': product_details.get('updatedAt', ''),
        "ratingDistribution": rating_dist_dict
    }

def handle_cold_start_user(product_popularity_df, db_products_list, n_recommendations=10, cache_duration_hours=120):
    """Handle recommendations for new users with no interaction history with caching"""
    try:
        cache_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'e-commerce-data', 
                                     'cold_start_cache.json')
        
        # Try to load from cache first
        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < timedelta(hours=cache_duration_hours):
                    logger.info("Returning cold start recommendations from cache")
                    return cache_data.get('recommendations', [])
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
                # Continue with normal processing if cache read fails
        
        # If we get here, either cache doesn't exist, is invalid, or had an error
        # Process recommendations normally
        if product_popularity_df is None or product_popularity_df.empty:
            logger.warning("No product popularity data available")
            return []
            
        if not db_products_list:
            logger.warning("No product details available")
            return []

        # For new users, return popular products
        popular_products = product_popularity_df.head(n_recommendations)
        if popular_products.empty:
            logger.warning("No popular products found")
            return []

        recommendations = []
        for _, row in popular_products.iterrows():
            product_id = row['product_id']
            if not product_id:
                continue
                
            # Find product details using product_id from db_products_list
            product_details = next((item for item in db_products_list if item.get('_id') == product_id), None)
            if product_details:
                recommendations.append(format_product_details(product_details, float(row['popularity_score'])))
        
        # Cache the results
        if recommendations:
            try:
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'recommendations': recommendations
                }
                with open(cache_file_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
                logger.info("Cold start recommendations cached successfully")
            except Exception as e:
                logger.warning(f"Error writing to cache file: {e}")
                # Continue even if caching fails
                
        return recommendations
    except Exception as e:
        logger.error(f"Error in handle_cold_start_user: {str(e)}")
        return []

