from typing import List, Dict, Any
import pandas as pd
from surprise import SVD, NMF, KNNBasic
import logging

logger = logging.getLogger(__name__)



class HybridRecommendationSystem:
    def __init__(self, svd_model, nmf_model, knn_model, product_similarity_df,
                 product_popularity, products_df, interactions_df):
        self.svd_model = svd_model
        self.nmf_model = nmf_model
        self.knn_model = knn_model
        self.product_similarity_df = product_similarity_df
        self.product_popularity = product_popularity # This should now be the time-decayed version
        self.products_df = products_df # This should now include one-hot encoded and popularity features
        self.interactions_df = interactions_df # This should now include user and temporal features

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

        print(f"Top collaborative: {cf_recs[:3]}")
        print(f"Top content-based: {cb_recs[:3]}")
        print(f"Top popular: {pop_recs[:3]}")


        # Get product details and convert numpy types
        recommendations_with_details = []
        for product_id, score in sorted_recs[:n_recommendations]:
            # Find product details using product_id
            product_details_row = self.products_df[self.products_df['product_id'] == product_id]
            if not product_details_row.empty:
                 product_details = product_details_row.iloc[0]
                 recommendations_with_details.append({
                    'product_id': product_id,
                    'product_name': product_details['name'],
                    'score': float(score), # Convert to float
                    'category': product_details['category'],
                    'brand': product_details['brand'],
                    'avgRating': float(product_details['avgRating']), # Convert to float
                    'numReviews': int(product_details['numReviews']) # Convert to int
                })

        return recommendations_with_details

    def get_recommendations_for_product(self, user_id, product_id, n_recommendations=10,
                                        content_weight=0.6, cf_weight=0.2, pop_weight=0.2):
        """
        Get recommendations for a user based on a specific product they are viewing.

        Parameters:
        - user_id: ID of the user.
        - product_id: ID of the product the user is currently viewing.
        - n_recommendations: Number of recommendations to return.
        - content_weight: Weight for content-based similarity.
        - cf_weight: Weight for collaborative filtering prediction.
        - pop_weight: Weight for popularity score.
        """
        if product_id not in self.product_similarity_df.index:
            print(f"Product {product_id} not found in similarity matrix. Returning popular products.")
            return handle_cold_start_user(self.product_popularity, self.products_df, n_recommendations)

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
            # Find product details using product_id
            product_details_row = self.products_df[self.products_df['product_id'] == product_id]
            if not product_details_row.empty:
                 product_details = product_details_row.iloc[0]
                 recommendations_with_details.append({
                    'product_id': product_id,
                    'product_name': product_details['name'],
                    'score': float(score), # Convert to float
                    'category': product_details['category'],
                    'brand': product_details['brand'],
                    'avgRating': float(product_details['avgRating']), # Convert to float
                    'numReviews': int(product_details['numReviews']) # Convert to int
                })

        return recommendations_with_details



def handle_cold_start_user(product_popularity_df, products_df, n_recommendations=10):
    """Handle recommendations for new users with no interaction history"""
    # For new users, return popular products
    popular_products = product_popularity_df.head(n_recommendations)

    recommendations = []
    for _, row in popular_products.iterrows():
        # Ensure product_id exists in products_df before accessing details
        product_details_row = products_df[products_df['product_id'] == row['product_id']]
        if not product_details_row.empty:
            product_details = product_details_row.iloc[0]
            recommendations.append({
                'product_id': row['product_id'],
                'product_name': row['name'],
                'score': float(row['popularity_score']), # Convert to float
                'category': product_details['category'],
                'brand': product_details['brand'],
                'avgRating': float(row['avgRating']), # Convert to float
                'numReviews': int(row['numReviews']) # Convert to int
            })

    return recommendations

