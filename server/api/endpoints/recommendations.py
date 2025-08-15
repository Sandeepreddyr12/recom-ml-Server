from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from models.schemas import ProductRecommendation, RecommendationResponse
from services.recommender import HybridRecommendationSystem
from services.model_loader import load_models

import pandas as pd
from services.recommender import handle_cold_start_user

logger = logging.getLogger(__name__)

router = APIRouter()

def get_recommender() -> HybridRecommendationSystem:
    """Dependency to get recommendation system instance"""
    try:
        models = load_models()
        return HybridRecommendationSystem(**models)
    except Exception as e:
        logger.error(f"Error initializing recommender: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize recommendation system")

@router.get("/recommendations/{user_id}", response_model=List[Dict[str, Any]])
async def get_user_recommendations(user_id: str, n_recommendations: int = 10):
    """
    Get recommendations for a specific user.

    Args:
        user_id: The ID of the user.
        n_recommendations: The number of recommendations to return.

    Returns:
        A list of recommended products with details.
    """
    try:
        models = load_models()
        # Check if user exists in interactions_df
        if user_id not in models['interactions_df']['userId'].unique():
            print(f"User {user_id} not found. Returning popular products.")
            # Pass product_popularity and products_df to handle_cold_start_user
            return handle_cold_start_user(models['product_popularity'], models['products_df'], n_recommendations)
        else:
            recommender = HybridRecommendationSystem(**models)
            recommendations = recommender.get_recommendations_for_user(user_id, n_recommendations)
            return recommendations
    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred while generating recommendations for user {user_id}: {e}")
        # Return an HTTP exception with status code 500 (Internal Server Error)
        raise HTTPException(status_code=500, detail="An internal server error occurred while generating recommendations.")
    

    
@router.get("/recommendations/{user_id}/{product_id}", 
            response_model=RecommendationResponse,
            tags=["recommendations"])
async def get_product_recommendations(
    user_id: str,
    product_id: str,
    n_recommendations: int = 10,
    recommender: HybridRecommendationSystem = Depends(get_recommender)
) -> RecommendationResponse:
    """
    Get recommendations for a specific product viewed by a user.

    Args:
        user_id: The ID of the user
        product_id: The ID of the product being viewed
        n_recommendations: Number of recommendations to return
        recommender: Recommendation system instance (injected)
        
    Returns:
        RecommendationResponse containing list of recommended products
        
    Raises:
        HTTPException: If error occurs during recommendation generation
    """
    try:
        recommendations = recommender.get_recommendations_for_product(
            user_id, 
            product_id, 
            n_recommendations
        )
        return RecommendationResponse(recommendations=recommendations)
    except Exception as e:
        logger.error(f"Error generating product recommendations for user {user_id} and product {product_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating product recommendations: {str(e)}"
        )
