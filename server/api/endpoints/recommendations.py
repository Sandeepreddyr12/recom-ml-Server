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
        # Validate input
        if not user_id or not isinstance(user_id, str):
            raise HTTPException(status_code=400, detail="Invalid user ID")
        if n_recommendations <= 0:
            raise HTTPException(status_code=400, detail="Number of recommendations must be positive")

        # Load models and data
        try:
            models = load_models()
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize recommendation system")

        # Validate required data is available
        if models.get('interactions_df') is None:
            logger.error("No interaction data available")
            raise HTTPException(status_code=500, detail="Recommendation system is not properly initialized")

        # Check if user exists in interactions_df
        if user_id not in models['interactions_df']['userId'].unique():
            logger.info(f"User {user_id} not found. Returning popular products.")
            recommendations = handle_cold_start_user(
                models.get('product_popularity'), 
                models.get('db_products', []), 
                n_recommendations
            )
            if not recommendations:
                logger.warning("No recommendations available for cold start user")
                return []
            return recommendations

        # Generate recommendations for existing user
        recommender = HybridRecommendationSystem(**models)
        recommendations = recommender.get_hybrid_recommendations(user_id, n_recommendations)
        
        if not recommendations:
            logger.warning(f"No recommendations generated for user {user_id}")
            return []
            
        return recommendations

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"An error occurred while generating recommendations for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An internal server error occurred while generating recommendations."
        )
    

    
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
