from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from models.schemas import ProductRecommendation, RecommendationResponse
from services.recommender import HybridRecommendationSystem
from services.model_loader import load_models

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

@router.get("/recommendations/{user_id}", 
            response_model=RecommendationResponse,
            tags=["recommendations"])
async def get_user_recommendations(
    user_id: str,
    n_recommendations: int = 10,
    recommender: HybridRecommendationSystem = Depends(get_recommender)
) -> RecommendationResponse:
    """
    Get personalized recommendations for a user.
    
    Args:
        user_id: The unique identifier of the user
        n_recommendations: Number of recommendations to return
        recommender: Recommendation system instance (injected)
        
    Returns:
        RecommendationResponse containing list of recommended products
        
    Raises:
        HTTPException: If error occurs during recommendation generation
    """
    try:
        recommendations = recommender.get_hybrid_recommendations(
            user_id, n_recommendations
        )
        return RecommendationResponse(recommendations=recommendations)
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )
