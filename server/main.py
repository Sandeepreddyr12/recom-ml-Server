from typing import Union
from fastapi import FastAPI, HTTPException
from services.model_loader import load_models
from services.recommender import HybridRecommendationSystem
import logging
import uvicorn

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with debug settings
app = FastAPI(
    title="E-commerce Recommendation API",
    description="API for hybrid product recommendations",
    version="1.0.0",
    debug=True
)

# Load models and initialize recommender system
try:
    logger.debug("Starting to load models...")
    models = load_models()
    recommender = HybridRecommendationSystem(**models)
    logger.info("Recommender system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommender system: {e}")
    raise


@app.get("/")
async def read_root():
    """Root endpoint"""
    logger.debug("Root endpoint called")
    return {"message": "E-commerce Recommendation System API"}


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, n_recommendations: int = 10):
    """
    Get personalized recommendations for a user

    Args:
        user_id: User ID for whom to generate recommendations
        n_recommendations: Number of recommendations to return (default: 10)
    """
    try:
        recommendations = recommender.get_hybrid_recommendations(
            user_id,
            n_recommendations
        )
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")