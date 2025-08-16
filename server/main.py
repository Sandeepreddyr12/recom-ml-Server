from fastapi import FastAPI
from api.endpoints import recommendations
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

# Include the recommendations router
app.include_router(
    recommendations.router,
    prefix="/api/v1",
    tags=["recommendations"]
)

@app.get("/")
async def read_root():
    """Root endpoint"""
    logger.debug("Root endpoint called")
    return {"message": "E-commerce Recommendation System API"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")