from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ProductRecommendation(BaseModel):
    """Schema for a single product recommendation"""
    product_id: str = Field(..., description="Unique identifier of the product")
    product_name: str = Field(..., description="Name of the product")
    score: float = Field(..., description="Recommendation score")
    category: str = Field(..., description="Product category")
    brand: str = Field(..., description="Product brand")
    avgRating: float = Field(..., description="Average rating of the product")
    numReviews: int = Field(..., description="Number of product reviews")

    class Config:
        schema_extra = {
            "example": {
                "product_id": "product_123",
                "product_name": "Nike Running Shoes",
                "score": 0.95,
                "category": "Shoes",
                "brand": "Nike",
                "avgRating": 4.5,
                "numReviews": 100
            }
        }

class RecommendationResponse(BaseModel):
    """Schema for recommendation response"""
    recommendations: List[ProductRecommendation]

class ErrorResponse(BaseModel):
    """Schema for error response"""
    detail: str
