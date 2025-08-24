from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ProductRecommendation(BaseModel):
    """Schema for a single product recommendation"""
    id: str = Field(..., description="Unique identifier of the product")
    name: str = Field(..., description="Name of the product")
    slug: str = Field(..., description="URL-friendly product name")
    category: str = Field(..., description="Product category")
    images: List[str] = Field(default=[], description="Product images")
    brand: str = Field(..., description="Product brand")
    description: str = Field(..., description="Product description")
    price: float = Field(..., description="Current price")
    listPrice: float = Field(..., description="Original list price")
    countInStock: int = Field(..., description="Available stock")
    tags: List[str] = Field(default=[], description="Product tags")
    colors: List[str] = Field(default=[], description="Available colors")
    sizes: List[str] = Field(default=[], description="Available sizes")
    avgRating: float = Field(..., description="Average rating of the product")
    numReviews: int = Field(..., description="Number of product reviews")
    ratingDistribution: List[Dict[str, int]] = Field(default=[], description="Distribution of ratings")
    numSales: int = Field(..., description="Number of sales")
    isPublished: bool = Field(..., description="Product visibility status")
    reviews: List[Any] = Field(default=[], description="Product reviews")
    # __v: int = Field(..., description="Version key")
    createdAt: str = Field(..., description="Creation timestamp")
    updatedAt: str = Field(..., description="Last update timestamp")

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
