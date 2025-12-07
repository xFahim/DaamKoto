"""Primary API router that includes all versioned routers."""

from fastapi import APIRouter
from app.api.v1.endpoints import facebook

api_router = APIRouter()

# Include versioned routers
api_router.include_router(
    facebook.router,
    prefix="/v1",
    tags=["facebook"],
)
