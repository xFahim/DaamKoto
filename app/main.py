"""FastAPI application entry point."""

from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(
    title="DaamKoto",
    description="A robust, scalable FastAPI project for integrating Facebook Messenger Webhooks",
    version="1.0.0",
)

# Include the primary API router
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to DaamKoto API"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
