"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.router import api_router
from app.services.rag_service import rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize services
    print("🚀 Starting up and initializing services...")
    await rag_service.initialize()
    yield
    # Shutdown: Clean up resources if needed
    print("🛑 Shutting down...")

app = FastAPI(
    title="DaamKoto",
    description="A robust, scalable FastAPI project for integrating Facebook Messenger Webhooks",
    version="1.0.0",
    lifespan=lifespan
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
