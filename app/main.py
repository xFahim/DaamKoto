"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.logging_config import setup_logging, get_logger
from app.api.router import api_router
from app.services.rag_service import rag_service
from app.services.agent_service import agent_service
from app.services.batching_service import message_batcher

# Initialize logging FIRST — before any other module logs anything
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize services
    logger.info("Starting up and initializing services...")
    agent_service.initialize()
    await rag_service.initialize()
    logger.info("All services initialized successfully.")
    yield
    # Shutdown: Clean up resources if needed
    logger.info("Shutting down...")
    await message_batcher.shutdown()
    logger.info("Shutdown complete.")

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
