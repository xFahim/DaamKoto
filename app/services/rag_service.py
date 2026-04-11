"""RAG service for vector-based product search.

This service handles:
  - Vertex AI multimodal embeddings (text → vector, image → vector)
  - Pinecone vector similarity search

It does NOT handle LLM response generation — that's the agent's job.
"""

import asyncio
from typing import List
import httpx
from pinecone import Pinecone
from app.core.config import settings
from app.core.logging_config import get_logger
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
import json
from google.oauth2 import service_account

logger = get_logger(__name__)


class RagService:
    """Service for embedding generation and vector search — no LLM needed."""

    def __init__(self):
        self.embedding_model = None
        self.pinecone_index = None

    async def initialize(self):
        """Initialize Vertex AI (embeddings) and Pinecone (vector store)."""

        # 1. Vertex AI — for multimodal embeddings only
        try:
            service_account_json = settings.gcp_service_account_json
            if not service_account_json:
                logger.warning("GCP_SERVICE_ACCOUNT_JSON not found. Embeddings will be unavailable.")
            else:
                info = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(info)
                vertexai.init(
                    project=info["project_id"],
                    location="asia-east1",
                    credentials=credentials
                )
                logger.info("Vertex AI authenticated (embeddings)")

                self.embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

        except Exception as e:
            logger.error(f"Vertex AI authentication failed: {e}")

        # 2. Pinecone — vector store
        try:
            pc = Pinecone(api_key=settings.pinecone_api_key)
            self.pinecone_index = pc.Index("chatpulse-multimodal")
            logger.info("Pinecone connected successfully")
        except Exception as e:
            logger.error(f"Pinecone connection failed: {e}")

    async def search_catalog(self, query: str, page_id: str) -> list[dict]:
        """
        Search Pinecone for product matches and return raw JSON data for the Agent.
        """
        namespace = f"store_{page_id}"
        
        try:
            query_embedding = await self.get_multimodal_embedding(text=query)
            if not query_embedding:
                logger.warning("No embedding generated for catalog search")
                return []
                
            if not self.pinecone_index:
                logger.error("Pinecone index not initialized for catalog search")
                return []

            query_response = self.pinecone_index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace=namespace,
            )
            
            good_matches = []
            MIN_SCORE = 0.08
            if query_response.matches:
                for match in query_response.matches:
                    score = match.score
                    metadata = match.metadata or {}
                    
                    if score < MIN_SCORE:
                        continue
                        
                    good_matches.append({
                        "id": metadata.get("ID", ""),
                        "name": metadata.get("name", "Unknown"),
                        "price": metadata.get("price", "N/A"),
                        "stock": metadata.get("stock", "N/A"),
                        "description": metadata.get("description", ""),
                        "product_url": metadata.get("product_url", ""),
                        "image_url": metadata.get("url", ""),
                        "score": score,
                    })
            
            logger.info(f"Catalog search: {len(good_matches)} matches for \"{query[:60]}\" in {namespace}")
            return good_matches
            
        except Exception as e:
            logger.error(f"RAG catalog search failed: {e}", exc_info=True)
            return []

    async def get_multimodal_embedding(
        self, text: str | None = None, image_url: str | None = None
    ) -> List[float]:
        """
        Generate 1408-dim vector using Google Vertex AI Multimodal Embedding.
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return []

        try:
            if image_url:
                # Fetch image bytes
                async with httpx.AsyncClient(timeout=30.0) as client:
                    img_response = await client.get(image_url)
                    img_response.raise_for_status()
                    image_bytes = img_response.content
                    
                image = VertexImage(image_bytes)
                
                # Run blocking Vertex AI call in thread pool
                embeddings = await asyncio.to_thread(
                    self.embedding_model.get_embeddings, image=image
                )
                return embeddings.image_embedding
                
            elif text:
                # Run blocking Vertex AI call in thread pool
                embeddings = await asyncio.to_thread(
                    self.embedding_model.get_embeddings, contextual_text=text[:2000]
                )
                return embeddings.text_embedding
                
            return []

        except Exception as e:
            logger.error(f"Vertex AI embedding error: {repr(e)}")
            return []


rag_service = RagService()
