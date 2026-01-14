"""Service layer for RAG ingestion processing."""

import uuid
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel
from pinecone import Pinecone
from typing import Any
import os
import json
from google.oauth2 import service_account
from app.core.config import settings

# Initialize Vertex AI
try:
    service_account_json = settings.gcp_service_account_json
    if service_account_json:
        info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(info)
        vertexai.init(
            project=info["project_id"],
            location="asia-east1",
            credentials=credentials
        )
except Exception as e:
    print(f"❌ Vertex AI Auth failed in IngestService: {e}")

# Initialize Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

# Connect to the multimodal index
index = pc.Index("chatpulse-multimodal")


class IngestService:
    """Service for handling RAG ingestion and vector storage."""

    def __init__(self):
        try:
            self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
        except Exception as e:
            print(f"⚠️ Failed to load MultiModalEmbeddingModel in IngestService: {e}")
            self.model = None

    async def process_and_upload(self, page_id: str, products: list) -> dict[str, Any]:
        """
        Process products and upload them to Pinecone as vectors.
        Handles GoodyBro JSON format with smart key mapping.
        """
        if not self.model:
            raise RuntimeError("Embedding model is not initialized")

        import asyncio  # Import inside method or at top level if checking imports

        namespace = f"store_{page_id}"
        vectors_to_upsert = []

        for product in products:
            # Smart Key Mapping with fallbacks
            name = (
                product.get("Product Name")
                or product.get("name")
                or "Unknown Product"
            )

            price = (
                product.get("price-item")
                or product.get("price")
                or "Unknown Price"
            )

            stock = product.get("badge") or "In Stock"
            
            product_url = product.get("product url", "")
            image_url = product.get("motion-reduce src") or product.get("image_url") or ""

            description = f"URL: {product_url} - Badge: {stock}"

            # ID
            vector_id = str(uuid.uuid4())

            # Construct rich embedding text (max 2000 chars roughly)
            text_to_embed = f"{name} {price} {stock} {description}"

            # Generate embedding using Vertex AI Multimodal
            try:
                # Text-only embedding for now since this endpoint receives JSON, not images files directly
                # If product has image_url, we COULD download and embed it, but for speed/simplicity of this endpoint
                # we'll stick to text embedding unless requested otherwise.
                # However, MultiModalEmbeddingModel produces 1408-dim vector for text too.
                
                # Run blocking call in thread
                embeddings = await asyncio.to_thread(
                    self.model.get_embeddings, contextual_text=text_to_embed
                )
                embedding_values = embeddings.text_embedding
            except Exception as e:
                print(f"❌ Error embedding product {name}: {e}")
                continue

            # Create vector record
            vector_record = {
                "id": vector_id,
                "values": embedding_values,
                "metadata": {
                    "name": name,
                    "price": price,
                    "stock": stock,
                    "description": description,
                    "page_id": page_id,
                    "url": image_url, # Adding image url to metadata
                    "product_url": product_url
                },
            }

            vectors_to_upsert.append(vector_record)

        # Upsert vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert, namespace=namespace)

        return {
            "success": True,
            "count": len(vectors_to_upsert),
            "namespace": namespace,
        }


ingest_service = IngestService()
