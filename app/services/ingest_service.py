"""Service layer for RAG ingestion processing."""

import uuid
import google.generativeai as genai
from pinecone import Pinecone
from typing import Any
from app.core.config import settings

# Configure Gemini AI
genai.configure(api_key=settings.gemini_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

# Connect to the specific index
index = pc.Index("chatpulse")


class IngestService:
    """Service for handling RAG ingestion and vector storage."""

    @staticmethod
    def process_and_upload(page_id: str, products: list) -> dict[str, Any]:
        """
        Process products and upload them to Pinecone as vectors.
        Handles GoodyBro JSON format with smart key mapping.

        Args:
            page_id: The page/store ID to associate with the products
            products: List of product dictionaries (GoodyBro format or standard format)

        Returns:
            Dictionary with count and namespace information
        """
        namespace = f"store_{page_id}"
        vectors_to_upsert = []

        for product in products:
            # Smart Key Mapping with fallbacks
            # Name: Try "Product Name", fallback to "name", fallback to "Unknown Product"
            name = (
                product.get("Product Name")
                or product.get("name")
                or "Unknown Product"
            )

            # Price: Try "price-item", fallback to "price", fallback to "Unknown Price"
            price = (
                product.get("price-item")
                or product.get("price")
                or "Unknown Price"
            )

            # Stock: Try "badge" (e.g., "-25% OFF"), fallback to "In Stock"
            stock = product.get("badge") or "In Stock"

            # Description: Construct dynamically from URL and badge
            product_url = product.get("product url", "")
            badge = product.get("badge", "")
            description = f"URL: {product_url} - Badge: {badge}"

            # ID: Generate UUID since JSON has no ID field
            vector_id = str(uuid.uuid4())

            # Construct rich embedding text
            text_to_embed = f"{name} {price} {stock} {description}"

            # Generate embedding using Gemini
            embedding_result = genai.embed_content(
                model="models/text-embedding-004",
                content=text_to_embed,
            )
            embedding = embedding_result["embedding"]

            # Create vector record with standardized metadata keys
            vector_record = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "name": name,
                    "price": price,
                    "stock": stock,
                    "description": description,
                    "page_id": page_id,
                },
            }

            vectors_to_upsert.append(vector_record)

        # Upsert vectors to the specific namespace
        index.upsert(vectors=vectors_to_upsert, namespace=namespace)

        return {
            "success": True,
            "count": len(vectors_to_upsert),
            "namespace": namespace,
        }


ingest_service = IngestService()

