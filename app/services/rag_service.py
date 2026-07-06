"""RAG service for vector-based product search via Supabase pgvector.

This service handles:
  - Generating text embeddings via gemini-embedding-2 (768-dim)
  - Vector similarity search via Supabase RPC (pgvector)

It does NOT handle LLM response generation — that's the agent's job.
"""

from google.genai import types
from app.core.dependencies import genai_client, get_supabase
from app.core.logging_config import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


class RagService:
    """Service for embedding generation and vector search via Supabase."""

    async def initialize(self):
        """Verify connectivity — lightweight startup check."""
        logger.info("RagService initialized (Supabase pgvector + gemini-embedding-2)")

    async def search_catalog(self, query: str, shop_id: str) -> list[dict]:
        """
        Search products via Supabase pgvector similarity.

        Generates a 768-dim embedding for the query text, then calls the
        Supabase RPC function `match_products_hybrid` (FTS + cosine similarity).
        """
        try:
            query_embedding = await self.get_text_embedding(text=query)
            if not query_embedding:
                logger.warning("No embedding generated for catalog search")
                return []

            # Call Supabase RPC function for hybrid search (FTS + pgvector)
            supabase = await get_supabase()
            result = await supabase.rpc("match_products_hybrid", {
                "query_text": query,
                "query_embedding": query_embedding,
                "match_count": 5,
                "filter_shop_id": shop_id,
            }).execute()

            products = []
            for row in (result.data or []):
                products.append({
                    "id": row.get("id"),
                    "name": row.get("name", "Unknown"),
                    "price": row.get("price", "N/A"),
                    "description": row.get("description", ""),
                    "image_url": row.get("image_url", ""),
                    "product_url": row.get("product_url", ""),
                    "score": row.get("similarity", 0),
                })

            logger.info(f"Catalog search: {len(products)} matches for \"{query[:60]}\" (shop={shop_id})")
            return products

        except Exception as e:
            logger.error(f"RAG catalog search failed: {e}", exc_info=True)
            return []

    async def get_text_embedding(self, text: str) -> list[float]:
        """Generate 768-dim embedding using gemini-embedding-2."""
        try:
            result = await genai_client.aio.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text[:2000],
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding generation error: {repr(e)}")
            return []


rag_service = RagService()
