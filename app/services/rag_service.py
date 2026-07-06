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

# attributes keys worth showing the agent; everything else (sku, currency,
# marketing copy) is noise that wastes tokens
_VARIANT_ATTR_KEYS = ("size", "color", "stock", "fabric")


def _compact_attributes(attrs) -> dict:
    """Reduce a product's attributes JSON to the fields the agent needs."""
    if not isinstance(attrs, dict):
        return {}
    out = {}
    for key in _VARIANT_ATTR_KEYS:
        value = attrs.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str) and len(value) > 60:
            value = value[:60] + "…"
        out[key] = value
    return out


class RagService:
    """Service for embedding generation and vector search via Supabase."""

    async def initialize(self):
        """Verify connectivity — lightweight startup check."""
        logger.info("RagService initialized (Supabase pgvector + gemini-embedding-2)")

    async def search_catalog(self, query: str, shop_id: str) -> list[dict]:
        """
        Search products via Supabase pgvector similarity, grouped by variant.

        Generates a 768-dim embedding for the query text, calls the Supabase
        RPC `match_products_hybrid` (FTS + cosine similarity), then expands
        each hit into its full variant family: in this catalog every size is
        a separate products row sharing the same name, and the RPC returns
        neither `attributes` (size/color/stock) nor the sibling sizes — so a
        second query fetches all rows with the matched names.

        Returns one dict per product NAME:
          name, price, description, image_url, all_image_urls (internal,
          for whitelisting), variants[{product_id, size, color, stock, ...}]
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

            # Unique matched names, best-match order
            names: list[str] = []
            for row in (result.data or []):
                n = row.get("name")
                if n and n not in names:
                    names.append(n)
            if not names:
                logger.info(f"Catalog search: 0 matches for \"{query[:60]}\" (shop={shop_id})")
                return []

            # Fetch ALL size/variant rows for the matched names, with attributes
            variant_rows = await supabase.table("products") \
                .select("id, name, price, description, image_url, attributes") \
                .eq("shop_id", shop_id) \
                .in_("name", names) \
                .execute()

            grouped: dict[str, dict] = {}
            for row in (variant_rows.data or []):
                name = row["name"]
                group = grouped.setdefault(name, {
                    "name": name,
                    "price": row.get("price"),
                    "description": row.get("description", ""),
                    "image_url": row.get("image_url") or "",
                    "all_image_urls": [],
                    "variants": [],
                })
                if row.get("image_url"):
                    if not group["image_url"]:
                        group["image_url"] = row["image_url"]
                    if row["image_url"] not in group["all_image_urls"]:
                        group["all_image_urls"].append(row["image_url"])

                if len(group["variants"]) >= 12:
                    continue  # runaway safety — no real product has more sizes
                variant = {"product_id": row["id"], **_compact_attributes(row.get("attributes"))}
                if row.get("price") is not None and row["price"] != group["price"]:
                    variant["price"] = row["price"]
                group["variants"].append(variant)

            # Preserve best-match order from the RPC
            products = [grouped[n] for n in names if n in grouped]

            logger.info(
                f"Catalog search: {len(products)} products "
                f"({sum(len(p['variants']) for p in products)} variants) "
                f"for \"{query[:60]}\" (shop={shop_id})"
            )
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
