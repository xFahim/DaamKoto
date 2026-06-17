"""Supabase webhook endpoint for product embedding generation.

When the Next.js admin dashboard inserts a new product into Supabase,
a database webhook fires to this endpoint. We immediately return 200 OK,
then generate the embedding in the background and update the product row.
"""

import httpx
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from google.genai import types
from app.core.dependencies import genai_client, get_supabase
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


# ── Payload Schema ───────────────────────────────────────────────────────

class ProductRecord(BaseModel):
    """Fields from the Supabase products table row."""
    id: str
    shop_id: str
    name: str
    description: str | None = None
    attributes: dict | list | None = None
    image_url: str | None = None


class SupabaseWebhookPayload(BaseModel):
    """Supabase database webhook payload (INSERT event)."""
    type: str = "INSERT"
    table: str = "products"
    schema_: str = Field(default="public", alias="schema")
    record: ProductRecord
    old_record: dict | None = None

    model_config = {"populate_by_name": True}


# ── Endpoint ─────────────────────────────────────────────────────────────

@router.post("/internal/webhook/supabase-product")
async def handle_product_webhook(
    payload: SupabaseWebhookPayload,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Receive a product insert webhook from Supabase.

    Returns 200 immediately to prevent webhook timeouts.
    Embedding generation runs as a FastAPI BackgroundTask.
    """
    product = payload.record
    logger.info(
        f"📦 Webhook received: product={product.id} shop={product.shop_id} name=\"{product.name[:60]}\""
    )

    background_tasks.add_task(_generate_and_store_embedding, product)

    return {"status": "accepted", "product_id": product.id}


# ── Background Task ─────────────────────────────────────────────────────

async def _generate_and_store_embedding(product: ProductRecord) -> None:
    """
    Generate a 768-dim embedding for a product and update the Supabase row.

    Steps:
      1. Combine text fields (name, description, attributes)
      2. Optionally fetch image bytes if image_url is present
      3. Call gemini-embedding-2 with text (+ image if available)
      4. Update the products table with the embedding vector
      5. Set embedding_status to 'completed' or 'failed'
    """
    product_id = product.id

    try:
        # ── 1. Prepare text content ──────────────────────────────────
        text_parts = [product.name]
        if product.description:
            text_parts.append(product.description)
        if product.attributes:
            if isinstance(product.attributes, dict):
                attr_text = ", ".join(f"{k}: {v}" for k, v in product.attributes.items())
            elif isinstance(product.attributes, list):
                attr_text = ", ".join(str(a) for a in product.attributes)
            else:
                attr_text = str(product.attributes)
            text_parts.append(attr_text)

        combined_text = " | ".join(text_parts)[:2000]

        # ── 2. Build content parts for embedding ─────────────────────
        # Start with text content
        contents = [combined_text]

        # If image_url present, fetch and include image bytes
        if product.image_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    img_response = await client.get(product.image_url)
                    img_response.raise_for_status()
                    image_bytes = img_response.content

                # Detect MIME type from response headers or default to JPEG
                content_type = img_response.headers.get("content-type", "image/jpeg")
                mime_type = content_type.split(";")[0].strip()

                # Add image as inline data part alongside the text
                contents = types.Content(parts=[
                    types.Part.from_text(text=combined_text),
                    types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
                ])

                logger.info(f"[{product_id}] Fetched image ({len(image_bytes)} bytes) for multimodal embedding")
            except Exception as img_err:
                logger.warning(f"[{product_id}] Failed to fetch image, falling back to text-only: {img_err}")
                # contents stays as [combined_text] — text-only embedding

        # ── 3. Generate embedding ────────────────────────────────────
        result = await genai_client.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
        )

        embedding_vector = result.embeddings[0].values

        if len(embedding_vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSIONS}-dim vector, got {len(embedding_vector)}-dim"
            )

        logger.info(f"[{product_id}] ✅ Embedding generated ({len(embedding_vector)}-dim)")

        # ── 4. Update Supabase ───────────────────────────────────────
        get_supabase().table("products").update({
            "embedding": embedding_vector,
            "embedding_status": "completed",
        }).eq("id", product_id).execute()

        logger.info(f"[{product_id}] ✅ Product embedding saved to Supabase")

    except Exception as e:
        logger.error(f"[{product_id}] ❌ Embedding generation failed: {e}", exc_info=True)

        # Mark as failed so the admin dashboard can surface the error
        try:
            get_supabase().table("products").update({
                "embedding_status": "failed",
            }).eq("id", product_id).execute()
        except Exception as update_err:
            logger.error(f"[{product_id}] Failed to update embedding_status to 'failed': {update_err}")
