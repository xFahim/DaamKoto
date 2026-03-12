import asyncio
from typing import List, Optional
import httpx
from pinecone import Pinecone
from app.core.config import settings
from google import genai
from google.genai import types
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
import os
import json
from google.oauth2 import service_account

# Initialize Vertex AI with explicit credentials
# We will do this inside the service initialization to avoid import-time errors

class RagService:
    """Service for handling RAG-based AI responses with vector retrieval."""

    def __init__(self):
        self.embedding_model = None
        self.pinecone_index = None
        self.client = None

    async def initialize(self):
        """Initialize Vertex AI, Gemini client, and Pinecone connections."""
        try:
            # 1. Get the JSON string from settings
            service_account_json = settings.gcp_service_account_json
            if not service_account_json:
                print("GCP_SERVICE_ACCOUNT_JSON not found. Vertex AI checks will fail.")
            else:
                info = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(info)
                vertexai.init(
                    project=info["project_id"],
                    location="asia-east1",
                    credentials=credentials
                )
                print("Vertex AI Authenticated successfully in RAG Service.")
                
                self.embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
                
        except Exception as e:
            print(f"Vertex AI Authentication failed in RAG Service: {e}")

        # Configure Gemini AI client for generation
        try:
            self.client = genai.Client(api_key=settings.gemini_api_key)
            print("Gemini AI Configured successfully.")
        except Exception as e:
            print(f"Gemini AI Configuration failed: {e}")

        try:
             # Initialize Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            # Connect to the multimodal index
            self.pinecone_index = pc.Index("chatpulse-multimodal")
            print("Pinecone Connected successfully.")
        except Exception as e:
            print(f"Pinecone Connection failed: {e}")

    async def generate_response(
        self, user_query: str, page_id: str, image_url: str | None = None
    ) -> str:
        """
        Generate an AI response using RAG (Retrieval-Augmented Generation).
        """
        # Dynamic namespace based on page_id
        namespace = f"store_{page_id}"

        try:
            # Step 1: Embed the user query or image
            query_embedding = await self.get_multimodal_embedding(
                text=user_query if not image_url else None,
                image_url=image_url
            )

            if not query_embedding:
                print("No embedding generated.")
                return "I'm sorry, I couldn't process the input to search for products."
            else:
                 print(f"Embedding generated. Dimension: {len(query_embedding)}")

            # Step 2: Retrieve candidates from Pinecone (fetch extra, filter by score)
            # Text embeddings vs image embeddings in index = lower cosine scores (~0.05-0.20)
            # Image embeddings vs image embeddings in index = higher cosine scores (~0.40-0.80)
            if image_url:
                MIN_SCORE = 0.45
                HIGH_CONFIDENCE = 0.65
            else:
                MIN_SCORE = 0.08
                HIGH_CONFIDENCE = 0.18
            print(f"Score thresholds: MIN={MIN_SCORE}, HIGH={HIGH_CONFIDENCE} ({'image' if image_url else 'text'} mode)")

            try:
                if not self.pinecone_index:
                     print("Pinecone index not initialized.")
                     return "I'm having trouble accessing the product catalog right now."

                print(f"Querying Pinecone in namespace '{namespace}'...")
                query_response = self.pinecone_index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True,
                    namespace=namespace,
                )
                print(f"Pinecone returned {len(query_response.matches)} raw matches.")
            except Exception as e:
                print(f"Pinecone query failed for namespace '{namespace}': {e}")
                return "I'm having trouble accessing the product catalog right now."

            # Step 3: Filter matches by score threshold
            good_matches = []
            if query_response.matches:
                for match in query_response.matches:
                    score = match.score
                    metadata = match.metadata or {}
                    name = metadata.get("name", "Unknown")

                    if score < MIN_SCORE:
                        print(f"   Skipped: {name} (Score: {score:.4f} < {MIN_SCORE})")
                        continue

                    print(f"   Match: {name} (Score: {score:.4f})")
                    good_matches.append({
                        "name": name,
                        "price": metadata.get("price", "N/A"),
                        "stock": metadata.get("stock", "N/A"),
                        "description": metadata.get("description", ""),
                        "product_url": metadata.get("product_url", metadata.get("url", "")),
                        "score": score,
                    })

            print(f"{len(good_matches)} matches passed score threshold.")

            # Step 4: Build prompt based on match quality
            tone_rule = (
                "\n\nLANGUAGE RULE (strict):\n"
                "- If the user writes in English, reply in English.\n"
                "- If they write in Banglish (Bengali in English letters), reply in Banglish.\n"
                "- If they write in Bangla (বাংলা), reply in Bangla.\n"
                "Match their tone and language exactly."
            )

            if not good_matches:
                # No relevant products found
                system_prompt = (
                    "You are a friendly store assistant on Messenger. "
                    "The user asked about a product but nothing matched in our catalog. "
                    "Let them know casually — don't be robotic. Maybe suggest they describe "
                    "it differently or ask if they want something else."
                    f"{tone_rule}"
                )
            elif len(good_matches) == 1 and good_matches[0]["score"] >= HIGH_CONFIDENCE:
                # Single strong match — confident suggestion
                p = good_matches[0]
                context = f"Name: {p['name']}, Price: {p['price']}, Stock: {p['stock']}, URL: {p['product_url']}"
                system_prompt = (
                    "You are a chill store assistant on Messenger. "
                    "You found ONE product that's a really strong match. "
                    "Suggest it naturally — like 'hey, you looking for this?' vibes. "
                    "Keep it short, include the product URL. "
                    "If they want more options, let them know they can ask.\n\n"
                    f"Product:\n{context}"
                    f"{tone_rule}"
                )
            else:
                # Multiple matches — show options casually
                context_lines = []
                for p in good_matches[:3]:  # Cap at 3 for display
                    context_lines.append(
                        f"Name: {p['name']}, Price: {p['price']}, Stock: {p['stock']}, URL: {p['product_url']}"
                    )
                context = "\n".join(context_lines)

                if image_url:
                    intro = "The user sent a product image. Here are the closest matches from inventory:"
                else:
                    intro = "Here are the matching products from the catalog:"

                system_prompt = (
                    "You are a chill store assistant on Messenger. "
                    f"{intro}\n"
                    "Suggest these naturally — like a friend helping them shop. "
                    "Include the product URL for each. Keep it brief, not a bullet-point list. "
                    "If only some are relevant, focus on the best one and mention the others as alternatives.\n\n"
                    f"Products:\n{context}"
                    f"{tone_rule}"
                )

            # Step 5: Generate response using Gemini
            full_prompt = f"{system_prompt}\n\nUser: {user_query or 'Find this product'}"
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
            return response.text.strip()

        except Exception as e:
            print(f"Error in RAG service: {str(e)}")
            return (
                "I apologize, but I'm having trouble processing your request right now. "
                "Please try again later or rephrase your question."
            )

    async def get_multimodal_embedding(
        self, text: str | None = None, image_url: str | None = None
    ) -> List[float]:
        """
        Generate 1408-dim vector using Google Vertex AI Multimodal Embedding.
        """
        if not self.embedding_model:
            print("Embedding model not initialized.")
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
            print(f"Error getting embedding from Vertex AI: {repr(e)}")
            return []


rag_service = RagService()
