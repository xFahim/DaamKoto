import asyncio
from typing import List, Optional
import httpx
from pinecone import Pinecone
from app.core.config import settings
import google.generativeai as genai
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

    async def initialize(self):
        """Initialize Vertex AI and Pinecone connections."""
        try:
            # 1. Get the JSON string from settings
            service_account_json = settings.gcp_service_account_json
            if not service_account_json:
                print("⚠️ GCP_SERVICE_ACCOUNT_JSON not found. Vertex AI checks will fail.")
            else:
                info = json.loads(service_account_json)
                credentials = service_account.Credentials.from_service_account_info(info)
                vertexai.init(
                    project=info["project_id"],
                    location="asia-east1",
                    credentials=credentials
                )
                print("✅ Vertex AI Authenticated successfully in RAG Service.")
                
                self.embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
                
        except Exception as e:
            print(f"❌ Vertex AI Authentication failed in RAG Service: {e}")

        # Configure Gemini AI for generation
        try:
            genai.configure(api_key=settings.gemini_api_key)
            print("✅ Gemini AI Configured successfully.")
        except Exception as e:
            print(f"❌ Gemini AI Configuration failed: {e}")

        try:
             # Initialize Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)
            # Connect to the multimodal index
            self.pinecone_index = pc.Index("chatpulse-multimodal")
            print("✅ Pinecone Connected successfully.")
        except Exception as e:
            print(f"❌ Pinecone Connection failed: {e}")

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
                print("⚠️ No embedding generated.")
                return "I'm sorry, I couldn't process the input to search for products."
            else:
                 print(f"✅ Embedding generated. Dimension: {len(query_embedding)}")

            # Step 2: Retrieve top 3 matches from Pinecone
            try:
                if not self.pinecone_index:
                     print("⚠️ Pinecone index not initialized.")
                     return "I'm having trouble accessing the product catalog right now."

                print(f"🔍 Querying Pinecone in namespace '{namespace}'...")
                query_response = self.pinecone_index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True,
                    namespace=namespace,
                )
                print(f"✅ Pinecone returned {len(query_response.matches)} matches.")
            except Exception as e:
                print(f"❌ Pinecone query failed for namespace '{namespace}': {e}")
                # Fallback to query without namespace if needed, or just return error
                return "I'm having trouble accessing the product catalog right now."

            # Step 3: Build context from retrieved products
            context_parts = []
            if query_response.matches:
                for match in query_response.matches:
                    metadata = match.metadata or {}
                    name = metadata.get("name", "Unknown")
                    price = metadata.get("price", "N/A")
                    stock = metadata.get("stock", "N/A")
                    description = metadata.get("description", "")
                    url = metadata.get("url", "No URL")
                    product_url = metadata.get("product_url", url)
                    score = match.score
                    
                    print(f"   Match: {name} (Score: {score:.4f})")
                    
                    context_parts.append(
                        f"Name: {name}, Price: {price}, Stock: {stock}, Description: {description}, URL: {product_url}"
                    )

            # Step 4: Construct the prompt
            if context_parts:
                print("📝 Constructing context with matches.")
                context = "\n".join(context_parts)
                
                if image_url:
                     intro = (
                        "The user has uploaded an image to search for a product. "
                        "We have analyzed the image and found the following matches from our inventory (most similar first):"
                     )
                     instruction = "Confirm you found these matching products."
                else:
                     intro = "Use the provided context to suggest products to the user."
                     instruction = "If the detailed context doesn't contain a relevant match, politely say so."

                system_prompt = (
                    "You are a helpful and concise sales assistant. "
                    f"{intro}\n"
                    "Your response should be friendly and human-like, not robotic. "
                    "Example: 'So, you're looking for this? We have this [Product Name] in stock.' "
                    "CRITICAL: You MUST include the full product URL for each item you suggest. "
                    f"\n\nContext Products:\n{context}\n\n"
                    f"{instruction}"
                )
            else:
                print("⚠️ No context parts found.")
                system_prompt = (
                    "You are a sales assistant. "
                    "Politely inform the user that you couldn't find any products matching their description right now within the current catalog."
                )

            # Step 5: Generate response using Gemini
            model = genai.GenerativeModel("gemini-2.5-flash")
            full_prompt = f"{system_prompt}\n\nUser: {user_query or 'Find this product'}"
            response = await model.generate_content_async(full_prompt)
            return response.text.strip()

        except Exception as e:
            print(f"❌ Error in RAG service: {str(e)}")
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
            print("❌ Embedding model not initialized.")
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
            print(f"❌ Error getting embedding from Vertex AI: {repr(e)}")
            return []


rag_service = RagService()
