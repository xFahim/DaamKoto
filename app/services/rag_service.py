"""Service layer for RAG (Retrieval-Augmented Generation) processing."""

import google.generativeai as genai
from pinecone import Pinecone
from app.core.config import settings

# Configure Gemini AI
genai.configure(api_key=settings.gemini_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

# Connect to the specific index
index = pc.Index("chatpulse")


class RagService:
    """Service for handling RAG-based AI responses with vector retrieval."""

    async def generate_response(self, user_query: str, page_id: str) -> str:
        """
        Generate an AI response using RAG (Retrieval-Augmented Generation).

        Args:
            user_query: The user's query/question
            page_id: The page/store ID (currently ignored in testing mode)

        Returns:
            The generated response text
        """
        # TESTING OVERRIDE: Hardcode namespace for testing
        namespace = "store_goodybro"

        try:
            # Step 1: Embed the user query
            embedding_result = genai.embed_content(
                model="models/text-embedding-004",
                content=user_query,
            )
            query_embedding = embedding_result["embedding"]

            # Step 2: Retrieve top 3 matches from Pinecone
            query_response = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                namespace=namespace,
            )

            # Step 3: Build context from retrieved products
            context_parts = []
            if query_response.matches:
                for match in query_response.matches:
                    metadata = match.metadata or {}
                    name = metadata.get("name", "Unknown")
                    price = metadata.get("price", "N/A")
                    stock = metadata.get("stock", "N/A")
                    context_parts.append(
                        f"Name: {name}, Price: {price}, Stock: {stock}"
                    )

            # Step 4: Construct the prompt
            if context_parts:
                context = "\n".join(context_parts)
                system_prompt = (
                    "You are a sales assistant. Use this context to answer the user's question: "
                    f"\n\n{context}\n\n"
                    "Provide helpful, accurate information based on the context provided. "
                    "If the context doesn't contain relevant information, politely say so."
                )
            else:
                # Fallback: No relevant products found
                system_prompt = (
                    "You are a sales assistant. Based on the available context, "
                    "you don't have information about the user's query. "
                    "Politely inform the user that you don't have that information available."
                )

            # Step 5: Generate response using Gemini
            model = genai.GenerativeModel("gemini-2.5-flash")
            full_prompt = f"{system_prompt}\n\nUser: {user_query}"
            response = model.generate_content(full_prompt)
            return response.text.strip()

        except Exception as e:
            print(f"Error in RAG service: {str(e)}")
            # Fallback error message
            return (
                "I apologize, but I'm having trouble processing your request right now. "
                "Please try again later or rephrase your question."
            )


rag_service = RagService()

