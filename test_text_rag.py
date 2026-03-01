import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.services.rag_service import rag_service

async def test_text_rag():
    # Test query
    query = "blue tshirt"
    
    print(f"🧪 Testing RAG with Text Query: '{query}'")
    print("-" * 50)

    # Initialize the service (required after refactor)
    await rag_service.initialize()
    
    try:
        response = await rag_service.generate_response(
            user_query=query,
            page_id="goodybro",
            image_url=None
        )
        
        print("\n🤖 RAG Response:")
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_text_rag())
