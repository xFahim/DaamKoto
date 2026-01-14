import asyncio
import os
import sys
from pprint import pprint

# Add project root to path
sys.path.append(os.getcwd())

try:
    from app.services.rag_service import rag_service
except Exception as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

async def test_vision_rag():
    # Hardcoded image URL for testing (GoodyBro product)
    test_image_url = "https://goodybro.com/cdn/shop/files/1_7b6dc657-3be9-4b0a-9249-a99234b569cc.png?v=1695207565&width=533" 
    
    print(f"🧪 Testing RAG with Image URL: {test_image_url}")
    print("-" * 50)
    
    # Initialize the service (required after refactor)
    await rag_service.initialize()
    
    try:
        response = await rag_service.generate_response(
            user_query="Find this product",
            page_id="goodybro",
            image_url=test_image_url
        )
        
        print("\n🤖 RAG Response:")
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision_rag())
