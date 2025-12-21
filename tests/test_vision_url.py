import asyncio
import os
from dotenv import load_dotenv
from app.services.image_service import image_service
from app.services.rag_service import rag_service

# Load environment variables
load_dotenv()

async def main():
    # Define test URL
    url = "https://goodybro.com/cdn/shop/files/RED_SM.jpg"

    # Step 1: Image Analysis
    print("📸 Testing Image Analysis...")
    try:
        description = await image_service.describe_image(url)
        # Step 2: Print Description
        print(f"Generated Description: {description}")

        # Step 3: RAG Search
        print("🧠 Testing RAG Search...")
        response = await rag_service.generate_response(description, "goodybro")
        
        # Step 4: Final Response
        print(f"Final Bot Response:\n{response}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
