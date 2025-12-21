"""Service layer for image analysis using Google Gemini."""

import httpx
import google.generativeai as genai
from app.core.config import settings

# Configure Gemini AI
genai.configure(api_key=settings.gemini_api_key)

# Initialize the model
model = genai.GenerativeModel("gemini-2.5-flash")


class ImageService:
    """Service for handling image analysis and description."""

    async def describe_image(self, image_url: str) -> str:
        """
        Analyze an image and generate a product description.

        Args:
            image_url: The URL of the image to analyze

        Returns:
            A formatted description string with product type and color
        """
        try:
            # Download the image
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                image_bytes = response.content

            # Prepare image blob for Gemini
            image_blob = {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            }

            # Create the prompt
            prompt = (
                "Analyze this product image. Describe the item, main color, material, "
                "and style in 3-4 keywords so I can search my inventory for it. "
                "Format: 'Product: [Type] | Color: [Color]'."
            )

            # Generate description using Gemini
            response = await model.generate_content_async([prompt, image_blob])
            return response.text

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return "unknown product"


image_service = ImageService()
