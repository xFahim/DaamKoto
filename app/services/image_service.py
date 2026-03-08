"""Service layer for image analysis using Google Gemini."""

import httpx
from google import genai
from google.genai import types
from app.core.config import settings

# Shared client instance
client = genai.Client(api_key=settings.gemini_api_key)


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
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(image_url)
                response.raise_for_status()
                image_bytes = response.content

            # Create the prompt
            prompt = (
                "Analyze this product image. Describe the item, main color, material, "
                "and style in 3-4 keywords so I can search my inventory for it. "
                "Format: 'Product: [Type] | Color: [Color]'."
            )

            # Generate description using Gemini
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
            return response.text

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return "unknown product"


image_service = ImageService()
