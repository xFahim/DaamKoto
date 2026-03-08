"""Handler for general conversation / chitchat messages."""

from google import genai
from google.genai import types
from app.core.config import settings
from app.services.messaging_service import messaging_service

# Shared client instance
client = genai.Client(api_key=settings.gemini_api_key)


class GeneralHandler:
    """Handles greetings, chitchat, and non-product queries with a direct Gemini call."""

    @staticmethod
    async def process(sender_id: str, message_text: str, page_id: str) -> None:
        """
        Generate a friendly conversational response (no RAG needed).

        Args:
            sender_id: The Facebook user ID who sent the message
            message_text: The user's message
            page_id: The Facebook page ID
        """
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"User: {message_text}",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a chill, friendly sales assistant for an online store. "
                        "Keep it short — 1-2 lines max. Sound like a real person texting, not a bot. "
                        "No emojis overload, just be natural.\n\n"
                        "LANGUAGE RULE (strict):\n"
                        "- If the user writes in English, reply in English.\n"
                        "- If they write in Banglish (Bengali in English letters like 'kemon acho'), reply in Banglish.\n"
                        "- If they write in Bangla (বাংলা), reply in Bangla.\n"
                        "Match their vibe and language exactly."
                    ),
                    max_output_tokens=150,
                ),
            )
            reply = response.text.strip()
        except Exception as e:
            print(f"❌ General handler error: {e}")
            reply = "Hey there! 👋 How can I help you today?"

        await messaging_service.send_message(
            recipient_id=sender_id,
            message_text=reply,
        )


general_handler = GeneralHandler()
