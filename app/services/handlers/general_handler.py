"""Handler for general conversation / chitchat messages."""

import google.generativeai as genai
from app.services.messaging_service import messaging_service


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
            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                system_instruction=(
                    "You are a friendly sales assistant for an online store. "
                    "Keep responses short, warm, and natural (1-2 sentences max). "
                    "If the user greets you, greet back and ask how you can help. "
                    "If they thank you, respond graciously. Stay in character as a store assistant."
                ),
            )
            response = await model.generate_content_async(
                f"User: {message_text}",
                generation_config={"max_output_tokens": 200},
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
