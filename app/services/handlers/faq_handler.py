"""Handler for FAQ-related messages with dummy store data."""

from google import genai
from google.genai import types
from app.core.config import settings
from app.services.messaging_service import messaging_service


# Dummy FAQ data — will be replaced with DB lookup per store later
FAQ_DATA = {
    "store_hours": "We are open Saturday to Thursday, 10 AM to 10 PM (GMT+6). Closed on Fridays.",
    "shipping": (
        "We offer delivery across Bangladesh. "
        "Inside Dhaka: 1-2 business days (৳60). "
        "Outside Dhaka: 3-5 business days (৳120). "
        "Free shipping on orders above ৳2000."
    ),
    "return_policy": (
        "We accept returns within 7 days of delivery. "
        "Items must be unused, unwashed, and in original packaging. "
        "Refunds are processed within 3-5 business days."
    ),
    "payment_methods": (
        "We accept bKash, Nagad, Rocket, bank transfer, and cash on delivery (COD). "
        "For bKash, send payment to 01XXXXXXXXX."
    ),
    "contact": (
        "You can reach us at:\n"
        "📞 Phone: 01XXXXXXXXX\n"
        "📧 Email: support@goodybro.com\n"
        "💬 Or just message us right here!"
    ),
    "store_location": "We are primarily an online store. No physical outlet at the moment.",
}

# Shared client instance
client = genai.Client(api_key=settings.gemini_api_key)


class FaqHandler:
    """Handles FAQ queries using dummy store data + Gemini for natural responses."""

    @staticmethod
    async def process(sender_id: str, message_text: str, page_id: str) -> None:
        """
        Look up FAQ data and generate a natural response via Gemini.

        Args:
            sender_id: The Facebook user ID who sent the message
            message_text: The user's message
            page_id: The Facebook page ID
        """
        # Build all FAQ context as a string
        faq_context = "\n".join(
            f"- {key.replace('_', ' ').title()}: {value}"
            for key, value in FAQ_DATA.items()
        )

        try:
            prompt = (
                f"Store FAQ Information:\n{faq_context}\n\n"
                f"User Question: {message_text}\n\n"
                "Answer using ONLY the FAQ info above. Keep it short and natural — "
                "like a real person replying on Messenger, not a help page. "
                "If the FAQ doesn't cover it, just say you'll check and get back.\n\n"
                "LANGUAGE RULE (strict):\n"
                "- If the user writes in English, reply in English.\n"
                "- If they write in Banglish (Bengali in English letters), reply in Banglish.\n"
                "- If they write in Bangla (বাংলা), reply in Bangla.\n"
                "Match their tone and language."
            )
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=200),
            )
            reply = response.text.strip()
        except Exception as e:
            print(f"❌ FAQ handler error: {e}")
            reply = (
                "I'm having a little trouble right now! "
                "You can reach us at support@goodybro.com for any questions. 😊"
            )

        await messaging_service.send_message(
            recipient_id=sender_id,
            message_text=reply,
        )


faq_handler = FaqHandler()
