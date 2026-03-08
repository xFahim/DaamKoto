"""Handler for FAQ-related messages with dummy store data."""

import google.generativeai as genai
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
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = (
                f"Store FAQ Information:\n{faq_context}\n\n"
                f"User Question: {message_text}\n\n"
                "Answer the user's question using ONLY the FAQ information above. "
                "Be concise, friendly, and helpful. If the FAQ doesn't cover their "
                "question, say you'll find out and get back to them."
            )
            response = await model.generate_content_async(
                prompt,
                generation_config={"max_output_tokens": 250},
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
