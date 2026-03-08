"""Placeholder handler for order status and complaint messages."""

from app.services.messaging_service import messaging_service


class ComplaintHandler:
    """Placeholder handler for order-related queries and complaints."""

    @staticmethod
    async def process(sender_id: str, message_text: str, page_id: str) -> None:
        """
        Send a placeholder response for order/complaint queries.
        Will be replaced with actual order lookup logic later.

        Args:
            sender_id: The Facebook user ID who sent the message
            message_text: The user's message
            page_id: The Facebook page ID
        """
        reply = (
            "I understand you have a concern about your order. 🙏\n\n"
            "Our support team will look into this and get back to you shortly. "
            "If it's urgent, you can reach us at:\n"
            "📞 01XXXXXXXXX\n"
            "📧 support@goodybro.com\n\n"
            "We appreciate your patience!"
        )

        await messaging_service.send_message(
            recipient_id=sender_id,
            message_text=reply,
        )


complaint_handler = ComplaintHandler()
