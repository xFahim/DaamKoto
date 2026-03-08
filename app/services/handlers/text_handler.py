"""Handler for processing text messages from Facebook Messenger."""

from app.services.intent_service import intent_service
from app.services.rag_service import rag_service
from app.services.messaging_service import messaging_service
from app.services.handlers.general_handler import general_handler
from app.services.handlers.faq_handler import faq_handler
from app.services.handlers.complaint_handler import complaint_handler


class TextHandler:
    """Handler for processing text-based messages with intent classification."""

    # Maps intent names to their handler functions
    INTENT_HANDLERS = {
        "general_chat": lambda self, **kw: general_handler.process(**kw),
        "answer_faq": lambda self, **kw: faq_handler.process(**kw),
        "handle_order_complaint": lambda self, **kw: complaint_handler.process(**kw),
    }

    async def process(
        self,
        sender_id: str,
        message_text: str,
        page_id: str,
    ) -> None:
        """
        Classify intent first, then route to the appropriate handler.

        Args:
            sender_id: The Facebook user ID who sent the message
            message_text: The text content of the message
            page_id: The Facebook page ID
        """
        try:
            # Show typing indicator immediately
            await messaging_service.send_typing_on(sender_id)

            # Step 1: Classify intent
            result = await intent_service.classify(message_text)
            intent = result["intent"]

            # Step 2: Route to handler
            handler_kwargs = {
                "sender_id": sender_id,
                "message_text": message_text,
                "page_id": page_id,
            }

            if intent == "search_products":
                # Product search → existing RAG pipeline
                response_text = await rag_service.generate_response(
                    user_query=message_text,
                    page_id=page_id,
                )
                await messaging_service.send_message(
                    recipient_id=sender_id,
                    message_text=response_text,
                )
            elif intent in self.INTENT_HANDLERS:
                await self.INTENT_HANDLERS[intent](self, **handler_kwargs)
            else:
                # Unknown intent — fall back to general chat
                print(f"⚠️ Unknown intent '{intent}', falling back to general_chat.")
                await general_handler.process(**handler_kwargs)

        except Exception as e:
            print(f"❌ Error in text handler: {str(e)}")
            await messaging_service.send_message(
                recipient_id=sender_id,
                message_text=(
                    "Sorry, I'm having trouble processing your message right now! "
                    "Please try again later!"
                ),
            )


text_handler = TextHandler()
