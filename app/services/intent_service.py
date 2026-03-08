"""Intent classification service using Gemini Function Calling."""

import google.generativeai as genai
from app.core.config import settings


# Tool declarations — Gemini picks which "function" to call based on user message
INTENT_TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="search_products",
                description=(
                    "Search the product catalog/inventory. Use this when the user is "
                    "looking for a specific product, asking about availability, price, "
                    "colors, sizes, or describing an item they want to buy. "
                    "Examples: 'red t-shirt', 'do you have sneakers?', 'show me hoodies under 2000'"
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "query": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The product search query extracted from the user message",
                        )
                    },
                    required=["query"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="general_chat",
                description=(
                    "Handle general conversation, greetings, chitchat, thank you messages, "
                    "goodbyes, or any message that is NOT about products, FAQs, or complaints. "
                    "Examples: 'hi', 'hello', 'thanks', 'how are you', 'ok', 'bye', 'good morning'"
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "message": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The user's message to respond to conversationally",
                        )
                    },
                    required=["message"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="answer_faq",
                description=(
                    "Answer frequently asked questions about the store — operating hours, "
                    "return policy, shipping info, payment methods, store location, contact info. "
                    "Examples: 'what are your hours?', 'do you accept bkash?', 'how long does delivery take?'"
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "topic": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The FAQ topic the user is asking about",
                        )
                    },
                    required=["topic"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="handle_order_complaint",
                description=(
                    "Handle order-related queries, complaints, issues with delivery, "
                    "refund requests, or problems with a purchased product. "
                    "Examples: 'where is my order?', 'I want a refund', 'my package arrived damaged'"
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "issue": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The order issue or complaint description",
                        )
                    },
                    required=["issue"],
                ),
            ),
        ]
    )
]


class IntentService:
    """Classifies user messages into intents using Gemini function calling."""

    def __init__(self):
        self.model = None

    def initialize(self):
        """Configure Gemini and create the classification model."""
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash",
            tools=INTENT_TOOLS,
            system_instruction=(
                "You are an intent classifier for an e-commerce store's Messenger chatbot. "
                "Your ONLY job is to classify the user's message by calling the correct function. "
                "You MUST call exactly one function for every message. Never respond with text."
            ),
        )
        print("✅ Intent classification service initialized.")

    async def classify(self, message: str) -> dict:
        """
        Classify a user message into an intent via Gemini function calling.

        Args:
            message: The user's text message

        Returns:
            dict with 'intent' (str) and 'params' (dict)
            Example: {"intent": "search_products", "params": {"query": "red t-shirt"}}
        """
        if not self.model:
            print("⚠️ Intent service not initialized, defaulting to general_chat.")
            return {"intent": "general_chat", "params": {"message": message}}

        try:
            response = await self.model.generate_content_async(
                message,
                tool_config={"function_calling_config": {"mode": "ANY"}},
            )

            # Extract the function call from the response
            candidate = response.candidates[0]
            fn_call = candidate.content.parts[0].function_call

            intent = fn_call.name
            params = dict(fn_call.args)

            print(f"🎯 Intent classified: {intent} | Params: {params}")
            return {"intent": intent, "params": params}

        except Exception as e:
            print(f"⚠️ Intent classification failed: {e}. Falling back to general_chat.")
            return {"intent": "general_chat", "params": {"message": message}}


intent_service = IntentService()
