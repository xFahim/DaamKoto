"""Intent classification service using Gemini Function Calling."""

from google import genai
from google.genai import types
from app.core.config import settings


# Tool declarations — Gemini picks which "function" to call based on user message
INTENT_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_products",
                description=(
                    "Search the product catalog/inventory. Use this when the user is "
                    "looking for a specific product, asking about availability, price, "
                    "colors, sizes, or describing an item they want to buy. "
                    "Examples: 'red t-shirt', 'do you have sneakers?', 'show me hoodies under 2000'"
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="The product search query extracted from the user message",
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="general_chat",
                description=(
                    "Handle general conversation, greetings, chitchat, thank you messages, "
                    "goodbyes, or any message that is NOT about products, FAQs, or complaints. "
                    "Examples: 'hi', 'hello', 'thanks', 'how are you', 'ok', 'bye', 'good morning'"
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "message": types.Schema(
                            type=types.Type.STRING,
                            description="The user's message to respond to conversationally",
                        )
                    },
                    required=["message"],
                ),
            ),
            types.FunctionDeclaration(
                name="answer_faq",
                description=(
                    "Answer frequently asked questions about the store — operating hours, "
                    "return policy, shipping info, payment methods, store location, contact info. "
                    "Examples: 'what are your hours?', 'do you accept bkash?', 'how long does delivery take?'"
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "topic": types.Schema(
                            type=types.Type.STRING,
                            description="The FAQ topic the user is asking about",
                        )
                    },
                    required=["topic"],
                ),
            ),
            types.FunctionDeclaration(
                name="handle_order_complaint",
                description=(
                    "Handle order-related queries, complaints, issues with delivery, "
                    "refund requests, or problems with a purchased product. "
                    "Examples: 'where is my order?', 'I want a refund', 'my package arrived damaged'"
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "issue": types.Schema(
                            type=types.Type.STRING,
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
        self.client = None

    def initialize(self):
        """Configure Gemini client for classification."""
        self.client = genai.Client(api_key=settings.gemini_api_key)
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
        if not self.client:
            print("⚠️ Intent service not initialized, defaulting to general_chat.")
            return {"intent": "general_chat", "params": {"message": message}}

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an intent classifier for an e-commerce store's Messenger chatbot. "
                        "Your ONLY job is to classify the user's message by calling the correct function. "
                        "You MUST call exactly one function for every message. Never respond with text."
                    ),
                    tools=INTENT_TOOLS,
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    ),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
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
