"""Test script for intent classification using Gemini function calling."""

import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.services.intent_service import intent_service


TEST_MESSAGES = [
    # General chat — should NOT trigger RAG
    ("hello", "general_chat"),
    ("hi there!", "general_chat"),
    ("thanks a lot", "general_chat"),
    ("ok bye", "general_chat"),
    ("good morning", "general_chat"),
    # Product search — should trigger RAG
    ("I want a red t-shirt", "search_products"),
    ("do you have sneakers?", "search_products"),
    ("show me hoodies under 2000", "search_products"),
    ("blue joggers", "search_products"),
    # FAQ — should use FAQ data
    ("what are your store hours?", "answer_faq"),
    ("do you accept bkash?", "answer_faq"),
    ("how long does delivery take?", "answer_faq"),
    ("what is your return policy?", "answer_faq"),
    # Order / Complaint — placeholder
    ("where is my order?", "handle_order_complaint"),
    ("I want a refund", "handle_order_complaint"),
    ("my package arrived damaged", "handle_order_complaint"),
]


async def test_intent_classification():
    print("🧪 Intent Classification Test")
    print("=" * 60)

    # Initialize the intent service
    intent_service.initialize()

    correct = 0
    total = len(TEST_MESSAGES)

    for i, (message, expected_intent) in enumerate(TEST_MESSAGES):
        result = await intent_service.classify(message)
        actual_intent = result["intent"]
        match = "✅" if actual_intent == expected_intent else "❌"
        if actual_intent == expected_intent:
            correct += 1

        print(f"  {match} \"{message}\"")
        print(f"     Expected: {expected_intent} | Got: {actual_intent}")
        print(f"     Params: {result['params']}")
        print()

        # Small delay to avoid rate limiting on free tier
        if i < total - 1:
            await asyncio.sleep(2)

    print("=" * 60)
    print(f"Results: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_intent_classification())
