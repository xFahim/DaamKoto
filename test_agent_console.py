"""Console test script for the DaamKoto Agent Flow without Facebook Webhooks."""

import asyncio
import sys

# 1. Mock the Facebook messaging layer so we don't spam errors about missing page tokens
from app.services.messaging_service import messaging_service
async def mock_typing_on(*args, **kwargs):
    print("... (Agent is thinking / running RAG tools in background) ...")
messaging_service.send_typing_on = mock_typing_on

async def mock_send_image(recipient_id, image_url):
    print(f"\n[IMAGE DISPATCHED TO MESSENGER UI]: {image_url}")
    return True
messaging_service.send_image = mock_send_image

# 2. Import agent architecture
from app.services.agent_service import agent_service
from app.core.config import settings

async def main():
    print("=" * 70)
    print("🤖 DaamKoto Agentic Console Tester (Local Mode)")
    print("=" * 70)
    
    if not settings.gemini_api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)

    agent_service.initialize()
    sender_id = "console_test_user_001"
    
    print("\n[System] Chat session started. Type 'quit' to exit.")
    print("[System] To simulate an image upload, type: image:https://example.com/img.jpg [optional text]")
    
    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ["quit", "exit"]:
            print("\nExiting chat. Bye!")
            break
            
        if not user_input.strip():
            continue

        # Handle image simulation hack
        image_url = None
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_url = parts[0][len("image:"):].strip()
            user_input = parts[1] if len(parts) > 1 else "What is this?"
            print(f"[System] Emulating upload for URL: {image_url}")

        reply = await agent_service.process(
            sender_id, 
            user_input, 
            image_urls=[image_url] if image_url else None
        )
        print(f"\nAgent: {reply}")

if __name__ == "__main__":
    asyncio.run(main())
