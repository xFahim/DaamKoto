"""Test script for RAG service brain functionality."""

import asyncio
from dotenv import load_dotenv
from app.services.rag_service import rag_service

# Load environment variables
load_dotenv()


async def test_rag_service():
    """Test the RAG service with user input."""

    print("=" * 70)
    print("RAG Service Brain Test")
    print("=" * 70)
    print()

    # Ask user for a question
    question = input("Enter your question: ").strip()

    if not question:
        print("No question provided. Exiting.")
        return

    print()
    print(f"Q: {question}")
    print()

    try:
        # Call the RAG service
        response = await rag_service.generate_response(
            user_query=question, page_id="dummy_page_id"
        )

        print(f"A: {response}")
        print()

    except Exception as e:
        print(f"Error: {str(e)}")
        print()

    print("=" * 70)
    print("Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_rag_service())
