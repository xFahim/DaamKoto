# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DaamKoto is a Facebook Messenger e-commerce chatbot that uses AI for product search and customer interaction. It combines:
- **FastAPI** webhook server for Facebook Messenger
- **Google Vertex AI** (multimodal embeddings, 1408-dim) + **Gemini 2.5-Flash** (generation, intent classification)
- **Pinecone** vector database for product similarity search (index: `chatpulse-multimodal`, namespace: `store_{page_id}`)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run production (Heroku)
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

# Test scripts (standalone, not a test framework)
python test_intent.py        # Test intent classification
python test_text_rag.py      # Test text-based RAG
python test_vision_rag.py    # Test image-based RAG
python ingest_multimodal.py  # Interactive product ingestion to Pinecone
```

## Required Environment Variables

Set in `.env`:
- `GEMINI_API_KEY` — Gemini API key
- `PINECONE_API_KEY` — Pinecone API key
- `FACEBOOK_PAGE_ACCESS_TOKEN` — Facebook Graph API token
- `FACEBOOK_VERIFY_TOKEN` — Webhook verification token
- `GCP_SERVICE_ACCOUNT_JSON` — Full JSON string of GCP service account (for Vertex AI)
- `PORT` — Server port (default 8000)

## Architecture

### Message Processing Flow

```
Facebook POST /api/v1/webhook
  → Returns {"status": "ok"} immediately (background task)
  → FacebookService.process_webhook_event()
  → MessageRouter.route_message() [by type: text / image]
      Text → IntentService.classify() [Gemini function calling]
               → search_products → RagService (text embedding → Pinecone)
               → general_chat   → GeneralHandler (Gemini generation)
               → answer_faq     → FaqHandler (hardcoded FAQ data + Gemini)
               → handle_order_complaint → ComplaintHandler (placeholder)
      Image → RagService (image URL → Vertex AI embedding → Pinecone)
  → MessagingService.send_message() [Facebook Graph API]
```

### Layer Responsibilities

- **`app/api/v1/endpoints/`** — HTTP concerns only (webhook verification, receive POST, return immediately)
- **`app/services/`** — External API integrations (Facebook, Gemini, Pinecone, Vertex AI)
- **`app/services/handlers/`** — Intent-specific business logic
- **`app/core/config.py`** — All settings via Pydantic BaseSettings from env vars
- **`app/schemas/facebook.py`** — Pydantic models for Facebook webhook payloads

### Key Implementation Details

- **Async-first**: All I/O is `async`/`await`. Vertex AI blocking calls use thread pools (`asyncio.get_event_loop().run_in_executor`).
- **Startup initialization**: `IntentService` and `RagService` are initialized in the FastAPI lifespan handler (`app/main.py`), not lazily.
- **Typing indicators**: Sent via Facebook API before processing each message.
- **Multi-language**: Gemini prompts explicitly support English, Banglish, and Bengali.
- **Page ID mapping**: Currently hardcoded to `"goodybro"` (marked TODO for multi-store support).
- **Embedding model**: `multimodalembedding@001` via Vertex AI — produces 1408-dim vectors for both text and images.

### Known Placeholders

- `ComplaintHandler` — returns a static response; meant to be replaced with real order lookup
- `FaqHandler` — uses hardcoded FAQ data; meant to be replaced with per-store DB lookup
- `goodybro.json` — sample product catalog used for ingestion testing
