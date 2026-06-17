# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DaamKoto is a **multi-tenant SaaS** Facebook Messenger e-commerce chatbot platform. It combines:
- **FastAPI** webhook server for Facebook Messenger (multi-page support)
- **Supabase** (pgvector) for product storage, orders, customers, and vector similarity search
- **Google GenAI SDK** — `gemini-embedding-2` (768-dim embeddings) + `gemini-3-flash-preview` (LLM generation, agent orchestration)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run production (Railway)
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

## Required Environment Variables

Set in `.env`:
- `GEMINI_API_KEY` — Google GenAI API key (shared for LLM + embeddings)
- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` — Supabase service role key (for backend writes)
- `FACEBOOK_VERIFY_TOKEN` — Webhook verification token
- `OPENAI_API_KEY` — (Optional) OpenAI key if using `LLM_PROVIDER=openai`
- `PORT` — Server port (default 8000)

**Note:** `FACEBOOK_PAGE_ACCESS_TOKEN` is no longer an env var — it's fetched dynamically per-tenant from `bot_settings` in Supabase.

## Architecture

### Multi-Tenant Resolution

```
Facebook POST /api/v1/webhook
  → entry.id (Facebook page ID)
  → resolve_tenant(entry.id) queries bot_settings WHERE facebook_page_id = entry.id
  → Returns TenantContext(shop_id, page_access_token, facebook_page_id)
  → TenantContext threads through the ENTIRE pipeline
  → 60-second TTL cache avoids repeated DB hits
```

### Message Processing Flow

```
Facebook POST /api/v1/webhook
  → Returns {"status": "ok"} immediately (background task)
  → FacebookService.process_webhook_event()
    → resolve_tenant(entry.id) → TenantContext
    → tenant.sender_id = messaging.sender.id
  → MessageRouter.route_message(sender_id, message, tenant)
      Text → InputGuard → MessageBatcher → TextHandler → AgentService
      Image → MessageBatcher → TextHandler → AgentService (image upload to Gemini Cloud)
  → AgentService._execute_tool(call_name, call_args, tenant)
      → shop_id injected server-side from tenant.shop_id
      → Tools NEVER receive shop_id from the LLM
  → MessagingService.send_message(sender_id, text, access_token=tenant.page_access_token)
```

### Tool Execution (Strict Tenant Isolation)

Tools are declared as pure Python stubs in `app/core/tools.py`. The SDK auto-parses signatures.
Actual execution happens in `AgentService._execute_tool()` which injects `tenant.shop_id` server-side.

| Tool | LLM-facing Args | Server-side Injection | Data Source |
|------|-----------------|----------------------|-------------|
| `search_products(query)` | query | tenant.shop_id | RagService → Supabase pgvector RPC |
| `get_company_policy(topic)` | topic | tenant.shop_id | bot_settings.store_policies |
| `check_order_status(order_number)` | order_number | tenant.shop_id | orders table (scoped by shop_id) |
| `execute_order(item_names, sizes, ...)` | item_names, sizes, address, phone | tenant.shop_id, tenant.sender_id | customers UPSERT + orders INSERT |
| `send_product_image(image_url)` | image_url | tenant.sender_id, tenant.page_access_token | MessagingService → Facebook Graph API |

### Product Embedding Pipeline

```
Next.js Admin Dashboard → Supabase INSERT (products table)
  → Supabase Webhook → POST /api/v1/internal/webhook/supabase-product
    → Return 200 OK immediately
    → BackgroundTask:
        → Combine text (name, description, attributes)
        → If image_url: fetch image bytes via httpx
        → gemini-embedding-2 (768-dim, text + optional image)
        → UPDATE products SET embedding=vector, embedding_status='completed'
        → On failure: SET embedding_status='failed'
```

### Layer Responsibilities

- **`app/core/tenant_context.py`** — TenantContext dataclass + resolve_tenant() with TTL cache
- **`app/core/dependencies.py`** — Shared Supabase and GenAI client singletons
- **`app/core/config.py`** — All settings via Pydantic BaseSettings from env vars
- **`app/core/tools.py`** — Tool function stubs (signatures only, no logic)
- **`app/core/openai_tools.py`** — OpenAI JSON schema mirrors of tools.py
- **`app/api/v1/endpoints/`** — HTTP concerns only (webhook verification, receive POST, return immediately)
- **`app/services/`** — External API integrations (Facebook, Gemini, Supabase)
- **`app/services/handlers/`** — Message routing and text/image handling
- **`app/schemas/facebook.py`** — Pydantic models for Facebook webhook payloads

### Key Implementation Details

- **Async-first**: All I/O is `async`/`await`.
- **TenantContext threading**: Resolved once at webhook entry, passed through every layer.
- **Strict tenant isolation**: Tools never receive `shop_id` from the LLM. It's injected in `_execute_tool()`.
- **Dynamic Facebook tokens**: `page_access_token` fetched from `bot_settings` per-tenant, not from env.
- **ReAct loop**: Manual 5-turn loop with `automatic_function_calling` disabled.
- **Typing indicators**: Sent via Facebook API before processing each message.
- **Multi-language**: Gemini prompts explicitly support English, Banglish, and Bengali.
- **Embedding model**: `gemini-embedding-2` via Google GenAI SDK — produces 768-dim vectors.
- **Vector store**: Supabase pgvector with `match_products` RPC function for cosine similarity search.

### Supabase Tables Used

| Table | Used By |
|-------|---------|
| `bot_settings` | resolve_tenant(), get_company_policy |
| `products` | search_products (via pgvector RPC), embedding webhook |
| `customers` | execute_order (UPSERT by messenger_psid + shop_id) |
| `orders` | execute_order (INSERT), check_order_status (SELECT) |

### Known Placeholders

- `ComplaintHandler` — returns a static response; meant to be replaced with real order lookup
- `FaqHandler` — uses hardcoded FAQ data; meant to be replaced with per-store DB lookup
