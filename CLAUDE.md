# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DaamKoto is a **multi-tenant SaaS** Facebook Messenger e-commerce chatbot platform. It combines:
- **FastAPI** webhook server for Facebook Messenger (multi-page support)
- **Supabase** (pgvector) for product storage, orders, customers, threads/messages, and vector similarity search
- **LLM providers**: OpenAI (default `gpt-5.4-mini`, current `LLM_PROVIDER`) or Google GenAI (default `gemini-3.5-flash`); agent models are env-overridable via `OPENAI_MODEL` / `GEMINI_MODEL`; embeddings always via `gemini-embedding-2` (768-dim)

Sibling repos (same parent folder, same Supabase project):
- `../tormoose` — Next.js merchant dashboard (Vercel). Owns auth, FB OAuth token exchange, bot settings UI.
- `../admin-space` — internal Streamlit admin (invites, catalog publishing, bot activation, embeddings backfill).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run production (Railway) — MUST stay single-worker (see Known Constraints)
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

## Required Environment Variables

Set in `.env`:
- `GEMINI_API_KEY` — Google GenAI API key (LLM + embeddings). NOTE: currently a FREE-TIER key (~100 embed items/min) — bulk embedding must be throttled.
- `OPENAI_API_KEY` — required when `LLM_PROVIDER=openai` (current production setting)
- `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` — Supabase service-role access
- `FACEBOOK_VERIFY_TOKEN` — Webhook GET verification token
- `FACEBOOK_APP_SECRET` — Meta App Secret. Enables X-Hub-Signature-256 verification on POST /webhook. If unset, the server logs a loud warning and accepts unsigned payloads (dev only — REQUIRED in production).
- `INTERNAL_WEBHOOK_SECRET` — shared secret; the Supabase product webhook must send it as `x-internal-secret` header. Same fail-open-with-warning behavior if unset.
- `PORT` — Server port (default 8000)

**Note:** `FACEBOOK_PAGE_ACCESS_TOKEN` is not an env var — it's fetched per-tenant from `bot_settings`.

## Architecture

### Multi-Tenant Resolution

```
Facebook POST /api/v1/webhook
  → X-Hub-Signature-256 HMAC verified against raw body (FACEBOOK_APP_SECRET)
  → entry.id (Facebook page ID)
  → resolve_tenant(entry.id) queries bot_settings WHERE page_id = entry.id
  → is_active=false → TenantInactiveError → events silently skipped
  → returns FROZEN TenantContext(shop_id, page_access_token, facebook_page_id)
  → per-message copy via tenant.for_sender(psid) — NEVER mutate a shared context
  → 60-second TTL cache (inactive tenants cached too; toggling takes ≤60s)
```

### Message Processing Flow

```
POST /api/v1/webhook → 200 immediately (asyncio background task)
  → FacebookService.process_webhook_event()
      → mid dedup (5-min TTLCache)
      → tenant = page_tenant.for_sender(sender_id)
  → MessageRouter.route_message()
      → reply-to enrichment (reply_context mid cache)
      → InputGuard: control/invisible char strip, injection DETECTION (logs,
        never mutates text), fixed-window rate limit (notifies user once/window)
  → MessageBatcher (key = "{shop_id}:{sender_id}")
      → 8s debounce (message_batch_timeout, resets per message); per-conversation asyncio.Lock serializes processing —
        an in-flight LLM run is never cancelled; later batches queue behind it
      → 90s processing timeout → logged, NOTHING sent to the user
  → TextHandler
      → logs user message to Supabase messages (fire-and-forget)
      → AgentService.process() → reply
      → artificial typing delay (1.5–4s) → send → log bot reply
```

**Errors are NEVER surfaced to the user** (policy, 2026-07-15): any internal
failure (LLM error, timeout, handler exception) is logged and the bot stays
silent — an empty reply from AgentService means "error already logged, send
nothing". Deliberate guard notices (rate-limit, too-long, unsupported message
type) still reply, as they are UX, not errors.

### Agent (AgentService)

- System instruction is composed PER REQUEST and passed as a parameter (never
  stored on the singleton): `ai_configurations.system_prompt` (tenant persona,
  2-min cache) + `PLATFORM_RULES` (language/formatting/order flow) + greeting
  hint + customer profile (2-min cache, invalidated on order).
- Memory: in-process TTLCache keyed `"{shop_id}:{sender_id}"` (600s TTL, 30-msg
  cap). On cache miss, history is REHYDRATED from the Supabase `messages` table.
- ReAct loop, MAX_TURNS=5. The FINAL turn forbids tool calls (Gemini:
  `FunctionCallingConfig(mode="NONE")`; OpenAI: `tool_choice="none"`) so the
  user always gets a text answer.
- History >15 messages → background summarization (serialized per sender via
  `_summarizing` set; keeps messages appended during the summarize call).

### Tools (strict tenant isolation, two-phase ordering)

Declared as Python stubs in `app/core/tools.py` (Gemini auto-parse) mirrored in
`app/core/openai_tools.py`. Execution in `AgentService._execute_tool()` injects
`tenant.shop_id` / `tenant.sender_id` server-side — the LLM NEVER supplies them.

| Tool | Notes |
|------|-------|
| `search_products(query)` | RagService → `match_products_hybrid` RPC (FTS + pgvector), then a second query expands hits into their full VARIANT family (every size is its own products row sharing the same `name`; `attributes` holds size/color/stock/fabric). Returns products grouped by name with a `variants` list (`product_id` + compacted attributes per size); whitelists all variant `image_url`s. There is no product_url — the table has no such column |
| `get_company_policy(topic)` | `bot_settings.store_policies` |
| `prepare_order(product_ids, quantities, delivery_address, contact_number, notes)` | Validates products against catalog, enforces `attributes.stock` per variant (rejects out-of-stock / over-stock), computes total, stores 15-min draft. Item names include the size. Does NOT write an order |
| `confirm_order()` | Consumes the draft (idempotent) → customers UPDATE + orders INSERT (`customer_id`, `total_amount`, `status='processing'`, delivery fields; the live `orders_status_check` constraint allows only processing/shipped/delivered/cancelled) + `order_items` rows |
| `check_order_status(order_number)` | Scoped to shop AND this customer's `customer_id` — customers can't read each other's orders |
| `send_product_image(image_url)` | URL must be in the whitelist from this conversation's search results (blocks injected/hallucinated URLs) |

### Conversation Persistence (persistence_service)

Every user message and bot reply is written to Supabase `messages` (via
customer → thread resolution, both cached) so the tormoose dashboard can show
chat history and the bot survives restarts. Writes are fire-and-forget — they
never block or fail the reply path.

### Product Embedding Pipeline

**There is NO automatic embedding pipeline.** No Supabase webhook is configured
(verified 2026-07-07) — this is why products historically sat at
`embedding_status='pending'`. After every catalog publish, an admin must run
the admin-space "🧠 Embeddings Backfill" module (throttled for the free-tier
Gemini quota); the publish flow shows a reminder.

`POST /api/v1/internal/webhook/supabase-product` (requires `x-internal-secret`
header) exists and embeds a single product per call. It is currently UNUSED —
wired for a future per-insert Supabase webhook. Do not create that webhook
while publishes are bulk inserts: it fires per row and would flood the
free-tier quota (~100 items/min).

### Layer Responsibilities

- **`app/core/tenant_context.py`** — frozen TenantContext + resolve_tenant() (TTL cache, is_active enforcement)
- **`app/core/dependencies.py`** — ASYNC Supabase client (`await get_supabase()`) + GenAI client singletons. All DB calls must be awaited — the sync client blocked the event loop
- **`app/core/config.py`** — Pydantic BaseSettings from env vars
- **`app/core/tools.py` / `app/core/openai_tools.py`** — tool schemas (keep in sync!)
- **`app/api/v1/endpoints/`** — HTTP concerns: signature verification, immediate 200s
- **`app/services/`** — agent, batching, guard, memory, messaging (2000-char chunking, shared httpx client, Graph v22.0), persistence, RAG, tenant_config
- **`app/services/handlers/`** — message_router + text_handler (images flow through the batcher into the agent; there is no separate image handler)
- **`app/schemas/facebook.py`** — Pydantic models for webhook payloads

### Supabase Tables Used

| Table | Used By |
|-------|---------|
| `bot_settings` | resolve_tenant (is_active!), get_company_policy |
| `ai_configurations` | tenant persona / greeting / fallback (tenant_config) |
| `products` | search RPC, prepare_order validation, embedding webhook |
| `customers` | persistence (get_or_create), profile context, confirm_order |
| `orders` + `order_items` | confirm_order (INSERT), check_order_status |
| `threads` + `messages` | conversation persistence + rehydration. Live enums (Supabase-only, not in repo SQL): `thread_status` = 'bot_active' (default) \| 'closed' — no 'open'; `sender_type` = 'customer' \| 'bot' \| 'human' — no 'agent' |

## Known Constraints

- **SINGLE PROCESS ONLY.** Memory, batching locks, rate limits, mid-dedup,
  order drafts, and image whitelists are in-process dicts/TTLCaches. Running
  `--workers 2` or multiple Railway replicas silently breaks batching and
  memory. Move this state to Redis before scaling out.
- Gemini key is free-tier: throttle any bulk embedding.
- `bot_settings.is_active` gates replies; activation is done from admin-space
  (auto on catalog publish) or the tormoose settings page.
