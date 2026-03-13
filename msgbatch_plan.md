# Implementing Message Batching and Debouncing

This plan outlines the architecture for batching and debouncing rapid, sequential text messages sent by Facebook Messenger users.

## Problem Statement
When a user sends multiple short messages in quick succession (e.g., "Hi", "Do you have this", "in red?"), the bot currently processes each text message individually. This places unnecessary load on the system (e.g. running 3 intent classification and 3 RAG queries) and might result in disconnected, out-of-context replies. 

## Proposed Solution
We'll implement an in-memory debouncing service using Python's `asyncio` to accumulate text messages from a specific `sender_id` over a configurable time window (e.g., 2 seconds). Once the user stops typing for that duration, the gathered messages are concatenated into a single query and passed to the `text_handler`.

## Proposed Changes

### 1. New Service: `app/services/batching_service.py`
#### [NEW] `batching_service.py`(file:///c:/COIDING/GIT%20PRO/DaamKoto/app/services/batching_service.py)
Create `MessageBatcher` with:
- `timeout` (e.g. 2.0 seconds).
- `_pending_messages: dict[str, list[str]]` to store text sequences per `sender_id`.
- `_timers: dict[str, asyncio.Task]` to track active countdowns.
- `add_text_message(sender_id, text, page_id)`: Appends text, cancels old timer, and starts a new async sleep timer.
- `_process_batch(sender_id, page_id)`: Callback when the timer completes. Combines messages with a newline `\n` and calls `text_handler.process()`.

### 2. Update Message Router
#### [MODIFY] [message_router.py](file:///c:/COIDING/GIT%20PRO/DaamKoto/app/services/handlers/message_router.py)(file:///c:/COIDING/GIT%20PRO/DaamKoto/app/services/handlers/message_router.py)
Change [route_message()](file:///c:/COIDING/GIT%20PRO/DaamKoto/app/services/handlers/message_router.py#12-55):
Instead of calling `await text_handler.process(...)` instantly, it will call `await batching_service.add_text_message(sender_id, message["text"], page_id)`.
*Note: Image messages will continue to be routed immediately unless specified otherwise.*

### 3. Prevent Circular Imports
Since `batching_service` needs to call `text_handler` and `message_router` might import `batching_service`, we'll structure the imports carefully (e.g., importing `text_handler` inside the batch processing method or at the module level if safe).

## Verification Plan

### Automated Tests
Currently, there are no webhook-level router integration tests, but we can verify our new `MessageBatcher` via a quick standalone script in the repository (e.g., `scripts/test_batching.py`) or simply running the FastAPI app.

### Manual Verification
1. Run the application locally with localtunnel/ngrok connected to Facebook Messenger.
2. Send 3 rapid messages from Messenger: "I want a shirt", "It must be blue", "Under 500 taka".
3. Check the console logs: Instead of 3 calls to [TextHandler](file:///c:/COIDING/GIT%20PRO/DaamKoto/app/services/handlers/text_handler.py#11-78), you should see the batching service collecting 3 messages and firing exactly *one* call to intent classification/RAG with the text:
   ```
   I want a shirt
   It must be blue
   Under 500 taka
   ```
4. Verify the user receives one cohesive AI response.
