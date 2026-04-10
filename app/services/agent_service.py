"""Multi-turn agentic service supporting Gemini and OpenAI providers."""

import json
import httpx
from app.core.config import settings
from app.core.tools import search_products, get_company_policy, execute_order, send_product_image
from app.services.memory_service import memory_service
from app.services.messaging_service import messaging_service


# System instruction shared by both providers
SYSTEM_INSTRUCTION = (
    "You are a helpful customer service and sales agent for an e-commerce store. "
    "Your job is to assist users, answer questions, and process orders.\n\n"
    "## ORDER PLACEMENT RULES\n"
    "If the user wants to buy something, YOU MUST follow this exact flow:\n"
    "1. Ask the user for ALL missing required details in a single message (delivery address, contact number, items, sizes).\n"
    "2. Once the user provides the details, validate them (e.g. ensure they didn't just type gibberish).\n"
    "3. DO NOT immediately call the order tool yet.\n"
    "4. Summarize the items, address, and phone number back to the user and explicitly ask: 'Do you confirm this order?'\n"
    "5. Only if the user explicitly confirms (e.g., 'yes', 'confirm', 'go ahead'), call the 'execute_order' tool.\n\n"
    "## TONE\n"
    "Keep responses concise. Only ask one clarifying question at a time."
)


class AgentService:
    """Agent orchestrator for handling user messages and tool execution."""

    def __init__(self):
        self.provider = None       # "gemini" or "openai"
        self.gemini_client = None
        self.openai_client = None
        # Provide the actual Python functions. The SDK parses their signatures and docstrings.
        self.tools = [search_products, get_company_policy, execute_order, send_product_image]

    def initialize(self):
        """Configure the LLM client based on the selected provider."""
        self.provider = settings.llm_provider.lower().strip()
        print(f"[Agent] Initializing with provider: {self.provider}")

        if self.provider == "openai":
            from openai import AsyncOpenAI
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            print("Agent service initialized with OpenAI (gpt-4o-mini).")
        else:
            from google import genai
            self.gemini_client = genai.Client(api_key=settings.gemini_api_key)
            self.provider = "gemini"
            print("Agent service initialized with Gemini.")

    async def _execute_tool(self, call_name: str, call_args: dict, sender_id: str, page_id: str) -> dict:
        """Dynamically execute the matched python tool."""
        print(f"Executing tool {call_name} with args {call_args}")

        # Send typing to show we're doing stuff
        await messaging_service.send_typing_on(sender_id)

        try:
            if call_name == "search_products":
                from app.services.rag_service import rag_service
                products = await rag_service.search_catalog(query=call_args.get("query", ""), page_id=page_id)
                result = {"products_found": products} if products else {"message": "No relevant products found in the catalog."}
            elif call_name == "get_company_policy":
                result = get_company_policy(**call_args)
            elif call_name == "execute_order":
                result = execute_order(**call_args)
            elif call_name == "send_product_image":
                url = call_args.get("image_url", "")
                print(f"[AGENT TOOL] Attempting to send image via Messenger Graph API. URL: '{url}'")
                if not url or url.lower() in ["none", "null", "undefined"]:
                    print("[AGENT TOOL] Aborting image dispatch: Expected a valid URL, got empty string.")
                    result = {"status": "Failed: You must provide a valid image_url string."}
                else:
                    success = await messaging_service.send_image(sender_id, url)
                    if success:
                        result = {"status": "Image successfully dispatched to the user interface."}
                    else:
                        result = {"status": "Failed to dispatch image to Facebook. Invalid URL format or Facebook API error."}
            else:
                result = {"error": f"Unknown tool: {call_name}"}
        except Exception as e:
            print(f"Error executing tool {call_name}: {str(e)}")
            result = {"error": str(e)}

        # Ensure result is always a dict
        if not isinstance(result, dict):
            result = {"result": result}

        return result

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────

    async def process(self, sender_id: str, message_text: str = "", image_urls: list[str] = None, page_id: str = "") -> str:
        """Process a message through the ReAct agent loop."""
        if self.provider == "openai":
            return await self._process_openai(sender_id, message_text, image_urls, page_id)
        else:
            return await self._process_gemini(sender_id, message_text, image_urls, page_id)

    # ─────────────────────────────────────────────────────────────────────
    #  Gemini path (unchanged logic)
    # ─────────────────────────────────────────────────────────────────────

    async def _process_gemini(self, sender_id: str, message_text: str, image_urls: list[str] | None, page_id: str) -> str:
        from google import genai
        from google.genai import types

        if not self.gemini_client:
            print("Agent service not initialized!")
            return "Server is currently unavailable."

        # 1. Load History (as Gemini Content objects)
        history = memory_service.get_gemini_history(sender_id)

        # 2. Append User Message
        parts = []
        if message_text:
            parts.append(types.Part.from_text(text=message_text))

        if image_urls:
            for url in image_urls:
                try:
                    import io
                    import tempfile
                    import os
                    import uuid

                    async with httpx.AsyncClient(timeout=10.0) as http_client:
                        response = await http_client.get(url)
                        response.raise_for_status()
                        image_bytes = response.content

                    # Write to temp file because the SDK upload function requires a valid file path or supported file-like object
                    temp_file_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
                    with open(temp_file_path, "wb") as f:
                        f.write(image_bytes)

                    uploaded_file = await self.gemini_client.aio.files.upload(
                        file=temp_file_path,
                        config={'mime_type': 'image/jpeg'}
                    )

                    # Cleanup temp
                    os.remove(temp_file_path)

                    # Use the Cloud URI so your server's context memory RAM stays tiny!
                    parts.append(types.Part.from_uri(uri=uploaded_file.uri, mime_type="image/jpeg"))
                    print(f"[AgentService] Successfully uploaded image to Gemini Cloud: {uploaded_file.uri}")

                except Exception as e:
                    print(f"Failed to upload image to agent context: {e}")
                    parts.append(types.Part.from_text(text="[User attached an image but the system failed to download it]"))

        if not parts:
            parts.append(types.Part.from_text(text="[Empty message]"))

        user_content = types.Content(role="user", parts=parts)
        memory_service.append_content(sender_id, user_content)
        history.append(user_content)

        # 3. Setup Agent Prompt
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=self.tools,
            temperature=0.7,
            # We explicitly handle the loop ourselves
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        # 4. Agent Execution Loop
        MAX_TURNS = 5
        for _ in range(MAX_TURNS):
            try:
                response = await self.gemini_client.aio.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=history,
                    config=config
                )
            except Exception as e:
                print(f"[Agent] Generation failed: {e}")
                err_reply = "I ran into a server error processing your request."
                memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=err_reply)]))
                return err_reply

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                print(f"[Agent Token Usage] Prompt: {usage.prompt_token_count or 0} | Output: {usage.candidates_token_count or 0} | Total: {usage.total_token_count or 0}")

            if not response.candidates:
                err_reply = "I'm not sure how to respond to that."
                memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=err_reply)]))
                return err_reply

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                err_reply = "I'm having trouble thinking."
                memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=err_reply)]))
                return err_reply

            # Copy content to avoid modifying references and append to history & memory cache
            memory_service.append_content(sender_id, candidate.content)
            history.append(candidate.content)

            # Check for tool calls
            tool_calls = [p.function_call for p in candidate.content.parts if p.function_call]

            if not tool_calls:
                # Text response generated, break loop and return the text
                text_parts = [p.text for p in candidate.content.parts if p.text]
                final_text = "\n".join(text_parts)
                return final_text

            # Execute tools
            tool_responses = []
            for call in tool_calls:
                result = await self._execute_tool(call.name, dict(call.args) if call.args else {}, sender_id, page_id)
                tool_responses.append(types.Part.from_function_response(
                    name=call.name,
                    response=result
                ))

            # Send tool responses back to model
            tool_content = types.Content(role="user", parts=tool_responses)
            memory_service.append_content(sender_id, tool_content)
            history.append(tool_content)

        # If it reached here, it exceeded the loop limit
        reply = "I had too much to process. Let's start over."
        memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
        return reply

    # ─────────────────────────────────────────────────────────────────────
    #  OpenAI path (same logic, different SDK)
    # ─────────────────────────────────────────────────────────────────────

    async def _process_openai(self, sender_id: str, message_text: str, image_urls: list[str] | None, page_id: str) -> str:
        from app.core.openai_tools import OPENAI_TOOLS

        if not self.openai_client:
            print("Agent service not initialized!")
            return "Server is currently unavailable."

        # 1. Load History (as OpenAI message dicts)
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        messages.extend(memory_service.get_openai_history(sender_id))

        # 2. Build user message
        if image_urls:
            # Multimodal message with images
            content_parts = []
            if message_text:
                content_parts.append({"type": "text", "text": message_text})
            for url in image_urls:
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            user_msg = {"role": "user", "content": content_parts}
        elif message_text:
            user_msg = {"role": "user", "content": message_text}
        else:
            user_msg = {"role": "user", "content": "[Empty message]"}

        # Store in provider-agnostic memory
        memory_service.append_content(sender_id, {
            "role": "user",
            "parts": self._openai_msg_to_parts(user_msg),
        })
        messages.append(user_msg)

        # 3. Agent Execution Loop
        MAX_TURNS = 5
        for _ in range(MAX_TURNS):
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=OPENAI_TOOLS,
                    tool_choice="auto",
                    temperature=0.7,
                )
            except Exception as e:
                print(f"[Agent] OpenAI generation failed: {e}")
                err_reply = "I ran into a server error processing your request."
                memory_service.append_content(sender_id, {
                    "role": "model",
                    "parts": [{"type": "text", "text": err_reply}],
                })
                return err_reply

            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                print(f"[Agent Token Usage] Prompt: {usage.prompt_tokens} | Output: {usage.completion_tokens} | Total: {usage.total_tokens}")

            choice = response.choices[0]
            assistant_msg = choice.message

            if not assistant_msg:
                err_reply = "I'm not sure how to respond to that."
                memory_service.append_content(sender_id, {
                    "role": "model",
                    "parts": [{"type": "text", "text": err_reply}],
                })
                return err_reply

            # Save assistant message to history
            assistant_dict = assistant_msg.model_dump(exclude_none=True)
            messages.append(assistant_dict)

            # Save to memory in our internal format
            internal_parts = []
            if assistant_msg.content:
                internal_parts.append({"type": "text", "text": assistant_msg.content})
            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    internal_parts.append({
                        "type": "function_call",
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments) if tc.function.arguments else {},
                        "tool_call_id": tc.id,
                    })
            if not internal_parts:
                internal_parts.append({"type": "text", "text": ""})

            memory_service.append_content(sender_id, {
                "role": "model",
                "parts": internal_parts,
            })

            # Check for tool calls
            if not assistant_msg.tool_calls:
                return assistant_msg.content or "I'm having trouble thinking."

            # Execute tools
            for tc in assistant_msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = await self._execute_tool(tc.function.name, args, sender_id, page_id)

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
                messages.append(tool_msg)

                # Save to internal memory
                memory_service.append_content(sender_id, {
                    "role": "user",
                    "parts": [{
                        "type": "function_response",
                        "name": tc.function.name,
                        "response": result,
                        "tool_call_id": tc.id,
                    }],
                })

        # Exceeded loop limit
        reply = "I had too much to process. Let's start over."
        memory_service.append_content(sender_id, {
            "role": "model",
            "parts": [{"type": "text", "text": reply}],
        })
        return reply

    @staticmethod
    def _openai_msg_to_parts(msg: dict) -> list[dict]:
        """Convert an OpenAI-style user message to our internal parts format."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    parts.append({"type": "file_data", "uri": item["image_url"]["url"], "mime_type": "image/jpeg"})
            return parts
        return [{"type": "text", "text": str(content)}]


agent_service = AgentService()
