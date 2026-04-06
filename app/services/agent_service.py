"""Multi-turn agentic service using Gemini."""

from google import genai
from google.genai import types
import httpx
from app.core.config import settings
from app.core.tools import search_products, get_company_policy, execute_order, send_product_image
from app.services.memory_service import memory_service
from app.services.messaging_service import messaging_service


class AgentService:
    """Agent orchestrator for handling user messages and tool execution."""

    def __init__(self):
        self.client = None
        # Provide the actual Python functions. The SDK parses their signatures and docstrings.
        self.tools = [search_products, get_company_policy, execute_order, send_product_image]

    def initialize(self):
        """Configure Gemini client for Agent operations."""
        self.client = genai.Client(api_key=settings.gemini_api_key)
        print("Agent service initialized.")

    async def _execute_tool(self, call: types.FunctionCall, sender_id: str, page_id: str) -> dict:
        """Dynamically execute the matched python tool."""
        name = call.name
        args = dict(call.args) if call.args else {}
        print(f"Executing tool {name} with args {args}")
        
        # Send typing to show we're doing stuff
        await messaging_service.send_typing_on(sender_id)
        
        try:
            if name == "search_products":
                from app.services.rag_service import rag_service
                products = await rag_service.search_catalog(query=args.get("query", ""), page_id=page_id)
                result = {"products_found": products} if products else {"message": "No relevant products found in the catalog."}
            elif name == "get_company_policy":
                result = get_company_policy(**args)
            elif name == "execute_order":
                result = execute_order(**args)
            elif name == "send_product_image":
                url = args.get("image_url", "")
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
                result = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            print(f"Error executing tool {name}: {str(e)}")
            result = {"error": str(e)}
            
        return result

    async def process(self, sender_id: str, message_text: str = "", image_urls: list[str] = None, page_id: str = "") -> str:
        """
        Process a message through the ReAct agent loop.
        """
        if not self.client:
            print("Agent service not initialized!")
            return "Server is currently unavailable."

        # 1. Load History
        history = memory_service.get_history(sender_id)
        
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
                    
                    uploaded_file = await self.client.aio.files.upload(
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
        system_instruction = (
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

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=self.tools,
            temperature=0.7,
            # We explicitly handle the loop ourselves
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        # 4. Agent Execution Loop
        MAX_TURNS = 5
        for _ in range(MAX_TURNS):
            try:
                response = await self.client.aio.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=history,
                    config=config
                )
            except Exception as e:
                print(f"[Agent] Generation failed: {e}")
                err_reply = "I ran into a server error processing your request."
                memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=err_reply)]))
                return err_reply

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
            # The API returns `parts` inside `content`
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
                result = await self._execute_tool(call, sender_id, page_id)
                
                # Gemini FunctionResponse strictly requires a dictionary payload
                if not isinstance(result, dict):
                    result = {"result": result}
                    
                tool_responses.append(types.Part.from_function_response(
                    name=call.name,
                    response=result
                ))
            
            # Send tool responses back to model.
            # Responses are generated BY the user (the system acting on behalf of the user/backend)
            tool_content = types.Content(role="user", parts=tool_responses)
            memory_service.append_content(sender_id, tool_content)
            history.append(tool_content)

        # If it reached here, it exceeded the loop limit
        reply = "I had too much to process. Let's start over."
        memory_service.append_content(sender_id, types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
        return reply


agent_service = AgentService()
