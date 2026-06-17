"""OpenAI function-calling tool definitions.

These mirror the exact same tools defined in tools.py but expressed as
JSON schemas that OpenAI's chat completions API expects.
The actual execution still routes through AgentService._execute_tool().
"""

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog for availability, price, colors, or sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search string (e.g. 'red t shirt', 'sneakers under 2000')."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_policy",
            "description": "Retrieve the store's policies about shipping, returns, operating hours, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic requested (e.g. 'operating hours', 'return policy', 'shipping')."
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_order",
            "description": "Place an order after the user explicitly confirms all details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_names": {
                        "type": "string",
                        "description": "The exact names or IDs of the items being purchased."
                    },
                    "sizes": {
                        "type": "string",
                        "description": "The required sizes or variants, if applicable."
                    },
                    "delivery_address": {
                        "type": "string",
                        "description": "The complete delivery address provided by the user."
                    },
                    "contact_number": {
                        "type": "string",
                        "description": "The user's contact phone number."
                    }
                },
                "required": ["item_names", "sizes", "delivery_address", "contact_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check the current status of an existing order by its order number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "The order number to look up (e.g. 'ORD-A1B2C3D4')."
                    }
                },
                "required": ["order_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_product_image",
            "description": "Send a product image physically to the user's chat screen. Use this when the user wants to see an item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "The direct URL of the image you want to send."
                    }
                },
                "required": ["image_url"]
            }
        }
    }
]
