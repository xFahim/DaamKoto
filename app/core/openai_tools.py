"""OpenAI function-calling tool definitions.

These mirror the exact same tools defined in tools.py but expressed as
JSON schemas that OpenAI's chat completions API expects.
The actual execution still routes through the Python functions in tools.py.
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
            "description": "Retrieve company policy info regarding hours, returns, shipping, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic requested (e.g. 'operating hours', 'return policy')."
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
            "description": "Process an order and insert it into the database once the user confirms all details.",
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
