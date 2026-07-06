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
            "description": (
                "Search the product catalog for availability, price, colors, or sizes. "
                "Each result has a 'variants' list — every size is a separate variant "
                "with its own product_id and stock. When preparing an order, use the "
                "product_id of the exact variant (size) the user wants."
            ),
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
            "name": "prepare_order",
            "description": (
                "Prepare an order draft once the user has provided all details. Does NOT place "
                "the order. Returns a summary with validated names and the exact total. You MUST "
                "relay this summary to the user and ask for explicit confirmation, then call "
                "confirm_order only after they clearly say yes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The variant product_id values from search_products results (pick the variant matching the user's size), in order."
                    },
                    "quantities": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Quantity for each product, same order as product_ids."
                    },
                    "delivery_address": {
                        "type": "string",
                        "description": "The complete delivery address provided by the user."
                    },
                    "contact_number": {
                        "type": "string",
                        "description": "The user's contact phone number."
                    },
                    "notes": {
                        "type": "string",
                        "description": "Sizes, variants, or special instructions (e.g. 'size L, blue')."
                    }
                },
                "required": ["product_ids", "quantities", "delivery_address", "contact_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order",
            "description": (
                "Place the order previously prepared with prepare_order. Call ONLY after the "
                "user explicitly confirms (e.g. 'yes', 'haan', 'confirm')."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check the current status of one of THIS customer's existing orders by order number.",
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
            "description": (
                "Send a product image to the user's chat screen. Use this when the user wants "
                "to see an item. Only image_url values from search_products results are allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {
                        "type": "string",
                        "description": "The image_url field of a product from search_products results."
                    }
                },
                "required": ["image_url"]
            }
        }
    }
]
