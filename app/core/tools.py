"""Tool definitions for Gemini native function calling.

These are the function SIGNATURES that the google-genai SDK auto-parses
to generate tool schemas. The actual execution logic lives in
AgentService._execute_tool() which injects tenant context server-side.

IMPORTANT: These functions must NEVER accept shop_id, page_id, or sender_id
as parameters. Tenant isolation is enforced in the execution bridge.
"""

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def search_products(query: str) -> list[dict]:
    """Search the product catalog for availability, price, colors, or sizes.

    Args:
        query: The search string (e.g. 'red t shirt', 'sneakers under 2000').

    Returns:
        List of matching products.
    """
    # Stub — real logic is in _execute_tool → RagService.search_catalog()
    return []


def get_company_policy(topic: str) -> str:
    """Retrieve the store's policies about shipping, returns, operating hours, etc.

    Args:
        topic: The topic requested (e.g. 'operating hours', 'return policy', 'shipping').

    Returns:
        A string containing the relevant store policy information.
    """
    # Stub — real logic is in _execute_tool → Supabase bot_settings.store_policies
    return ""


def execute_order(item_names: str, sizes: str, delivery_address: str, contact_number: str) -> dict:
    """Place an order after the user explicitly confirms all details.

    Args:
        item_names: The exact names or IDs of the items being purchased.
        sizes: The required sizes or variants, if applicable.
        delivery_address: The complete delivery address provided by the user.
        contact_number: The user's contact phone number.

    Returns:
        A dictionary containing the order status and order number.
    """
    # Stub — real logic is in _execute_tool → Supabase customers + orders INSERT
    return {}


def check_order_status(order_number: str) -> dict:
    """Check the current status of an existing order by its order number.

    Args:
        order_number: The order number to look up (e.g. 'ORD-A1B2C3D4').

    Returns:
        A dictionary containing order details and current status.
    """
    # Stub — real logic is in _execute_tool → Supabase orders SELECT
    return {}


def send_product_image(image_url: str) -> dict:
    """Send a product image physically to the user's chat screen. Use this when the user wants to see an item.

    Args:
        image_url: The direct URL of the image you want to send.

    Returns:
        Status indicating if the image was successfully dispatched.
    """
    # Stub — real logic is in _execute_tool → MessagingService.send_image()
    return {}
