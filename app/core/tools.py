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
        List of matching products with name, price, description, image_url and a
        'variants' list. EVERY SIZE IS A SEPARATE VARIANT with its own product_id
        and stock — when preparing an order, use the product_id of the exact
        variant (size) the user wants.
    """
    # Stub — real logic is in _execute_tool → RagService.search_catalog()
    return []


def get_company_policy(topic: str) -> str:
    """Retrieve store info written by the owner: shipping, returns, operating hours,
    store location/address, about the business, and contact details.

    Args:
        topic: The topic requested (e.g. 'operating hours', 'return policy', 'shipping',
            'location', 'about us', 'contact').

    Returns:
        A string containing the relevant store policy information.
    """
    # Stub — real logic is in _execute_tool → Supabase bot_settings.store_policies
    return ""


def prepare_order(
    product_ids: list[str],
    quantities: list[int],
    delivery_address: str,
    contact_number: str,
    notes: str = "",
) -> dict:
    """Prepare an order draft once the user has provided all details. Does NOT place the order.

    Returns a summary with validated names and the exact total. You MUST relay
    this summary to the user and ask for explicit confirmation, then call
    confirm_order only after they clearly say yes.

    Args:
        product_ids: The variant product_id values from search_products results
            (pick the variant matching the user's size), in order.
        quantities: Quantity for each product, same order as product_ids.
        delivery_address: The complete delivery address provided by the user.
        contact_number: The user's contact phone number.
        notes: Sizes, variants, or special instructions (e.g. 'size L, blue').

    Returns:
        A dictionary with the order summary (items, unit prices, total) to relay.
    """
    # Stub — real logic is in _execute_tool (validates products, computes total, stores draft)
    return {}


def confirm_order() -> dict:
    """Place the order previously prepared with prepare_order. Call ONLY after the user explicitly confirms (e.g. 'yes', 'haan', 'confirm').

    Returns:
        A dictionary containing the order status and order number.
    """
    # Stub — real logic is in _execute_tool → Supabase customers/orders/order_items
    return {}


def check_order_status(order_number: str) -> dict:
    """Check the current status of one of THIS customer's existing orders by order number.

    Args:
        order_number: The order number to look up (e.g. 'ORD-A1B2C3D4').

    Returns:
        A dictionary containing order details and current status.
    """
    # Stub — real logic is in _execute_tool → Supabase orders SELECT
    return {}


def send_product_image(image_url: str) -> dict:
    """Send a product image to the user's chat screen. Use this when the user wants to see an item.

    Only image_url values that came back from search_products results are allowed.

    Args:
        image_url: The image_url field of a product from search_products results.

    Returns:
        Status indicating if the image was successfully dispatched.
    """
    # Stub — real logic is in _execute_tool → MessagingService.send_image()
    return {}
