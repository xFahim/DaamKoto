"""Tools module for Gemini agentic workflows."""

from app.core.logging_config import get_logger

logger = get_logger(__name__)

def search_products(query: str) -> list[dict]:
    """Search the product catalog for availability, price, colors, or sizes.

    Args:
        query: The search string (e.g. 'red t shirt', 'sneakers under 2000').

    Returns:
        List of matching products.
    """
    logger.debug(f"search_products called with query: {query}")
    # Dummy mock response, we will wire this to RAG later
    return [
        {"product_id": "P101", "name": "Red Classic T-Shirt", "price": 500, "sizes": ["M", "L", "XL"], "in_stock": True, "image_url": "https://dummyimage.com/400x400/ff0000/ffffff.png&text=Red+T-Shirt"},
        {"product_id": "P102", "name": "Premium Red Hoodie", "price": 1500, "sizes": ["L"], "in_stock": True, "image_url": "https://dummyimage.com/400x400/cc0000/ffffff.png&text=Red+Hoodie"}
    ]

def get_company_policy(topic: str) -> str:
    """Retrieve dummy company policy info regarding hours, returns, shipping, etc.

    Args:
        topic: The topic requested (e.g. 'operating hours', 'return policy').

    Returns:
        A string documenting the policy.
    """
    logger.debug(f"get_company_policy called with topic: {topic}")
    topic_lower = topic.lower()
    if "hour" in topic_lower or "time" in topic_lower:
        return "We are open from 10:00 AM to 10:00 PM everyday."
    if "return" in topic_lower or "refund" in topic_lower:
        return "We accept returns within 7 days of purchase, provided the tag is intact."
    if "ship" in topic_lower or "deliver" in topic_lower:
        return "Delivery takes 2-3 days inside Dhaka and 4-5 days outside Dhaka."
    return "Please contact our support team at 01700000000 for more details."

def execute_order(item_names: str, sizes: str, delivery_address: str, contact_number: str) -> dict:
    """Process an order and insert it into the database once the user confirms all details.

    Args:
        item_names: The exact names or IDs of the items being purchased.
        sizes: The required sizes or variants, if applicable.
        delivery_address: The complete delivery address provided by the user.
        contact_number: The user's contact phone number.

    Returns:
        A dictionary containing the order status and order ID.
    """
    logger.debug(f"execute_order called: items={item_names}, sizes={sizes}, address={delivery_address}, phone={contact_number}")
    # Mocking a DB insert
    return {
        "status": "success",
        "order_id": "ORD-58392",
        "message": f"Order successfully placed for {item_names} to {delivery_address}."
    }

def send_product_image(image_url: str) -> dict:
    """Send a product image physically to the user's chat screen. Use this when the user wants to see an item.
    
    Args:
        image_url: The direct URL of the image you want to send.
        
    Returns:
        Status indicating if the image was successfully dispatched.
    """
    logger.debug(f"send_product_image called with url: {image_url}")
    return {"status": "Image successfully dispatched to the user interface."}
