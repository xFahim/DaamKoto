import asyncio
import json
import uuid
import httpx
from pinecone import Pinecone, ServerlessSpec
import os
import sys
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from app.core.config import settings

# Configuration
INDEX_NAME = "chatpulse-multimodal"
NAMESPACE = "store_goodybro"
BATCH_SIZE = 10
DIMENSION = 1408  # Vertex AI Multimodal Embedding Dimension

# Initialize Vertex AI
try:
    service_account_json = settings.gcp_service_account_json
    if not service_account_json:
        print("⚠️ GCP_SERVICE_ACCOUNT_JSON not found.")
        sys.exit(1)
    else:
        info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(info)
        vertexai.init(
            project=info["project_id"],
            location="asia-east1",
            credentials=credentials
        )
        print("✅ Vertex AI Authenticated.")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    sys.exit(1)

# Initialize Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

async def manage_index():
    """Interactive index management."""
    existing_indexes = pc.list_indexes().names()
    
    if INDEX_NAME in existing_indexes:
        print(f"\n⚠️ Index '{INDEX_NAME}' already exists.")
        user_input = input("❓ Do you want to DELETE and RECREATE this index? (y/n): ").strip().lower()
        
        if user_input == 'y':
            print(f"🗑️ Deleting index '{INDEX_NAME}'...")
            pc.delete_index(INDEX_NAME)
            print("✅ Index deleted.")
        else:
            print("Using existing index.")
            return

    # Create Index
    print(f"🆕 Creating index '{INDEX_NAME}' (Dimension: {DIMENSION}, Metric: cosine)...")
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("✅ Index created successfully.")
    except Exception as e:
        print(f"⚠️ Index creation check: {e}")

    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        print("Waiting for index to be ready...")
        await asyncio.sleep(2)
    print("✅ Index is ready!")

def get_embedding(model, image_bytes: bytes = None, text: str = None):
    """Generate embedding using Vertex AI."""
    try:
        if image_bytes:
            image = VertexImage(image_bytes)
            embeddings = model.get_embeddings(image=image)
            return embeddings.image_embedding
        elif text:
            # Vertex AI text embedding (multimodal)
            embeddings = model.get_embeddings(contextual_text=text)
            return embeddings.text_embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

async def ingest_products():
    print(f"\n🚀 Starting Multimodal Ingestion Pipeline")
    print(f"Target: Index '{INDEX_NAME}', Namespace '{NAMESPACE}'")
    
    # 1. Manage Index (Interactive)
    await manage_index()
    
    # 2. Connect to Index
    index = pc.Index(INDEX_NAME)

    # 3. Load Products
    if not os.path.exists("goodybro.json"):
        print("❌ goodybro.json not found in current directory.")
        return

    with open("goodybro.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"📦 Loaded {len(products)} products from JSON.")

    # 4. Input Confirmation
    confirm = input(f"❓ Ready to ingest {len(products)} products? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Operation cancelled.")
        return

    # 5. Load Model
    print("🧠 Loading Vertex AI Multimodal Embedding Model...")
    try:
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 6. Ingestion Loop
    async with httpx.AsyncClient() as client:
        vectors_to_upsert = []
        
        for i, product in enumerate(products):
            try:
                # Extract fields with fallbacks
                name = product.get("Product Name") or product.get("name") or "Unknown"
                price = product.get("price-item") or product.get("price") or "N/A"
                stock = product.get("badge") or "In Stock"
                product_url = product.get("product url")
                
                # Image URL
                image_url = product.get("motion-reduce src") or product.get("image_url") or product.get("src")
                
                # Validation
                if not image_url:
                    print(f"⚠️ Skipping '{name}': No image URL found.")
                    continue
                
                if image_url.startswith("//"):
                    image_url = "https:" + image_url

                print(f"Processing {i+1}/{len(products)}: {name}...")
                
                # Fetch Image
                try:
                    img_resp = await client.get(image_url, timeout=10.0)
                    img_resp.raise_for_status()
                    image_bytes = img_resp.content
                except Exception as e:
                    print(f"❌ Failed to download image {image_url}: {e}")
                    continue

                # Get Embedding (choose strategy: Image priority, or text if needed)
                # Here we strictly embed the IMAGE for visual search, or you could average them.
                # Current request implies "work with image for product as well", so image embedding is key.
                embedding = await asyncio.to_thread(get_embedding, model, image_bytes=image_bytes)
                
                if not embedding:
                    print(f"⚠️ Failed to generate embedding for '{name}'")
                    continue
                
                # Metadata
                metadata = {
                    "name": name,
                    "price": price,
                    "stock": stock,
                    "description": f"{name} - {stock}",
                    "url": image_url,
                    "product_url": product_url or image_url,
                    "page_id": "goodybro"
                }

                vectors_to_upsert.append({
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": metadata
                })
                
                # Upsert Batch
                if len(vectors_to_upsert) >= BATCH_SIZE:
                    print(f"⬆️ Upserting batch of {len(vectors_to_upsert)}...")
                    index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
                    vectors_to_upsert = []
                
                # Rate Limiting for Free Tier (approx 40 requests/min)
                await asyncio.sleep(1.5)
            
            except Exception as e:
                print(f"❌ Error processing item {i}: {e}")
                continue

        # Final Batch
        if vectors_to_upsert:
            print(f"⬆️ Upserting final batch of {len(vectors_to_upsert)}...")
            index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)

    print("\n✅ Ingestion Pipeline Complete!")

if __name__ == "__main__":
    asyncio.run(ingest_products())
