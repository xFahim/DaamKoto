"""Test script for RAG ingestion pipeline."""

import os
import sys
from pathlib import Path
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Configuration
API_URL = "http://127.0.0.1:8000/api/v1/ingest"
PAGE_ID = "goodybro"
JSON_FILE = project_root / "goodybro.json"  # Path relative to project root
INDEX_NAME = "chatpulse"


def upload_data():
    """Upload data from goodybro.json to the ingestion API."""
    print(f"Uploading data from {JSON_FILE}...")
    
    try:
        # Check if file exists
        if not JSON_FILE.exists():
            print(f"Error: {JSON_FILE} not found in the project root directory.")
            return
        
        # Open and read the JSON file
        with open(JSON_FILE, "rb") as f:
            files = {"file": (JSON_FILE.name, f, "application/json")}
            data = {"page_id": PAGE_ID}
            
            # Send POST request
            response = requests.post(API_URL, files=files, data=data)
            
            # Print response
            print(f"\nStatus Code: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                print("\n✓ Upload successful!")
            else:
                print("\n✗ Upload failed!")
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"Error: {str(e)}")


def delete_namespace():
    """Delete the namespace from Pinecone."""
    print(f"Deleting namespace 'store_{PAGE_ID}' from Pinecone...")
    
    try:
        # Get Pinecone API key from environment
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            print("Error: PINECONE_API_KEY not found in environment variables.")
            return
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Connect to the index
        index = pc.Index(INDEX_NAME)
        
        # Delete the namespace
        namespace = f"store_{PAGE_ID}"
        index.delete(delete_all=True, namespace=namespace)
        
        print(f"\n✓ Namespace '{namespace}' has been wiped successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("=" * 50)
    print("RAG Ingestion Pipeline Test Script")
    print("=" * 50)
    print("\nOptions:")
    print("  1. Upload data")
    print("  2. Delete namespace")
    print("=" * 50)
    
    choice = input("\nType '1' to Upload, '2' to Delete Namespace: ").strip()
    
    if choice == "1":
        upload_data()
    elif choice == "2":
        delete_namespace()
    else:
        print("Invalid choice. Please run the script again and enter '1' or '2'.")

