
import os
import json
import traceback
from dotenv import load_dotenv

# Try to import required Google Cloud libraries
try:
    import vertexai
    from google.oauth2 import service_account
except ImportError:
    print("Error: Required libraries 'google-cloud-aiplatform' or 'google-auth' not found.")
    print("Please install them using: pip install google-cloud-aiplatform google-auth")
    exit(1)

# Load environment variables from .env file
load_dotenv()

def initialize_vertex():
    print("Attempting to initialize Vertex AI...")
    
    # Load the JSON string from environment variable
    service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    
    if not service_account_json:
        print("Error: GCP_SERVICE_ACCOUNT_JSON environment variable is not set.")
        # Debug: check if .env was loaded
        if os.path.exists(".env"):
            print("Found .env file.")
        else:
            print("Warning: .env file not found in current directory.")
        return

    try:
        # Parse the string into a dictionary and create credentials
        print("Parsing service account JSON...")
        info = json.loads(service_account_json)
        
        print("Creating credentials...")
        credentials = service_account.Credentials.from_service_account_info(info)
        
        # Initialize Vertex AI
        print("Initializing vertexai...")
        vertexai.init(
            project=info["project_id"], 
            location="us-central1", 
            credentials=credentials
        )
        print("✓ Vertex AI Initialized successfully.")
        
    except json.JSONDecodeError as e:
         print(f"Error parsing JSON from GCP_SERVICE_ACCOUNT_JSON: {e}")
    except KeyError as e:
        print(f"Error: Missing key in service account JSON: {e}")
    except Exception as e:
        print(f"Failed to initialize Vertex AI: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    initialize_vertex()
