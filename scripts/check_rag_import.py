
import sys
import os
sys.path.append(os.getcwd())

try:
    from app.services.rag_service import rag_service
    print("✓ Successfully imported rag_service")
except ImportError as e:
    print(f"✗ Failed to import rag_service: {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error importing rag_service: {e}")
    exit(1)
