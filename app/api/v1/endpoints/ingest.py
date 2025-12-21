"""RAG ingestion endpoints."""

import json
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from app.services.ingest_service import ingest_service

router = APIRouter()


@router.post("/ingest")
async def ingest_products(
    page_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """
    Ingest products from a JSON file and upload them to Pinecone.

    This endpoint accepts a JSON file containing a list of products,
    processes them into embeddings, and stores them in Pinecone.

    Args:
        page_id: The page/store ID to associate with the products
        file: JSON file containing a list of products

    Returns:
        Dictionary with success status, count, and namespace information

    Raises:
        HTTPException: If the file is invalid, not JSON, or not a list
    """
    # Validate file type
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a JSON file",
        )

    try:
        # Read and parse the JSON file
        contents = await file.read()
        products = json.loads(contents.decode("utf-8"))

        # Validate that it is a list
        if not isinstance(products, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON file must contain a list of products",
            )

        # Process and upload to Pinecone
        result = ingest_service.process_and_upload(page_id=page_id, products=products)

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )


