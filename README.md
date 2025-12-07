# DaamKoto

A robust, scalable FastAPI project for integrating Facebook Messenger Webhooks, with architecture designed for easy expansion to WhatsApp, Instagram, and AI automation.

## Features

- Facebook Messenger Webhook integration
- Structured for multi-platform expansion (WhatsApp, Instagram)
- Type-hinted codebase following modern Python best practices
- Environment-based configuration using Pydantic Settings

## Project Structure

```
DaamKoto/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # Settings and configuration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py           # Primary API router
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           └── facebook.py # Facebook webhook endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   └── facebook_service.py # Facebook business logic
│   └── schemas/
│       ├── __init__.py
│       └── facebook.py         # Pydantic models for Facebook webhooks
├── .env                        # Environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   - Copy `.env` and set your `FACEBOOK_VERIFY_TOKEN`

4. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

### Facebook Webhook

- `GET /api/v1/webhook` - Webhook verification endpoint
- `POST /api/v1/webhook` - Webhook message reception endpoint

## Development

The project follows a clean architecture pattern with separation of concerns:

- **Schemas**: Data validation models
- **Services**: Business logic
- **API Endpoints**: Request/response handling
- **Core**: Configuration and shared utilities

## Future Expansion

The architecture is designed to easily accommodate:

- WhatsApp integration
- Instagram integration
- AI automation features
