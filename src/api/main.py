from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.utils.observability import initialize_observability, Logger
from src.api.routes import router as api_router
import os

# Initialize Observability
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
initialize_observability(environment=ENVIRONMENT)

logger = Logger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Tennis Prediction API",
        description="ML-powered tennis match prediction service.",
        version="1.0.0"
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Production: Restrict to frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Root redirect
    @app.get("/")
    def root():
        return {"message": "Tennis Prediction API. Go to /docs for Swagger UI.", "version": "1.0.0"}

    logger.log_event("api_startup_complete", environment=ENVIRONMENT)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
