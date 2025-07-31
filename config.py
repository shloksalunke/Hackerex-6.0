from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Loads and validates all required environment variables."""
    MISTRAL_AI_API_KEY: str
    PLATFORM_API_KEY: str

    class Config:
        env_file = ".env"

# Create a single, globally accessible settings instance
settings = Settings()