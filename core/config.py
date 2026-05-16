from pydantic_settings import BaseSettings


# OopCompanion:suppressRename


class Settings(BaseSettings):
    # Taille maximale des images acceptées par l'API (10 MB)
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024

    class Config:
        env_file = ".env"


settings = Settings()
