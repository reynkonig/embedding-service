from embedding_service.api import create_app
from embedding_service.cli import main
from embedding_service.config import ServerSettings

__all__ = [
    "ServerSettings",
    "create_app",
    "main",
]


if __name__ == "__main__":
    main()
