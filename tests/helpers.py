from embedding_service.config import ServerSettings

TEST_MODEL_ID = "example/embedding-model"
TEST_MODEL_REVISION = "test-revision"
TEST_DIMENSIONS = 2


def make_settings(**overrides: object) -> ServerSettings:
    values: dict[str, object] = {
        "model_id": TEST_MODEL_ID,
        "dimensions": TEST_DIMENSIONS,
    }
    values.update(overrides)
    return ServerSettings(**values)  # type: ignore[arg-type]
