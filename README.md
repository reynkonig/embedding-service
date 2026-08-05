# Universal embedding server

OpenAI-compatible HTTP service for one configured Hugging Face embedding model.
It supports CPU inference, one model replica per GPU, or a model sharded across
multiple GPUs.

## Docker

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f embedding-server
```

The API listens on `127.0.0.1:8091` by default. Model files are stored in the
`embedding-server-hf-cache` Docker volume.

Configure replicas with `EMBEDDING_DEVICES`:

```dotenv
# One GPU
EMBEDDING_DEVICES=cuda:0

# Three replicas
EMBEDDING_DEVICES=cuda:0,cuda:1,cuda:2
```

To shard one model across all visible GPUs:

```dotenv
EMBEDDING_DEVICES=auto
EMBEDDING_DEVICE_MAP=auto
```

Replicas increase throughput; sharding increases the memory available to one
model. Do not increase the Uvicorn worker count because every worker creates its
own model pool.

Docker GPU access requires the NVIDIA Container Toolkit.

## Local run

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2 \
EMBEDDING_DIMENSIONS=384 \
uv run embedding-server --devices cpu
```

## API

```bash
curl http://127.0.0.1:8091/healthz

curl http://127.0.0.1:8091/readyz

curl http://127.0.0.1:8091/v1/embeddings \
  -H "Authorization: Bearer $EMBEDDING_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input_type": "document",
    "input": ["First text", "Second text"]
  }'

curl -X POST http://127.0.0.1:8091/admin/warmup \
  -H "Authorization: Bearer $EMBEDDING_API_KEY"
```

`GET /v1/models` returns the configured model and requires authentication.
Health checks do not load the model. Readiness returns HTTP 503 until a worker
is available.

## Configuration

Each deployment serves one fixed model. `EMBEDDING_MODEL_ID` and
`EMBEDDING_DIMENSIONS` are required.

| Variable | Purpose |
| --- | --- |
| `EMBEDDING_MODEL_ID` | Hugging Face repository or local path |
| `EMBEDDING_MODEL_REVISION` | Model revision |
| `EMBEDDING_DIMENSIONS` | Output vector size |
| `EMBEDDING_BACKEND` | `native`, `mean`, `cls`, `last_token`, or `max` |
| `EMBEDDING_QUERY_PREFIX` | Prefix for query input |
| `EMBEDDING_DOCUMENT_PREFIX` | Prefix for document input |
| `EMBEDDING_NORMALIZE` | L2-normalize output vectors |
| `EMBEDDING_DEVICES` | Replica devices |
| `EMBEDDING_DEVICE_MAP` | Transformers device map for one sharded model |
| `EMBEDDING_DTYPE` | `auto`, `float32`, `float16`, or `bfloat16` |
| `EMBEDDING_MAX_LENGTH` | Tokenizer length limit |
| `EMBEDDING_PRELOAD` | Load replicas during startup |
| `EMBEDDING_API_KEY` | Bearer token required for non-loopback binding |

`mean` is the default backend. Use `native` only when the model returns a pooled
embedding from `return_embeddings=True`.

Remote model code is disabled. Enabling `EMBEDDING_TRUST_REMOTE_CODE` requires
`EMBEDDING_MODEL_REVISION` to be a full 40-character commit SHA.

Changing the model, revision, dimensions, backend, prefixes, or normalization
requires rebuilding existing vector indexes.

Use TLS or an SSH tunnel when exposing the service over a network.
