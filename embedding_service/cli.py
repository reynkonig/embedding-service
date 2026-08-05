from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from .api import create_app
from .config import (
    SUPPORTED_ATTENTION,
    SUPPORTED_BACKENDS,
    SUPPORTED_DTYPES,
    ServerSettings,
)
from .pool import InProcessBackend


def _device_list(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    return devices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one fixed Hugging Face embedding model through an "
        "OpenAI-compatible API",
    )
    parser.add_argument("--host", help="listen address (environment default)")
    parser.add_argument("--port", type=int, help="listen port")
    parser.add_argument("--model", "--model-id", dest="model_id")
    parser.add_argument("--revision", "--model-revision", dest="model_revision")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--backend", choices=sorted(SUPPORTED_BACKENDS))
    parser.add_argument("--query-prefix")
    parser.add_argument("--document-prefix")
    parser.add_argument(
        "--devices",
        "--device",
        dest="devices",
        type=_device_list,
        metavar="DEVICE[,DEVICE...]",
        help="replica devices, for example cuda:0,cuda:1",
    )
    parser.add_argument(
        "--device-map",
        help="Transformers device_map for one sharded worker, usually auto",
    )
    parser.add_argument("--dtype", choices=sorted(SUPPORTED_DTYPES))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--attn-implementation",
        choices=sorted(SUPPORTED_ATTENTION),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="load all replicas before accepting normal traffic",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="debug mode for one device; disables process isolation",
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    return parser


def settings_from_args(args: argparse.Namespace) -> ServerSettings:
    updates: dict[str, object] = {}
    fields = (
        "host",
        "port",
        "model_id",
        "model_revision",
        "dimensions",
        "backend",
        "query_prefix",
        "document_prefix",
        "devices",
        "device_map",
        "dtype",
        "batch_size",
        "max_length",
        "attn_implementation",
        "trust_remote_code",
        "local_files_only",
        "preload",
    )
    for field in fields:
        value = getattr(args, field)
        if value is not None:
            updates[field] = value
    if args.devices is not None and args.device_map is None:
        updates["device_map"] = None
    elif args.device_map is not None and args.devices is None:
        updates["devices"] = ("auto",)
    if args.normalize is not None:
        updates["normalize_embeddings"] = args.normalize

    return ServerSettings.from_env(**updates)


def main(argv: Sequence[str] | None = None) -> None:
    import uvicorn

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        backend = InProcessBackend(settings) if args.in_process else None
    except ValueError as error:
        parser.error(str(error))

    uvicorn.run(
        create_app(settings, backend=backend),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
