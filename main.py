import argparse

import uvicorn

from litert_proxy import config
from litert_proxy.config import TOOL_CONTEXT_MODE, TOOL_CONTEXT_MODES


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI-compatible LiteRT-LM server with selectable "
            "tool-context handling."
        )
    )
    parser.add_argument(
        "--tool-context-mode",
        choices=sorted(TOOL_CONTEXT_MODES),
        default=TOOL_CONTEXT_MODE,
        help=(
            "merged keeps one live KV cache across tool calls; "
            "separate rebuilds a fresh full-history context for "
            "each tool turn. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--no-web-ui",
        action="store_true",
        help="Disable the built-in browser chat interface.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config.TOOL_CONTEXT_MODE = args.tool_context_mode
    config.WEB_UI_ENABLED = config.WEB_UI_ENABLED and not args.no_web_ui

    from litert_proxy.server import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )
