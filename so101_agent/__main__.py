from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from dotenv import load_dotenv

from .config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SO101 vision-and-voice agent.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config. Defaults to config.toml or SO101_CONFIG.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    settings = load_settings(args.config)
    os.environ["SO101_CONFIG"] = str(settings.config_path)
    load_dotenv(settings.root / ".env")

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "so101_agent.server:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=args.log_level,
        factory=False,
    )


if __name__ == "__main__":
    main()

