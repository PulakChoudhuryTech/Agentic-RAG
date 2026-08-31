"""Basic logging setup. DEBUG level (verbose trace output) when DEBUG_MODE
is on, INFO otherwise."""

from __future__ import annotations

import logging

from backend.app.config import Settings


def configure_logging(settings: Settings) -> None:
    level = logging.DEBUG if settings.debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
