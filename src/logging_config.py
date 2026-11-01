import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

# libraries that write a lot of DEBUG messages we don't need
NOISY_LIBRARIES = [
    "httpx", "httpcore", "urllib3", "chromadb", "posthog",
    "watchdog", "asyncio", "PIL", "matplotlib",
]


def setup_logging(reset=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if reset:
        LOG_FILE.write_text("", encoding="utf-8")  # start a fresh log file

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)