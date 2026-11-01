"""Launch the AI Assistant chat UI (Streamlit)."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]                   # src/ (holds the shared logging_config.py)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logging_config import setup_logging  # noqa: E402

setup_logging(reset=True)
logger = logging.getLogger("launch")

# launch.py and app.py live in the same folder (src/chatbot_ui).
APP_DIR = Path(__file__).resolve().parent
APP_PATH = APP_DIR / "app.py"


def main():
    parser = argparse.ArgumentParser(description="Open the AI Assistant chat interface.")
    parser.add_argument("--port", type=int, default=8501, help="port to serve on (default: 8501)")
    parser.add_argument("--headless", action="store_true", help="don't open a browser tab automatically")
    args = parser.parse_args()

    command = [
        sys.executable, "-m", "streamlit", "run", str(APP_PATH),
        "--server.port", str(args.port),
    ]
    if args.headless:
        command += ["--server.headless", "true"]

    logger.info("Launching Streamlit on port %s (headless=%s)", args.port, args.headless)
    print(f"Opening the AI Assistant on http://localhost:{args.port} ...")
    try:
        # cwd = this folder so app.py and its imports resolve from here
        subprocess.run(command, cwd=APP_DIR, check=True)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
        print("\nStopped.")
    except subprocess.CalledProcessError:
        logger.exception("Streamlit exited with an error")
        raise


if __name__ == "__main__":
    main()