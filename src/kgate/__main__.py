"""Allow running the CLI with ``python -m kgate``."""

import sys

from kgate.cli import main

if __name__ == "__main__":
    sys.exit(main())
