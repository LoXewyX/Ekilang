"""Entry point for running Ekilang as a module (python -m ekilang)."""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
