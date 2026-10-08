"""Allow ``python -m app`` to launch the desktop application."""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
