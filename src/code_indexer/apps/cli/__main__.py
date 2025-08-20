"""
Entry point for running the code indexer as a module.

This allows the CLI to be run with:
    python -m code_indexer.apps.cli
"""

from .main import main

if __name__ == "__main__":
    main()
