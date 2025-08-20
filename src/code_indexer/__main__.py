"""
Code Indexer - Entry point for running as a module.

This allows the entire package to be run with:
    python -m code_indexer
"""

from .apps.cli.main import main

if __name__ == "__main__":
    main()
