"""
Entry point for running the MCP server as a module.

This allows the MCP server to be run with:
    python -m code_indexer.apps.mcp
"""

from .main import main

if __name__ == "__main__":
    main()
