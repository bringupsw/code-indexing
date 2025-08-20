"""
MCP (Model Context Protocol) Server Application

This module provides an MCP server that exposes the semantic code indexer
functionality to Claude Desktop and other MCP-compatible clients.

The server provides tools for:
- Indexing Python codebases with semantic analysis
- Searching code using natural language queries
- Generating code analytics and insights
- Finding similar code patterns
- Managing multiple project indexes
"""

from .main import main

__all__ = ["main"]
