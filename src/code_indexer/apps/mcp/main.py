#!/usr/bin/env python3
"""
MCP Server for Semantic Code Indexer

This server exposes the semantic code indexer capabilities to Claude Desktop
via the Model Context Protocol (MCP).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
except ImportError:
    print("MCP not installed. Install with: pip install mcp")
    print("Install with: pip install mcp")
    sys.exit(1)

# Import our semantic code indexer
from ...libs.common.pipeline import CodebaseSemanticPipeline
from ...libs.common.models import CodeEntity
from ...libs.common.embeddings import CodeEmbeddingGenerator

# Import setup functionality
from .setup_claude import setup_claude_desktop

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-code-indexer-mcp")

# Global pipeline instance
pipeline: Optional[CodebaseSemanticPipeline] = None
current_index_path: Optional[str] = None

# MCP Server instance
server = Server("semantic-code-indexer")


def parse_arguments():
    """Parse command line arguments for the MCP server."""
    parser = argparse.ArgumentParser(
        description="Semantic Code Indexer MCP Server - Expose code analysis to Claude Desktop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the MCP server (typically called by Claude Desktop)
  code-indexer-mcp

  # Test the server locally
  code-indexer-mcp --test

  # Setup Claude Desktop configuration
  code-indexer-mcp --setup-claude-desktop

Configuration:
  The MCP server is typically configured in Claude Desktop's settings:
  
  Location: ~/Library/Application Support/Claude/claude_desktop_config.json
  
  {
    "mcpServers": {
      "semantic-code-indexer": {
        "command": "code-indexer-mcp",
        "env": {
          "PYTHONPATH": "/path/to/your/project"
        }
      }
    }
  }

Tools Available (Read-Only):
  - get_project_context: Get information about the indexed project and check query relevance
  - search_code: Search code using natural language queries
  - ask_agent: Ask questions about your codebase
  - get_code_analytics: Generate comprehensive analytics
  - find_similar_code: Find similar code patterns
  - get_entity_details: Get detailed info about functions/classes
  - list_indexes: List available code indexes

Note: This server operates in read-only mode. Use the CLI tool to create indexes.

For more information, visit: https://github.com/bringupsw/code-indexing
""",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - run basic functionality checks",
    )

    parser.add_argument(
        "--setup-claude-desktop",
        action="store_true",
        help="Setup Claude Desktop configuration for the MCP server",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args()


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools for the semantic code indexer (read-only)."""
    return [
        Tool(
            name="search_code",
            description="Search the indexed codebase using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'authentication functions', 'database connections')",
                    },
                    "index_path": {
                        "type": "string",
                        "description": "Path to the index to search (optional, uses last indexed path if not specified)",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["semantic", "graph", "hybrid", "contextual"],
                        "description": "Type of search to perform",
                        "default": "hybrid",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "domain": {
                        "type": "string",
                        "description": "Filter by business domain (optional)",
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "Minimum importance score",
                        "default": 0.0,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask_agent",
            description="Ask natural language questions about the codebase",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the codebase",
                    },
                    "index_path": {
                        "type": "string",
                        "description": "Path to the index to query (optional, uses last indexed path if not specified)",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["natural", "structured", "json"],
                        "description": "Response format",
                        "default": "structured",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="get_code_analytics",
            description="Generate comprehensive analytics about the codebase",
            inputSchema={
                "type": "object",
                "properties": {
                    "index_path": {
                        "type": "string",
                        "description": "Path to the index to analyze (optional, uses last indexed path if not specified)",
                    },
                    "include_quality": {
                        "type": "boolean",
                        "description": "Include code quality metrics",
                        "default": True,
                    },
                    "include_security": {
                        "type": "boolean",
                        "description": "Include security analysis",
                        "default": True,
                    },
                    "include_architecture": {
                        "type": "boolean",
                        "description": "Include architectural patterns analysis",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="find_similar_code",
            description="Find code entities similar to a given function or class",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the function, class, or entity to find similar code for",
                    },
                    "index_path": {
                        "type": "string",
                        "description": "Path to the index to search (optional, uses last indexed path if not specified)",
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0 to 1.0)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of similar entities to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["entity_name"],
            },
        ),
        Tool(
            name="get_entity_details",
            description="Get detailed information about a specific code entity",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the entity to get details for",
                    },
                    "index_path": {
                        "type": "string",
                        "description": "Path to the index to search (optional, uses last indexed path if not specified)",
                    },
                    "include_source": {
                        "type": "boolean",
                        "description": "Include full source code",
                        "default": True,
                    },
                    "include_dependencies": {
                        "type": "boolean",
                        "description": "Include dependency information",
                        "default": True,
                    },
                },
                "required": ["entity_name"],
            },
        ),
        Tool(
            name="list_indexes",
            description="List available code indexes and show current status",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search for indexes (optional, defaults to current directory)",
                        "default": ".",
                    }
                },
            },
        ),
        Tool(
            name="get_project_context",
            description="Get information about what project is currently indexed and available for analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "check_query_relevance": {
                        "type": "string",
                        "description": "Optional query to check if it's relevant to the indexed project",
                    }
                },
            },
        ),
    ]


def ensure_pipeline(index_path: str = None) -> CodebaseSemanticPipeline:
    """Ensure we have a pipeline instance, using DEFAULT_INDEX_PATH if available."""
    global pipeline, current_index_path

    # Use DEFAULT_INDEX_PATH from environment if no specific path provided
    if not index_path:
        index_path = os.getenv("DEFAULT_INDEX_PATH")

    if index_path and index_path != current_index_path:
        current_index_path = index_path
        pipeline = None

    if pipeline is None:
        output_dir = index_path or "./code_index"
        pipeline = CodebaseSemanticPipeline(
            output_dir=output_dir, context_config_dir="./context_config"
        )
        current_index_path = output_dir

    return pipeline


def get_project_info(index_path: str = None) -> Dict[str, Any]:
    """Get project information from the index."""
    try:
        if not index_path:
            index_path = current_index_path or os.getenv("DEFAULT_INDEX_PATH", "./code_index")

        # Try to read summary.json for project info
        summary_path = os.path.join(index_path, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
                return {
                    "project_name": summary.get("project_name", "Unknown Project"),
                    "total_files": summary.get("total_files", 0),
                    "total_functions": summary.get("total_functions", 0),
                    "total_classes": summary.get("total_classes", 0),
                    "indexed_at": summary.get("created_at", "Unknown"),
                    "source_path": summary.get("source_path", "Unknown"),
                }
        else:
            # Fallback: extract project name from path
            project_name = os.path.basename(index_path.rstrip("/"))
            if project_name.startswith("code-indexed-"):
                project_name = project_name[13:]  # Remove "code-indexed-" prefix

            return {
                "project_name": project_name,
                "total_files": "Unknown",
                "total_functions": "Unknown",
                "total_classes": "Unknown",
                "indexed_at": "Unknown",
                "source_path": "Unknown",
            }
    except Exception as e:
        return {"project_name": "Unknown Project", "error": str(e)}


def is_query_relevant_to_project(query: str, project_info: Dict[str, Any]) -> tuple[bool, str]:
    """
    Check if a query is relevant to the indexed project and provide context.

    Args:
        query (str): The user's query
        project_info (Dict): Project information from summary.json

    Returns:
        tuple[bool, str]: (is_relevant, context_message)
    """
    project_name = project_info.get("project_name", "").lower()
    query_lower = query.lower()

    # Keywords that indicate code-related queries
    code_keywords = [
        "function",
        "class",
        "method",
        "code",
        "implementation",
        "algorithm",
        "authentication",
        "database",
        "api",
        "endpoint",
        "security",
        "login",
        "user",
        "admin",
        "permission",
        "model",
        "view",
        "controller",
        "service",
        "import",
        "module",
        "library",
        "dependency",
        "bug",
        "error",
        "exception",
        "test",
        "unit test",
        "integration",
        "performance",
        "optimization",
        "refactor",
        "design pattern",
        "architecture",
        "component",
        "interface",
    ]

    # Check if query mentions the specific project
    project_mentioned = project_name in query_lower if project_name else False

    # Check if query contains code-related keywords
    contains_code_keywords = any(keyword in query_lower for keyword in code_keywords)

    # Determine relevance
    if project_mentioned:
        return (
            True,
            f"✅ Query specifically mentions '{project_info['project_name']}' - this is the indexed project.",
        )
    elif contains_code_keywords:
        return (
            True,
            f"📋 This appears to be a code-related query. I have the '{project_info['project_name']}' codebase indexed and can help with questions about it.",
        )
    else:
        return (
            False,
            f"ℹ️ This query doesn't appear to be about code analysis. I have the '{project_info['project_name']}' codebase indexed if you'd like to ask questions about that project.",
        )


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls from Claude (read-only operations only)."""
    try:
        if name == "search_code":
            return await handle_search_code(arguments)
        elif name == "ask_agent":
            return await handle_ask_agent(arguments)
        elif name == "get_code_analytics":
            return await handle_get_analytics(arguments)
        elif name == "find_similar_code":
            return await handle_find_similar(arguments)
        elif name == "get_entity_details":
            return await handle_get_entity_details(arguments)
        elif name == "list_indexes":
            return await handle_list_indexes(arguments)
        elif name == "get_project_context":
            return await handle_get_project_context(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_search_code(args: Dict[str, Any]) -> List[TextContent]:
    """Handle code search."""
    query = args["query"]
    index_path = args.get("index_path")  # Optional index path
    search_type = args.get("search_type", "hybrid")
    top_k = args.get("top_k", 10)
    domain = args.get("domain")
    min_importance = args.get("min_importance", 0.0)

    try:
        pipeline = ensure_pipeline(index_path)

        # Check if we have an index to search
        if not pipeline or not hasattr(pipeline, "knowledge_graph"):
            index_location = index_path or current_index_path or "./code_index"
            return [
                TextContent(
                    type="text",
                    text=f"❌ No index found at '{index_location}'. Please index a codebase first using the 'index_codebase' tool.",
                )
            ]

        # Perform search
        results = pipeline.search(
            query=query,
            search_type=search_type,
            top_k=top_k,
            domain=domain,
            min_importance=min_importance,
        )

        if not results:
            return [TextContent(type="text", text=f"No results found for query: '{query}'")]

        # Format results
        index_location = (
            index_path or current_index_path or os.getenv("DEFAULT_INDEX_PATH", "./code_index")
        )
        project_info = get_project_info(index_location)

        response = f"🔍 **Search Results for**: '{query}'\n"
        response += f"� **Project**: {project_info['project_name']} | **Search Type**: {search_type} | **Results**: {len(results)}\n\n"

        for i, result in enumerate(results[:top_k], 1):
            entity = result.get("entity", {})
            score = result.get("score", 0)

            response += f"**{i}. {entity.get('name', 'Unknown')}** (Score: {score:.3f})\n"
            response += f"   📁 File: `{entity.get('file_path', 'N/A')}`\n"
            response += f"   🏷️ Type: {entity.get('type', 'N/A')}\n"

            if entity.get("docstring"):
                response += f"   📝 {entity['docstring'][:100]}...\n"

            if entity.get("business_domain"):
                response += f"   🏢 Domain: {entity['business_domain']}\n"

            response += "\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error searching code: {str(e)}")]


async def handle_ask_agent(args: Dict[str, Any]) -> List[TextContent]:
    """Handle natural language questions."""
    question = args["question"]
    index_path = args.get("index_path")  # Optional index path
    format_type = args.get("format", "structured")

    try:
        pipeline = ensure_pipeline(index_path)

        # Check if we have an index to query
        if not pipeline or not hasattr(pipeline, "knowledge_graph"):
            index_location = (
                index_path or current_index_path or os.getenv("DEFAULT_INDEX_PATH", "./code_index")
            )
            return [
                TextContent(
                    type="text",
                    text=f"❌ No index found at '{index_location}'. Please ensure the index exists and is accessible.",
                )
            ]

        # Get project info and check relevance
        index_location = (
            index_path or current_index_path or os.getenv("DEFAULT_INDEX_PATH", "./code_index")
        )
        project_info = get_project_info(index_location)
        is_relevant, relevance_msg = is_query_relevant_to_project(question, project_info)

        # Provide context if query seems irrelevant
        if not is_relevant:
            response = f"🤖 **AI Agent Response** | **Project**: {project_info['project_name']}\n\n"
            response += f"**Question**: {question}\n\n"
            response += f"{relevance_msg}\n\n"
            response += f"**What I can help with regarding {project_info['project_name']}**:\n"
            response += f"- Code structure and architecture questions\n"
            response += f"- Function and class explanations\n"
            response += f"- Security and authentication patterns\n"
            response += f"- Database and API implementations\n"
            response += f"- Code quality and complexity analysis\n"
            response += f"- Dependencies and imports\n\n"
            response += f"**Example questions**:\n"
            response += f"- \"How does authentication work in {project_info['project_name']}?\"\n"
            response += f'- "Show me the main API endpoints"\n'
            response += f'- "What are the most complex functions?"\n'
            response += f'- "Find database models and schemas"\n'

            return [TextContent(type="text", text=response)]

        # Query the AI agent
        agent_response = pipeline.query_agent(question, format_type=format_type)

        formatted_response = (
            f"🤖 **AI Agent Response** | **Project**: {project_info['project_name']}\n\n"
        )
        formatted_response += f"**Question**: {question}\n\n"
        formatted_response += f"{relevance_msg}\n\n"
        formatted_response += f"**Answer**:\n{agent_response}"

        return [TextContent(type="text", text=formatted_response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error querying agent: {str(e)}")]


async def handle_get_analytics(args: Dict[str, Any]) -> List[TextContent]:
    """Handle analytics generation."""
    index_path = args.get("index_path")  # Optional index path
    include_quality = args.get("include_quality", True)
    include_security = args.get("include_security", True)
    include_architecture = args.get("include_architecture", True)

    try:
        pipeline = ensure_pipeline(index_path)

        # Check if we have an index to analyze
        if not pipeline or not hasattr(pipeline, "knowledge_graph"):
            index_location = index_path or current_index_path or "./code_index"
            return [
                TextContent(
                    type="text",
                    text=f"❌ No index found at '{index_location}'. Please index a codebase first using the 'index_codebase' tool.",
                )
            ]

        # Generate analytics
        analytics = pipeline.get_contextual_analytics()

        index_location = index_path or current_index_path or "./code_index"
        response = f"📊 **Codebase Analytics** (Index: {index_location})\n\n"

        # Basic metrics
        if analytics.get("basic_metrics"):
            metrics = analytics["basic_metrics"]
            response += "**📈 Basic Metrics**:\n"
            response += f"- Total files: {metrics.get('total_files', 'N/A')}\n"
            response += f"- Total functions: {metrics.get('total_functions', 'N/A')}\n"
            response += f"- Total classes: {metrics.get('total_classes', 'N/A')}\n"
            response += f"- Average complexity: {metrics.get('avg_complexity', 'N/A')}\n\n"

        # Quality metrics
        if include_quality and analytics.get("quality_metrics"):
            quality = analytics["quality_metrics"]
            response += "**✨ Quality Metrics**:\n"
            response += (
                f"- High complexity functions: {quality.get('high_complexity_count', 'N/A')}\n"
            )
            response += (
                f"- Functions without docstrings: {quality.get('missing_docstrings', 'N/A')}\n"
            )
            response += f"- Code coverage estimate: {quality.get('coverage_estimate', 'N/A')}\n\n"

        # Security analysis
        if include_security and analytics.get("security_analysis"):
            security = analytics["security_analysis"]
            response += "**🔒 Security Analysis**:\n"
            response += (
                f"- Security-critical functions: {security.get('critical_functions', 'N/A')}\n"
            )
            response += f"- Potential vulnerabilities: {security.get('vulnerabilities', 'N/A')}\n\n"

        # Architecture patterns
        if include_architecture and analytics.get("architecture_patterns"):
            patterns = analytics["architecture_patterns"]
            response += "**🏗️ Architecture Patterns**:\n"
            for pattern, count in patterns.items():
                response += f"- {pattern}: {count}\n"
            response += "\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating analytics: {str(e)}")]


async def handle_find_similar(args: Dict[str, Any]) -> List[TextContent]:
    """Handle similarity search."""
    entity_name = args["entity_name"]
    index_path = args.get("index_path")  # Optional index path
    threshold = args.get("similarity_threshold", 0.7)
    max_results = args.get("max_results", 10)

    try:
        pipeline = ensure_pipeline(index_path)

        # Check if we have an index to search
        if not pipeline or not hasattr(pipeline, "knowledge_graph"):
            index_location = index_path or current_index_path or "./code_index"
            return [
                TextContent(
                    type="text",
                    text=f"❌ No index found at '{index_location}'. Please index a codebase first using the 'index_codebase' tool.",
                )
            ]

        # Find similar entities
        similar_entities = pipeline.find_similar_entities(
            entity_name=entity_name, threshold=threshold, max_results=max_results
        )

        if not similar_entities:
            return [
                TextContent(
                    type="text",
                    text=f"No similar entities found for '{entity_name}' with threshold {threshold}",
                )
            ]

        index_location = index_path or current_index_path or "./code_index"
        response = f"🔄 **Similar Code to**: '{entity_name}' (Index: {index_location})\n"
        response += f"🎯 **Threshold**: {threshold} | **Results**: {len(similar_entities)}\n\n"

        for i, (entity, similarity) in enumerate(similar_entities, 1):
            response += f"**{i}. {entity.get('name', 'Unknown')}** (Similarity: {similarity:.3f})\n"
            response += f"   📁 File: `{entity.get('file_path', 'N/A')}`\n"
            response += f"   🏷️ Type: {entity.get('type', 'N/A')}\n"

            if entity.get("docstring"):
                response += f"   📝 {entity['docstring'][:80]}...\n"

            response += "\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error finding similar code: {str(e)}")]


async def handle_get_entity_details(args: Dict[str, Any]) -> List[TextContent]:
    """Handle entity details request."""
    entity_name = args["entity_name"]
    index_path = args.get("index_path")  # Optional index path
    include_source = args.get("include_source", True)
    include_dependencies = args.get("include_dependencies", True)

    try:
        pipeline = ensure_pipeline(index_path)

        # Check if we have an index to search
        if not pipeline or not hasattr(pipeline, "knowledge_graph"):
            index_location = index_path or current_index_path or "./code_index"
            return [
                TextContent(
                    type="text",
                    text=f"❌ No index found at '{index_location}'. Please index a codebase first using the 'index_codebase' tool.",
                )
            ]

        # Get entity details
        entity_details = pipeline.get_entity_details(
            entity_name=entity_name,
            include_source=include_source,
            include_dependencies=include_dependencies,
        )

        if not entity_details:
            return [TextContent(type="text", text=f"Entity '{entity_name}' not found in the index")]

        entity = entity_details
        index_location = index_path or current_index_path or "./code_index"
        response = (
            f"📋 **Entity Details**: {entity.get('name', 'Unknown')} (Index: {index_location})\n\n"
        )

        # Basic info
        response += f"**🏷️ Type**: {entity.get('type', 'N/A')}\n"
        response += f"**📁 File**: `{entity.get('file_path', 'N/A')}`\n"
        response += (
            f"**📍 Lines**: {entity.get('line_start', 'N/A')}-{entity.get('line_end', 'N/A')}\n"
        )
        response += f"**🔢 Complexity**: {entity.get('complexity', 'N/A')}\n"

        if entity.get("business_domain"):
            response += f"**🏢 Domain**: {entity['business_domain']}\n"

        if entity.get("docstring"):
            response += f"\n**📝 Documentation**:\n{entity['docstring']}\n"

        if entity.get("parameters"):
            response += f"\n**⚙️ Parameters**: {', '.join(entity['parameters'])}\n"

        if include_dependencies and entity.get("imports"):
            response += f"\n**📦 Imports**: {', '.join(entity['imports'])}\n"

        if include_source and entity.get("source_code"):
            response += f"\n**💻 Source Code**:\n```python\n{entity['source_code']}\n```\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting entity details: {str(e)}")]


async def handle_list_indexes(args: Dict[str, Any]) -> List[TextContent]:
    """Handle listing available indexes with project information."""
    directory = args.get("directory", ".")

    try:
        import glob
        from pathlib import Path

        # Look for index directories (containing knowledge.db files)
        search_pattern = os.path.join(directory, "**/knowledge.db")
        index_files = glob.glob(search_pattern, recursive=True)

        response = f"📂 **Available Code Indexes** (Read-Only Mode)\n\n"

        # Show default index from environment
        default_index = os.getenv("DEFAULT_INDEX_PATH")
        if default_index:
            project_info = get_project_info(default_index)
            response += f"🎯 **Default Index**: {default_index}\n"
            response += f"📋 **Project**: {project_info['project_name']}\n\n"

        if current_index_path and current_index_path != default_index:
            response += f"🔄 **Current Index**: {current_index_path}\n\n"

        if not index_files:
            response += f"❌ No indexes found in '{directory}'\n"
            response += f"💡 Indexes must be created externally using the CLI tool.\n"
        else:
            response += f"Found {len(index_files)} index(es):\n\n"

            for i, index_file in enumerate(index_files, 1):
                index_dir = os.path.dirname(index_file)
                project_info = get_project_info(index_dir)

                # Get file stats
                stat = os.stat(index_file)
                size_mb = stat.st_size / (1024 * 1024)
                modified = Path(index_file).stat().st_mtime
                import datetime

                mod_time = datetime.datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")

                # Check if this is the current or default index
                markers = []
                if index_dir == current_index_path:
                    markers.append("🔄 Current")
                if index_dir == default_index:
                    markers.append("🎯 Default")
                marker_text = f" ({', '.join(markers)})" if markers else ""

                response += f"**{i}. {project_info['project_name']}**{marker_text}\n"
                response += f"   📁 Path: {index_dir}\n"
                response += f"   📊 Size: {size_mb:.1f} MB | Modified: {mod_time}\n"

                if project_info.get("total_files") != "Unknown":
                    response += f"   � Files: {project_info['total_files']} | Functions: {project_info['total_functions']} | Classes: {project_info['total_classes']}\n"

                if project_info.get("source_path") != "Unknown":
                    response += f"   � Source: {project_info['source_path']}\n"

                response += "\n"

        response += f"\n💡 **Usage Tips** (Read-Only Mode):\n"
        response += f"• Use `index_path` parameter in search tools to specify which index to use\n"
        response += f"• If no `index_path` is specified, the default index is used\n"
        response += f"• This server is read-only - use the CLI tool to create new indexes\n"
        response += f"• Set DEFAULT_INDEX_PATH environment variable to specify default index\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing indexes: {str(e)}")]


async def handle_get_project_context(args: Dict[str, Any]) -> List[TextContent]:
    """Handle project context information request."""
    check_query = args.get("check_query_relevance")

    try:
        # Get default index path
        default_index = os.getenv("DEFAULT_INDEX_PATH")
        current_project_info = None

        if default_index:
            current_project_info = get_project_info(default_index)
        elif current_index_path:
            current_project_info = get_project_info(current_index_path)

        if not current_project_info:
            return [
                TextContent(
                    type="text",
                    text="❌ No project is currently indexed or available. Please use the CLI tool to index a codebase first.",
                )
            ]

        response = f"📋 **Current Project Context**\n\n"
        response += f"**Project**: {current_project_info['project_name']}\n"

        if current_project_info.get("source_path") != "Unknown":
            response += f"**Source**: {current_project_info['source_path']}\n"

        if current_project_info.get("indexed_at") != "Unknown":
            response += f"**Indexed**: {current_project_info['indexed_at']}\n"

        if current_project_info.get("total_files") != "Unknown":
            response += f"\n**Statistics**:\n"
            response += f"- Files: {current_project_info['total_files']}\n"
            response += f"- Functions: {current_project_info['total_functions']}\n"
            response += f"- Classes: {current_project_info['total_classes']}\n"

        response += f"\n**Available Tools**:\n"
        response += f"- `search_code`: Search for specific code patterns\n"
        response += f"- `ask_agent`: Ask natural language questions about the code\n"
        response += f"- `get_code_analytics`: Get code quality and complexity metrics\n"
        response += f"- `find_similar_code`: Find similar functions or classes\n"
        response += (
            f"- `get_entity_details`: Get detailed information about specific functions/classes\n"
        )

        # Check query relevance if provided
        if check_query:
            is_relevant, relevance_msg = is_query_relevant_to_project(
                check_query, current_project_info
            )
            response += f"\n**Query Relevance Check**:\n"
            response += f'Query: "{check_query}"\n'
            response += f"{relevance_msg}\n"

            if not is_relevant:
                response += f"\n💡 **Suggestion**: Try asking about:\n"
                response += f"- \"Show me authentication functions in {current_project_info['project_name']}\"\n"
                response += f'- "What are the most complex functions in this codebase?"\n'
                response += f'- "Find database-related code"\n'
                response += f'- "Show me the main API endpoints"\n'
        else:
            response += f"\n💡 **Example Questions**:\n"
            response += f"- \"Show me authentication functions in {current_project_info['project_name']}\"\n"
            response += f'- "What are the most complex functions?"\n'
            response += f'- "Find database-related code"\n'
            response += f'- "Show me security-critical functions"\n'

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting project context: {str(e)}")]


async def run_server():
    """Run the MCP server."""
    logger.info("Starting Semantic Code Indexer MCP Server")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def test_mode():
    """Test the MCP server functionality."""
    print("🧪 Testing Semantic Code Indexer MCP Server")
    print("=" * 50)

    # Test imports
    try:
        from ...libs.common.pipeline import CodebaseSemanticPipeline

        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Test MCP imports
    try:
        from mcp.server import Server

        print("✅ MCP imports successful")
    except ImportError as e:
        print(f"❌ MCP import error: {e}")
        print("   Install with: pip install mcp")
        return False

    # Test tool listing
    try:
        import asyncio

        tools = asyncio.run(list_tools())
        print(f"✅ Found {len(tools)} tools available")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"❌ Tool listing error: {e}")
        return False

    print("\n✅ MCP server is ready for Claude Desktop!")
    return True


def main():
    """Main entry point for the MCP server CLI."""
    args = parse_arguments()

    # Set up logging
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.test:
        success = test_mode()
        sys.exit(0 if success else 1)
    elif args.setup_claude_desktop:
        success = setup_claude_desktop()
        sys.exit(0 if success else 1)
    else:
        # Run the MCP server
        asyncio.run(run_server())


if __name__ == "__main__":
    main()
